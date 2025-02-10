import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
import logging
from .preprocessing import MetricPreprocessor, LogPreprocessor, create_sliding_windows

logger = logging.getLogger(__name__)

class OCEANDataset(Dataset):
    """
    Dataset class for OCEAN model
    Handles both historical and streaming data
    """
    def __init__(
        self,
        metrics_data: np.ndarray,
        logs_data: np.ndarray,
        kpi_data: np.ndarray,
        window_size: int = 100,
        stride: int = 10,
        device: str = 'cuda'
    ):
        """
        Initialize dataset
        Args:
            metrics_data: Metrics array (n_entities, n_metrics, sequence_length)
            logs_data: Logs array (n_entities, n_logs, sequence_length)
            kpi_data: KPI array (sequence_length,)
            window_size: Size of sliding windows
            stride: Stride between windows
            device: Device to store tensors on
        """
        self.device = device
        
        # Create sliding windows
        self.metrics_windows = create_sliding_windows(metrics_data, window_size, stride)
        self.logs_windows = create_sliding_windows(logs_data, window_size, stride)
        
        # Handle KPI data
        n_windows = len(self.metrics_windows)
        self.kpi_indices = np.arange(window_size - 1, window_size - 1 + n_windows * stride, stride)
        self.kpi_values = kpi_data[self.kpi_indices]
        
    def __len__(self) -> int:
        return len(self.metrics_windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metrics = torch.FloatTensor(self.metrics_windows[idx]).to(self.device)
        logs = torch.FloatTensor(self.logs_windows[idx]).to(self.device)
        kpi = torch.FloatTensor([self.kpi_values[idx]]).to(self.device)
        
        return metrics, logs, kpi

class OCEANDataLoader:
    """
    Data loader class for OCEAN model
    Handles data loading, preprocessing, and batch creation
    """
    def __init__(
        self,
        data_dir: str,
        window_size: int = 100,
        stride: int = 10,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = 'cuda'
    ):
        """
        Initialize data loader
        Args:
            data_dir: Directory containing data files
            window_size: Size of sliding windows
            stride: Stride between windows
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes
            device: Device to store tensors on
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        # Initialize preprocessors
        self.metric_preprocessor = MetricPreprocessor(
            window_size=window_size,
            stride=stride
        )
        self.log_preprocessor = LogPreprocessor(
            window_size=window_size,
            stride=stride
        )
        
        # Load and preprocess historical data
        self._load_historical_data()
        
    def _load_historical_data(self) -> None:
        """Load and preprocess historical data"""
        # Load historical data
        metrics_df = pd.read_csv(self.data_dir / 'historical_metrics.csv')
        logs_df = pd.read_csv(self.data_dir / 'historical_logs.csv')
        kpi_df = pd.read_csv(self.data_dir / 'historical_kpi.csv')
        
        # Fit preprocessors
        self.metric_preprocessor.fit(metrics_df)
        self.log_preprocessor.fit(logs_df)
        
        # Transform data
        self.historical_metrics = self.metric_preprocessor.transform(metrics_df)
        self.historical_logs = self.log_preprocessor.transform(logs_df)
        self.historical_kpi = kpi_df['value'].values
        
        logger.info(f"Loaded historical data: {self.historical_metrics.shape[0]} entities, "
                   f"{self.historical_metrics.shape[1]} metrics, {self.historical_metrics.shape[2]} timestamps")
    
    def get_historical_dataloader(self) -> DataLoader:
        """Create DataLoader for historical data"""
        dataset = OCEANDataset(
            metrics_data=np.array(self.historical_metrics),
            logs_data=np.array(self.historical_logs),
            kpi_data=np.array(self.historical_kpi),
            window_size=self.window_size,
            stride=self.stride,
            device=self.device
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def process_streaming_batch(
        self,
        metrics_df: pd.DataFrame,
        logs_df: pd.DataFrame,
        kpi_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process streaming batch of data
        Args:
            metrics_df: DataFrame with current metrics
            logs_df: DataFrame with current logs
            kpi_value: Current KPI value
        Returns:
            Tuple of (metrics_tensor, logs_tensor, kpi_tensor)
        """
        # Preprocess metrics and logs
        metrics_array = self.metric_preprocessor.transform(metrics_df)
        logs_array = self.log_preprocessor.transform(logs_df)
        
        # Convert to tensors
        metrics_tensor = torch.FloatTensor(metrics_array).to(self.device)
        logs_tensor = torch.FloatTensor(logs_array).to(self.device)
        kpi_tensor = torch.FloatTensor([kpi_value]).to(self.device)
        
        return metrics_tensor, logs_tensor, kpi_tensor
    
    @property
    def n_entities(self) -> int:
        """Number of system entities"""
        return self.historical_metrics.shape[0]
    
    @property
    def n_metrics(self) -> int:
        """Number of metric features"""
        return self.historical_metrics.shape[1]
    
    @property
    def n_logs(self) -> int:
        """Number of log features"""
        return self.historical_logs.shape[1]
