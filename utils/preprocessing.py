import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricPreprocessor:
    """
    Preprocessor for system metrics data
    Handles normalization, missing values, and outliers
    """
    def __init__(
        self,
        window_size: int = 100,
        stride: int = 10,
        normalize: bool = True,
        outlier_threshold: float = 3.0
    ):
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.outlier_threshold = outlier_threshold
        self.scalers: Dict[str, StandardScaler] = {}
        
    def fit(self, metrics_df: pd.DataFrame) -> None:
        """
        Fit preprocessor on training data
        Args:
            metrics_df: DataFrame with metrics (timestamp, entity_id, metric1, metric2, ...)
        """
        if self.normalize:
            for column in metrics_df.select_dtypes(include=[np.number]).columns:
                if column not in ['timestamp', 'entity_id']:
                    scaler = StandardScaler()
                    scaler.fit(np.array(metrics_df[column]).reshape(-1, 1))
                    self.scalers[column] = scaler
    
    def handle_missing_values(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in metrics data"""
        # Forward fill then backward fill
        metrics_df = metrics_df.ffill().bfill()
        
        # If still has missing values, fill with zeros
        if metrics_df.isna().any().any():
            logger.warning("Some missing values couldn't be interpolated, filling with zeros")
            metrics_df = metrics_df.fillna(0)
            
        return metrics_df
    
    def remove_outliers(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method"""
        for column in metrics_df.select_dtypes(include=[np.number]).columns:
            if column not in ['timestamp', 'entity_id']:
                z_scores = np.abs((metrics_df[column] - metrics_df[column].mean()) / metrics_df[column].std())
                metrics_df.loc[z_scores > self.outlier_threshold, column] = metrics_df[column].mean()
        return metrics_df
    
    def transform(self, metrics_df: pd.DataFrame) -> np.ndarray:
        """
        Transform metrics data
        Args:
            metrics_df: DataFrame with metrics
        Returns:
            Preprocessed metrics array of shape (n_entities, n_metrics, sequence_length)
        """
        # Handle missing values
        metrics_df = self.handle_missing_values(metrics_df)
        
        # Remove outliers
        metrics_df = self.remove_outliers(metrics_df)
        
        # Normalize if required
        if self.normalize:
            for column, scaler in self.scalers.items():
                metrics_df[column] = scaler.transform(np.array(metrics_df[column]).reshape(-1, 1))
        
        # Reshape to required format
        n_entities = metrics_df['entity_id'].nunique()
        n_metrics = len(metrics_df.columns) - 2  # Excluding timestamp and entity_id
        sequence_length = len(metrics_df) // n_entities
        
        metrics_array = metrics_df.drop(['timestamp', 'entity_id'], axis=1).values
        metrics_array = metrics_array.reshape(n_entities, sequence_length, n_metrics)
        metrics_array = metrics_array.transpose(0, 2, 1)  # (n_entities, n_metrics, sequence_length)
        
        return metrics_array

class LogPreprocessor:
    """
    Preprocessor for system logs data
    Handles log parsing, feature extraction, and normalization
    """
    def __init__(
        self,
        window_size: int = 100,
        stride: int = 10,
        normalize: bool = True,
        max_features: int = 100
    ):
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.max_features = max_features
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        
    def extract_log_features(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from log data
        - Log frequency
        - Error/Warning counts
        - Golden signals (error, exception, critical, fatal)
        """
        features_df = pd.DataFrame()
        features_df['timestamp'] = logs_df['timestamp'].unique()
        
        # Group by timestamp and entity_id
        grouped = logs_df.groupby(['timestamp', 'entity_id'])
        
        # Log frequency
        features_df['log_frequency'] = grouped.size().unstack(fill_value=0)
        
        # Error/Warning counts
        error_patterns = ['error', 'exception', 'critical', 'fatal', 'failed', 'timeout']
        for pattern in error_patterns:
            col_name = f'{pattern}_count'
            features_df[col_name] = grouped.apply(
                lambda x: x['message'].str.contains(pattern, case=False).sum()
            ).unstack(fill_value=0)
            
        return features_df
    
    def fit(self, logs_df: pd.DataFrame) -> None:
        """
        Fit preprocessor on training data
        Args:
            logs_df: DataFrame with logs (timestamp, entity_id, message, level, ...)
        """
        # Extract features
        features_df = self.extract_log_features(logs_df)
        self.feature_columns = features_df.columns.tolist()
        
        # Fit scaler
        if self.normalize:
            self.scaler.fit(features_df[self.feature_columns].values)
    
    def transform(self, logs_df: pd.DataFrame) -> np.ndarray:
        """
        Transform logs data
        Args:
            logs_df: DataFrame with logs
        Returns:
            Preprocessed logs array of shape (n_entities, n_features, sequence_length)
        """
        # Extract features
        features_df = self.extract_log_features(logs_df)
        
        # Normalize if required
        if self.normalize:
            features_df[self.feature_columns] = self.scaler.transform(
                features_df[self.feature_columns].values
            )
        
        # Reshape to required format
        n_entities = len(features_df.columns) // len(self.feature_columns)
        n_features = len(self.feature_columns)
        sequence_length = len(features_df)
        
        logs_array = features_df[self.feature_columns].values
        logs_array = logs_array.reshape(sequence_length, n_entities, n_features)
        logs_array = logs_array.transpose(1, 2, 0)  # (n_entities, n_features, sequence_length)
        
        return logs_array

def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    stride: int
) -> np.ndarray:
    """
    Create sliding windows from data
    Args:
        data: Input array of shape (n_entities, n_features, sequence_length)
        window_size: Size of each window
        stride: Stride between windows
    Returns:
        Array of windows of shape (n_windows, n_entities, n_features, window_size)
    """
    n_entities, n_features, sequence_length = data.shape
    n_windows = (sequence_length - window_size) // stride + 1
    
    windows = np.zeros((n_windows, n_entities, n_features, window_size))
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[:, :, start_idx:end_idx]
        
    return windows
