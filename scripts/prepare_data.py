import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import zipfile
import tarfile
from typing import List
from pydantic import BaseModel, Field

from config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetConfig(BaseModel):
    """Configuration for dataset preparation"""
    name: str
    url: str
    file_type: str = Field(..., description="File type (csv/zip/tar)")
    metrics_columns: List[str]
    log_columns: List[str]
    kpi_threshold: float = Field(90.0, description="Threshold for KPI anomaly")

class DatasetPreparation:
    """
    Prepares datasets used in the OCEAN paper:
    1. AIOps Challenge Dataset (KPI anomaly detection)
    2. Azure Public Dataset (Cloud service monitoring)
    3. Alibaba Cluster Trace (Container monitoring)
    """
    
    DATASET_CONFIGS = {
        'aiops': DatasetConfig(
            name='aiops',
            url="https://github.com/NetManAIOps/KPI-Anomaly-Detection/raw/master/data/kpi_data.csv",
            file_type='csv',
            metrics_columns=['value'],
            log_columns=[],
            kpi_threshold=0.5
        ),
        'azure': DatasetConfig(
            name='azure',
            url="https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdataset/trace_data.zip",
            file_type='zip',
            metrics_columns=['cpu_usage', 'memory_usage'],
            log_columns=['status'],
            kpi_threshold=90.0
        ),
        'alibaba': DatasetConfig(
            name='alibaba',
            url="https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/trace_2018.tar.gz",
            file_type='tar',
            metrics_columns=['cpu_util', 'mem_util', 'disk_util'],
            log_columns=['event_type', 'status'],
            kpi_threshold=90.0
        )
    }
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.setup()  # Setup directories
            
    def download_file(self, url: str, target_path: Path) -> None:
        """Download file from URL with progress bar"""
        if target_path.exists():
            logger.info(f"File already exists: {target_path}")
            return
            
        logger.info(f"Downloading {url} to {target_path}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                total_size = total_size // 1024  # KB
                for data in response.iter_content(chunk_size=1024):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded // total_size)
                    print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded//1024}KB/{total_size//1024}KB", end='')
        print()
        
    def prepare_dataset(self, dataset_name: str) -> None:
        """Prepare specified dataset"""
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        dataset_config = self.DATASET_CONFIGS[dataset_name]
        dataset_dir = self.config.paths.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Download dataset
        target_path = dataset_dir / f"raw_data.{dataset_config.file_type}"
        self.download_file(dataset_config.url, target_path)
        
        # Extract if needed
        if dataset_config.file_type == 'zip':
            with zipfile.ZipFile(target_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
        elif dataset_config.file_type == 'tar':
            with tarfile.open(target_path, 'r:gz') as tar:
                tar.extractall(dataset_dir)
                
        # Process dataset based on type
        if dataset_name == 'aiops':
            self._prepare_aiops(dataset_dir, dataset_config)
        elif dataset_name == 'azure':
            self._prepare_azure(dataset_dir, dataset_config)
        elif dataset_name == 'alibaba':
            self._prepare_alibaba(dataset_dir, dataset_config)
            
        logger.info(f"{dataset_name} dataset preparation completed")
        
    def _prepare_aiops(self, dataset_dir: Path, config: DatasetConfig) -> None:
        """Prepare AIOps dataset"""
        df = pd.read_csv(dataset_dir / "raw_data.csv")
        
        # Split into metrics, logs, and KPI
        metrics_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'entity_id': df['KPI ID'],
            'value': df['value']
        })
        
        # Create dummy logs (since original dataset doesn't have logs)
        logs_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'entity_id': df['KPI ID'],
            'message': 'System status normal',
            'level': 'INFO'
        })
        
        # Create KPI values (using anomaly labels as KPI)
        kpi_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'value': df['label']
        })
        
        self._save_processed_data(dataset_dir, metrics_df, logs_df, kpi_df, df)
        
    def _prepare_azure(self, dataset_dir: Path, config: DatasetConfig) -> None:
        """Prepare Azure dataset"""
        vm_df = pd.read_csv(dataset_dir / 'vm_cpu_readings.csv')
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'timestamp': vm_df['timestamp'],
            'entity_id': vm_df['vm_id'],
            'cpu_usage': vm_df['cpu_usage'],
            'memory_usage': vm_df['memory_usage']
        })
        
        # Create logs from VM status changes
        status_df = pd.read_csv(dataset_dir / 'vm_status_changes.csv')
        logs_df = pd.DataFrame({
            'timestamp': status_df['timestamp'],
            'entity_id': status_df['vm_id'],
            'message': status_df['status'],
            'level': 'INFO'
        })
        
        # Create KPI values (using CPU threshold violations as KPI)
        kpi_df = pd.DataFrame({
            'timestamp': vm_df['timestamp'],
            'value': (vm_df['cpu_usage'] > config.kpi_threshold).astype(int)
        })
        
        self._save_processed_data(dataset_dir, metrics_df, logs_df, kpi_df, vm_df)
        
    def _prepare_alibaba(self, dataset_dir: Path, config: DatasetConfig) -> None:
        """Prepare Alibaba dataset"""
        container_df = pd.read_csv(dataset_dir / 'container_usage.csv')
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'timestamp': container_df['timestamp'],
            'entity_id': container_df['container_id'],
            'cpu_usage': container_df['cpu_util'],
            'memory_usage': container_df['mem_util'],
            'disk_usage': container_df['disk_util']
        })
        
        # Create logs from container events
        events_df = pd.read_csv(dataset_dir / 'container_events.csv')
        logs_df = pd.DataFrame({
            'timestamp': events_df['timestamp'],
            'entity_id': events_df['container_id'],
            'message': events_df['event_type'],
            'level': events_df['status']
        })
        
        # Create KPI values (using resource utilization threshold as KPI)
        kpi_df = pd.DataFrame({
            'timestamp': container_df['timestamp'],
            'value': ((container_df['cpu_util'] > config.kpi_threshold) | 
                     (container_df['mem_util'] > config.kpi_threshold) |
                     (container_df['disk_util'] > config.kpi_threshold)).astype(int)
        })
        
        self._save_processed_data(dataset_dir, metrics_df, logs_df, kpi_df, container_df)
        
    def _save_processed_data(
        self,
        dataset_dir: Path,
        metrics_df: pd.DataFrame,
        logs_df: pd.DataFrame,
        kpi_df: pd.DataFrame,
        raw_df: pd.DataFrame
    ) -> None:
        """Save processed data files"""
        metrics_df.to_csv(dataset_dir / 'historical_metrics.csv', index=False)
        logs_df.to_csv(dataset_dir / 'historical_logs.csv', index=False)
        kpi_df.to_csv(dataset_dir / 'historical_kpi.csv', index=False)
        
        # Save ground truth if available
        if 'label' in raw_df.columns:
            ground_truth = {
                idx: [i for i, label in enumerate(group['label']) if label == 1]
                for idx, group in raw_df.groupby('entity_id')
            }
            ground_truth = np.array(ground_truth, dtype=object)
            np.save(dataset_dir / 'ground_truth.npy', ground_truth)
        
    def prepare_all_datasets(self) -> None:
        """Prepare all datasets"""
        for dataset in self.config.datasets:
            self.prepare_dataset(dataset)

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for OCEAN experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['aiops', 'azure', 'alibaba', 'all'],
                      default='all', help='Dataset to prepare')
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    # Initialize data preparation
    data_prep = DatasetPreparation(config)
    
    if args.dataset == 'all':
        data_prep.prepare_all_datasets()
    else:
        data_prep.prepare_dataset(args.dataset)

if __name__ == '__main__':
    main() 