import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
from huggingface_hub import snapshot_download
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetConfig(BaseModel):
    """Configuration for RCA datasets"""
    name: str
    url: str
    description: str
    n_faults: int
    data_format: Dict[str, List[str]]  # Expected columns/features for each data type
    hf_repo_id: Optional[str] = None  # Hugging Face repository ID

class RCADatasets:
    """
    Manages the downloading and preprocessing of datasets used in OCEAN paper:
    1. Product Review Dataset (4 system faults)
    2. Online Boutique Dataset (5 system faults)
    3. Train Ticket Dataset (5 system faults)
    """
    
    DATASET_CONFIGS = {
        'product_review': DatasetConfig(
            name='product_review',
            url='https://lemma-rca.github.io/docs/data.html',
            description='Microservice system for online product reviews',
            n_faults=4,
            hf_repo_id='Lemma-RCA-NEC/Product_Review_Original',
            data_format={
                'metrics': ['cpu_usage', 'memory_usage', 'latency', 'request_rate'],
                'logs': ['log_template', 'timestamp', 'severity'],
                'kpi': ['response_time', 'error_rate']
            }
        ),
        'online_boutique': DatasetConfig(
            name='online_boutique',
            url='https://github.com/GoogleCloudPlatform/microservices-demo',
            description='Google Cloud microservice demo for e-commerce',
            n_faults=5,
            data_format={
                'metrics': ['cpu_usage', 'memory_usage', 'network_in', 'network_out'],
                'logs': ['log_template', 'timestamp', 'severity', 'service'],
                'kpi': ['latency', 'error_rate']
            }
        ),
        'train_ticket': DatasetConfig(
            name='train_ticket',
            url='https://github.com/FudanSELab/train-ticket',
            description='Railway ticketing microservice system',
            n_faults=5,
            data_format={
                'metrics': ['cpu_usage', 'memory_usage', 'throughput'],
                'logs': ['log_template', 'timestamp', 'severity', 'service'],
                'kpi': ['response_time', 'success_rate']
            }
        )
    }
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def check_dataset_files(self, dataset_dir: Path) -> bool:
        """
        Check if the dataset directory contains actual data files in the expected format.
        Returns True if the dataset appears to be properly downloaded and processed.
        """
        # Check for at least one incident directory
        incident_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('incident_')]
        if not incident_dirs:
            return False
            
        # Check for required files in the first incident directory
        required_files = ['metrics.npy', 'logs.npy', 'kpi.npy', 'root_cause.npy']
        first_incident = incident_dirs[0]
        return all((first_incident / f).exists() for f in required_files)
    
    def download_from_huggingface(self, dataset_name: str, dataset_dir: Path) -> bool:
        """
        Download dataset from Hugging Face Hub.
        Returns True if successful, False otherwise.
        """
        config = self.DATASET_CONFIGS[dataset_name]
        if not config.hf_repo_id:
            return False
            
        try:
            logger.info(f"Downloading {dataset_name} dataset from Hugging Face Hub...")
            logger.info("Note: This dataset requires Hugging Face authentication.")
            logger.info("Please run 'huggingface-cli login' if you haven't already.")
            
            # Try to download the entire repository snapshot
            try:
                # Download the entire repository
                snapshot_path = snapshot_download(
                    repo_id=config.hf_repo_id,
                    repo_type="dataset",
                    local_dir=dataset_dir,
                    allow_patterns=["*.zip"],  # Only download zip files
                    token=True  # Use token from huggingface-cli login
                )
                
                logger.info(f"Successfully downloaded repository to {snapshot_path}")
                
                # Extract all zip files
                for zip_file in dataset_dir.glob("*.zip"):
                    try:
                        logger.info(f"Extracting {zip_file}...")
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(dataset_dir)
                        # Optionally remove zip file after extraction
                        zip_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to extract {zip_file}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Failed to download repository: {str(e)}")
                logger.info("Please ensure you:")
                logger.info("1. Have installed huggingface_hub with LFS support: pip install huggingface_hub[cli]")
                logger.info("2. Have logged in: huggingface-cli login")
                logger.info("3. Have Git LFS installed: git lfs install")
                return False
            
            # Check if we have any data
            if not self.check_dataset_files(dataset_dir):
                raise Exception("No valid data files found after extraction")
                    
            logger.info(f"Successfully downloaded and extracted {dataset_name} dataset to {dataset_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset from Hugging Face Hub: {str(e)}")
            logger.info("\nTo manually download the dataset:")
            logger.info(f"1. Visit https://huggingface.co/datasets/{config.hf_repo_id}")
            logger.info("2. Login to Hugging Face")
            logger.info("3. Download the zip files")
            logger.info(f"4. Extract them to: {dataset_dir}")
            return False
    
    def download_dataset(self, dataset_name: str) -> Path:
        """
        Download and prepare dataset. For Product Review dataset, attempts to download
        from Hugging Face Hub first. For other datasets, provides manual instructions.
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        config = self.DATASET_CONFIGS[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        if self.check_dataset_files(dataset_dir):
            logger.info(f"Dataset {dataset_name} is already downloaded and processed at {dataset_dir}")
            return dataset_dir
        
        # Try Hugging Face download for Product Review dataset
        if dataset_name == 'product_review':
            if self.download_from_huggingface(dataset_name, dataset_dir):
                return dataset_dir
                
            logger.info("\nFailed to download from Hugging Face Hub. Falling back to manual instructions.")
            
        # Provide manual instructions if automatic download fails or for other datasets
        logger.info(f"\nDataset {dataset_name} needs to be downloaded manually:")
        
        if dataset_name == 'product_review':
            logger.info(
                "Product Review Dataset Instructions:\n"
                f"1. Visit {config.url}\n"
                "2. Fill out the form to request dataset access\n"
                "3. Download the dataset when received\n"
                f"4. Extract the files to: {dataset_dir}\n"
                "5. Ensure each incident is in its own directory with required files:\n"
                "   - metrics.npy: (n_entities, n_metrics, time_steps)\n"
                "   - logs.npy: (n_entities, n_logs, time_steps)\n"
                "   - kpi.npy: (time_steps,)\n"
                "   - root_cause.npy: Ground truth labels"
            )
        elif dataset_name == 'online_boutique':
            logger.info(
                "Online Boutique Dataset Instructions:\n"
                f"1. Visit {config.url}\n"
                "2. Follow instructions to set up the demo environment\n"
                "3. Use provided scripts to collect metrics and logs\n"
                f"4. Process and save data to: {dataset_dir}\n"
                "5. Format data according to OCEAN requirements"
            )
        elif dataset_name == 'train_ticket':
            logger.info(
                "Train Ticket Dataset Instructions:\n"
                f"1. Visit {config.url}\n"
                "2. Clone and set up the Train Ticket system\n"
                "3. Collect system metrics and logs during fault injection\n"
                f"4. Process and save data to: {dataset_dir}\n"
                "5. Format data according to OCEAN requirements"
            )
        
        return dataset_dir

def generate_synthetic_data(
    n_incidents: int = 100,
    n_entities: int = 10,
    n_metrics: int = 4,
    n_logs: int = 3,
    time_steps: int = 100,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data for testing and development
    
    Args:
        n_incidents: Number of incidents to generate
        n_entities: Number of system entities (services/pods)
        n_metrics: Number of metrics per entity
        n_logs: Number of log features per entity
        time_steps: Number of time steps per incident
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing synthetic data arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate metric data with temporal patterns
    metrics = np.random.randn(n_incidents, n_entities, n_metrics, time_steps)
    
    # Add anomaly patterns for some entities during incidents
    for i in range(n_incidents):
        # Randomly select root cause entity
        root_cause = np.random.randint(0, n_entities)
        # Add anomaly pattern to metrics
        start_t = np.random.randint(0, time_steps // 2)
        metrics[i, root_cause, :, start_t:] += np.random.randn(n_metrics, time_steps - start_t) * 2
    
    # Generate log data (simplified as frequency features)
    logs = np.random.poisson(lam=1.0, size=(n_incidents, n_entities, n_logs, time_steps))
    
    # Generate KPI data (affected by root cause anomalies)
    kpi = np.zeros((n_incidents, time_steps))
    for i in range(n_incidents):
        anomaly_start = np.random.randint(0, time_steps // 2)
        kpi[i, anomaly_start:] = 1
    
    # Generate root cause labels
    root_causes = np.random.randint(0, n_entities, size=n_incidents)
    
    return {
        'metrics': metrics,
        'logs': logs,
        'kpi': kpi,
        'root_causes': root_causes
    }

class RCADataset(Dataset):
    """
    PyTorch Dataset for Root Cause Analysis data
    
    Can load either real data from downloaded datasets or generate synthetic data
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_name: str,
        incident_ids: Optional[List[str]] = None,
        use_synthetic: bool = False,
        synthetic_config: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.use_synthetic = use_synthetic
        
        if use_synthetic:
            logger.info("Using synthetic data for development")
            synthetic_config = synthetic_config or {}
            self.data = generate_synthetic_data(**synthetic_config)
            self.incident_ids = list(range(len(self.data['metrics'])))
        else:
            if not incident_ids:
                # Try to find all incident directories
                dataset_dir = self.data_dir / dataset_name
                if not dataset_dir.exists():
                    raise ValueError(f"Dataset directory not found: {dataset_dir}")
                incident_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                self.incident_ids = [d.name for d in incident_dirs]
            else:
                self.incident_ids = incident_ids
    
    def __len__(self) -> int:
        return len(self.incident_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        incident_id = self.incident_ids[idx]
        
        if self.use_synthetic:
            # Return synthetic data
            return {
                'metrics': torch.tensor(self.data['metrics'][idx], dtype=torch.float32),
                'logs': torch.tensor(self.data['logs'][idx], dtype=torch.float32),
                'kpi': torch.tensor(self.data['kpi'][idx], dtype=torch.float32),
                'root_cause': torch.tensor(self.data['root_causes'][idx], dtype=torch.long),
                'incident_id': str(incident_id)  # Convert to string for consistency
            }
        else:
            # Load real data from files
            incident_dir = self.data_dir / self.dataset_name / str(incident_id)
            
            try:
                metrics = np.load(incident_dir / 'metrics.npy')
                logs = np.load(incident_dir / 'logs.npy')
                kpi = np.load(incident_dir / 'kpi.npy')
                root_cause = np.load(incident_dir / 'root_cause.npy')
                
                return {
                    'metrics': torch.tensor(metrics, dtype=torch.float32),
                    'logs': torch.tensor(logs, dtype=torch.float32),
                    'kpi': torch.tensor(kpi, dtype=torch.float32),
                    'root_cause': torch.tensor(root_cause, dtype=torch.long),
                    'incident_id': str(incident_id)
                }
            except Exception as e:
                logger.error(f"Error loading data for incident {incident_id}: {e}")
                raise

def create_dataloader(
    dataset: RCADataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for the RCA dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) 