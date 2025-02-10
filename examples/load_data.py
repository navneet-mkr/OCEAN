import logging
import torch
from pathlib import Path
from data_loading import RCADatasets, RCADataset, create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Setup data directory
    data_dir = Path("data")
    
    # Initialize dataset manager
    datasets = RCADatasets(data_dir)
    
    # Try to download Product Review dataset
    logger.info("Attempting to download Product Review dataset...")
    dataset_dir = datasets.download_dataset('product_review')
    
    # Check if we have real data, otherwise use synthetic
    if any(dataset_dir.iterdir()):
        logger.info("Using real Product Review dataset")
        dataset = RCADataset(data_dir, 'product_review', use_synthetic=False)
    else:
        logger.info("No real data found. Using synthetic data for development")
        dataset = RCADataset(
            data_dir,
            'product_review',
            use_synthetic=True,
            synthetic_config={
                'n_incidents': 100,
                'n_entities': 10,
                'n_metrics': 4,
                'n_logs': 3,
                'time_steps': 100
            }
        )
    
    # Create data loader
    dataloader = create_dataloader(dataset, batch_size=32)
    
    # Print information about the dataset
    logger.info(f"Dataset size: {len(dataset)} incidents")
    
    # Load and print a batch to verify data format
    batch = next(iter(dataloader))
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            logger.info(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
        else:
            logger.info(f"{key}: type={type(tensor)}")

if __name__ == "__main__":
    main() 