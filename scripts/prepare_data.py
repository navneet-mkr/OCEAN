import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

import argparse
from data_loading import RCADatasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Download and prepare datasets for OCEAN experiments')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory to store the datasets')
    parser.add_argument('--dataset', type=str, 
                      choices=['product_review', 'online_boutique', 'train_ticket', 'all'],
                      default='all', help='Dataset to prepare')
    args = parser.parse_args()
    
    # Initialize dataset manager
    datasets = RCADatasets(args.data_dir)
    
    if args.dataset == 'all':
        for dataset_name in datasets.DATASET_CONFIGS:
            datasets.download_dataset(dataset_name)
    else:
        datasets.download_dataset(args.dataset)
    
    logger.info("Dataset preparation completed. Please follow the instructions above to download the datasets.")

if __name__ == '__main__':
    main() 