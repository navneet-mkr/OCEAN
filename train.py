"""
OCEAN Model Training Script
==========================

This script trains the OCEAN (Observability-driven Causal Explanation for ANomaly) model
using PyTorch Lightning with Hydra configuration management.

Supported Datasets:
------------------
1. Product Review Dataset (4 system faults)
2. Online Boutique Dataset (5 system faults)
3. Train Ticket Dataset (5 system faults)

Requirements:
------------
- PyTorch
- PyTorch Lightning
- Hydra
- Weights & Biases (for logging)
- pandas
- numpy

Usage:
------
1. Basic training with default configuration:
   ```
   python train.py
   ```

2. Override specific configuration values:
   ```
   python train.py training.batch_size=64 training.learning_rate=0.0001
   ```

3. Use a different dataset:
   ```
   python train.py data=online_boutique
   ```

4. Train with synthetic data:
   ```
   python train.py data.use_synthetic=true
   ```

Configuration:
-------------
The configuration is managed by Hydra and split across multiple files:

1. config/config.yaml: Main configuration file
   - Experiment settings
   - Training parameters
   - Paths and logging
   - WandB integration

2. config/model/ocean.yaml: Model architecture configuration
   - Network architecture
   - Loss functions
   - Optimization settings

3. config/data/{dataset}.yaml: Dataset-specific configurations
   - Data processing parameters
   - Augmentation settings
   - Feature selection

4. config/trainer/default.yaml: PyTorch Lightning Trainer settings
   - Training devices
   - Distributed training
   - Profiling options

Environment Variables:
-------------------
- WANDB_ENTITY: WandB team/user name
- DATA_DIR: Base directory for datasets (default: 'data')

Features:
--------
- Hydra configuration management
- Automatic device selection (CPU/GPU)
- Model checkpointing
- Early stopping
- WandB logging integration
- Support for synthetic data generation
- Train/validation/test split
- Hyperparameter optimization support
- Multirun capabilities
"""

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
import logging
import platform
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
from datetime import datetime
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from models.ocean_module import OCEANModule
from data_loading import OCEANDataModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ocean_training.log')
    ]
)
logger = logging.getLogger(__name__)

def get_system_info():
    """Gather system information for experiment tracking."""
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total / (1024 ** 3),  # GB
    }
    
    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                system_info['gpu_info'] = [{
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'driver': gpu.driver
                } for gpu in gpus]
        except:
            pass
    
    if 'gpu_info' not in system_info:
        system_info['gpu_info'] = 'No GPU information available'
    
    return system_info

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""
    logger.info(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create directories
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    log_dir = Path(cfg.paths.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    data_module = OCEANDataModule(
        data_dir=cfg.paths.data_dir,
        dataset_name=cfg.data.dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        use_synthetic=cfg.data.use_synthetic
    )
    
    # Setup data module to get dimensions
    data_module.setup()
    
    # Initialize model
    model = OCEANModule(
        n_entities=data_module.n_entities,
        n_metrics=data_module.n_metrics,
        n_logs=data_module.n_logs,
        hidden_dim=cfg.model.hidden_dim,
        learning_rate=cfg.training.learning_rate
    )
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='ocean-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=cfg.training.patience,
        mode='min'
    )
    
    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = cfg.get('experiment_name', f"ocean_{cfg.data.dataset}_{timestamp}")
    
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=experiment_name,
        save_dir=log_dir,
        log_model=True,
        config={
            'model_config': OmegaConf.to_container(cfg.model, resolve=True),
            'training_config': OmegaConf.to_container(cfg.training, resolve=True),
            'data_config': OmegaConf.to_container(cfg.data, resolve=True),
            'system_info': get_system_info()
        }
    )
    
    # Log model architecture diagram
    wandb_logger.watch(model, log='all', log_freq=100)
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=cfg.training.gradient_clip_val,
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler=cfg.trainer.profiler,
        log_every_n_steps=cfg.training.log_interval,
        val_check_interval=cfg.training.eval_interval
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Log final metrics and artifacts
    if isinstance(trainer.checkpoint_callback, ModelCheckpoint):
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            wandb_logger.experiment.log_artifact(
                best_model_path,
                name='best_model',
                type='model'
            )

if __name__ == '__main__':
    main()
