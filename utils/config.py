from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Model architecture and training configuration"""
    # Architecture
    n_temporal_layers: int = Field(3, description="Number of temporal convolutional layers")
    hidden_dim: int = Field(128, description="Hidden dimension size")
    n_heads: int = Field(4, description="Number of attention heads")
    dropout: float = Field(0.1, description="Dropout rate")
    activation: str = Field("gelu", description="Activation function")
    
    # Contrastive Learning
    temperature: float = Field(0.1, description="Temperature for contrastive learning")
    contrast_mode: str = Field("all", description="Contrastive learning mode")
    base_temperature: float = Field(0.07, description="Base temperature")
    
    # Graph Neural Network
    gnn_layers: int = Field(2, description="Number of GNN layers")
    gnn_hidden_dim: int = Field(64, description="GNN hidden dimension")
    edge_dim: int = Field(16, description="Edge feature dimension")
    
    # Root Cause Analysis
    restart_prob: float = Field(0.3, description="Random walk restart probability")
    top_k: int = Field(5, description="Number of top root causes to identify")
    threshold: float = Field(0.5, description="Detection threshold")

class TrainingConfig(BaseModel):
    """Training configuration"""
    # Basic Training
    num_epochs: int = Field(100, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size")
    learning_rate: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(0.0001, description="Weight decay")
    
    # Learning Rate Schedule
    lr_scheduler: str = Field("cosine", description="Learning rate scheduler")
    warmup_epochs: int = Field(10, description="Number of warmup epochs")
    min_lr: float = Field(0.00001, description="Minimum learning rate")
    
    # Early Stopping
    patience: int = Field(10, description="Early stopping patience")
    min_delta: float = Field(0.001, description="Minimum change for early stopping")
    
    # Gradient Clipping
    clip_grad_norm: float = Field(1.0, description="Gradient clipping norm")
    
    # Mixed Precision
    use_amp: bool = Field(True, description="Use automatic mixed precision")
    
    # Logging
    log_interval: int = Field(10, description="Logging interval")
    eval_interval: int = Field(100, description="Evaluation interval")

class DataConfig(BaseModel):
    """Data loading and preprocessing configuration"""
    # Data Loading
    num_workers: int = Field(4, description="Number of data loading workers")
    prefetch_factor: int = Field(2, description="Data loader prefetch factor")
    pin_memory: bool = Field(True, description="Pin memory in data loader")
    
    # Preprocessing
    window_size: int = Field(100, description="Sliding window size")
    stride: int = Field(10, description="Stride between windows")
    normalize: bool = Field(True, description="Normalize input data")
    standardize: bool = Field(True, description="Standardize input data")
    
    # Augmentation
    use_augmentation: bool = Field(True, description="Use data augmentation")
    noise_std: float = Field(0.01, description="Standard deviation of Gaussian noise")
    mask_prob: float = Field(0.15, description="Masking probability")
    
    # Feature Selection
    max_log_features: int = Field(100, description="Maximum number of log features")
    metric_aggregation: str = Field("mean", description="Metric aggregation method")

class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    metrics: list[str] = Field(
        ["precision@1", "precision@5", "map@3", "map@5", "mrr"],
        description="Evaluation metrics"
    )
    threshold: float = Field(0.5, description="Evaluation threshold")
    save_predictions: bool = Field(True, description="Save model predictions")

class Config(BaseModel):
    """Complete configuration"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvaluationConfig

def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return Config(
        model=ModelConfig(**config_dict["model"]),
        training=TrainingConfig(**config_dict["training"]),
        data=DataConfig(**config_dict["data"]),
        evaluation=EvaluationConfig(**config_dict["evaluation"])
    )

def save_config(config: Config, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.model_dump()
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """Update configuration with new values"""
    config_dict = config.model_dump()
    
    for key_path, value in updates.items():
        keys = key_path.split('.')
        current = config_dict
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    
    return Config(**config_dict) 