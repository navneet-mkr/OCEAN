from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from pathlib import Path
import yaml

class ModelConfig(BaseModel):
    """Model hyperparameters configuration"""
    n_entities: int = Field(..., description="Number of system entities")
    n_metrics: int = Field(..., description="Number of metric features")
    n_logs: int = Field(..., description="Number of log features")
    hidden_dim: int = Field(64, description="Hidden dimension size")
    n_temporal_layers: int = Field(2, description="Number of temporal conv layers")
    temperature: float = Field(0.1, description="Temperature for contrastive learning")
    dropout: float = Field(0.1, description="Dropout rate")
    beta: float = Field(0.5, description="Transition probability coefficient")
    restart_prob: float = Field(0.3, description="Random walk restart probability")
    top_k: int = Field(5, description="Number of top root causes to identify")
    rbo_threshold: float = Field(0.9, description="Threshold for stopping criterion")
    lambda_temporal: float = Field(1.0, description="Weight for temporal loss")
    lambda_sparsity: float = Field(0.1, description="Weight for sparsity loss")
    lambda_acyclicity: float = Field(1.0, description="Weight for acyclicity loss")

class DataConfig(BaseModel):
    """Data processing configuration"""
    window_size: int = Field(100, description="Size of sliding windows")
    stride: int = Field(10, description="Stride between windows")
    normalize: bool = Field(True, description="Whether to normalize data")
    outlier_threshold: float = Field(3.0, description="Threshold for outlier removal")
    max_features: int = Field(100, description="Maximum number of log features")

class TrainingConfig(BaseModel):
    """Training configuration"""
    num_epochs: int = Field(100, description="Number of training epochs")
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(1e-3, description="Learning rate")
    device: str = Field("cuda", description="Device to use (cuda/cpu)")
    eval_interval: int = Field(1, description="Epochs between evaluations")
    save_interval: int = Field(5, description="Epochs between saving checkpoints")

class PathConfig(BaseModel):
    """Path configuration"""
    data_dir: Path = Field(..., description="Data directory")
    output_dir: Path = Field(..., description="Output directory")
    checkpoint_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    results_dir: Optional[Path] = None

    @model_validator(mode='before')
    def validate_paths(cls, values):
        for key, value in values.items():
            if isinstance(value, str):
                values[key] = Path(value)
        return values

    def setup_directories(self) -> None:
        """Create all necessary directories"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.checkpoint_dir:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        if not self.log_dir:
            self.log_dir = self.output_dir / "logs"
        if not self.results_dir:
            self.results_dir = self.output_dir / "results"
            
        for d in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            d.mkdir(exist_ok=True)

class ExperimentConfig(BaseModel):
    """Complete experiment configuration"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    paths: PathConfig
    datasets: List[str] = Field(["aiops", "azure", "alibaba"], description="Datasets to use")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Convert path strings to Path objects
        paths = config_dict.get('paths', {})
        for key in paths:
            if isinstance(paths[key], str):
                paths[key] = Path(paths[key])
        return cls(**config_dict)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = self.dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2)

    def setup(self) -> None:
        """Setup all configurations"""
        self.paths.setup_directories() 