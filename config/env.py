from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class EnvConfig(BaseModel):
    """Environment configuration"""
    # Paths
    DATA_DIR: Path = Field(
        Path(os.getenv("OCEAN_DATA_DIR", "data")),
        description="Data directory path"
    )
    OUTPUT_DIR: Path = Field(
        Path(os.getenv("OCEAN_OUTPUT_DIR", "experiments")),
        description="Output directory path"
    )
    LOG_DIR: Path = Field(
        Path(os.getenv("OCEAN_LOG_DIR", "logs")),
        description="Log directory path"
    )
    
    # Model
    DEVICE: str = Field(
        os.getenv("OCEAN_DEVICE", "cuda"),
        description="Device to use (cuda/cpu)"
    )
    NUM_EPOCHS: int = Field(
        int(os.getenv("OCEAN_NUM_EPOCHS", "100")),
        description="Number of training epochs"
    )
    BATCH_SIZE: int = Field(
        int(os.getenv("OCEAN_BATCH_SIZE", "32")),
        description="Training batch size"
    )
    LEARNING_RATE: float = Field(
        float(os.getenv("OCEAN_LEARNING_RATE", "0.001")),
        description="Learning rate"
    )
    
    # Logging
    LOG_LEVEL: str = Field(
        os.getenv("OCEAN_LOG_LEVEL", "INFO"),
        description="Logging level"
    )
    
    def setup_directories(self) -> None:
        """Create all necessary directories"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Global environment configuration instance
env = EnvConfig() 