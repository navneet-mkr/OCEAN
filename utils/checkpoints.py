from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import torch
import logging
from datetime import datetime
from .config import Config

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpoints with metadata and versioning."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Config] = None,
        max_checkpoints: int = 5,
        save_best_only: bool = True,
        metric_name: str = "val_loss",
        metric_mode: str = "min"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            model: PyTorch model to checkpoint
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler (optional)
            config: Model configuration (optional)
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save checkpoints that improve the metric
            metric_name: Name of metric to monitor
            metric_mode: 'min' or 'max' for metric optimization direction
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        # Load checkpoint history
        self.history_file = self.checkpoint_dir / "checkpoint_history.json"
        self.history = self._load_history()
        
        # Track best metric value
        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')
    
    def _load_history(self) -> Dict[str, Any]:
        """Load checkpoint history from JSON file."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {
            "checkpoints": [],
            "best_checkpoint": None,
            "best_metric": None
        }
    
    def _save_history(self) -> None:
        """Save checkpoint history to JSON file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best metric."""
        if self.metric_mode == 'min':
            return current < best
        return current > best
    
    def save(
        self,
        epoch: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Save model checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
            metadata: Additional metadata to save
        
        Returns:
            Path to saved checkpoint or None if no checkpoint was saved
        """
        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            logger.warning(f"Metric {self.metric_name} not found in metrics dict")
            return None
        
        # Check if we should save
        if self.save_best_only and not self._is_better_metric(current_metric, self.best_metric):
            return None
        
        # Update best metric
        if self._is_better_metric(current_metric, self.best_metric):
            self.best_metric = current_metric
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'metadata': metadata or {},
            'config': self.config.model_dump() if self.config else None,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Generate checkpoint filename
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update history
        self.history['checkpoints'].append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        })
        
        # Update best checkpoint if needed
        if self._is_better_metric(current_metric, self.history.get('best_metric', float('inf') if self.metric_mode == 'min' else float('-inf'))):
            self.history['best_checkpoint'] = str(checkpoint_path)
            self.history['best_metric'] = current_metric
        
        # Remove old checkpoints if needed
        if len(self.history['checkpoints']) > self.max_checkpoints:
            oldest = self.history['checkpoints'].pop(0)
            try:
                Path(oldest['path']).unlink()
                logger.info(f"Removed old checkpoint: {oldest['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")
        
        self._save_history()
        return checkpoint_path
    
    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to specific checkpoint to load
            load_best: Load the best checkpoint if no path specified
        
        Returns:
            Dictionary containing checkpoint data
        """
        if checkpoint_path is None and load_best:
            if self.history['best_checkpoint'] is None:
                raise ValueError("No best checkpoint found")
            checkpoint_path = self.history['best_checkpoint']
        
        if checkpoint_path is None:
            raise ValueError("Must specify checkpoint_path or set load_best=True")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.history['checkpoints']:
            return None
        return Path(self.history['checkpoints'][-1]['path'])
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self.history['best_checkpoint'] is None:
            return None
        return Path(self.history['best_checkpoint'])
    
    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        return self.history['checkpoints'] 