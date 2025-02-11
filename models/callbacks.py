from typing import Any, Dict, Optional, Sequence
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

class OnlineLearningCallback(Callback):
    """
    Callback for online learning that handles:
    1. Sliding window of recent performance
    2. Concept drift detection
    3. Model adaptation tracking
    """
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.1,
        adaptation_cooldown: int = 10,
        metrics_to_track: Sequence[str] = ('val_loss', 'val_accuracy')
    ):
        """
        Args:
            window_size: Number of batches to keep in sliding window
            drift_threshold: Threshold for concept drift detection
            adaptation_cooldown: Minimum batches between adaptations
            metrics_to_track: Metrics to monitor for drift detection
        """
        super().__init__()
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.adaptation_cooldown = adaptation_cooldown
        self.metrics_to_track = metrics_to_track
        
        # Initialize sliding windows for each metric
        self.metric_windows = {metric: [] for metric in metrics_to_track}
        self.baseline_stats = {}
        self.last_adaptation = 0
        self.current_batch = 0
        
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int
    ) -> None:
        """Update sliding windows and check for concept drift after each batch."""
        self.current_batch += 1
        
        # Update metric windows
        for metric in self.metrics_to_track:
            if metric in outputs:
                value = outputs[metric].item() if torch.is_tensor(outputs[metric]) else outputs[metric]
                self.metric_windows[metric].append(value)
                if len(self.metric_windows[metric]) > self.window_size:
                    self.metric_windows[metric].pop(0)
        
        # Check for concept drift
        if self._detect_drift() and self._can_adapt():
            self._trigger_adaptation(trainer, pl_module)
    
    def _detect_drift(self) -> bool:
        """Detect concept drift using statistical tests."""
        if not all(len(window) >= self.window_size for window in self.metric_windows.values()):
            return False
            
        for metric, window in self.metric_windows.items():
            if metric not in self.baseline_stats:
                # Initialize baseline statistics
                self.baseline_stats[metric] = {
                    'mean': np.mean(window),
                    'std': np.std(window)
                }
                continue
            
            # Calculate current statistics
            current_mean = np.mean(window)
            baseline_mean = self.baseline_stats[metric]['mean']
            baseline_std = self.baseline_stats[metric]['std']
            
            # Check for significant deviation
            z_score = abs(current_mean - baseline_mean) / (baseline_std + 1e-8)
            if z_score > self.drift_threshold:
                return True
        
        return False
    
    def _can_adapt(self) -> bool:
        """Check if enough time has passed since last adaptation."""
        return (self.current_batch - self.last_adaptation) >= self.adaptation_cooldown
    
    def _trigger_adaptation(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Trigger model adaptation in response to concept drift."""
        self.last_adaptation = self.current_batch
        
        # Update baseline statistics
        for metric, window in self.metric_windows.items():
            self.baseline_stats[metric] = {
                'mean': np.mean(window),
                'std': np.std(window)
            }
        
        # Log adaptation event
        if trainer.logger is not None and hasattr(trainer.logger, "log_metrics"):
            # Convert metrics to flat dictionary with float values
            metrics_dict = {
                'adaptation_triggered': 1.0,
                'adaptation_batch': float(self.current_batch)
            }
            # Add flattened drift metrics
            for metric, stats in self.baseline_stats.items():
                metrics_dict[f'drift_{metric}_mean'] = float(stats['mean'])
                metrics_dict[f'drift_{metric}_std'] = float(stats['std'])
            
            trainer.logger.log_metrics(metrics_dict)

class OnlineCheckpointCallback(Callback):
    """
    Enhanced checkpointing for online learning that:
    1. Maintains a rolling window of checkpoints
    2. Saves metadata about data distribution
    3. Tracks adaptation points
    """
    
    def __init__(
        self,
        dirpath: str,
        window_size: int = 5,
        save_interval: int = 100,
        metadata_keys: Sequence[str] = ('loss', 'accuracy', 'drift_detected')
    ):
        """
        Args:
            dirpath: Directory to save checkpoints
            window_size: Number of checkpoints to keep
            save_interval: Batches between checkpoints
            metadata_keys: Additional metadata to save with checkpoints
        """
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.save_interval = save_interval
        self.metadata_keys = metadata_keys
        
        # Load existing checkpoint history
        self.history_file = self.dirpath / 'checkpoint_history.json'
        self.history = self._load_history()
        
    def _load_history(self) -> Dict:
        """Load checkpoint history from JSON file."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {'checkpoints': [], 'metadata': {}}
    
    def _save_history(self) -> None:
        """Save checkpoint history to JSON file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int
    ) -> None:
        """Save checkpoint if interval is reached."""
        if (batch_idx + 1) % self.save_interval == 0:
            self._save_checkpoint(trainer, pl_module, outputs)
    
    def _save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any]
    ) -> None:
        """Save checkpoint with metadata."""
        timestamp = datetime.now().isoformat()
        checkpoint_name = f'checkpoint_{int(time.time())}.ckpt'
        checkpoint_path = self.dirpath / checkpoint_name
        
        # Collect metadata
        metadata = {
            'timestamp': timestamp,
            'global_step': trainer.global_step,
            'batch_idx': trainer.fit_loop.batch_idx,
        }
        
        # Add additional metadata from outputs
        for key in self.metadata_keys:
            if key in outputs:
                value = outputs[key]
                if torch.is_tensor(value):
                    value = value.item()
                metadata[key] = value
        
        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)
        
        # Update history
        self.history['checkpoints'].append({
            'path': str(checkpoint_path),
            'metadata': metadata
        })
        
        # Remove old checkpoints if needed
        while len(self.history['checkpoints']) > self.window_size:
            oldest = self.history['checkpoints'].pop(0)
            try:
                Path(oldest['path']).unlink()
            except Exception as e:
                print(f"Failed to remove old checkpoint: {e}")
        
        # Save updated history
        self._save_history()

class OnlineValidationCallback(Callback):
    """
    Callback for online validation that:
    1. Maintains separate validation windows for different data distributions
    2. Adjusts validation frequency based on drift detection
    3. Tracks performance on both recent and historical data
    """
    
    def __init__(
        self,
        initial_val_interval: int = 100,
        min_val_interval: int = 50,
        max_val_interval: int = 200,
        drift_sensitivity: float = 0.1
    ):
        """
        Args:
            initial_val_interval: Initial batches between validations
            min_val_interval: Minimum batches between validations
            max_val_interval: Maximum batches between validations
            drift_sensitivity: How quickly to adjust validation frequency
        """
        super().__init__()
        self.initial_val_interval = initial_val_interval
        self.min_val_interval = min_val_interval
        self.max_val_interval = max_val_interval
        self.drift_sensitivity = drift_sensitivity
        
        self.current_val_interval = initial_val_interval
        self.last_validation = 0
        self.validation_history = []
        
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int
    ) -> None:
        """Check if validation should be performed."""
        if self._should_validate(batch_idx):
            self._trigger_validation(trainer, outputs)
            self._adjust_validation_interval(outputs)
    
    def _should_validate(self, batch_idx: int) -> bool:
        """Determine if validation should be performed."""
        return (batch_idx - self.last_validation) >= self.current_val_interval
    
    def _trigger_validation(self, trainer: pl.Trainer, outputs: Dict[str, Any]) -> None:
        """Perform validation and record results."""
        trainer.validate()
        self.last_validation = trainer.fit_loop.batch_idx
        
        # Record validation results
        val_results = {
            'batch_idx': self.last_validation,
            'timestamp': datetime.now().isoformat()
        }
        val_results.update(outputs)
        self.validation_history.append(val_results)
    
    def _adjust_validation_interval(self, outputs: Dict[str, Any]) -> None:
        """Adjust validation interval based on recent performance."""
        if len(self.validation_history) < 2:
            return
            
        # Calculate performance change
        if 'val_loss' in outputs:
            current_loss = outputs['val_loss']
            prev_loss = self.validation_history[-2].get('val_loss', current_loss)
            
            # Increase frequency if performance is degrading
            if current_loss > prev_loss:
                self.current_val_interval = max(
                    self.min_val_interval,
                    int(self.current_val_interval * (1 - self.drift_sensitivity))
                )
            # Decrease frequency if performance is stable
            else:
                self.current_val_interval = min(
                    self.max_val_interval,
                    int(self.current_val_interval * (1 + self.drift_sensitivity))
                ) 