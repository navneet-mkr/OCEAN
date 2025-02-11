from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    from lightning.pytorch.loggers import WandbLogger

from .ocean import OCEAN

class OCEANModule(L.LightningModule):
    """PyTorch Lightning module for OCEAN model."""
    
    def __init__(
        self,
        n_entities: int,
        n_metrics: int,
        n_logs: int,
        hidden_dim: int = 64,
        n_temporal_layers: int = 2,
        temperature: float = 0.1,
        dropout: float = 0.1,
        beta: float = 0.5,
        restart_prob: float = 0.3,
        top_k: int = 5,
        rbo_threshold: float = 0.9,
        lambda_temporal: float = 1.0,
        lambda_sparsity: float = 0.1,
        lambda_acyclicity: float = 1.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 10,
        **kwargs
    ):
        """Initialize OCEAN module."""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        
        # Initialize OCEAN model
        self.model = OCEAN(
            n_entities=n_entities,
            n_metrics=n_metrics,
            n_logs=n_logs,
            hidden_dim=hidden_dim,
            n_temporal_layers=n_temporal_layers,
            temperature=temperature,
            dropout=dropout,
            beta=beta,
            restart_prob=restart_prob,
            top_k=top_k,
            rbo_threshold=rbo_threshold,
            lambda_temporal=lambda_temporal,
            lambda_sparsity=lambda_sparsity,
            lambda_acyclicity=lambda_acyclicity
        )
        
        # Track best metrics
        self.best_val_loss = float('inf')
    
    def forward(self, metrics: torch.Tensor, logs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.model(metrics, logs, mask)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        metrics = batch['metrics']
        logs = batch['logs']
        root_cause = batch['root_cause']
        mask = batch.get('mask', None)
        
        # Forward pass
        scores = self(metrics, logs, mask)
        loss = nn.CrossEntropyLoss()(scores, root_cause)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        metrics = batch['metrics']
        logs = batch['logs']
        root_cause = batch['root_cause']
        mask = batch.get('mask', None)
        
        # Forward pass
        scores = self(metrics, logs, mask)
        loss = nn.CrossEntropyLoss()(scores, root_cause)
        
        # Calculate metrics
        preds = torch.argmax(scores, dim=1)
        accuracy = (preds == root_cause).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        # Log predictions to wandb if using WandbLogger
        if isinstance(self.logger, WandbLogger) and batch_idx == 0:
            self.logger.experiment.log({
                "val_predictions": self.logger.experiment.Histogram(preds.cpu().numpy()),
                "val_targets": self.logger.experiment.Histogram(root_cause.cpu().numpy())
            })
    
    def configure_optimizers(self) -> Union[Optimizer, Tuple[list[Optimizer], list[dict]]]:
        """Configure optimizers and schedulers."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['learning_rate'],
            weight_decay=self.hparams['weight_decay']
        )
        
        # Scheduler
        if self.hparams['lr_scheduler'] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - self.hparams['warmup_epochs'],
                eta_min=1e-6
            )
        else:
            # Cast verbose to int since ReduceLROnPlateau accepts both bool and int
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=0  # type: ignore
            )
        
        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.hparams['warmup_epochs']
        )
        
        # Chain schedulers
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [warmup_scheduler, scheduler]
        )
        
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint."""
        # Add custom info to checkpoint
        checkpoint['best_val_loss'] = self.best_val_loss
        if isinstance(self.logger, WandbLogger):
            checkpoint['metadata'] = {
                'wandb_run_id': self.logger.experiment.id,
                'wandb_run_name': self.logger.experiment.name
            }
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        metrics = batch['metrics']
        logs = batch['logs']
        mask = batch.get('mask', None)
        return self(metrics, logs, mask) 