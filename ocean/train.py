import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, Optional, List, Tuple
import argparse
from tqdm import tqdm

from models.ocean import OCEAN
from utils.data_loader import OCEANDataLoader
from evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCEANTrainer:
    """
    Trainer class for OCEAN model
    Handles both historical training and online inference
    """
    def __init__(
        self,
        model: OCEAN,
        data_loader: OCEANDataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.learning_rate = learning_rate
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses: List[Dict[str, float]] = []
        
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses
        }
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            self.checkpoint_dir / 'latest_checkpoint.pt'
        )
        
        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint,
                self.checkpoint_dir / 'best_checkpoint.pt'
            )
            
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train model for one epoch
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        epoch_losses: Dict[str, float] = {}
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (metrics, logs, kpi) in enumerate(pbar):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                old_adj = torch.zeros(
                    (self.model.n_entities, self.model.n_entities),
                    device=self.device
                )
                
                _, _, _, loss_components = self.model(
                    metrics=metrics,
                    logs=logs,
                    old_adj=old_adj,
                    kpi_idx=0,  # Use first entity as KPI for training
                    is_training=True
                )
                
                # Backward pass
                total_loss = loss_components['total_loss']
                total_loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update average losses
                for loss_name, loss_value in loss_components.items():
                    if loss_name not in epoch_losses:
                        epoch_losses[loss_name] = 0.0
                    epoch_losses[loss_name] += loss_value
                    
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}"
                })
        
        # Compute averages
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches
            
        return epoch_losses
    
    def train(
        self,
        num_epochs: int,
        eval_interval: int = 1,
        save_interval: int = 5
    ) -> None:
        """
        Train model on historical data
        Args:
            num_epochs: Number of epochs to train
            eval_interval: Epochs between evaluations
            save_interval: Epochs between saving checkpoints
        """
        logger.info("Starting historical training phase...")
        
        # Get training data loader
        train_loader = self.data_loader.get_historical_dataloader()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            epoch_losses = self.train_epoch(train_loader, epoch)
            self.train_losses.append(epoch_losses)
            
            # Log losses
            loss_str = ' '.join(f"{k}: {v:.4f}" for k, v in epoch_losses.items())
            logger.info(f"Epoch {epoch} - {loss_str}")
            
            # Save losses
            with open(self.log_dir / 'train_losses.json', 'w') as f:
                json.dump(self.train_losses, f, indent=2)
            
            # Evaluate if needed
            if epoch % eval_interval == 0:
                eval_metrics = evaluate_model(self.model, self.data_loader)
                logger.info(f"Evaluation metrics: {eval_metrics}")
                
                # Update best model if needed
                if eval_metrics['total_loss'] < self.best_loss:
                    self.best_loss = eval_metrics['total_loss']
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint if needed
            if epoch % save_interval == 0:
                self.save_checkpoint()
                
        logger.info("Historical training completed!")
    
    def online_inference(
        self,
        metrics_df: pd.DataFrame,
        logs_df: pd.DataFrame,
        kpi_value: float,
        kpi_idx: int,
        old_adj: Optional[torch.Tensor] = None
    ) -> Tuple[List[int], List[float], bool]:
        """
        Perform online inference on streaming data
        Args:
            metrics_df: Current metrics data
            logs_df: Current logs data
            kpi_value: Current KPI value
            kpi_idx: Index of KPI node
            old_adj: Previous adjacency matrix (optional)
        Returns:
            Tuple of (ranked_indices, ranked_probs, should_stop)
        """
        self.model.eval()
        
        # Process streaming batch
        metrics_tensor, logs_tensor, kpi_tensor = self.data_loader.process_streaming_batch(
            metrics_df, logs_df, kpi_value
        )
        
        # Use zero adjacency matrix if not provided
        if old_adj is None:
            old_adj = torch.zeros(
                (self.model.n_entities, self.model.n_entities),
                device=self.device
            )
        
        # Forward pass
        with torch.no_grad():
            ranked_indices, ranked_probs, should_stop, _ = self.model(
                metrics=metrics_tensor,
                logs=logs_tensor,
                old_adj=old_adj,
                kpi_idx=kpi_idx,
                is_training=False
            )
            
        return ranked_indices, ranked_probs, should_stop

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train OCEAN model')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--stride', type=int, default=10, help='Stride')
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = OCEANDataLoader(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Initialize model
    model = OCEAN(
        n_entities=data_loader.n_entities,
        n_metrics=data_loader.n_metrics,
        n_logs=data_loader.n_logs
    ).to(args.device)
    
    # Initialize trainer
    trainer = OCEANTrainer(
        model=model,
        data_loader=data_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Train model
    trainer.train(num_epochs=args.num_epochs)

if __name__ == '__main__':
    main()
