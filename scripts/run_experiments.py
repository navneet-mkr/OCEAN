import os
import argparse
import logging
import json
from pathlib import Path
import torch
from typing import Dict

from models.ocean import OCEAN
from utils.data_loader import OCEANDataLoader
from evaluate import evaluate_model
from config.models import ExperimentConfig, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Runs experiments to replicate results from the OCEAN paper
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.setup()  # Setup directories
        
    def setup_dataset(self, dataset_name: str) -> OCEANDataLoader:
        """Setup data loader for specified dataset"""
        dataset_dir = self.config.paths.data_dir / dataset_name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
            
        return OCEANDataLoader(
            data_dir=str(dataset_dir),
            batch_size=self.config.training.batch_size,
            window_size=self.config.data.window_size,
            stride=self.config.data.stride,
            device=self.config.training.device
        )
        
    def train_and_evaluate(
        self,
        dataset_name: str,
        model: OCEAN,
        data_loader: OCEANDataLoader
    ) -> Dict[str, float]:
        """Train and evaluate model on specified dataset"""
        # Setup paths
        if not self.config.paths:
            raise ValueError("Paths not set in config")
        if not self.config.paths.checkpoint_dir:
            raise ValueError("Checkpoint directory not set in config")
        if not self.config.paths.log_dir:
            raise ValueError("Log directory not set in config")
        if not self.config.paths.data_dir:
            raise ValueError("Data directory not set in config")
        
        checkpoint_path = self.config.paths.checkpoint_dir / f"{dataset_name}_best.pt"
        log_path = self.config.paths.log_dir / f"{dataset_name}_training.json"
        ground_truth_path = self.config.paths.data_dir / dataset_name / 'ground_truth.npy'
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.training.learning_rate
        )
        
        # Training loop
        best_loss = float('inf')
        training_log = []
        
        logger.info(f"Starting training on {dataset_name} dataset")
        for epoch in range(self.config.training.num_epochs):
            model.train()
            epoch_losses = []
            
            # Get training data loader
            train_loader = data_loader.get_historical_dataloader()
            
            for metrics, logs, kpi in train_loader:
                # Zero gradients
                optimizer.zero_grad()
                
                # Initialize adjacency matrix
                old_adj = torch.zeros(
                    (model.n_entities, model.n_entities),
                    device=self.config.training.device
                )
                
                # Forward pass
                _, _, _, loss_components = model(
                    metrics=metrics,
                    logs=logs,
                    old_adj=old_adj,
                    kpi_idx=0,
                    is_training=True
                )
                
                # Backward pass
                total_loss = loss_components['total_loss']
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # Compute average loss
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            # Evaluate model
            eval_metrics = evaluate_model(
                model,
                data_loader,
                str(ground_truth_path) if ground_truth_path.exists() else None
            )
            
            # Log progress
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, MAP@10 = {eval_metrics['map@10']:.4f}")
            
            # Save checkpoint if best so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)
            
            # Save training log
            training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                **eval_metrics
            })
            
            if epoch % self.config.training.save_interval == 0:
                with open(log_path, 'w') as f:
                    json.dump(training_log, f, indent=2)
        
        # Load best model for final evaluation
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        final_metrics = evaluate_model(
            model,
            data_loader,
            str(ground_truth_path) if ground_truth_path.exists() else None
        )
        
        return final_metrics
    
    def run_all_experiments(self) -> None:
        """Run experiments on all datasets"""

        # Setup paths
        if not self.config.paths.results_dir:
            raise ValueError("Results directory not set in config")
        
        
        results = {}
        
        for dataset in self.config.datasets:
            logger.info(f"\nRunning experiments on {dataset} dataset")
            
            # Setup data loader
            data_loader = self.setup_dataset(dataset)
            
            # Update model config with dataset-specific parameters
            model_config = ModelConfig(
                n_entities=data_loader.n_entities,
                n_metrics=data_loader.n_metrics,
                n_logs=data_loader.n_logs,
                **self.config.model.dict(exclude={'n_entities', 'n_metrics', 'n_logs'})
            )
            
            # Initialize model
            model = OCEAN(**model_config.dict()).to(self.config.training.device)
            
            # Train and evaluate
            metrics = self.train_and_evaluate(dataset, model, data_loader)
            results[dataset] = metrics
            
            # Save results
            results_path = self.config.paths.results_dir / f"{dataset}_results.json"
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        # Save summary of all results
        summary_path = self.config.paths.results_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Print summary
        logger.info("\nExperiment Results Summary:")
        for dataset, metrics in results.items():
            logger.info(f"\n{dataset.upper()} Dataset:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run OCEAN experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_all_experiments()

if __name__ == '__main__':
    main() 