import pytest
import torch
import torch.nn as nn
from pathlib import Path
from utils.checkpoints import CheckpointManager
from utils.config import Config, ModelConfig, TrainingConfig, DataConfig, EvaluationConfig

class SimpleModel(nn.Module):
    """Simple model for testing checkpoints."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleModel()

@pytest.fixture
def optimizer(model):
    """Create an optimizer for testing."""
    return torch.optim.Adam(model.parameters(), lr=0.001)

@pytest.fixture
def scheduler(optimizer):
    """Create a learning rate scheduler for testing."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

@pytest.fixture
def config():
    """Create a test configuration."""
    model_config = ModelConfig(
        n_temporal_layers=3,
        hidden_dim=128,
        n_heads=4,
        dropout=0.1,
        activation="gelu",
        temperature=0.1,
        contrast_mode="all",
        base_temperature=0.07,
        gnn_layers=2,
        gnn_hidden_dim=64,
        edge_dim=16,
        restart_prob=0.3,
        top_k=5,
        threshold=0.5
    )
    
    training_config = TrainingConfig(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        lr_scheduler="cosine",
        warmup_epochs=10,
        min_lr=0.00001,
        patience=10,
        min_delta=0.001,
        clip_grad_norm=1.0,
        use_amp=True,
        log_interval=10,
        eval_interval=100
    )
    
    data_config = DataConfig(
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        window_size=100,
        stride=10,
        normalize=True,
        standardize=True,
        use_augmentation=True,
        noise_std=0.01,
        mask_prob=0.15,
        max_log_features=100,
        metric_aggregation="mean"
    )
    
    eval_config = EvaluationConfig(
        metrics=["precision@1", "precision@5", "map@3", "map@5", "mrr"],
        threshold=0.5,
        save_predictions=True
    )
    
    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
        evaluation=eval_config
    )

@pytest.fixture
def checkpoint_manager(tmp_path, model, optimizer, scheduler, config):
    """Create a checkpoint manager for testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        max_checkpoints=3
    )

@pytest.mark.unit
def test_checkpoint_save(checkpoint_manager):
    """Test saving checkpoints."""
    # Save first checkpoint
    metrics = {"val_loss": 0.5, "val_acc": 0.8}
    path = checkpoint_manager.save(epoch=1, metrics=metrics)
    
    assert path.exists()
    assert len(checkpoint_manager.list_checkpoints()) == 1
    assert checkpoint_manager.get_best_checkpoint() == path
    
    # Save second checkpoint with worse metric
    metrics = {"val_loss": 0.6, "val_acc": 0.79}
    path2 = checkpoint_manager.save(epoch=2, metrics=metrics)
    
    assert path2 is None  # Should not save due to worse metric
    assert len(checkpoint_manager.list_checkpoints()) == 1
    
    # Save third checkpoint with better metric
    metrics = {"val_loss": 0.4, "val_acc": 0.85}
    path3 = checkpoint_manager.save(epoch=3, metrics=metrics)
    
    assert path3.exists()
    assert len(checkpoint_manager.list_checkpoints()) == 2
    assert checkpoint_manager.get_best_checkpoint() == path3

@pytest.mark.unit
def test_checkpoint_load(checkpoint_manager):
    """Test loading checkpoints."""
    # Save a checkpoint
    metrics = {"val_loss": 0.5, "val_acc": 0.8}
    metadata = {"git_commit": "abc123"}
    path = checkpoint_manager.save(epoch=1, metrics=metrics, metadata=metadata)
    
    # Modify model parameters
    for param in checkpoint_manager.model.parameters():
        nn.init.zeros_(param)
    
    # Load checkpoint
    checkpoint = checkpoint_manager.load(path)
    
    assert checkpoint['epoch'] == 1
    assert checkpoint['metrics'] == metrics
    assert checkpoint['metadata'] == metadata
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'scheduler_state_dict' in checkpoint
    assert 'config' in checkpoint
    assert 'timestamp' in checkpoint

@pytest.mark.unit
def test_checkpoint_rotation(checkpoint_manager):
    """Test checkpoint rotation with max_checkpoints limit."""
    # Save multiple checkpoints
    metrics = [
        {"val_loss": 0.5},
        {"val_loss": 0.4},
        {"val_loss": 0.3},
        {"val_loss": 0.2}
    ]
    
    paths = []
    for i, m in enumerate(metrics, 1):
        path = checkpoint_manager.save(epoch=i, metrics=m)
        if path:
            paths.append(path)
    
    # Check that only max_checkpoints are kept
    assert len(checkpoint_manager.list_checkpoints()) == 3
    assert not paths[0].exists()  # First checkpoint should be deleted
    assert paths[-1].exists()  # Latest checkpoint should exist
    
    # Check that best checkpoint is preserved
    assert checkpoint_manager.get_best_checkpoint() == paths[-1]

@pytest.mark.unit
def test_checkpoint_history(checkpoint_manager):
    """Test checkpoint history tracking."""
    # Save checkpoints
    metrics = {"val_loss": 0.5}
    path1 = checkpoint_manager.save(epoch=1, metrics=metrics)
    
    history = checkpoint_manager.list_checkpoints()
    assert len(history) == 1
    assert history[0]['epoch'] == 1
    assert history[0]['metrics'] == metrics
    assert history[0]['path'] == str(path1)
    assert 'timestamp' in history[0]
    
    # Check history file exists
    assert checkpoint_manager.history_file.exists()
    
    # Create new manager instance with same directory
    new_manager = CheckpointManager(
        checkpoint_dir=checkpoint_manager.checkpoint_dir,
        model=checkpoint_manager.model,
        optimizer=checkpoint_manager.optimizer
    )
    
    # Check history is loaded
    assert len(new_manager.list_checkpoints()) == 1
    assert new_manager.get_best_checkpoint() == path1 