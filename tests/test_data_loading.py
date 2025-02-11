import pytest
import torch
from pathlib import Path
from data_loading import RCADataset, RCADatasets, generate_synthetic_data

@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary directory for test data"""
    return tmp_path / "data"

@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing"""
    return generate_synthetic_data(
        n_incidents=10,
        n_entities=5,
        n_metrics=3,
        n_logs=2,
        time_steps=50,
        seed=42
    )

@pytest.mark.unit
def test_synthetic_data_generation(synthetic_data):
    """Test synthetic data generation"""
    assert 'metrics' in synthetic_data
    assert 'logs' in synthetic_data
    assert 'kpi' in synthetic_data
    assert 'root_causes' in synthetic_data
    
    assert synthetic_data['metrics'].shape == (10, 5, 3, 50)
    assert synthetic_data['logs'].shape == (10, 5, 2, 50)
    assert synthetic_data['kpi'].shape == (10, 50)
    assert synthetic_data['root_causes'].shape == (10,)

@pytest.mark.unit
def test_dataset_synthetic(data_dir):
    """Test RCADataset with synthetic data"""
    dataset = RCADataset(
        data_dir=data_dir,
        dataset_name='test',
        use_synthetic=True,
        synthetic_config={
            'n_incidents': 5,
            'n_entities': 3,
            'time_steps': 20
        }
    )
    
    assert len(dataset) == 5
    
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert all(k in sample for k in ['metrics', 'logs', 'kpi', 'root_cause', 'incident_id'])
    assert all(isinstance(v, torch.Tensor) for k, v in sample.items() if k != 'incident_id')

@pytest.mark.integration
def test_dataset_manager(data_dir):
    """Test RCADatasets manager"""
    manager = RCADatasets(data_dir)
    
    # Check dataset configurations
    assert 'product_review' in manager.DATASET_CONFIGS
    assert 'online_boutique' in manager.DATASET_CONFIGS
    assert 'train_ticket' in manager.DATASET_CONFIGS
    
    config = manager.DATASET_CONFIGS['product_review']
    assert config.name == 'product_review'
    assert config.n_faults == 4
    assert 'metrics' in config.data_format
    assert 'logs' in config.data_format
    assert 'kpi' in config.data_format 