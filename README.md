# OCEAN: Online Multi-modal Root Cause Analysis

Implementation of the paper "ONLINE MULTI-MODAL ROOT CAUSE ANALYSIS" (arXiv:2410.10021v1).

## Overview

OCEAN is a novel framework for online root cause analysis in microservice systems that:
- Learns temporal and causal dependencies between system entities
- Integrates both system metrics and logs for better analysis
- Performs online root cause analysis with streaming data
- Uses contrastive learning for multi-modal feature fusion
- Supports PyTorch Lightning for efficient training and experiment tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Git LFS (will be installed by setup script if missing)
- Hydra (for configuration management)
- Weights & Biases (for experiment tracking)

### Quick Setup

The easiest way to get started is to use our setup script:

```bash
# Clone the repository
git clone https://github.com/navneet-mkr/OCEAN
cd ocean

# On Unix/Mac:
chmod +x setup.sh
./setup.sh

# On Windows:
setup.bat
```

The setup script will:
1. Check and install prerequisites (including Git LFS)
2. Create a virtual environment
3. Install all dependencies
4. Configure Hugging Face authentication
5. Download and prepare datasets

### Manual Setup

If you prefer to set things up manually:

```bash
# Clone the repository
git clone https://github.com/yourusername/ocean.git
cd ocean

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Git LFS
git lfs install

# Login to Hugging Face
huggingface-cli login
```

## Project Structure

```
ocean/
├── data/                      # Dataset directory
│   ├── product_review/       # Product Review dataset
│   ├── online_boutique/      # Online Boutique dataset
│   └── train_ticket/         # Train Ticket dataset
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── temporal.py           # Temporal causal structure learning
│   ├── attention.py          # Multi-factor attention mechanism
│   ├── fusion.py            # Graph fusion with contrastive learning
│   └── ocean.py             # Main OCEAN model
├── config/                    # Hydra configuration files
│   ├── config.yaml          # Main configuration
│   ├── model/               # Model configurations
│   ├── data/                # Dataset configurations
│   └── trainer/             # Training configurations
├── scripts/                   # Utility scripts
│   ├── prepare_data.py       # Dataset preparation
│   └── run_experiments.py    # Experiment runner
├── data_loading.py           # Data loading utilities
├── requirements.txt          # Project dependencies
├── .env.example              # Environment variable template
└── README.md                # Project documentation
```

### Environment Setup

After installation, copy the environment template and set your variables:

```bash
cp .env.example .env
```

Key environment variables:
- `WANDB_ENTITY`: Your Weights & Biases team/username
- `DATA_DIR`: Base directory for datasets (default: 'data')
- `OUTPUT_DIR`: Directory for experiment outputs (default: 'outputs')

## Data Preparation

The OCEAN model uses three real-world microservice datasets:

1. **Product Review Dataset** (4 system faults)
   - Option 1: Automatic download from Hugging Face Hub (Recommended)
     ```bash
     # Ensure you've installed Git LFS and logged into Hugging Face first
     python scripts/prepare_data.py --dataset product_review
     ```
     This will automatically download and extract the dataset from [Lemma-RCA-NEC/Product_Review_Original](https://huggingface.co/datasets/Lemma-RCA-NEC/Product_Review_Original)

   - Option 2: Manual download
     If automatic download fails:
     1. Visit https://lemma-rca.github.io/docs/data.html
     2. Fill out the form to request access
     3. Download and extract the data
     4. Format according to OCEAN requirements

2. **Online Boutique Dataset** (5 system faults)
   - Google Cloud microservice demo for e-commerce
   - Reference: https://github.com/GoogleCloudPlatform/microservices-demo
   - Features:
     - System metrics: CPU, memory, network I/O
     - System logs: Service-level logs with severity
     - KPIs: Latency and error rate

3. **Train Ticket Dataset** (5 system faults)
   - Railway ticketing microservice system
   - Reference: https://github.com/FudanSELab/train-ticket
   - Features:
     - System metrics: CPU, memory, throughput
     - System logs: Service logs with severity levels
     - KPIs: Response time and success rate

### Download Datasets

The datasets used in OCEAN can be obtained in different ways:

1. **Product Review Dataset** (4 system faults)
   - Option 1: Automatic download from Hugging Face Hub
     ```bash
     python scripts/prepare_data.py --dataset product_review
     ```
     This will automatically download the dataset from [Lemma-RCA-NEC/Product_Review_Original](https://huggingface.co/datasets/Lemma-RCA-NEC/Product_Review_Original)

   - Option 2: Manual download
     1. Visit https://lemma-rca.github.io/docs/data.html
     2. Fill out the form to request access
     3. Download and extract the data
     4. Format according to OCEAN requirements

2. **Online Boutique Dataset** (5 system faults)
   ```bash
   python scripts/prepare_data.py --dataset online_boutique
   ```
   This dataset requires:
   1. Setting up the Google Cloud Online Boutique demo
   2. Collecting metrics and logs during operation
   3. Processing the data into OCEAN format

3. **Train Ticket Dataset** (5 system faults)
   ```bash
   python scripts/prepare_data.py --dataset train_ticket
   ```
   This dataset requires:
   1. Setting up the Train Ticket microservice system
   2. Running fault injection experiments
   3. Collecting and processing the data

The `prepare_data.py` script will create the necessary directory structure and provide detailed instructions for each dataset.

### Expected Data Format

After downloading and processing, each dataset should be organized as follows:
```
data/
├── product_review/
│   ├── incident_1/           # Each incident in its own directory
│   │   ├── metrics.npy      # Shape: (n_entities, n_metrics, time_steps)
│   │   ├── logs.npy         # Shape: (n_entities, n_logs, time_steps)
│   │   ├── kpi.npy          # Shape: (time_steps,)
│   │   └── root_cause.npy   # Ground truth labels
│   ├── incident_2/
│   └── ...
├── online_boutique/         # Similar structure for Online Boutique
└── train_ticket/           # Similar structure for Train Ticket
```

For development without real data, you can use synthetic data generation:

```python
from data_loading import RCADataset

# Use synthetic data
dataset = RCADataset(
    data_dir="data",
    dataset_name="product_review",  # or any other dataset name
    use_synthetic=True,
    synthetic_config={
        'n_incidents': 100,
        'n_entities': 10,
        'n_metrics': 4,
        'n_logs': 3,
        'time_steps': 100
    }
)
```

## Model Architecture

OCEAN consists of four main components:

1. **Long-Term Temporal Causal Structure Learning**
   - Uses dilated temporal convolutions to capture long-range dependencies
   - Learns initial causal structure from historical data
   - Outputs temporal causal adjacency matrix

2. **Multi-factor Attention Mechanism**
   - Analyzes correlations between different factors (metrics and logs)
   - Computes attention weights for each modality
   - Enhances important features for root cause analysis

3. **Graph Fusion with Contrastive Learning**
   - Fuses information from metrics and logs
   - Uses contrastive learning to align different modalities
   - Generates unified causal graph representation

4. **Root Cause Identification**
   - Implements random walk with restart algorithm
   - Propagates anomaly signals through causal graph
   - Ranks potential root causes based on propagation scores

## Training

### Configuration

The model can be configured through Hydra's configuration system or command-line overrides:

```yaml
# config/config.yaml
defaults:
  - model: ocean
  - trainer: default
  - data: product_review

training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  num_workers: 4
  gradient_clip_val: 1.0
```

### Training Process

The model can be trained using PyTorch Lightning with Hydra configuration:

```bash
# Basic training with default config
python train.py

# Override training parameters
python train.py training.batch_size=64 training.learning_rate=0.0001

# Use a different dataset
python train.py data=online_boutique

# Train with synthetic data
python train.py data.use_synthetic=true
```

For programmatic training:

```python
from models.ocean_module import OCEANModule
from data_loading import OCEANDataModule
import lightning as L

# Initialize data module
data_module = OCEANDataModule(
    data_dir="data",
    dataset_name="product_review",
    batch_size=32,
    num_workers=4
)

# Initialize model
model = OCEANModule(
    n_entities=data_module.n_entities,
    n_metrics=data_module.n_metrics,
    n_logs=data_module.n_logs,
    hidden_dim=64
)

# Initialize trainer
trainer = L.Trainer(
    max_epochs=100,
    accelerator='auto',
    devices=1,
    gradient_clip_val=1.0
)

# Train model
trainer.fit(model, data_module)
```

The training process includes:
- Automatic train/validation/test splitting
- Model checkpointing based on validation loss
- Early stopping to prevent overfitting
- Progress tracking and metric logging
- Multi-GPU support through PyTorch Lightning
- WandB integration for experiment tracking
- Hydra configuration management for experiments

## Evaluation

OCEAN is evaluated using the following metrics:

1. **Precision@K**
   - Measures precision of top-K root cause predictions
   - Reported for K = 1, 5, 10

2. **Mean Average Precision (MAP@K)**
   - Average precision across all incidents
   - Reported for K = 3, 5, 10

3. **Mean Reciprocal Rank (MRR)**
   - Average reciprocal rank of true root cause
   - Measures ranking quality

### Running Evaluation

```python
from models.ocean import evaluate_model

# Load test dataset
test_dataset = RCADataset(
    data_dir="data",
    dataset_name="product_review",
    split="test"
)

# Evaluate model
metrics = evaluate_model(
    model=model,
    dataset=test_dataset,
    device=device
)

print(f"P@1: {metrics['p@1']:.3f}")
print(f"MAP@5: {metrics['map@5']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
```

## Online Inference

OCEAN supports online inference for streaming data:

```python
# Initialize model
model = OCEAN(n_entities=10, n_metrics=4, n_logs=3)
model.load_state_dict(torch.load('checkpoints/model.pt'))

# Process streaming data
ranked_causes, scores = model.infer(
    metrics=current_metrics,
    logs=current_logs,
    kpi=current_kpi
)

# Get top-K root causes
top_k_causes = ranked_causes[:5]
print(f"Top-5 potential root causes: {top_k_causes}")
```
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

1. OCEAN Paper: "ONLINE MULTI-MODAL ROOT CAUSE ANALYSIS" (arXiv:2410.10021v1)
2. Product Review Dataset: https://lemma-rca.github.io/docs/data.html
3. Online Boutique: https://github.com/GoogleCloudPlatform/microservices-demo
4. Train Ticket: https://github.com/FudanSELab/train-ticket

## License

This project is licensed under the MIT License - see the LICENSE file for details.

