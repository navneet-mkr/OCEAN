# OCEAN: Online Multi-Modal Root Cause Analysis

PyTorch implementation of OCEAN (Online Multi-modal Causal Structure LEArNing) framework for root cause analysis in microservice systems, based on the paper "ONLINE MULTI-MODAL ROOT CAUSE ANALYSIS" (Zheng et al., 2024).

## Overview

OCEAN is a novel framework for online root cause analysis that consists of four main modules:

1. Long-Term Temporal Causal Structure Learning
2. Representation Learning with Multi-factor Attention
3. Graph Fusion with Contrastive Multi-modal Learning
4. Network Propagation-Based Root Cause Identification

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd ocean

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
ocean/
├── models/
│   ├── __init__.py
│   ├── temporal_learning.py     # Long-term temporal causal structure learning
│   ├── attention.py            # Multi-factor attention mechanism
│   ├── graph_fusion.py         # Graph fusion with contrastive learning
│   └── root_cause.py          # Root cause identification
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   └── evaluation.py          # Evaluation metrics
├── data/                      # Data directory
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## Usage

### Data Preparation

1. Prepare your data in the following format:
   ```
   data/
   ├── historical_metrics.csv  # Historical system metrics
   ├── historical_logs.csv     # Historical system logs
   └── historical_kpi.csv      # Historical KPI values
   ```

2. Data format requirements:
   - `historical_metrics.csv`: Contains columns [timestamp, entity_id, metric1, metric2, ...]
   - `historical_logs.csv`: Contains columns [timestamp, entity_id, message, level, ...]
   - `historical_kpi.csv`: Contains columns [timestamp, value]

### Training

1. Train the model on historical data:
   ```bash
   python train.py \
       --data_dir data \
       --checkpoint_dir checkpoints \
       --log_dir logs \
       --device cuda \
       --num_epochs 100 \
       --batch_size 32 \
       --learning_rate 1e-3 \
       --window_size 100 \
       --stride 10
   ```

2. Training arguments:
   - `--data_dir`: Directory containing the data files
   - `--checkpoint_dir`: Directory to save model checkpoints
   - `--log_dir`: Directory to save training logs
   - `--device`: Device to use (cuda/cpu)
   - `--num_epochs`: Number of training epochs
   - `--batch_size`: Training batch size
   - `--learning_rate`: Learning rate for optimization
   - `--window_size`: Size of sliding windows
   - `--stride`: Stride between windows

### Evaluation

1. Evaluate a trained model:
   ```bash
   python evaluate.py \
       --data_dir data \
       --checkpoint_path checkpoints/best_checkpoint.pt \
       --ground_truth_file data/ground_truth.npy \
       --device cuda
   ```

2. Evaluation metrics:
   - Precision@K (K=1,5,10)
   - MAP@K (K=3,5,10)
   - MRR (Mean Reciprocal Rank)

### Online Inference

The model supports online inference for streaming data:

```python
from models.ocean import OCEAN
from utils.data_loader import OCEANDataLoader

# Initialize model and load checkpoint
model = OCEAN(n_entities, n_metrics, n_logs)
model.load_state_dict(torch.load('checkpoints/best_checkpoint.pt'))

# Initialize data loader
data_loader = OCEANDataLoader(data_dir='data')

# Process streaming data
ranked_indices, ranked_probs, should_stop = model.forward(
    metrics=current_metrics,
    logs=current_logs,
    old_adj=previous_adj,
    kpi_idx=anomaly_entity_idx,
    is_training=False
)
```

### Model Components

The OCEAN framework consists of four main modules:

1. **Long-Term Temporal Causal Structure Learning**
   - Captures temporal dependencies using dilated convolutions
   - Learns initial causal structure from historical data

2. **Multi-factor Attention**
   - Analyzes correlations between different factors
   - Computes importance weights for metrics and logs

3. **Graph Fusion with Contrastive Learning**
   - Fuses information from multiple modalities
   - Uses contrastive learning for better representations

4. **Root Cause Identification**
   - Implements random walk with restart algorithm
   - Ranks potential root causes based on propagation scores

## Features

- Dilated convolutional neural networks for capturing long-term temporal dependencies
- Multi-factor attention mechanism for analyzing correlations among different factors
- Contrastive multi-modal learning for effective graph fusion
- Network propagation-based root cause identification

## Citation

```bibtex
[Paper citation will be added when published]
```

## License

[License information]

## Running Experiments

To replicate the results from the paper, follow these steps:

1. **Prepare Datasets**
   ```bash
   # Download and prepare all datasets
   python scripts/prepare_data.py --data_dir data --dataset all
   
   # Or prepare specific datasets
   python scripts/prepare_data.py --data_dir data --dataset aiops
   python scripts/prepare_data.py --data_dir data --dataset azure
   python scripts/prepare_data.py --data_dir data --dataset alibaba
   ```

2. **Run Experiments**
   ```bash
   # Run experiments on all datasets
   python scripts/run_experiments.py \
       --data_dir data \
       --output_dir experiments \
       --device cuda \
       --num_epochs 100 \
       --batch_size 32 \
       --learning_rate 1e-3
   ```

3. **Results**
   The experiment results will be saved in the following structure:
   ```
   experiments/
   ├── checkpoints/           # Model checkpoints
   │   ├── aiops_best.pt
   │   ├── azure_best.pt
   │   └── alibaba_best.pt
   ├── logs/                  # Training logs
   │   ├── aiops_training.json
   │   ├── azure_training.json
   │   └── alibaba_training.json
   └── results/               # Evaluation results
       ├── aiops_results.json
       ├── azure_results.json
       ├── alibaba_results.json
       └── summary.json
   ```

4. **Datasets Used**
   - **AIOps Challenge Dataset**: KPI anomaly detection dataset
   - **Azure Public Dataset**: Cloud service monitoring data
   - **Alibaba Cluster Trace**: Container monitoring data

5. **Evaluation Metrics**
   The experiments report the following metrics:
   - Precision@K (K=1,5,10)
   - MAP@K (Mean Average Precision, K=3,5,10)
   - MRR (Mean Reciprocal Rank)

6. **Hyperparameters**
   The default hyperparameters are set according to the paper:
   - Learning rate: 1e-3
   - Batch size: 32
   - Number of epochs: 100
   - Window size: 100
   - Stride: 10
   - Temperature: 0.1
   - Dropout: 0.1
