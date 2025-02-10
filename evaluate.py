import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from models.ocean import OCEAN
from utils.data_loader import OCEANDataLoader

logger = logging.getLogger(__name__)

def compute_precision_at_k(
    predicted: List[int],
    actual: List[int],
    k: int
) -> float:
    """
    Compute Precision@K metric
    Args:
        predicted: List of predicted indices
        actual: List of actual indices
        k: K value
    Returns:
        Precision@K score
    """
    if not actual:
        return 0.0
        
    k = min(k, len(predicted))
    predicted_set = set(predicted[:k])
    actual_set = set(actual)
    
    return len(predicted_set.intersection(actual_set)) / k

def compute_map_at_k(
    predicted: List[int],
    actual: List[int],
    k: int
) -> float:
    """
    Compute MAP@K (Mean Average Precision at K) metric
    Args:
        predicted: List of predicted indices
        actual: List of actual indices
        k: K value
    Returns:
        MAP@K score
    """
    if not actual:
        return 0.0
        
    k = min(k, len(predicted))
    ap = 0.0
    hits = 0
    
    for i in range(k):
        if predicted[i] in actual:
            hits += 1
            ap += hits / (i + 1)
            
    return ap / min(k, len(actual))

def compute_mrr(
    predicted: List[int],
    actual: List[int]
) -> float:
    """
    Compute MRR (Mean Reciprocal Rank) metric
    Args:
        predicted: List of predicted indices
        actual: List of actual indices
    Returns:
        MRR score
    """
    if not actual:
        return 0.0
        
    for rank, pred_idx in enumerate(predicted, 1):
        if pred_idx in actual:
            return 1.0 / rank
            
    return 0.0

def evaluate_batch(
    model: OCEAN,
    metrics: torch.Tensor,
    logs: torch.Tensor,
    kpi: torch.Tensor,
    ground_truth: List[int],
    old_adj: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate model on a single batch
    Args:
        model: OCEAN model
        metrics: Metric tensor
        logs: Log tensor
        kpi: KPI tensor
        ground_truth: List of ground truth root cause indices
        old_adj: Previous adjacency matrix
    Returns:
        Dictionary of evaluation metrics
    """
    # Forward pass
    ranked_indices, ranked_probs, _, loss_components = model(
        metrics=metrics,
        logs=logs,
        old_adj=old_adj,
        kpi_idx=0,  # Use first entity as KPI for evaluation
        is_training=False
    )
    
    # Compute metrics
    eval_metrics = {
        'precision@1': compute_precision_at_k(ranked_indices, ground_truth, k=1),
        'precision@5': compute_precision_at_k(ranked_indices, ground_truth, k=5),
        'precision@10': compute_precision_at_k(ranked_indices, ground_truth, k=10),
        'map@3': compute_map_at_k(ranked_indices, ground_truth, k=3),
        'map@5': compute_map_at_k(ranked_indices, ground_truth, k=5),
        'map@10': compute_map_at_k(ranked_indices, ground_truth, k=10),
        'mrr': compute_mrr(ranked_indices, ground_truth)
    }
    
    # Add loss components
    eval_metrics.update({
        f"loss_{k}": v.item() for k, v in loss_components.items()
    })
    
    return eval_metrics

def evaluate_model(
    model: OCEAN,
    data_loader: OCEANDataLoader,
    ground_truth_file: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation/test data
    Args:
        model: OCEAN model
        data_loader: Data loader instance
        ground_truth_file: Path to ground truth file (optional)
    Returns:
        Dictionary of averaged evaluation metrics
    """
    model.eval()
    all_metrics: Dict[str, List[float]] = {}
    
    # Load ground truth if provided
    if ground_truth_file:
        ground_truth = np.load(ground_truth_file, allow_pickle=True).item()
    else:
        # Use dummy ground truth for training evaluation
        ground_truth = {0: [1, 2, 3]}  # Example: KPI 0 has root causes [1, 2, 3]
    
    # Get validation data loader
    val_loader = data_loader.get_historical_dataloader()
    
    with torch.no_grad():
        for metrics, logs, kpi in tqdm(val_loader, desc='Evaluating'):
            # Get ground truth for current batch
            batch_ground_truth = ground_truth.get(kpi.item(), [])
            
            # Initialize adjacency matrix
            old_adj = torch.zeros(
                (model.n_entities, model.n_entities),
                device=metrics.device
            )
            
            # Evaluate batch
            batch_metrics = evaluate_batch(
                model, metrics, logs, kpi,
                batch_ground_truth, old_adj
            )
            
            # Accumulate metrics
            for name, value in batch_metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)
    
    # Compute averages
    return {
        name: float(np.mean(values)) for name, values in all_metrics.items()
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate OCEAN model')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--ground_truth_file', type=str, help='Ground truth file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = OCEANDataLoader(
        data_dir=args.data_dir,
        device=args.device
    )
    
    # Initialize model
    model = OCEAN(
        n_entities=data_loader.n_entities,
        n_metrics=data_loader.n_metrics,
        n_logs=data_loader.n_logs
    ).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    metrics = evaluate_model(
        model,
        data_loader,
        args.ground_truth_file
    )
    
    # Print results
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
