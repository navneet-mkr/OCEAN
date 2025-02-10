import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
import numpy as np

from .temporal_learning import TemporalCausalLearning
from .attention import MultiFactorAttention
from .graph_fusion import GraphFusion
from .root_cause import RootCauseIdentification

class OCEAN(nn.Module):
    """
    OCEAN: Online Multi-modal Causal Structure LEArNing
    Main model class that integrates all four modules:
    1. Long-Term Temporal Causal Structure Learning
    2. Representation Learning with Multi-factor Attention
    3. Graph Fusion with Contrastive Multi-modal Learning
    4. Network Propagation-Based Root Cause Identification
    """
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
        lambda_acyclicity: float = 1.0
    ):
        """
        Initialize OCEAN model
        Args:
            n_entities: Number of system entities
            n_metrics: Number of metric features
            n_logs: Number of log features
            hidden_dim: Hidden dimension size
            n_temporal_layers: Number of temporal conv layers
            temperature: Temperature for contrastive learning
            dropout: Dropout rate
            beta: Transition probability coefficient
            restart_prob: Random walk restart probability
            top_k: Number of top root causes to identify
            rbo_threshold: Threshold for stopping criterion
            lambda_temporal: Weight for temporal loss
            lambda_sparsity: Weight for sparsity loss
            lambda_acyclicity: Weight for acyclicity loss
        """
        super(OCEAN, self).__init__()
        
        self.n_entities = n_entities
        self.hidden_dim = hidden_dim
        self.lambda_temporal = lambda_temporal
        self.lambda_sparsity = lambda_sparsity
        self.lambda_acyclicity = lambda_acyclicity
        
        # Initialize all modules
        self.temporal_learning = TemporalCausalLearning(
            n_entities=n_entities,
            n_metrics=n_metrics,
            n_logs=n_logs,
            hidden_dim=hidden_dim,
            n_layers=n_temporal_layers
        )
        
        self.attention = MultiFactorAttention(
            n_metrics=n_metrics,
            n_logs=n_logs,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.graph_fusion = GraphFusion(
            hidden_dim=hidden_dim,
            n_entities=n_entities,
            temperature=temperature,
            dropout=dropout
        )
        
        self.root_cause = RootCauseIdentification(
            n_entities=n_entities,
            beta=beta,
            restart_prob=restart_prob,
            top_k=top_k,
            rbo_threshold=rbo_threshold
        )
        
        # Initialize previous rankings list for stopping criterion
        self.previous_rankings: List[List[int]] = []
        
    def compute_loss(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        metric_repr: torch.Tensor,
        log_repr: torch.Tensor,
        recovered_metric: torch.Tensor,
        recovered_log: torch.Tensor,
        mi_loss: torch.Tensor,
        sparsity_loss: torch.Tensor,
        acyclicity_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual loss components
        """
        # Temporal reconstruction loss
        temporal_loss = (
            torch.mean((metrics - recovered_metric) ** 2) +
            torch.mean((logs - recovered_log) ** 2)
        )
        
        # Total loss (Eq. 18)
        total_loss = (
            mi_loss +
            self.lambda_temporal * temporal_loss +
            self.lambda_sparsity * sparsity_loss +
            self.lambda_acyclicity * acyclicity_loss
        )
        
        # Loss components dictionary for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'mi_loss': mi_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'acyclicity_loss': acyclicity_loss.item()
        }
        
        return total_loss, loss_components
        
    def forward(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        old_adj: torch.Tensor,
        kpi_idx: int,
        batch_mask: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[List[int], List[float], bool, Optional[Dict[str, float]]]:
        """
        Forward pass of OCEAN model
        Args:
            metrics: Metric data (n_entities, n_metrics, sequence_length)
            logs: Log data (n_entities, n_logs, sequence_length)
            old_adj: Previous adjacency matrix
            kpi_idx: Index of the KPI node
            batch_mask: Optional mask for valid batch entries
            is_training: Whether in training mode
        Returns:
            Tuple containing:
            - Ranked list of root cause indices
            - Corresponding probabilities
            - Boolean indicating whether to stop
            - Optional dictionary of loss components (during training)
        """
        # 1. Temporal Causal Structure Learning
        metric_repr, log_repr, adj_metric, adj_log = self.temporal_learning(
            metrics, logs, old_adj
        )
        
        # 2. Multi-factor Attention
        weighted_metric, weighted_log, recovered_metric, recovered_log, similarity = self.attention(
            metric_repr, log_repr
        )
        
        # 3. Graph Fusion
        mi_loss, sparsity_loss, acyclicity_loss, fused_adj = self.graph_fusion(
            weighted_metric,
            weighted_log,
            old_adj,
            adj_metric - old_adj,  # delta_adj_metric
            adj_log - old_adj,     # delta_adj_log
            similarity,
            batch_mask
        )
        
        # Compute loss during training
        loss_components = None
        if is_training:
            total_loss, loss_components = self.compute_loss(
                metrics, logs,
                metric_repr, log_repr,
                recovered_metric, recovered_log,
                mi_loss, sparsity_loss, acyclicity_loss
            )
        
        # 4. Root Cause Identification
        ranked_indices, ranked_probs, should_stop = self.root_cause(
            fused_adj,
            kpi_idx,
            self.previous_rankings
        )
        
        # Update previous rankings
        self.previous_rankings.append(ranked_indices)
        
        return ranked_indices, ranked_probs, should_stop, loss_components
    
    def reset_rankings(self):
        """Reset previous rankings list"""
        self.previous_rankings = []
