import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GraphFusion(nn.Module):
    """
    Graph Fusion Module with Contrastive Multi-modal Learning
    Implements equations 14-17 from the paper for fusing information across modalities
    """
    def __init__(
        self,
        hidden_dim: int,
        n_entities: int,
        temperature: float = 0.1,
        dropout: float = 0.1
    ):
        super(GraphFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_entities = n_entities
        self.temperature = temperature
        
        # Projection networks for contrastive learning
        self.metric_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.log_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def compute_mutual_info(
        self,
        metric_repr: torch.Tensor,
        log_repr: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mutual information between metric and log representations (Eq. 14, 15)
        Args:
            metric_repr: Metric representations (n_entities, hidden_dim)
            log_repr: Log representations (n_entities, hidden_dim)
            batch_mask: Optional mask for valid batch entries
        Returns:
            Mutual information loss
        """
        # Project representations
        metric_proj = self.metric_proj(metric_repr)  # (n_entities, hidden_dim)
        log_proj = self.log_proj(log_repr)  # (n_entities, hidden_dim)
        
        # Normalize projections
        metric_proj = F.normalize(metric_proj, dim=-1)
        log_proj = F.normalize(log_proj, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(metric_proj, log_proj.T) / self.temperature  # (n_entities, n_entities)
        
        # InfoNCE loss
        if batch_mask is not None:
            sim_matrix = sim_matrix * batch_mask
            
        labels = torch.arange(metric_repr.size(0)).to(metric_repr.device)
        loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        
        return loss / 2.0
    
    def compute_modality_importance(
        self,
        similarity_matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance weights for each modality (Eq. 16)
        Args:
            similarity_matrices: Similarity matrices from attention module (n_entities, n_metrics, n_logs)
        Returns:
            Metric modality importance weight
        """
        # Compute importance scores based on similarity matrices
        metric_score = torch.sum(
            torch.exp(torch.sum(similarity_matrices, dim=-1)),
            dim=-1
        )  # (n_entities,)
        
        log_score = torch.sum(
            torch.exp(torch.sum(similarity_matrices, dim=-2)),
            dim=-1
        )  # (n_entities,)
        
        # Normalize to get metric importance weight
        metric_weight = metric_score / (metric_score + log_score)
        
        return metric_weight.mean()
    
    def fuse_adjacency_matrices(
        self,
        old_adj: torch.Tensor,
        delta_adj_metric: torch.Tensor,
        delta_adj_log: torch.Tensor,
        similarity_matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse adjacency matrices from different modalities (Eq. 17)
        Args:
            old_adj: Previous adjacency matrix
            delta_adj_metric: Change in metric adjacency matrix
            delta_adj_log: Change in log adjacency matrix
            similarity_matrices: Similarity matrices from attention module
        Returns:
            Fused adjacency matrix
        """
        # Compute modality importance weight
        s_metric = self.compute_modality_importance(similarity_matrices)
        
        # Fuse adjacency matrices
        adj_metric = old_adj + delta_adj_metric
        adj_log = old_adj + delta_adj_log
        
        # Weighted combination
        fused_adj = (1 - s_metric) * adj_log + s_metric * adj_metric
        
        return fused_adj
    
    def compute_sparsity_loss(
        self,
        delta_adj_metric: torch.Tensor,
        delta_adj_log: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sparsity regularization loss
        Args:
            delta_adj_metric: Change in metric adjacency matrix
            delta_adj_log: Change in log adjacency matrix
        Returns:
            Sparsity loss
        """
        return torch.norm(delta_adj_metric, p=1) + torch.norm(delta_adj_log, p=1)
    
    def compute_acyclicity_loss(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute acyclicity constraint loss using trace exponential
        Args:
            adj: Adjacency matrix
        Returns:
            Acyclicity loss
        """
        # Compute matrix exponential
        exp_adj = torch.matrix_exp(adj * adj)  # Hadamard product for positive entries
        
        # Compute trace
        trace = torch.trace(exp_adj)
        
        return trace - self.n_entities
    
    def forward(
        self,
        metric_repr: torch.Tensor,
        log_repr: torch.Tensor,
        old_adj: torch.Tensor,
        delta_adj_metric: torch.Tensor,
        delta_adj_log: torch.Tensor,
        similarity_matrices: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of graph fusion module
        Args:
            metric_repr: Metric representations
            log_repr: Log representations
            old_adj: Previous adjacency matrix
            delta_adj_metric: Change in metric adjacency matrix
            delta_adj_log: Change in log adjacency matrix
            similarity_matrices: Similarity matrices from attention module
            batch_mask: Optional mask for valid batch entries
        Returns:
            Tuple containing:
            - Mutual information loss
            - Sparsity loss
            - Acyclicity loss
            - Fused adjacency matrix
        """
        # Compute mutual information loss (Eq. 14, 15)
        mi_loss = self.compute_mutual_info(metric_repr, log_repr, batch_mask)
        
        # Fuse adjacency matrices (Eq. 16, 17)
        fused_adj = self.fuse_adjacency_matrices(
            old_adj,
            delta_adj_metric,
            delta_adj_log,
            similarity_matrices
        )
        
        # Compute regularization losses
        sparsity_loss = self.compute_sparsity_loss(delta_adj_metric, delta_adj_log)
        acyclicity_loss = self.compute_acyclicity_loss(fused_adj)
        
        return mi_loss, sparsity_loss, acyclicity_loss, fused_adj
