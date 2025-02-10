import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MultiFactorAttention(nn.Module):
    """
    Multi-factor Attention Module for analyzing correlations between modalities
    and reassessing importance of different factors
    """
    def __init__(
        self,
        n_metrics: int,
        n_logs: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super(MultiFactorAttention, self).__init__()
        
        self.n_metrics = n_metrics
        self.n_logs = n_logs
        self.hidden_dim = hidden_dim
        
        # Similarity transformation matrices (Eq. 9)
        self.W_similarity = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W_similarity)
        
        # Factor importance transformation (Eq. 10)
        self.W_metric = nn.Linear(hidden_dim, hidden_dim)
        self.W_log = nn.Linear(hidden_dim, hidden_dim)
        self.W_metric_2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_log_2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention vectors (Eq. 10)
        self.w_metric = nn.Parameter(torch.Tensor(hidden_dim))
        self.w_log = nn.Parameter(torch.Tensor(hidden_dim))
        nn.init.xavier_uniform_(self.w_metric.view(1, -1))
        nn.init.xavier_uniform_(self.w_log.view(1, -1))
        
        # MLPs for recovering factors (Eq. 12)
        self.mlp_metric = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_metrics * hidden_dim)
        )
        
        self.mlp_log = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_logs * hidden_dim)
        )
        
    def compute_similarity_matrix(
        self,
        metric_repr: torch.Tensor,
        log_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-factor similarity matrix (Eq. 9)
        Args:
            metric_repr: Metric representations (n_entities, hidden_dim)
            log_repr: Log representations (n_entities, hidden_dim)
        Returns:
            Similarity matrix (n_entities, n_metrics, n_logs)
        """
        # Transform representations
        similarity = torch.tanh(
            torch.matmul(
                torch.matmul(metric_repr, self.W_similarity),
                log_repr.transpose(-2, -1)
            )
        )
        return similarity
    
    def compute_attention_weights(
        self,
        metric_repr: torch.Tensor,
        log_repr: torch.Tensor,
        similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights for each factor (Eq. 10)
        Args:
            metric_repr: Metric representations (n_entities, hidden_dim)
            log_repr: Log representations (n_entities, hidden_dim)
            similarity: Similarity matrix (n_entities, n_metrics, n_logs)
        Returns:
            Tuple of attention weights for metrics and logs
        """
        # Compute Z matrices (Eq. 10)
        Z_metric = torch.tanh(
            self.W_metric(metric_repr) + 
            torch.matmul(similarity, self.W_log(log_repr))
        )
        
        Z_log = torch.tanh(
            self.W_log_2(log_repr) + 
            torch.matmul(similarity.transpose(-2, -1), self.W_metric_2(metric_repr))
        )
        
        # Compute attention weights (Eq. 10)
        a_metric = F.softmax(torch.matmul(Z_metric, self.w_metric), dim=-1)
        a_log = F.softmax(torch.matmul(Z_log, self.w_log), dim=-1)
        
        return a_metric, a_log
    
    def forward(
        self,
        metric_repr: torch.Tensor,
        log_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-factor attention module
        Args:
            metric_repr: Metric representations (n_entities, hidden_dim)
            log_repr: Log representations (n_entities, hidden_dim)
        Returns:
            Tuple containing:
            - Weighted metric representations
            - Weighted log representations
            - Recovered metric factors
            - Recovered log factors
            - Similarity matrix
        """
        # Compute similarity matrix (Eq. 9)
        similarity = self.compute_similarity_matrix(metric_repr, log_repr)
        
        # Compute attention weights (Eq. 10)
        a_metric, a_log = self.compute_attention_weights(metric_repr, log_repr, similarity)
        
        # Apply attention weights (Eq. 11)
        weighted_metric = metric_repr * a_metric.unsqueeze(-1)
        weighted_log = log_repr * a_log.unsqueeze(-1)
        
        # Recover factors (Eq. 12)
        recovered_metric = self.mlp_metric(weighted_metric)
        recovered_log = self.mlp_log(weighted_log)
        
        # Reshape recovered factors to match original dimensions
        recovered_metric = recovered_metric.view(-1, self.n_metrics, self.hidden_dim)
        recovered_log = recovered_log.view(-1, self.n_logs, self.hidden_dim)
        
        return weighted_metric, weighted_log, recovered_metric, recovered_log, similarity
