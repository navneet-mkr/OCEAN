import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DilatedConvBlock(nn.Module):
    """
    Dilated Convolutional Block for capturing long-term temporal dependencies
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2  # Maintain sequence length
        )
        self.norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of dilated conv block
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
        Returns:
            Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        return F.relu(self.norm(self.conv(x)))

class TemporalCausalLearning(nn.Module):
    """
    Long-Term Temporal Causal Structure Learning Module
    """
    def __init__(
        self,
        n_entities: int,
        n_metrics: int,
        n_logs: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        kernel_size: int = 3
    ):
        super(TemporalCausalLearning, self).__init__()
        
        self.n_entities = n_entities
        self.n_metrics = n_metrics
        self.n_logs = n_logs
        self.hidden_dim = hidden_dim
        
        # Dilated convolution layers for metrics
        self.metric_convs = nn.ModuleList([
            DilatedConvBlock(
                in_channels=n_metrics if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2**i
            ) for i in range(n_layers)
        ])
        
        # Dilated convolution layers for logs
        self.log_convs = nn.ModuleList([
            DilatedConvBlock(
                in_channels=n_logs if i == 0 else hidden_dim,
                kernel_size=kernel_size,
                out_channels=hidden_dim,
                dilation=2**i
            ) for i in range(n_layers)
        ])
        
        # GNN layers for causal structure learning
        self.gnn_metric = nn.Linear(hidden_dim, hidden_dim)
        self.gnn_log = nn.Linear(hidden_dim, hidden_dim)
        
        # Adjacency matrix updates
        self.delta_adj_metric = nn.Parameter(torch.zeros(n_entities, n_entities))
        self.delta_adj_log = nn.Parameter(torch.zeros(n_entities, n_entities))
        
    def temporal_conv(
        self, 
        x: torch.Tensor, 
        conv_layers: nn.ModuleList
    ) -> torch.Tensor:
        """
        Apply temporal convolution layers
        Args:
            x: Input tensor of shape (batch_size, n_features, sequence_length)
            conv_layers: List of dilated convolution layers
        Returns:
            Output tensor after temporal convolutions
        """
        for conv in conv_layers:
            x = conv(x)
        return x
    
    def gnn_propagation(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        gnn_layer: nn.Module
    ) -> torch.Tensor:
        """
        Perform GNN message passing
        Args:
            x: Node features
            adj: Adjacency matrix
            gnn_layer: GNN layer for transformation
        Returns:
            Updated node features
        """
        # Normalize adjacency matrix
        deg = torch.sum(adj, dim=1)
        deg = torch.clamp(deg, min=1.0)  # Avoid division by zero
        norm_adj = adj / deg.unsqueeze(-1)
        
        # Message passing
        return F.relu(gnn_layer(torch.matmul(norm_adj, x)))
    
    def forward(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        old_adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of temporal causal learning module
        Args:
            metrics: Metric data of shape (n_entities, n_metrics, sequence_length)
            logs: Log data of shape (n_entities, n_logs, sequence_length)
            old_adj: Previous adjacency matrix of shape (n_entities, n_entities)
        Returns:
            Tuple containing:
            - Processed metric representations
            - Processed log representations
            - Updated metric adjacency matrix
            - Updated log adjacency matrix
        """
        # Apply temporal convolutions
        metric_repr = self.temporal_conv(metrics, self.metric_convs)  # (n_entities, hidden_dim, seq_len)
        log_repr = self.temporal_conv(logs, self.log_convs)  # (n_entities, hidden_dim, seq_len)
        
        # Pool temporal dimension to get node features
        metric_repr = torch.mean(metric_repr, dim=-1)  # (n_entities, hidden_dim)
        log_repr = torch.mean(log_repr, dim=-1)  # (n_entities, hidden_dim)
        
        # Update adjacency matrices
        adj_metric = old_adj + self.delta_adj_metric
        adj_log = old_adj + self.delta_adj_log
        
        # Apply GNN layers
        metric_repr = self.gnn_propagation(metric_repr, adj_metric, self.gnn_metric)
        log_repr = self.gnn_propagation(log_repr, adj_log, self.gnn_log)
        
        return metric_repr, log_repr, adj_metric, adj_log
