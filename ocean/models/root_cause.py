import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np

class RootCauseIdentification(nn.Module):
    """
    Network Propagation-Based Root Cause Identification Module
    Implements random walk with restart algorithm for root cause ranking
    """
    def __init__(
        self,
        n_entities: int,
        beta: float = 0.5,
        restart_prob: float = 0.3,
        max_iter: int = 100,
        convergence_threshold: float = 1e-6,
        top_k: int = 5,
        rbo_threshold: float = 0.9
    ):
        super(RootCauseIdentification, self).__init__()
        
        self.n_entities = n_entities
        self.beta = beta  # Transition probability coefficient
        self.restart_prob = restart_prob  # Probability of restarting from KPI node
        self.max_iter = max_iter  # Maximum iterations for random walk
        self.convergence_threshold = convergence_threshold
        self.top_k = top_k  # Number of top root causes to identify
        self.rbo_threshold = rbo_threshold  # Threshold for stopping criterion
        
    def compute_transition_matrix(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute transition probability matrix (Eq. 19)
        Args:
            adj: Adjacency matrix (n_entities, n_entities)
        Returns:
            Transition probability matrix
        """
        # Normalize adjacency matrix by column sum
        col_sum = torch.sum(adj, dim=0)
        col_sum = torch.clamp(col_sum, min=1e-12)  # Prevent division by zero
        
        # Compute transition probabilities
        transition_matrix = self.beta * (adj / col_sum)
        
        return transition_matrix
    
    def random_walk_with_restart(
        self,
        transition_matrix: torch.Tensor,
        kpi_idx: int
    ) -> torch.Tensor:
        """
        Perform random walk with restart (Eq. 20)
        Args:
            transition_matrix: Transition probability matrix
            kpi_idx: Index of the KPI node
        Returns:
            Steady-state probability distribution
        """
        # Initialize starting probability distribution
        r0 = torch.zeros(self.n_entities, device=transition_matrix.device)
        r0[kpi_idx] = 1.0
        
        # Initialize current probability distribution
        rt = r0.clone()
        
        # Perform random walk iterations
        for _ in range(self.max_iter):
            rt_next = (1 - self.restart_prob) * torch.matmul(transition_matrix, rt) + self.restart_prob * r0
            
            # Check convergence
            if torch.norm(rt_next - rt) < self.convergence_threshold:
                break
                
            rt = rt_next
        
        return rt
    
    def rank_root_causes(
        self,
        steady_state_prob: torch.Tensor,
        exclude_indices: Optional[List[int]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Rank system entities based on steady-state probabilities
        Args:
            steady_state_prob: Steady-state probability distribution
            exclude_indices: Optional list of indices to exclude from ranking (e.g., KPI node)
        Returns:
            Tuple of (ranked entity indices, corresponding probabilities)
        """
        probs = steady_state_prob.cpu().numpy()
        if exclude_indices:
            probs[exclude_indices] = -np.inf
            
        # Get top-k indices and probabilities
        indices = np.argsort(-probs)[:self.top_k]
        top_k_indices = [int(i) for i in indices]  # Convert to list of ints
        top_k_probs = [float(-probs[i]) for i in top_k_indices]  # Convert to list of floats
        
        return top_k_indices, top_k_probs
    
    def compute_rank_similarity(
        self,
        current_ranking: List[int],
        previous_ranking: List[int]
    ) -> float:
        """
        Compute rank-biased overlap (RBO) between current and previous rankings (Eq. 21)
        Args:
            current_ranking: Current list of ranked indices
            previous_ranking: Previous list of ranked indices
        Returns:
            RBO similarity score
        """
        if not previous_ranking:
            return 0.0
            
        p = 0.9  # Persistence parameter for RBO
        rbo = 0.0
        weight = 1.0
        
        for k in range(min(len(current_ranking), len(previous_ranking))):
            # Compute overlap at depth k+1
            current_set = set(current_ranking[:k+1])
            previous_set = set(previous_ranking[:k+1])
            overlap = len(current_set.intersection(previous_set)) / (k + 1)
            
            rbo += weight * overlap
            weight *= p
            
        return rbo
    
    def check_stopping_criterion(
        self,
        current_ranking: List[int],
        previous_rankings: List[List[int]],
        n_consecutive: int = 3
    ) -> bool:
        """
        Check if the online RCA process should stop
        Args:
            current_ranking: Current list of ranked indices
            previous_rankings: List of previous rankings
            n_consecutive: Number of consecutive similar rankings required
        Returns:
            Boolean indicating whether to stop
        """
        if len(previous_rankings) < n_consecutive:
            return False
            
        # Check last n_consecutive rankings
        for prev_ranking in previous_rankings[-n_consecutive:]:
            similarity = self.compute_rank_similarity(current_ranking, prev_ranking)
            if similarity < self.rbo_threshold:
                return False
                
        return True
    
    def forward(
        self,
        adj: torch.Tensor,
        kpi_idx: int,
        previous_rankings: Optional[List[List[int]]] = None
    ) -> Tuple[List[int], List[float], bool]:
        """
        Forward pass to identify root causes
        Args:
            adj: Adjacency matrix
            kpi_idx: Index of the KPI node
            previous_rankings: Optional list of previous rankings for stopping criterion
        Returns:
            Tuple containing:
            - Ranked list of root cause indices
            - Corresponding probabilities
            - Boolean indicating whether to stop the online RCA process
        """
        # Compute transition matrix
        transition_matrix = self.compute_transition_matrix(adj)
        
        # Perform random walk with restart
        steady_state_prob = self.random_walk_with_restart(transition_matrix, kpi_idx)
        
        # Rank root causes
        ranked_indices, ranked_probs = self.rank_root_causes(
            steady_state_prob,
            exclude_indices=[kpi_idx]
        )
        
        # Check stopping criterion
        should_stop = False
        if previous_rankings is not None:
            should_stop = self.check_stopping_criterion(ranked_indices, previous_rankings)
            
        return ranked_indices, ranked_probs, should_stop
