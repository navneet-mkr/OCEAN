from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class MetricData(BaseModel):
    """Metric data for a single entity"""
    timestamp: int
    entity_id: str
    values: Dict[str, float]

class LogData(BaseModel):
    """Log data for a single entity"""
    timestamp: int
    entity_id: str
    message: str
    level: str

class InferenceRequest(BaseModel):
    """Request model for root cause inference"""
    metrics: List[MetricData] = Field(..., description="List of metric data points")
    logs: List[LogData] = Field(..., description="List of log data points")
    kpi_entity_id: str = Field(..., description="Entity ID showing anomaly")
    previous_adj_matrix: Optional[List[List[float]]] = Field(None, description="Previous adjacency matrix")

class RootCause(BaseModel):
    """Single root cause with probability"""
    entity_id: str
    probability: float

class InferenceResponse(BaseModel):
    """Response model for root cause inference"""
    root_causes: List[RootCause] = Field(..., description="Ranked list of root causes")
    should_stop: bool = Field(..., description="Whether to stop the online RCA process")
    confidence: float = Field(..., description="Confidence in the results") 