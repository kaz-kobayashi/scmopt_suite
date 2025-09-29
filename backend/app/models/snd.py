from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class SNDOptimizationRequest(BaseModel):
    """Request model for SND optimization"""
    dc_data: Optional[List[Dict[str, Any]]] = None
    od_data: Optional[Dict[str, Any]] = None
    cost_per_distance: float = Field(default=20, description="Cost per km")
    cost_per_time: float = Field(default=8000, description="Cost per hour")
    capacity: float = Field(default=1000, description="Vehicle capacity")
    max_cpu_time: float = Field(default=10, description="Maximum CPU time in seconds")
    use_scaling: bool = Field(default=True, description="Use gradient scaling method")
    k_paths: int = Field(default=10, description="Number of k-shortest paths")
    alpha: float = Field(default=0.5, description="Gradient scaling parameter")
    max_iterations: int = Field(default=100, description="Maximum iterations for scaling")
    use_osrm: bool = Field(default=False, description="Use OSRM for real distances")


class DCLocation(BaseModel):
    """Distribution Center data model"""
    name: str
    lat: float
    lon: float
    capacity: float = Field(alias="ub")
    transfer_cost: float = Field(alias="vc")
    fixed_cost: float = Field(alias="fc")


class PathResult(BaseModel):
    """Path result model"""
    origin: str
    destination: str
    path: List[str]
    cost: float


class VehicleResult(BaseModel):
    """Vehicle assignment result model"""
    from_id: int
    to_id: int
    from_name: str
    to_name: str
    flow: float
    number: float


class CostBreakdown(BaseModel):
    """Cost breakdown model"""
    transfer_cost: float
    vehicle_cost: float
    total_cost: float


class SNDOptimizationResult(BaseModel):
    """SND optimization result model"""
    status: str
    total_cost: float
    paths: List[PathResult]
    vehicles: List[VehicleResult]
    cost_breakdown: CostBreakdown
    computation_time: float
    iterations: Optional[int] = None
    paths_generated: int


class SNDVisualizationRequest(BaseModel):
    """Request model for SND visualization"""
    session_id: str
    destination_filter: Optional[str] = None


class NetworkNode(BaseModel):
    """Network node for visualization"""
    id: str
    name: str
    lat: float
    lon: float
    node_type: str  # "dc" or "destination"


class NetworkEdge(BaseModel):
    """Network edge for visualization"""
    from_node: str
    to_node: str
    edge_type: str  # "path" or "vehicle"
    weight: float
    color: str
    width: float


class SNDVisualizationResult(BaseModel):
    """SND visualization result model"""
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    center_lat: float
    center_lon: float
    zoom_level: int


class SNDDataUploadResult(BaseModel):
    """Result model for data upload"""
    session_id: str
    dc_count: int
    od_pairs: int
    message: str


class SNDExportRequest(BaseModel):
    """Request model for exporting results"""
    session_id: str
    format: str = Field(default="csv", description="Export format: csv or excel")
    include_visualization: bool = Field(default=False)


class SolverError(Exception):
    """Custom exception for solver errors"""
    pass