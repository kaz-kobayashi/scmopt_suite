"""
Logistics Network Design (LND) models for API requests and responses
Implements data models for MELOS Excel integration endpoints
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import numpy as np

class ExcelTemplateRequest(BaseModel):
    """Request model for creating Excel templates"""
    template_type: str = "basic"  # basic, advanced, full
    include_sample_data: bool = False
    
class ExcelWorkflowRequest(BaseModel):
    """Request model for Excel workflow operations"""
    workbook_data: bytes
    operation: str  # "add_sheets", "prepare_dataframes", "solve", "add_results"
    parameters: Optional[Dict[str, Any]] = None

class LNDSolveRequest(BaseModel):
    """Request model for solving LND problems"""
    customers: List[Dict[str, Union[str, float]]]
    warehouses: List[Dict[str, Union[str, float]]]
    plants: List[Dict[str, Union[str, float]]]
    products: List[Dict[str, Union[str, float]]]
    demand: List[Dict[str, Union[str, float]]]
    production: List[Dict[str, Union[str, float]]]
    solver_type: str = "multi_source"  # multi_source, single_source
    parameters: Optional[Dict[str, Any]] = None

class ExcelTemplateResponse(BaseModel):
    """Response model for Excel template creation"""
    success: bool
    message: str
    template_data: Optional[bytes] = None
    
class ExcelWorkflowResponse(BaseModel):
    """Response model for Excel workflow operations"""
    success: bool
    message: str
    result_data: Optional[bytes] = None
    analysis_results: Optional[Dict[str, Any]] = None

class LNDSolveResponse(BaseModel):
    """Response model for LND problem solutions"""
    success: bool
    message: str
    optimization_results: Optional[Dict[str, Any]] = None
    cost_breakdown: Optional[Dict[str, float]] = None
    selected_facilities: Optional[List[Dict[str, Any]]] = None
    customer_assignments: Optional[List[Dict[str, Any]]] = None

class CustomerAggregationRequest(BaseModel):
    """Request model for customer aggregation"""
    customers: List[Dict[str, Union[str, float]]]
    products: List[Dict[str, Union[str, float]]]
    demand: List[Dict[str, Union[str, float]]]
    num_clusters: int = 10
    method: str = "hierarchical"  # hierarchical, kmeans, weiszfeld

class CustomerAggregationResponse(BaseModel):
    """Response model for customer aggregation"""
    success: bool
    message: str
    aggregated_customers: Optional[List[Dict[str, Any]]] = None
    aggregated_demand: Optional[List[Dict[str, Any]]] = None
    cluster_assignments: Optional[List[int]] = None

class NetworkGenerationRequest(BaseModel):
    """Request model for network generation"""
    customers: List[Dict[str, Union[str, float]]]
    warehouses: List[Dict[str, Union[str, float]]]
    plants: List[Dict[str, Union[str, float]]]
    distance_threshold_plant_dc: float = 99999.0
    distance_threshold_dc_customer: float = 99999.0
    unit_transport_cost: float = 1.0
    unit_delivery_cost: float = 10.0
    average_speed: float = 50.0  # km/h for time calculation

class NetworkGenerationResponse(BaseModel):
    """Response model for network generation"""
    success: bool
    message: str
    transportation_network: Optional[List[Dict[str, Any]]] = None
    network_statistics: Optional[Dict[str, Any]] = None

class AdvancedLNDPRequest(BaseModel):
    """Request model for advanced LNDP (Abstract Logistics) model"""
    nodes: List[str]
    arcs: List[Dict[str, Any]]
    products: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    bom: List[Dict[str, Any]]  # Bill of Materials
    demand: List[Dict[str, Any]]
    constraints: Optional[Dict[str, Any]] = None
    carbon_footprint_limit: Optional[float] = None

class AdvancedLNDPResponse(BaseModel):
    """Response model for advanced LNDP solutions"""
    success: bool
    message: str
    optimization_results: Optional[Dict[str, Any]] = None
    cost_breakdown: Optional[Dict[str, float]] = None
    carbon_footprint: Optional[float] = None
    resource_utilization: Optional[Dict[str, float]] = None

class VRPIntegrationRequest(BaseModel):
    """Request model for LND-VRP integration"""
    lnd_solution: Dict[str, Any]
    customer_data: List[Dict[str, Any]]
    vehicle_specifications: List[Dict[str, Any]]
    time_windows: Optional[List[Dict[str, Any]]] = None
    
class VRPIntegrationResponse(BaseModel):
    """Response model for VRP integration"""
    success: bool
    message: str
    vrp_solutions: Optional[List[Dict[str, Any]]] = None
    routing_statistics: Optional[Dict[str, Any]] = None
    integrated_cost: Optional[float] = None

class CO2CalculationRequest(BaseModel):
    """Request model for CO2 emission calculations"""
    capacity: float  # Vehicle capacity in kg
    load_rate: float = 0.5  # Load rate (0-1)
    is_diesel: bool = False  # True for diesel, False for gasoline
    distance: float  # Distance in km
    
class CO2CalculationResponse(BaseModel):
    """Response model for CO2 calculations"""
    success: bool
    message: str
    fuel_consumption: Optional[float] = None  # liters per ton-km
    co2_emission: Optional[float] = None  # grams of CO2