"""
VRP Data Models - Pydantic models for Vehicle Routing Problem variants

This module defines comprehensive data models for all VRP variants supported by PyVRP:
- Basic Capacitated VRP (CVRP)
- VRP with Time Windows (VRPTW)  
- Multi-Depot VRP (MDVRP)
- Pickup and Delivery VRP (PDVRP)
- Prize-Collecting VRP (PC-VRP)
- VRPLIB format support
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import datetime


class VRPVariant(str, Enum):
    """Enumeration of supported VRP variants"""
    CVRP = "CVRP"
    VRPTW = "VRPTW" 
    MDVRP = "MDVRP"
    PDVRP = "PDVRP"
    PC_VRP = "PC-VRP"
    VRPLIB = "VRPLIB"


class LocationModel(BaseModel):
    """Basic location model for VRP"""
    name: str = Field(description="Location name or identifier")
    lat: float = Field(description="Latitude coordinate", ge=-90, le=90)
    lon: float = Field(description="Longitude coordinate", ge=-180, le=180)
    demand: float = Field(default=0.0, description="Demand at this location", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Customer_1",
                "lat": 35.6762,
                "lon": 139.6503,
                "demand": 10.0
            }
        }


class TimeWindow(BaseModel):
    """Time window constraint for VRPTW"""
    earliest: float = Field(description="Earliest service time (hours)", ge=0)
    latest: float = Field(description="Latest service time (hours)", ge=0)
    
    @validator('latest')
    def validate_time_window(cls, v, values):
        if 'earliest' in values and v < values['earliest']:
            raise ValueError('Latest time must be greater than or equal to earliest time')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "earliest": 8.0,
                "latest": 17.0
            }
        }


class PickupDeliveryPair(BaseModel):
    """Pickup-delivery pair for PDVRP"""
    pickup_location_idx: int = Field(description="Index of pickup location", ge=0)
    delivery_location_idx: int = Field(description="Index of delivery location", ge=0)
    demand: float = Field(description="Pickup/delivery demand", gt=0)
    
    @validator('delivery_location_idx')
    def validate_different_locations(cls, v, values):
        if 'pickup_location_idx' in values and v == values['pickup_location_idx']:
            raise ValueError('Pickup and delivery locations must be different')
        return v


class DepotModel(BaseModel):
    """Depot model for multi-depot VRP"""
    name: str = Field(description="Depot name or identifier")
    lat: float = Field(description="Latitude coordinate", ge=-90, le=90)
    lon: float = Field(description="Longitude coordinate", ge=-180, le=180)
    capacity: float = Field(description="Depot capacity", gt=0)
    num_vehicles: int = Field(description="Number of vehicles at depot", ge=1)
    time_window: Optional[TimeWindow] = Field(default=None, description="Operating time window")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Depot_Tokyo",
                "lat": 35.6762,
                "lon": 139.6503,
                "capacity": 1000.0,
                "num_vehicles": 5
            }
        }


# Request Models for each VRP variant

class CVRPRequest(BaseModel):
    """Request model for basic Capacitated VRP"""
    locations: List[LocationModel] = Field(description="List of all locations including depot")
    depot_index: int = Field(default=0, description="Index of depot location", ge=0)
    vehicle_capacity: float = Field(description="Vehicle capacity", gt=0)
    num_vehicles: Optional[int] = Field(default=None, description="Maximum number of vehicles")
    max_runtime: int = Field(default=30, description="Maximum runtime in seconds", ge=1, le=3600)
    
    @validator('depot_index')
    def validate_depot_index(cls, v, values):
        if 'locations' in values and v >= len(values['locations']):
            raise ValueError('Depot index must be within locations list bounds')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "locations": [
                    {"name": "Depot", "lat": 35.6762, "lon": 139.6503, "demand": 0},
                    {"name": "Customer_1", "lat": 35.6854, "lon": 139.7531, "demand": 10},
                    {"name": "Customer_2", "lat": 35.6586, "lon": 139.7454, "demand": 15}
                ],
                "depot_index": 0,
                "vehicle_capacity": 100.0,
                "num_vehicles": 3,
                "max_runtime": 60
            }
        }


class VRPTWRequest(BaseModel):
    """Request model for VRP with Time Windows"""
    locations: List[LocationModel] = Field(description="List of all locations")
    time_windows: List[TimeWindow] = Field(description="Time windows for each location")
    service_times: List[float] = Field(description="Service time at each location (hours)")
    depot_index: int = Field(default=0, description="Index of depot location", ge=0)
    vehicle_capacity: float = Field(description="Vehicle capacity", gt=0)
    num_vehicles: Optional[int] = Field(default=None, description="Maximum number of vehicles")
    max_runtime: int = Field(default=60, description="Maximum runtime in seconds", ge=1, le=3600)
    
    @validator('time_windows', 'service_times')
    def validate_list_lengths(cls, v, values):
        if 'locations' in values and len(v) != len(values['locations']):
            raise ValueError('List length must match number of locations')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "locations": [
                    {"name": "Depot", "lat": 35.6762, "lon": 139.6503, "demand": 0},
                    {"name": "Customer_1", "lat": 35.6854, "lon": 139.7531, "demand": 10}
                ],
                "time_windows": [
                    {"earliest": 0.0, "latest": 24.0},
                    {"earliest": 9.0, "latest": 17.0}
                ],
                "service_times": [0.0, 0.5],
                "depot_index": 0,
                "vehicle_capacity": 100.0,
                "max_runtime": 120
            }
        }


class MDVRPRequest(BaseModel):
    """Request model for Multi-Depot VRP"""
    locations: List[LocationModel] = Field(description="All locations including depots and customers")
    depots: List[DepotModel] = Field(description="Depot specifications")
    depot_indices: List[int] = Field(description="Indices of depot locations in locations list")
    max_runtime: int = Field(default=60, description="Maximum runtime in seconds", ge=1, le=3600)
    
    @validator('depot_indices')
    def validate_depot_indices(cls, v, values):
        if 'locations' in values:
            if any(idx >= len(values['locations']) for idx in v):
                raise ValueError('All depot indices must be within locations list bounds')
            if len(v) != len(set(v)):
                raise ValueError('Depot indices must be unique')
        return v
    
    @validator('depots')
    def validate_depots_match_indices(cls, v, values):
        if 'depot_indices' in values and len(v) != len(values['depot_indices']):
            raise ValueError('Number of depots must match number of depot indices')
        return v


class PDVRPRequest(BaseModel):
    """Request model for Pickup and Delivery VRP"""
    locations: List[LocationModel] = Field(description="All locations including depot")
    pickup_delivery_pairs: List[PickupDeliveryPair] = Field(description="Pickup-delivery pairs")
    depot_index: int = Field(default=0, description="Index of depot location", ge=0)
    vehicle_capacity: float = Field(description="Vehicle capacity", gt=0)
    max_runtime: int = Field(default=60, description="Maximum runtime in seconds", ge=1, le=3600)
    
    @validator('pickup_delivery_pairs')
    def validate_pd_pairs(cls, v, values):
        if 'locations' in values:
            max_idx = len(values['locations']) - 1
            for pair in v:
                if pair.pickup_location_idx > max_idx or pair.delivery_location_idx > max_idx:
                    raise ValueError('Pickup/delivery location indices must be within locations list bounds')
        return v


class PCVRPRequest(BaseModel):
    """Request model for Prize-Collecting VRP"""
    locations: List[LocationModel] = Field(description="All locations including depot")
    prizes: List[float] = Field(description="Prize for visiting each location")
    depot_index: int = Field(default=0, description="Index of depot location", ge=0)
    vehicle_capacity: float = Field(description="Vehicle capacity", gt=0)
    min_prize: float = Field(description="Minimum total prize to collect", ge=0)
    max_runtime: int = Field(default=60, description="Maximum runtime in seconds", ge=1, le=3600)
    
    @validator('prizes')
    def validate_prizes_length(cls, v, values):
        if 'locations' in values and len(v) != len(values['locations']):
            raise ValueError('Prizes list length must match number of locations')
        return v


class VRPLIBRequest(BaseModel):
    """Request model for VRPLIB instance solving"""
    file_content: str = Field(description="VRPLIB file content")
    file_name: str = Field(description="Original filename")
    max_runtime: int = Field(default=300, description="Maximum runtime in seconds", ge=1, le=7200)
    
    class Config:
        schema_extra = {
            "example": {
                "file_content": "NAME : E-n22-k4\nTYPE : CVRP\nDIMENSION : 22\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : 6000\n...",
                "file_name": "E-n22-k4.vrp",
                "max_runtime": 300
            }
        }


# Result Models

class RouteModel(BaseModel):
    """Individual route model"""
    route_id: int = Field(description="Route identifier")
    sequence: List[int] = Field(description="Sequence of location indices")
    locations: List[str] = Field(description="Sequence of location names")
    distance: float = Field(description="Total route distance (km)", ge=0)
    total_demand: float = Field(description="Total demand served", ge=0)
    num_stops: int = Field(description="Number of customer stops", ge=0)
    arrival_times: Optional[List[float]] = Field(default=None, description="Arrival times at each location")
    total_prize: Optional[float] = Field(default=None, description="Total prize collected (PC-VRP)")
    capacity_utilization: Optional[float] = Field(default=None, description="Capacity utilization ratio")
    pickup_delivery_info: Optional[List[Dict[str, Any]]] = Field(default=None, description="PD pair information")


class VRPSolution(BaseModel):
    """Base VRP solution model"""
    status: str = Field(description="Solution status: optimal, feasible, infeasible, error")
    objective_value: float = Field(description="Objective function value", ge=0)
    routes: List[RouteModel] = Field(description="List of routes")
    total_distance: float = Field(description="Total solution distance (km)", ge=0)
    num_vehicles_used: int = Field(description="Number of vehicles used", ge=0)
    computation_time: float = Field(description="Computation time (seconds)", ge=0)
    solver: str = Field(description="Solver used")
    problem_type: VRPVariant = Field(description="VRP variant solved")
    message: Optional[str] = Field(default=None, description="Additional message or error details")


class CVRPResult(VRPSolution):
    """CVRP solution result"""
    total_demand_served: float = Field(description="Total demand served", ge=0)
    average_capacity_utilization: Optional[float] = Field(default=None, description="Average capacity utilization")


class VRPTWResult(VRPSolution):
    """VRPTW solution result"""
    total_demand_served: float = Field(description="Total demand served", ge=0)
    time_window_violations: int = Field(default=0, description="Number of time window violations", ge=0)
    latest_return_time: Optional[float] = Field(default=None, description="Latest vehicle return time")


class MDVRPResult(VRPSolution):
    """MDVRP solution result"""
    routes_by_depot: Dict[str, List[RouteModel]] = Field(description="Routes grouped by depot")
    total_demand_served: float = Field(description="Total demand served", ge=0)
    depot_utilization: Optional[Dict[str, float]] = Field(default=None, description="Utilization per depot")


class PDVRPResult(VRPSolution):
    """PDVRP solution result"""
    pickup_delivery_pairs: int = Field(description="Number of pickup-delivery pairs", ge=0)
    pairs_served: Optional[int] = Field(default=None, description="Number of pairs successfully served")
    constraint_violations: Optional[int] = Field(default=0, description="Number of PD constraint violations")


class PCVRPResult(VRPSolution):
    """PC-VRP solution result"""
    total_prize: float = Field(description="Total prize collected", ge=0)
    min_prize_met: bool = Field(description="Whether minimum prize constraint is satisfied")
    prize_efficiency: Optional[float] = Field(default=None, description="Prize per distance ratio")


class VRPLIBResult(VRPSolution):
    """VRPLIB solution result"""
    instance_name: str = Field(description="VRPLIB instance name")
    instance_type: str = Field(description="Problem type from VRPLIB file")
    best_known_value: Optional[float] = Field(default=None, description="Best known solution value")
    gap: Optional[float] = Field(default=None, description="Gap from best known solution (%)")


# Analysis and Comparison Models

class SolutionComparison(BaseModel):
    """Model for comparing multiple VRP solutions"""
    solutions: List[VRPSolution] = Field(description="List of solutions to compare")
    best_solution_idx: int = Field(description="Index of best solution", ge=0)
    comparison_metrics: Dict[str, Any] = Field(description="Comparison metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "solutions": [],
                "best_solution_idx": 0,
                "comparison_metrics": {
                    "distance_improvement": 12.5,
                    "vehicle_reduction": 1,
                    "runtime_comparison": [30.2, 45.1, 25.8]
                }
            }
        }


class VRPBenchmark(BaseModel):
    """Model for VRP benchmarking results"""
    instance_name: str = Field(description="Benchmark instance name")
    problem_size: Dict[str, int] = Field(description="Problem size characteristics")
    algorithm_results: Dict[str, VRPSolution] = Field(description="Results by algorithm")
    performance_metrics: Dict[str, float] = Field(description="Performance comparison metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "instance_name": "E-n22-k4",
                "problem_size": {"customers": 21, "vehicles": 4, "capacity": 6000},
                "algorithm_results": {},
                "performance_metrics": {
                    "pyvrp_runtime": 15.2,
                    "heuristic_runtime": 0.8,
                    "pyvrp_distance": 375.2,
                    "heuristic_distance": 412.1
                }
            }
        }


# Data Upload and File Handling Models

class VRPDataUpload(BaseModel):
    """Model for VRP data file upload"""
    file_content: str = Field(description="CSV or JSON file content")
    file_name: str = Field(description="Original filename")
    data_type: str = Field(description="Type: locations, time_windows, demands, etc.")
    
    class Config:
        schema_extra = {
            "example": {
                "file_content": "name,lat,lon,demand\nDepot,35.6762,139.6503,0\nCustomer_1,35.6854,139.7531,10",
                "file_name": "locations.csv",
                "data_type": "locations"
            }
        }


class VRPDataValidation(BaseModel):
    """Model for VRP data validation results"""
    is_valid: bool = Field(description="Whether data is valid")
    errors: List[str] = Field(description="List of validation errors")
    warnings: List[str] = Field(description="List of validation warnings")
    data_summary: Dict[str, Any] = Field(description="Summary of validated data")
    
    class Config:
        schema_extra = {
            "example": {
                "is_valid": True,
                "errors": [],
                "warnings": ["Customer_2 has very high demand"],
                "data_summary": {
                    "num_locations": 10,
                    "total_demand": 150.5,
                    "depot_count": 1
                }
            }
        }


# Configuration and Settings Models

class VRPSolverConfig(BaseModel):
    """Configuration for VRP solvers"""
    solver_type: str = Field(default="pyvrp", description="Solver type: pyvrp, heuristic")
    max_runtime: int = Field(default=60, description="Maximum runtime in seconds", ge=1, le=7200)
    seed: int = Field(default=42, description="Random seed for reproducibility", ge=0)
    num_threads: Optional[int] = Field(default=None, description="Number of threads to use")
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        schema_extra = {
            "example": {
                "solver_type": "pyvrp",
                "max_runtime": 120,
                "seed": 42,
                "num_threads": 4,
                "log_level": "INFO"
            }
        }


class VRPVisualizationConfig(BaseModel):
    """Configuration for VRP visualization"""
    show_routes: bool = Field(default=True, description="Show route lines")
    show_demands: bool = Field(default=True, description="Show demand values")
    route_colors: Optional[List[str]] = Field(default=None, description="Custom route colors")
    map_style: str = Field(default="OpenStreetMap", description="Map tile style")
    
    class Config:
        schema_extra = {
            "example": {
                "show_routes": True,
                "show_demands": True,
                "route_colors": ["#FF0000", "#00FF00", "#0000FF"],
                "map_style": "OpenStreetMap"
            }
        }


# Error and Exception Models

class VRPError(BaseModel):
    """VRP error model"""
    error_type: str = Field(description="Type of error")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error_type": "ValidationError",
                "message": "Invalid depot index",
                "details": {"depot_index": 10, "max_valid_index": 5},
                "timestamp": "2024-01-15T10:30:00"
            }
        }


# Export and Reporting Models

class VRPExportRequest(BaseModel):
    """Request for exporting VRP results"""
    solution_id: str = Field(description="Solution identifier")
    export_format: str = Field(default="json", description="Export format: json, csv, excel")
    include_visualization: bool = Field(default=False, description="Include visualization data")
    include_statistics: bool = Field(default=True, description="Include performance statistics")


class VRPReport(BaseModel):
    """Comprehensive VRP analysis report"""
    problem_summary: Dict[str, Any] = Field(description="Problem characteristics summary")
    solution_summary: Dict[str, Any] = Field(description="Solution summary statistics")
    route_details: List[Dict[str, Any]] = Field(description="Detailed route information")
    performance_metrics: Dict[str, float] = Field(description="Performance and efficiency metrics")
    recommendations: List[str] = Field(description="Optimization recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "problem_summary": {
                    "problem_type": "CVRP",
                    "num_customers": 20,
                    "total_demand": 180.5
                },
                "solution_summary": {
                    "total_distance": 245.7,
                    "num_vehicles": 3,
                    "avg_capacity_utilization": 0.85
                },
                "route_details": [],
                "performance_metrics": {
                    "distance_per_customer": 12.29,
                    "capacity_efficiency": 0.85
                },
                "recommendations": [
                    "Consider reducing vehicle capacity to improve utilization",
                    "Customer clustering could reduce total distance"
                ]
            }
        }