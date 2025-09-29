"""
VRP Unified API Models - Pydantic models for unified PyVRP API

This module defines the unified data models for the PyVRP API that accepts
a single JSON request format for all VRP variants.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from enum import Enum


class MultipleTimeWindow(BaseModel):
    """Multiple time windows for a single client"""
    early: int = Field(description="Time window start (minutes from midnight)")
    late: int = Field(description="Time window end (minutes from midnight)")
    
    @validator('late')
    def validate_time_window(cls, v, values):
        if 'early' in values and v < values['early']:
            raise ValueError('late must be greater than or equal to early')
        return v


class ClientGroup(BaseModel):
    """Client group definition for advanced grouping constraints"""
    group_id: str = Field(description="Unique group identifier")
    client_indices: List[int] = Field(description="Indices of clients in this group")
    required: bool = Field(default=False, description="Must visit at least one client from group")
    mutually_exclusive: bool = Field(default=False, description="Visit exactly one client from group")
    penalty: Optional[float] = Field(default=None, description="Penalty for not satisfying group constraint")


class ClientModel(BaseModel):
    """Unified client model for all VRP variants with full PyVRP support"""
    x: int = Field(description="X coordinate (integer)")
    y: int = Field(description="Y coordinate (integer)")
    delivery: Union[int, List[int]] = Field(default=0, description="Delivery demand (integer or list for multi-dimensional)")
    pickup: Union[int, List[int]] = Field(default=0, description="Pickup demand (integer or list for multi-dimensional)")
    service_duration: int = Field(default=10, description="Service time in minutes")
    tw_early: Optional[int] = Field(default=0, description="Primary time window start (minutes from midnight)")
    tw_late: Optional[int] = Field(default=1440, description="Primary time window end (minutes from midnight)")
    time_windows: Optional[List[MultipleTimeWindow]] = Field(default=None, description="Multiple time windows")
    release_time: Optional[int] = Field(default=0, description="Release time (earliest time client becomes available)")
    prize: Optional[int] = Field(default=0, description="Prize for visiting (PC-VRP)")
    required: bool = Field(default=True, description="Must visit flag")
    group_id: Optional[str] = Field(default=None, description="Client group membership")
    allowed_vehicle_types: Optional[List[int]] = Field(default=None, description="Allowed vehicle type indices")
    priority: Optional[int] = Field(default=1, description="Client priority level (1=highest)")
    service_time_multiplier: Optional[float] = Field(default=1.0, description="Service time multiplier")
    
    @validator('tw_late')
    def validate_time_window(cls, v, values):
        if 'tw_early' in values and v < values['tw_early']:
            raise ValueError('tw_late must be greater than or equal to tw_early')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "x": 100,
                "y": 200,
                "delivery": 50,
                "pickup": 0,
                "service_duration": 10,
                "tw_early": 480,
                "tw_late": 1020,
                "prize": 100,
                "required": True
            }
        }


class DepotModel(BaseModel):
    """Unified depot model with full PyVRP support"""
    x: int = Field(description="X coordinate (integer)")
    y: int = Field(description="Y coordinate (integer)")
    tw_early: Optional[int] = Field(default=0, description="Depot operating start time (minutes)")
    tw_late: Optional[int] = Field(default=1440, description="Depot operating end time (minutes)")
    capacity: Optional[Union[int, List[int]]] = Field(default=None, description="Depot capacity constraints")
    is_reload_depot: bool = Field(default=False, description="Can vehicles reload at this depot")
    reload_time: Optional[int] = Field(default=0, description="Time required for reload operation (minutes)")
    depot_type: Optional[str] = Field(default="main", description="Depot type: main, satellite, reload_only")
    
    class Config:
        schema_extra = {
            "example": {
                "x": 0,
                "y": 0
            }
        }


class VehicleTypeModel(BaseModel):
    """Unified vehicle type model with full PyVRP support"""
    num_available: int = Field(description="Number of available vehicles")
    capacity: Union[int, List[int]] = Field(description="Vehicle capacity (integer or list for multi-dimensional)")
    start_depot: int = Field(description="Start depot index")
    end_depot: Optional[int] = Field(default=None, description="End depot index (None for return to start)")
    fixed_cost: int = Field(default=0, description="Fixed cost for using vehicle")
    unit_distance_cost: Optional[float] = Field(default=1.0, description="Cost per unit distance")
    unit_duration_cost: Optional[float] = Field(default=0.0, description="Cost per unit time")
    tw_early: Optional[int] = Field(default=0, description="Shift start time (minutes)")
    tw_late: Optional[int] = Field(default=1440, description="Shift end time (minutes)")
    max_duration: Optional[int] = Field(default=480, description="Maximum route duration (minutes)")
    max_distance: Optional[int] = Field(default=200000, description="Maximum route distance (meters)")
    profile: Optional[str] = Field(default="default", description="Routing profile (car, truck, bicycle, etc.)")
    
    # Reload capabilities
    can_reload: bool = Field(default=False, description="Vehicle can reload during route")
    max_reloads: Optional[int] = Field(default=None, description="Maximum number of reloads per route")
    reload_depots: Optional[List[int]] = Field(default=None, description="Depot indices where vehicle can reload")
    
    # Break requirements
    max_work_duration: Optional[int] = Field(default=None, description="Maximum work duration before break required")
    break_duration: Optional[int] = Field(default=None, description="Required break duration (minutes)")
    
    # Advanced constraints
    forbidden_locations: Optional[List[int]] = Field(default=None, description="Client indices this vehicle cannot visit")
    required_locations: Optional[List[int]] = Field(default=None, description="Client indices this vehicle must visit if used")
    
    @validator('end_depot')
    def validate_end_depot(cls, v, values):
        if v is None:
            # If end_depot is None, it defaults to start_depot
            return values.get('start_depot', 0)
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "num_available": 5,
                "capacity": 1000,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 100,
                "tw_early": 0,
                "tw_late": 1440,
                "max_duration": 480,
                "max_distance": 200000
            }
        }


class RoutingProfile(BaseModel):
    """Routing profile definition for different vehicle types"""
    profile_name: str = Field(description="Profile identifier (e.g., 'car', 'truck', 'bicycle')")
    distance_matrix: List[List[int]] = Field(description="Distance matrix for this profile (meters)")
    duration_matrix: List[List[int]] = Field(description="Duration matrix for this profile (minutes)")
    description: Optional[str] = Field(default=None, description="Profile description")


class SolverConfig(BaseModel):
    """Advanced solver configuration options"""
    max_runtime: int = Field(default=60, description="Maximum runtime in seconds", ge=1, le=7200)
    
    # Stopping criteria
    max_iterations: Optional[int] = Field(default=None, description="Maximum number of iterations")
    target_objective: Optional[float] = Field(default=None, description="Target objective value")
    time_limit_intensification: Optional[int] = Field(default=None, description="Time limit for intensification phase")
    
    # Algorithm parameters
    population_size: Optional[int] = Field(default=25, description="Population size for genetic algorithm")
    min_population_size: Optional[int] = Field(default=10, description="Minimum population size")
    generation_size: Optional[int] = Field(default=40, description="Generation size")
    nb_iter_diversity_management: Optional[int] = Field(default=500, description="Iterations for diversity management")
    
    # Local search
    nb_granular: Optional[int] = Field(default=20, description="Granular neighborhood size")
    
    # Seed and reproducibility
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    # Penalty management
    penalty_capacity: Optional[float] = Field(default=100.0, description="Penalty for capacity violations")
    penalty_time_window: Optional[float] = Field(default=100.0, description="Penalty for time window violations")
    penalty_distance: Optional[float] = Field(default=1.0, description="Penalty for distance violations")
    penalty_duration: Optional[float] = Field(default=1.0, description="Penalty for duration violations")


class VRPProblemData(BaseModel):
    """Unified VRP problem data for all variants with full PyVRP support"""
    clients: List[ClientModel] = Field(description="List of clients/customers")
    depots: List[DepotModel] = Field(description="List of depots")
    vehicle_types: List[VehicleTypeModel] = Field(description="List of vehicle types")
    
    # Multiple matrix support
    distance_matrix: Optional[List[List[int]]] = Field(default=None, description="Default distance matrix in meters")
    duration_matrix: Optional[List[List[int]]] = Field(default=None, description="Default duration matrix in minutes")
    routing_profiles: Optional[List[RoutingProfile]] = Field(default=None, description="Multiple routing profiles")
    
    # Client groups
    client_groups: Optional[List[ClientGroup]] = Field(default=None, description="Client group definitions")
    
    # Solver configuration
    solver_config: Optional[SolverConfig] = Field(default=None, description="Advanced solver configuration")
    
    # Compatibility
    max_runtime: int = Field(default=60, description="Maximum runtime in seconds (deprecated - use solver_config)", ge=1, le=7200)
    
    @validator('clients')
    def validate_clients_not_empty(cls, v):
        if not v:
            raise ValueError('At least one client is required')
        return v
    
    @validator('depots')
    def validate_depots_not_empty(cls, v):
        if not v:
            raise ValueError('At least one depot is required')
        return v
    
    @validator('vehicle_types')
    def validate_vehicle_types_not_empty(cls, v):
        if not v:
            raise ValueError('At least one vehicle type is required')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "clients": [
                    {
                        "x": 100,
                        "y": 200,
                        "delivery": 50,
                        "service_duration": 10,
                        "tw_early": 480,
                        "tw_late": 1020
                    }
                ],
                "depots": [
                    {"x": 0, "y": 0}
                ],
                "vehicle_types": [
                    {
                        "num_available": 5,
                        "capacity": 1000,
                        "start_depot": 0,
                        "max_duration": 480
                    }
                ],
                "max_runtime": 60
            }
        }


class ReloadInfo(BaseModel):
    """Information about reload operations in a route"""
    depot_index: int = Field(description="Depot where reload occurred")
    position_in_route: int = Field(description="Position in route sequence")
    reload_duration: int = Field(description="Time spent reloading (minutes)")
    capacity_before: Union[int, List[int]] = Field(description="Vehicle capacity before reload")
    capacity_after: Union[int, List[int]] = Field(description="Vehicle capacity after reload")


class BreakInfo(BaseModel):
    """Information about driver breaks in a route"""
    position_in_route: int = Field(description="Position in route sequence")
    break_duration: int = Field(description="Break duration (minutes)")
    break_start_time: int = Field(description="Break start time (minutes from start)")


class UnifiedRouteModel(BaseModel):
    """Unified route model for solution with full PyVRP support"""
    vehicle_type: int = Field(description="Vehicle type index used")
    vehicle_id: Optional[int] = Field(default=None, description="Specific vehicle ID within type")
    start_depot: int = Field(description="Starting depot index")
    end_depot: int = Field(description="Ending depot index")
    clients: List[int] = Field(description="Sequence of client indices visited")
    
    # Cost and distance information
    distance: int = Field(description="Total route distance in meters")
    duration: int = Field(description="Total route duration in minutes")
    fixed_cost: float = Field(description="Fixed cost for using this vehicle")
    variable_cost: float = Field(description="Variable cost (distance + duration)")
    total_cost: float = Field(description="Total route cost")
    
    # Capacity and demand
    demand_served: Union[int, List[int]] = Field(description="Total demand served")
    max_load: Union[int, List[int]] = Field(description="Maximum load during route")
    capacity_utilization: float = Field(description="Capacity utilization ratio")
    
    # Timing information
    start_time: int = Field(description="Route start time (minutes from midnight)")
    end_time: int = Field(description="Route end time (minutes from midnight)")
    arrival_times: Optional[List[int]] = Field(default=None, description="Arrival times at each location")
    departure_times: Optional[List[int]] = Field(default=None, description="Departure times from each location")
    waiting_times: Optional[List[int]] = Field(default=None, description="Waiting times at each location")
    
    # Advanced features
    reloads: Optional[List[ReloadInfo]] = Field(default=None, description="Reload operations performed")
    breaks: Optional[List[BreakInfo]] = Field(default=None, description="Driver breaks taken")
    
    # Constraint violations
    capacity_violations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Capacity constraint violations")
    time_window_violations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Time window violations")
    
    # Route quality metrics
    num_clients: int = Field(description="Number of clients served")
    empty_distance: int = Field(description="Distance traveled empty (meters)")
    loaded_distance: int = Field(description="Distance traveled loaded (meters)")
    routing_profile: Optional[str] = Field(default=None, description="Routing profile used")
    
    class Config:
        schema_extra = {
            "example": {
                "vehicle_type": 0,
                "depot": 0,
                "clients": [2, 4, 1],
                "distance": 4567,
                "duration": 234,
                "demand_served": 75
            }
        }


class SolutionStatistics(BaseModel):
    """Detailed solution statistics"""
    total_clients: int = Field(description="Total number of clients")
    clients_served: int = Field(description="Number of clients served")
    clients_unserved: int = Field(description="Number of unserved clients")
    unserved_client_indices: Optional[List[int]] = Field(default=None, description="Indices of unserved clients")
    
    total_distance: int = Field(description="Total distance across all routes (meters)")
    total_duration: int = Field(description="Total duration across all routes (minutes)")
    total_fixed_cost: float = Field(description="Total fixed costs")
    total_variable_cost: float = Field(description="Total variable costs")
    
    vehicles_used: int = Field(description="Number of vehicles used")
    vehicles_available: int = Field(description="Number of vehicles available")
    
    average_capacity_utilization: float = Field(description="Average capacity utilization across routes")
    max_route_duration: int = Field(description="Maximum route duration (minutes)")
    min_route_duration: int = Field(description="Minimum route duration (minutes)")
    
    total_waiting_time: int = Field(description="Total waiting time across all routes (minutes)")
    total_reloads: int = Field(description="Total number of reload operations")
    total_breaks: int = Field(description="Total number of breaks taken")
    
    constraint_violations: Dict[str, int] = Field(description="Count of different constraint violations")
    penalty_costs: Dict[str, float] = Field(description="Penalty costs by violation type")


class AlgorithmInfo(BaseModel):
    """Information about the algorithm execution"""
    iterations: int = Field(description="Number of iterations performed")
    best_iteration: int = Field(description="Iteration where best solution was found")
    population_size: int = Field(description="Final population size")
    diversity_score: Optional[float] = Field(default=None, description="Population diversity score")
    convergence_info: Optional[Dict[str, Any]] = Field(default=None, description="Convergence information")


class UnifiedVRPSolution(BaseModel):
    """Unified VRP solution model with full PyVRP support"""
    status: str = Field(description="Solution status: optimal, feasible, infeasible, error")
    objective_value: float = Field(description="Objective function value")
    routes: List[UnifiedRouteModel] = Field(description="List of routes")
    
    # Timing and solver information
    computation_time: float = Field(description="Computation time in seconds")
    solver: str = Field(default="PyVRP", description="Solver used")
    solver_version: Optional[str] = Field(default=None, description="Solver version")
    
    # Detailed statistics
    statistics: Optional[SolutionStatistics] = Field(default=None, description="Detailed solution statistics")
    algorithm_info: Optional[AlgorithmInfo] = Field(default=None, description="Algorithm execution information")
    
    # Problem information
    problem_type: Optional[str] = Field(default=None, description="VRP variant (CVRP, VRPTW, etc.)")
    problem_size: Optional[Dict[str, int]] = Field(default=None, description="Problem size characteristics")
    
    # Quality metrics
    gap_to_best_known: Optional[float] = Field(default=None, description="Gap to best known solution (%)")
    solution_quality: Optional[str] = Field(default=None, description="Solution quality assessment")
    
    # Warnings and messages
    warnings: Optional[List[str]] = Field(default=None, description="Solution warnings")
    messages: Optional[List[str]] = Field(default=None, description="Additional messages")
    
    # Validation results
    is_feasible: bool = Field(description="Whether solution is feasible")
    feasibility_report: Optional[Dict[str, Any]] = Field(default=None, description="Detailed feasibility analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "optimal",
                "objective_value": 12345,
                "routes": [
                    {
                        "vehicle_type": 0,
                        "depot": 0,
                        "clients": [2, 4, 1],
                        "distance": 4567,
                        "duration": 234,
                        "demand_served": 75
                    }
                ],
                "computation_time": 1.23,
                "solver": "PyVRP"
            }
        }