"""
Multi-objective optimization models for VRP
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class ObjectiveType(str, Enum):
    """Types of optimization objectives"""
    MINIMIZE_DISTANCE = "minimize_distance"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_VEHICLES = "minimize_vehicles"
    MINIMIZE_CO2 = "minimize_co2"
    MAXIMIZE_SERVICE_LEVEL = "maximize_service_level"
    MINIMIZE_WAITING_TIME = "minimize_waiting_time"
    BALANCE_WORKLOAD = "balance_workload"
    MINIMIZE_FUEL_CONSUMPTION = "minimize_fuel_consumption"
    MAXIMIZE_PROFIT = "maximize_profit"

class ObjectiveFunction(BaseModel):
    """Single objective function definition"""
    type: ObjectiveType = Field(description="Type of objective")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Objective weight (0-1)")
    priority: int = Field(1, ge=1, le=10, description="Objective priority (1=highest)")
    target_value: Optional[float] = Field(None, description="Target value for goal programming")
    tolerance: Optional[float] = Field(None, description="Tolerance for target value")
    
    @validator('weight')
    def validate_weight(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Weight must be between 0.0 and 1.0')
        return v

class MultiObjectiveConfig(BaseModel):
    """Multi-objective optimization configuration"""
    objectives: List[ObjectiveFunction] = Field(description="List of objectives to optimize")
    method: str = Field("weighted_sum", description="Multi-objective optimization method")
    normalize_objectives: bool = Field(True, description="Normalize objectives before combining")
    pareto_solutions: bool = Field(False, description="Generate Pareto-optimal solutions")
    max_pareto_solutions: int = Field(10, description="Maximum number of Pareto solutions")
    
    @validator('objectives')
    def validate_objectives(cls, v):
        if len(v) < 1:
            raise ValueError('At least one objective must be specified')
        
        # Check weight sum if using weighted sum method
        total_weight = sum(obj.weight for obj in v)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f'Total weight must equal 1.0, got {total_weight}')
        
        # Check for duplicate objectives
        obj_types = [obj.type for obj in v]
        if len(obj_types) != len(set(obj_types)):
            raise ValueError('Duplicate objective types not allowed')
        
        return v

class CostStructure(BaseModel):
    """Cost structure for cost-based objectives"""
    distance_cost_per_km: float = Field(0.5, description="Cost per kilometer traveled")
    time_cost_per_hour: float = Field(25.0, description="Cost per hour (driver wages)")
    vehicle_fixed_cost: float = Field(100.0, description="Fixed cost per vehicle used")
    fuel_cost_per_liter: float = Field(1.5, description="Fuel cost per liter")
    fuel_consumption_per_km: float = Field(0.08, description="Fuel consumption per km")
    co2_cost_per_kg: float = Field(0.05, description="CO2 emission cost per kg")
    overtime_cost_multiplier: float = Field(1.5, description="Overtime cost multiplier")
    waiting_time_cost_per_hour: float = Field(15.0, description="Cost of waiting time per hour")

class ServiceLevelConfig(BaseModel):
    """Service level configuration for service-based objectives"""
    on_time_delivery_weight: float = Field(0.4, description="Weight for on-time delivery")
    customer_satisfaction_weight: float = Field(0.3, description="Weight for customer satisfaction")
    service_quality_weight: float = Field(0.3, description="Weight for service quality")
    time_window_violation_penalty: float = Field(100.0, description="Penalty for time window violations")
    preferred_time_bonus: float = Field(10.0, description="Bonus for preferred time delivery")

class EnvironmentalConfig(BaseModel):
    """Environmental impact configuration"""
    co2_emission_factor: float = Field(2.31, description="CO2 kg per liter of fuel")
    fuel_efficiency_factor: float = Field(1.0, description="Vehicle fuel efficiency factor")
    eco_driving_bonus: float = Field(0.1, description="Eco-driving efficiency bonus")
    electric_vehicle_factor: float = Field(0.0, description="Electric vehicle emission factor")

class WorkloadBalanceConfig(BaseModel):
    """Workload balance configuration"""
    max_workload_deviation: float = Field(0.2, description="Maximum allowed workload deviation")
    balance_metric: str = Field("distance", description="Metric to balance (distance, time, stops)")
    penalty_per_deviation: float = Field(50.0, description="Penalty per unit of deviation")
    fair_distribution_bonus: float = Field(20.0, description="Bonus for fair distribution")

class MultiObjectiveResult(BaseModel):
    """Result of multi-objective optimization"""
    total_score: float = Field(description="Combined objective score")
    objective_values: Dict[str, float] = Field(description="Individual objective values")
    normalized_values: Dict[str, float] = Field(description="Normalized objective values")
    weighted_contributions: Dict[str, float] = Field(description="Weighted contributions of each objective")
    pareto_rank: Optional[int] = Field(None, description="Pareto rank if applicable")
    dominated_solutions: Optional[int] = Field(None, description="Number of dominated solutions")
    is_pareto_optimal: Optional[bool] = Field(None, description="Whether solution is Pareto optimal")

class ParetoSolution(BaseModel):
    """Single solution in Pareto frontier"""
    solution_id: str = Field(description="Unique solution identifier")
    objective_values: Dict[str, float] = Field(description="Objective function values")
    routes: List[Dict[str, Any]] = Field(description="Route solution")
    performance_metrics: Dict[str, Any] = Field(description="Additional performance metrics")
    trade_offs: Dict[str, str] = Field(description="Trade-off descriptions")

class ParetoFrontier(BaseModel):
    """Set of Pareto-optimal solutions"""
    solutions: List[ParetoSolution] = Field(description="Pareto-optimal solutions")
    dominated_solutions_count: int = Field(description="Number of dominated solutions found")
    frontier_size: int = Field(description="Size of Pareto frontier")
    objective_ranges: Dict[str, Dict[str, float]] = Field(description="Min/max ranges for each objective")
    trade_off_analysis: Dict[str, str] = Field(description="Trade-off analysis between objectives")

class OptimizationConstraint(BaseModel):
    """Advanced optimization constraint"""
    constraint_type: str = Field(description="Type of constraint")
    parameters: Dict[str, Any] = Field(description="Constraint parameters")
    violation_penalty: float = Field(1000.0, description="Penalty for constraint violation")
    soft_constraint: bool = Field(False, description="Whether constraint can be violated")
    priority: int = Field(1, description="Constraint priority")

class AdvancedVRPObjectives(BaseModel):
    """Advanced VRP objectives configuration"""
    multi_objective: MultiObjectiveConfig = Field(description="Multi-objective configuration")
    cost_structure: Optional[CostStructure] = Field(None, description="Cost structure for cost objectives")
    service_level: Optional[ServiceLevelConfig] = Field(None, description="Service level configuration")
    environmental: Optional[EnvironmentalConfig] = Field(None, description="Environmental configuration")
    workload_balance: Optional[WorkloadBalanceConfig] = Field(None, description="Workload balance configuration")
    additional_constraints: List[OptimizationConstraint] = Field([], description="Additional constraints")
    
    class Config:
        schema_extra = {
            "example": {
                "multi_objective": {
                    "objectives": [
                        {"type": "minimize_distance", "weight": 0.4, "priority": 1},
                        {"type": "minimize_cost", "weight": 0.3, "priority": 2},
                        {"type": "minimize_co2", "weight": 0.3, "priority": 3}
                    ],
                    "method": "weighted_sum",
                    "normalize_objectives": True
                },
                "cost_structure": {
                    "distance_cost_per_km": 0.5,
                    "time_cost_per_hour": 25.0,
                    "vehicle_fixed_cost": 100.0
                },
                "environmental": {
                    "co2_emission_factor": 2.31,
                    "fuel_efficiency_factor": 1.0
                }
            }
        }