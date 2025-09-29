"""
Advanced constraints models for commercial VRP
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from datetime import datetime, time

class ConstraintType(str, Enum):
    """Types of VRP constraints"""
    # Driver constraints
    DRIVER_BREAK = "driver_break"
    DRIVER_SHIFT = "driver_shift"
    DRIVER_SKILLS = "driver_skills"
    DRIVER_OVERTIME = "driver_overtime"
    
    # Vehicle constraints
    VEHICLE_COMPATIBILITY = "vehicle_compatibility"
    VEHICLE_EQUIPMENT = "vehicle_equipment"
    VEHICLE_ACCESS = "vehicle_access"
    MULTI_COMPARTMENT = "multi_compartment"
    
    # Customer constraints
    CUSTOMER_PRIORITY = "customer_priority"
    CUSTOMER_EXCLUSION = "customer_exclusion"
    SERVICE_DURATION = "service_duration"
    PRECEDENCE = "precedence"
    
    # Temporal constraints
    MULTIPLE_TIME_WINDOWS = "multiple_time_windows"
    PERIODIC_VISITS = "periodic_visits"
    TIME_DEPENDENCY = "time_dependency"
    
    # Capacity constraints
    MULTI_DIMENSIONAL_CAPACITY = "multi_dimensional_capacity"
    COMPARTMENT_COMPATIBILITY = "compartment_compatibility"
    LOADING_CONSTRAINTS = "loading_constraints"
    
    # Route constraints
    ROUTE_LENGTH_LIMIT = "route_length_limit"
    ROUTE_DURATION_LIMIT = "route_duration_limit"
    ROUTE_BALANCE = "route_balance"
    FORBIDDEN_SEQUENCES = "forbidden_sequences"

class DriverBreakConstraint(BaseModel):
    """Driver break requirements"""
    mandatory: bool = Field(True, description="Whether breaks are mandatory")
    min_work_before_break: int = Field(240, description="Minimum work time before break (minutes)")
    break_duration: int = Field(30, description="Break duration (minutes)")
    max_work_without_break: int = Field(360, description="Maximum work without break (minutes)")
    flexible_break_window: int = Field(60, description="Flexible break window (minutes)")
    multiple_breaks: bool = Field(True, description="Allow multiple breaks per route")

class DriverShiftConstraint(BaseModel):
    """Driver shift constraints"""
    max_shift_duration: int = Field(480, description="Maximum shift duration (minutes)")
    min_shift_duration: int = Field(120, description="Minimum shift duration (minutes)")
    overtime_allowed: bool = Field(True, description="Whether overtime is allowed")
    max_overtime: int = Field(120, description="Maximum overtime (minutes)")
    overtime_penalty: float = Field(50.0, description="Overtime penalty per minute")
    night_shift_premium: float = Field(1.5, description="Night shift premium multiplier")
    
class DriverSkillConstraint(BaseModel):
    """Driver skill requirements"""
    required_skills: List[str] = Field([], description="Required driver skills")
    skill_levels: Dict[str, int] = Field({}, description="Minimum skill levels (1-5)")
    certification_expiry: Dict[str, str] = Field({}, description="Certification expiry dates")
    training_requirements: List[str] = Field([], description="Training requirements")

class VehicleCompatibilityConstraint(BaseModel):
    """Vehicle compatibility constraints"""
    allowed_vehicle_types: List[str] = Field([], description="Allowed vehicle types")
    forbidden_vehicle_types: List[str] = Field([], description="Forbidden vehicle types")
    min_vehicle_capacity: float = Field(0.0, description="Minimum vehicle capacity")
    max_vehicle_capacity: float = Field(float('inf'), description="Maximum vehicle capacity")
    required_equipment: List[str] = Field([], description="Required vehicle equipment")

class VehicleAccessConstraint(BaseModel):
    """Vehicle access restrictions"""
    weight_restrictions: Dict[str, float] = Field({}, description="Weight restrictions by road type")
    height_restrictions: Dict[str, float] = Field({}, description="Height restrictions")
    width_restrictions: Dict[str, float] = Field({}, description="Width restrictions")
    environmental_zones: List[str] = Field([], description="Environmental zone restrictions")
    toll_avoidance: bool = Field(False, description="Avoid toll roads")
    urban_access_restrictions: Dict[str, Any] = Field({}, description="Urban access restrictions")

class MultiCompartmentConstraint(BaseModel):
    """Multi-compartment vehicle constraints"""
    compartments: Dict[str, Dict[str, Any]] = Field({}, description="Compartment definitions")
    product_compartment_mapping: Dict[str, str] = Field({}, description="Product to compartment mapping")
    cross_contamination_rules: List[Dict[str, Any]] = Field([], description="Cross-contamination rules")
    loading_sequence_rules: List[Dict[str, Any]] = Field([], description="Loading sequence rules")

class CustomerPriorityConstraint(BaseModel):
    """Customer priority constraints"""
    priority_levels: Dict[str, int] = Field({}, description="Customer priority levels (1=highest)")
    vip_customers: List[str] = Field([], description="VIP customer IDs")
    priority_penalty: float = Field(100.0, description="Penalty for not serving high priority customers")
    mandatory_customers: List[str] = Field([], description="Customers that must be served")

class PrecedenceConstraint(BaseModel):
    """Precedence constraints between customers"""
    precedence_pairs: List[Tuple[str, str]] = Field([], description="(before, after) customer pairs")
    same_route_required: bool = Field(True, description="Must be on same route")
    max_time_between: Optional[int] = Field(None, description="Maximum time between visits (minutes)")
    min_time_between: Optional[int] = Field(None, description="Minimum time between visits (minutes)")

class MultipleTimeWindowConstraint(BaseModel):
    """Multiple time windows for customers"""
    time_windows: List[Tuple[int, int]] = Field([], description="List of (start, end) time windows")
    preferred_windows: List[int] = Field([], description="Indices of preferred time windows")
    preference_bonus: float = Field(10.0, description="Bonus for preferred time windows")
    window_violation_penalty: float = Field(100.0, description="Penalty for window violations")

class PeriodicVisitConstraint(BaseModel):
    """Periodic visit constraints"""
    visit_frequency: int = Field(1, description="Required visits per planning period")
    min_interval: int = Field(0, description="Minimum days between visits")
    max_interval: int = Field(7, description="Maximum days between visits")
    preferred_days: List[int] = Field([], description="Preferred days of week (0=Monday)")
    forbidden_days: List[int] = Field([], description="Forbidden days of week")

class MultiDimensionalCapacityConstraint(BaseModel):
    """Multi-dimensional capacity constraints"""
    dimensions: Dict[str, float] = Field({}, description="Capacity dimensions (weight, volume, etc.)")
    dimension_units: Dict[str, str] = Field({}, description="Units for each dimension")
    mixed_loading_rules: Dict[str, Any] = Field({}, description="Mixed loading compatibility rules")
    stacking_constraints: Dict[str, Any] = Field({}, description="Stacking constraints")

class LoadingConstraint(BaseModel):
    """Loading sequence and constraints"""
    loading_time_per_unit: float = Field(5.0, description="Loading time per unit (minutes)")
    unloading_time_per_unit: float = Field(3.0, description="Unloading time per unit (minutes)")
    fifo_required: bool = Field(False, description="First-in-first-out loading required")
    lifo_required: bool = Field(False, description="Last-in-first-out loading required")
    fragile_items_on_top: bool = Field(True, description="Fragile items must be loaded on top")

class RouteConstraint(BaseModel):
    """Route-level constraints"""
    max_route_distance: Optional[float] = Field(None, description="Maximum route distance")
    max_route_duration: Optional[int] = Field(None, description="Maximum route duration (minutes)")
    max_stops_per_route: Optional[int] = Field(None, description="Maximum stops per route")
    min_stops_per_route: Optional[int] = Field(1, description="Minimum stops per route")
    route_balance_tolerance: float = Field(0.2, description="Route balance tolerance")

class ForbiddenSequenceConstraint(BaseModel):
    """Forbidden customer sequences"""
    forbidden_sequences: List[List[str]] = Field([], description="Lists of forbidden customer sequences")
    forbidden_pairs: List[Tuple[str, str]] = Field([], description="Pairs of customers that cannot be consecutive")
    required_separations: Dict[str, Dict[str, int]] = Field({}, description="Required minimum separations")

class AdvancedConstraints(BaseModel):
    """Collection of advanced VRP constraints"""
    # Driver constraints
    driver_breaks: Optional[DriverBreakConstraint] = Field(None)
    driver_shifts: Optional[DriverShiftConstraint] = Field(None)
    driver_skills: Optional[DriverSkillConstraint] = Field(None)
    
    # Vehicle constraints
    vehicle_compatibility: Optional[VehicleCompatibilityConstraint] = Field(None)
    vehicle_access: Optional[VehicleAccessConstraint] = Field(None)
    multi_compartment: Optional[MultiCompartmentConstraint] = Field(None)
    
    # Customer constraints
    customer_priority: Optional[CustomerPriorityConstraint] = Field(None)
    precedence: Optional[PrecedenceConstraint] = Field(None)
    
    # Temporal constraints
    multiple_time_windows: Dict[str, MultipleTimeWindowConstraint] = Field({}, description="Customer ID to time windows")
    periodic_visits: Dict[str, PeriodicVisitConstraint] = Field({}, description="Customer ID to visit patterns")
    
    # Capacity constraints
    multi_dimensional_capacity: Optional[MultiDimensionalCapacityConstraint] = Field(None)
    loading_constraints: Optional[LoadingConstraint] = Field(None)
    
    # Route constraints
    route_constraints: Optional[RouteConstraint] = Field(None)
    forbidden_sequences: Optional[ForbiddenSequenceConstraint] = Field(None)
    
    # Custom constraints
    custom_constraints: List[Dict[str, Any]] = Field([], description="Custom constraint definitions")
    
    class Config:
        schema_extra = {
            "example": {
                "driver_breaks": {
                    "mandatory": True,
                    "min_work_before_break": 240,
                    "break_duration": 30
                },
                "vehicle_compatibility": {
                    "allowed_vehicle_types": ["truck", "van"],
                    "required_equipment": ["refrigeration", "tailgate"]
                },
                "customer_priority": {
                    "vip_customers": ["CUST_001", "CUST_002"],
                    "priority_levels": {"CUST_001": 1, "CUST_002": 1}
                },
                "route_constraints": {
                    "max_route_distance": 200.0,
                    "max_route_duration": 480,
                    "max_stops_per_route": 15
                }
            }
        }

class ConstraintViolation(BaseModel):
    """Constraint violation information"""
    constraint_type: ConstraintType
    violation_description: str
    severity: str = Field("medium", description="Severity: low, medium, high, critical")
    penalty_cost: float = Field(0.0, description="Penalty cost for this violation")
    affected_entities: List[str] = Field([], description="Affected routes, customers, vehicles, etc.")
    suggested_resolution: Optional[str] = Field(None, description="Suggested resolution")

class ConstraintValidationResult(BaseModel):
    """Result of constraint validation"""
    is_valid: bool
    violations: List[ConstraintViolation]
    total_penalty: float
    constraint_satisfaction_score: float = Field(description="Score from 0-100")
    recommendations: List[str] = Field([], description="Recommendations for improvement")

class ConstraintConfiguration(BaseModel):
    """Global constraint configuration"""
    strict_mode: bool = Field(False, description="Reject solutions with any violations")
    penalty_weights: Dict[str, float] = Field({}, description="Penalty weights by constraint type")
    violation_tolerance: float = Field(0.1, description="Acceptable violation rate")
    priority_enforcement: bool = Field(True, description="Enforce constraint priorities")
    constraint_relaxation: Dict[str, bool] = Field({}, description="Relaxation settings per constraint")