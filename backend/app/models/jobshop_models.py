from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any, Literal
from enum import Enum


class ProblemType(str, Enum):
    job_shop = "job_shop"
    flexible_job_shop = "flexible_job_shop"
    flow_shop = "flow_shop"
    hybrid_flow_shop = "hybrid_flow_shop"
    project_scheduling = "project_scheduling"


class OptimizationObjective(str, Enum):
    makespan = "makespan"
    total_completion_time = "total_completion_time"
    total_weighted_completion_time = "total_weighted_completion_time"
    maximum_lateness = "maximum_lateness"
    total_tardiness = "total_tardiness"
    weighted_tardiness = "weighted_tardiness"


class ConstraintType(str, Enum):
    precedence = "precedence"
    resource_capacity = "resource_capacity"
    time_windows = "time_windows"
    setup_times = "setup_times"
    release_times = "release_times"
    due_dates = "due_dates"
    transportation = "transportation"


# Core Models
class Operation(BaseModel):
    id: str
    job_id: str
    machine_id: Optional[str] = None  # For flexible scheduling
    duration: int = Field(..., ge=1, description="Duration in time units")
    position_in_job: int = Field(..., ge=0, description="Position in job sequence")
    setup_time: Optional[int] = Field(0, ge=0, description="Setup time before operation")
    eligible_machines: Optional[List[str]] = Field(default=None, description="Machines that can process this operation")
    earliest_start: Optional[int] = Field(0, ge=0, description="Earliest start time")
    latest_finish: Optional[int] = Field(None, ge=0, description="Latest finish time")
    skill_requirements: Optional[List[str]] = Field(default=None, description="Required skills")


class Job(BaseModel):
    id: str
    name: Optional[str] = None
    operations: List[Operation]
    priority: int = Field(1, ge=1, description="Job priority (higher number = higher priority)")
    weight: float = Field(1.0, ge=0, description="Weight for objective function")
    release_time: int = Field(0, ge=0, description="Job release time")
    due_date: Optional[int] = Field(None, ge=0, description="Job due date")
    deadline: Optional[int] = Field(None, ge=0, description="Hard deadline")


class Machine(BaseModel):
    id: str
    name: Optional[str] = None
    capacity: int = Field(1, ge=1, description="Machine capacity")
    available_from: int = Field(0, ge=0, description="Machine availability start time")
    available_until: Optional[int] = Field(None, ge=0, description="Machine availability end time")
    setup_matrix: Optional[Dict[str, Dict[str, int]]] = Field(default=None, description="Setup times between operations")
    skills: Optional[List[str]] = Field(default=None, description="Machine skills/capabilities")
    maintenance_windows: Optional[List[Dict[str, int]]] = Field(default=None, description="Maintenance time windows")


class Resource(BaseModel):
    id: str
    name: Optional[str] = None
    capacity: int = Field(1, ge=1, description="Resource capacity")
    renewable: bool = Field(True, description="Whether resource is renewable")
    cost_per_unit: Optional[float] = Field(0.0, ge=0, description="Cost per unit of resource")


class TimeWindow(BaseModel):
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    
    class Config:
        validate_assignment = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.start >= self.end:
            raise ValueError("Start time must be less than end time")


# Request Models
class JobShopProblem(BaseModel):
    problem_type: ProblemType = ProblemType.job_shop
    jobs: List[Job]
    machines: List[Machine]
    resources: Optional[List[Resource]] = Field(default=None)
    optimization_objective: OptimizationObjective = OptimizationObjective.makespan
    time_horizon: Optional[int] = Field(None, ge=1, description="Maximum scheduling horizon")
    
    # Constraint settings
    allow_preemption: bool = Field(False, description="Allow operation preemption")
    transportation_times: Optional[Dict[str, Dict[str, int]]] = Field(default=None)
    setup_times_included: bool = Field(False, description="Whether setup times are included")
    
    # Optimization parameters
    max_solve_time_seconds: int = Field(300, ge=1, description="Maximum solving time")
    optimality_gap_tolerance: float = Field(0.01, ge=0, le=1, description="Optimality gap tolerance")
    
    class Config:
        use_enum_values = True


class FlexibleJobShopProblem(JobShopProblem):
    problem_type: Literal[ProblemType.flexible_job_shop] = ProblemType.flexible_job_shop
    machine_eligibility: Dict[str, List[str]] = Field(..., description="Operation to eligible machines mapping")


class FlowShopProblem(JobShopProblem):
    problem_type: Literal[ProblemType.flow_shop] = ProblemType.flow_shop
    machine_sequence: List[str] = Field(..., description="Sequence of machines that all jobs must follow")
    

class HybridFlowShopProblem(JobShopProblem):
    problem_type: Literal[ProblemType.hybrid_flow_shop] = ProblemType.hybrid_flow_shop
    stages: List[Dict[str, Any]] = Field(..., description="Stages with parallel machines")
    stage_sequence: List[str] = Field(..., description="Sequence of stages that all jobs must follow")
    

class ProjectSchedulingProblem(BaseModel):
    problem_type: Literal[ProblemType.project_scheduling] = ProblemType.project_scheduling
    activities: List[Dict[str, Any]] = Field(..., description="Project activities")
    precedence_relations: List[Dict[str, str]] = Field(..., description="Activity precedence relations")
    resources: List[Resource]
    optimization_objective: OptimizationObjective = OptimizationObjective.makespan
    max_solve_time_seconds: int = Field(300, ge=1)


# Solution Models
class ScheduledOperation(BaseModel):
    operation_id: str
    job_id: str
    machine_id: str
    start_time: int
    end_time: int
    duration: int
    setup_time: Optional[int] = 0


class JobSchedule(BaseModel):
    job_id: str
    operations: List[ScheduledOperation]
    start_time: int
    completion_time: int
    tardiness: int = 0
    lateness: int = 0


class MachineSchedule(BaseModel):
    machine_id: str
    operations: List[ScheduledOperation]
    utilization: float = Field(..., ge=0, le=1)
    idle_time: int = Field(..., ge=0)


class ResourceUsage(BaseModel):
    resource_id: str
    time_periods: List[Dict[str, Union[int, float]]]
    max_usage: int
    average_usage: float


class SolutionMetrics(BaseModel):
    makespan: int
    total_completion_time: int
    total_tardiness: int
    total_weighted_tardiness: float
    maximum_lateness: int
    average_machine_utilization: float
    objective_value: float
    solve_time_seconds: float
    optimality_gap: Optional[float] = None
    feasible: bool = True


class JobShopSolution(BaseModel):
    problem_type: str
    job_schedules: List[JobSchedule]
    machine_schedules: List[MachineSchedule]
    resource_usage: Optional[List[ResourceUsage]] = None
    metrics: SolutionMetrics
    gantt_chart_data: Optional[Dict[str, Any]] = None
    solution_status: str = "OPTIMAL"
    
    # Additional analysis
    critical_path: Optional[List[str]] = None
    bottleneck_machines: Optional[List[str]] = None
    improvement_suggestions: Optional[List[str]] = None
    
    # New advanced analysis fields
    bottleneck_analysis: Optional[Dict[str, Any]] = None
    advanced_kpis: Optional[Dict[str, Any]] = None


# Configuration Models
class SolverConfig(BaseModel):
    solver_name: str = "CP-SAT"
    time_limit_seconds: int = Field(300, ge=1)
    num_workers: int = Field(1, ge=1)
    log_level: int = Field(1, ge=0, le=4)
    optimality_gap: float = Field(0.01, ge=0, le=1)
    search_branching: str = Field("AUTOMATIC", description="Search branching strategy")
    use_heuristics: bool = Field(True, description="Use heuristic search")


class AnalysisConfig(BaseModel):
    include_gantt_chart: bool = Field(True)
    include_utilization_analysis: bool = Field(True)
    include_bottleneck_analysis: bool = Field(True)
    include_critical_path: bool = Field(True)
    include_improvement_suggestions: bool = Field(True)


# Batch Processing Models
class BatchJobShopRequest(BaseModel):
    problems: List[JobShopProblem]
    solver_config: SolverConfig = SolverConfig()
    analysis_config: AnalysisConfig = AnalysisConfig()


class BatchJobShopResponse(BaseModel):
    solutions: List[JobShopSolution]
    comparison_metrics: Optional[Dict[str, Any]] = None
    batch_statistics: Optional[Dict[str, Any]] = None


# Multi-objective Models
class MultiObjectiveWeights(BaseModel):
    makespan_weight: float = Field(1.0, ge=0)
    tardiness_weight: float = Field(0.0, ge=0)
    completion_time_weight: float = Field(0.0, ge=0)
    resource_cost_weight: float = Field(0.0, ge=0)


class MultiObjectiveProblem(JobShopProblem):
    objective_weights: MultiObjectiveWeights = MultiObjectiveWeights()
    pareto_analysis: bool = Field(False, description="Perform Pareto frontier analysis")


# Real-time Scheduling Models
class ReschedulingTrigger(BaseModel):
    trigger_type: str  # "new_job", "machine_breakdown", "operation_delay", "priority_change"
    timestamp: int
    affected_entities: List[str]  # Job IDs, Machine IDs, etc.
    parameters: Optional[Dict[str, Any]] = None


class RealTimeSchedulingRequest(BaseModel):
    current_schedule: JobShopSolution
    trigger: ReschedulingTrigger
    rescheduling_scope: str = "full"  # "full", "partial", "local"
    preserve_started_operations: bool = Field(True)
    max_rescheduling_time: int = Field(60, description="Max rescheduling time in seconds")


# Request Models for API endpoints
class JobShopSolveRequest(BaseModel):
    problem: JobShopProblem
    solver_config: Optional[SolverConfig] = None
    analysis_config: Optional[AnalysisConfig] = None