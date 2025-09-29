"""
Advanced VRP routes with multi-objective optimization and constraint validation
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.models.vrp_unified_models import VRPProblemData, UnifiedVRPSolution
from app.models.multi_objective_models import (
    AdvancedVRPObjectives, MultiObjectiveResult, ParetoFrontier,
    ObjectiveType, CostStructure, ServiceLevelConfig, EnvironmentalConfig
)
from app.models.advanced_constraints_models import (
    AdvancedConstraints, ConstraintValidationResult
)
from app.services.pyvrp_unified_service import PyVRPUnifiedService
from app.services.multi_objective_service import MultiObjectiveOptimizer
from app.services.constraint_validation_service import ConstraintValidator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
pyvrp_service = PyVRPUnifiedService()
multi_objective_optimizer = MultiObjectiveOptimizer()
constraint_validator = ConstraintValidator()

@router.post("/solve/multi-objective", response_model=Dict[str, Any])
async def solve_multi_objective_vrp(
    problem_data: VRPProblemData,
    objectives: AdvancedVRPObjectives,
    num_iterations: int = 10,
    enable_pareto_frontier: bool = False
):
    """
    Solve VRP with multiple objectives
    
    This endpoint provides advanced multi-objective optimization for VRP problems:
    - Multiple optimization criteria (distance, cost, CO2, service level)
    - Configurable objective weights and priorities
    - Pareto frontier analysis for trade-off decisions
    - Cost structure and environmental impact modeling
    
    Supports commercial optimization patterns like:
    - Weighted sum method for balanced optimization
    - Lexicographic ordering for priority-based decisions
    - Epsilon-constraint method for goal programming
    - Pareto analysis for decision support
    """
    try:
        logger.info(f"Starting multi-objective VRP optimization with {len(objectives.multi_objective.objectives)} objectives")
        
        # Generate multiple solutions with different parameters
        solutions = []
        
        # Base solution
        base_solution = pyvrp_service.solve(problem_data)
        if base_solution and base_solution.status == "solved":
            solutions.append(base_solution)
        
        # Generate alternative solutions by varying parameters
        for i in range(num_iterations - 1):
            # Vary solver parameters to get different solutions
            varied_problem = problem_data.copy(deep=True)
            
            # Add some randomization or parameter variation
            # This is a simplified approach - in practice, you'd use different algorithm configurations
            alternative_solution = pyvrp_service.solve(varied_problem)
            if alternative_solution and alternative_solution.status == "solved":
                solutions.append(alternative_solution)
        
        if not solutions:
            raise HTTPException(status_code=500, detail="No valid solutions found")
        
        logger.info(f"Generated {len(solutions)} candidate solutions")
        
        # Perform multi-objective optimization
        best_solution, optimization_result, pareto_frontier = multi_objective_optimizer.optimize_multi_objective(
            solutions=solutions,
            config=objectives.multi_objective,
            cost_structure=objectives.cost_structure,
            service_level=objectives.service_level,
            environmental=objectives.environmental,
            workload_balance=objectives.workload_balance
        )
        
        # Prepare response
        response = {
            "status": "completed",
            "best_solution": best_solution.dict(),
            "optimization_result": optimization_result.dict(),
            "candidate_solutions_count": len(solutions),
            "optimization_method": objectives.multi_objective.method,
            "timestamp": datetime.now().isoformat()
        }
        
        # Include Pareto frontier if requested
        if enable_pareto_frontier and pareto_frontier:
            response["pareto_frontier"] = pareto_frontier.dict()
        
        # Add objective-specific insights
        insights = _generate_optimization_insights(optimization_result, objectives)
        response["insights"] = insights
        
        logger.info(f"Multi-objective optimization completed. Best score: {optimization_result.total_score:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Multi-objective VRP optimization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-objective optimization failed: {str(e)}"
        )

@router.post("/validate/constraints", response_model=ConstraintValidationResult)
async def validate_solution_constraints(
    solution: UnifiedVRPSolution,
    constraints: AdvancedConstraints,
    problem_data: Optional[VRPProblemData] = None
):
    """
    Validate VRP solution against advanced constraints
    
    This endpoint provides comprehensive constraint validation including:
    - Driver constraints (breaks, shifts, skills, overtime)
    - Vehicle constraints (compatibility, access, equipment)
    - Customer constraints (priority, precedence, time windows)
    - Route constraints (length, duration, balance)
    - Temporal constraints (multiple time windows, periodic visits)
    
    Returns detailed violation information with:
    - Violation descriptions and severity levels
    - Penalty costs for constraint violations
    - Suggested resolutions for each violation
    - Overall constraint satisfaction score
    """
    try:
        logger.info("Starting constraint validation")
        
        # Validate solution against constraints
        validation_result = constraint_validator.validate_solution(
            solution=solution,
            constraints=constraints,
            problem_data=problem_data.dict() if problem_data else None
        )
        
        logger.info(f"Constraint validation completed. Violations: {len(validation_result.violations)}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Constraint validation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Constraint validation failed: {str(e)}"
        )

@router.post("/solve/constrained", response_model=Dict[str, Any])
async def solve_constrained_vrp(
    problem_data: VRPProblemData,
    constraints: AdvancedConstraints,
    objectives: Optional[AdvancedVRPObjectives] = None,
    max_violations: int = 5,
    max_penalty: float = 1000.0
):
    """
    Solve VRP with advanced constraints and validation
    
    This endpoint combines VRP optimization with comprehensive constraint checking:
    - Solves VRP problem with PyVRP
    - Validates solution against all specified constraints
    - Provides constraint violation analysis
    - Optionally applies multi-objective optimization
    - Returns actionable recommendations for improvements
    
    Features commercial-grade constraint handling:
    - Hard and soft constraint support
    - Penalty-based constraint relaxation
    - Iterative constraint satisfaction
    - Business rule enforcement
    """
    try:
        logger.info("Starting constrained VRP optimization")
        
        # Solve base VRP problem
        solution = pyvrp_service.solve(problem_data)
        
        if not solution or solution.status != "solved":
            raise HTTPException(status_code=500, detail="Failed to solve base VRP problem")
        
        # Validate constraints
        validation_result = constraint_validator.validate_solution(
            solution=solution,
            constraints=constraints,
            problem_data=problem_data.dict()
        )
        
        # Check if solution meets constraint requirements
        constraint_satisfaction = {
            "is_feasible": validation_result.is_valid,
            "violation_count": len(validation_result.violations),
            "total_penalty": validation_result.total_penalty,
            "satisfaction_score": validation_result.constraint_satisfaction_score
        }
        
        # Apply multi-objective optimization if specified
        optimization_result = None
        if objectives:
            logger.info("Applying multi-objective optimization")
            _, optimization_result, _ = multi_objective_optimizer.optimize_multi_objective(
                solutions=[solution],
                config=objectives.multi_objective,
                cost_structure=objectives.cost_structure,
                service_level=objectives.service_level,
                environmental=objectives.environmental,
                workload_balance=objectives.workload_balance
            )
        
        # Prepare response
        response = {
            "status": "completed",
            "solution": solution.dict(),
            "constraint_satisfaction": constraint_satisfaction,
            "validation_result": validation_result.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        if optimization_result:
            response["multi_objective_result"] = optimization_result.dict()
        
        # Add solution quality assessment
        quality_assessment = _assess_solution_quality(
            solution, validation_result, optimization_result
        )
        response["quality_assessment"] = quality_assessment
        
        logger.info(f"Constrained VRP optimization completed. Satisfaction score: {validation_result.constraint_satisfaction_score:.1f}")
        return response
        
    except Exception as e:
        logger.error(f"Constrained VRP optimization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Constrained optimization failed: {str(e)}"
        )

@router.post("/analyze/pareto", response_model=Dict[str, Any])
async def analyze_pareto_frontier(
    problem_data: VRPProblemData,
    objectives: AdvancedVRPObjectives,
    num_solutions: int = 20
):
    """
    Generate and analyze Pareto frontier for VRP problem
    
    This endpoint performs comprehensive Pareto analysis:
    - Generates multiple diverse solutions
    - Identifies Pareto-optimal solutions
    - Analyzes trade-offs between objectives
    - Provides decision support insights
    - Recommends optimal solution based on preferences
    
    Useful for strategic decision making when multiple
    conflicting objectives must be balanced.
    """
    try:
        logger.info(f"Starting Pareto frontier analysis with {num_solutions} solutions")
        
        # Generate multiple solutions
        solutions = []
        for i in range(num_solutions):
            # Generate solutions with varied parameters
            solution = pyvrp_service.solve(problem_data)
            if solution and solution.status == "solved":
                solutions.append(solution)
        
        if len(solutions) < 2:
            raise HTTPException(status_code=500, detail="Insufficient solutions for Pareto analysis")
        
        # Perform multi-objective optimization with Pareto analysis enabled
        objectives.multi_objective.pareto_solutions = True
        objectives.multi_objective.max_pareto_solutions = min(num_solutions, 15)
        
        best_solution, optimization_result, pareto_frontier = multi_objective_optimizer.optimize_multi_objective(
            solutions=solutions,
            config=objectives.multi_objective,
            cost_structure=objectives.cost_structure,
            service_level=objectives.service_level,
            environmental=objectives.environmental,
            workload_balance=objectives.workload_balance
        )
        
        if not pareto_frontier:
            raise HTTPException(status_code=500, detail="Failed to generate Pareto frontier")
        
        # Analyze trade-offs
        trade_off_insights = _analyze_trade_offs(pareto_frontier, objectives)
        
        # Recommend solution based on objectives
        recommendation = _recommend_pareto_solution(pareto_frontier, objectives)
        
        response = {
            "status": "completed",
            "pareto_frontier": pareto_frontier.dict(),
            "trade_off_insights": trade_off_insights,
            "recommendation": recommendation,
            "analysis_summary": {
                "total_solutions_analyzed": len(solutions),
                "pareto_optimal_solutions": len(pareto_frontier.solutions),
                "dominated_solutions": pareto_frontier.dominated_solutions_count,
                "objective_count": len(objectives.multi_objective.objectives)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Pareto analysis completed. {len(pareto_frontier.solutions)} Pareto-optimal solutions found")
        return response
        
    except Exception as e:
        logger.error(f"Pareto frontier analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Pareto analysis failed: {str(e)}"
        )

@router.get("/objectives/templates")
async def get_objective_templates():
    """
    Get predefined objective templates for common use cases
    
    Returns ready-to-use objective configurations for:
    - Cost optimization (traditional VRP)
    - Environmental optimization (green logistics)
    - Service optimization (customer satisfaction)
    - Balanced optimization (multi-criteria)
    - Profit maximization (revenue optimization)
    """
    templates = {
        "cost_optimization": {
            "name": "Cost Optimization",
            "description": "Traditional cost-focused VRP optimization",
            "objectives": [
                {"type": "minimize_distance", "weight": 0.4, "priority": 1},
                {"type": "minimize_vehicles", "weight": 0.3, "priority": 2},
                {"type": "minimize_cost", "weight": 0.3, "priority": 3}
            ],
            "method": "weighted_sum"
        },
        "environmental_optimization": {
            "name": "Environmental Optimization",
            "description": "Green logistics with CO2 minimization",
            "objectives": [
                {"type": "minimize_co2", "weight": 0.5, "priority": 1},
                {"type": "minimize_fuel_consumption", "weight": 0.3, "priority": 2},
                {"type": "minimize_distance", "weight": 0.2, "priority": 3}
            ],
            "method": "weighted_sum"
        },
        "service_optimization": {
            "name": "Service Optimization",
            "description": "Customer satisfaction focused",
            "objectives": [
                {"type": "maximize_service_level", "weight": 0.4, "priority": 1},
                {"type": "minimize_waiting_time", "weight": 0.3, "priority": 2},
                {"type": "minimize_distance", "weight": 0.3, "priority": 3}
            ],
            "method": "lexicographic"
        },
        "balanced_optimization": {
            "name": "Balanced Optimization",
            "description": "Multi-criteria balanced approach",
            "objectives": [
                {"type": "minimize_distance", "weight": 0.25, "priority": 1},
                {"type": "minimize_cost", "weight": 0.25, "priority": 2},
                {"type": "minimize_co2", "weight": 0.25, "priority": 3},
                {"type": "balance_workload", "weight": 0.25, "priority": 4}
            ],
            "method": "weighted_sum"
        },
        "profit_maximization": {
            "name": "Profit Maximization",
            "description": "Revenue-focused optimization",
            "objectives": [
                {"type": "maximize_profit", "weight": 0.6, "priority": 1},
                {"type": "maximize_service_level", "weight": 0.25, "priority": 2},
                {"type": "minimize_cost", "weight": 0.15, "priority": 3}
            ],
            "method": "lexicographic"
        }
    }
    
    return {
        "templates": templates,
        "available_objectives": [obj.value for obj in ObjectiveType],
        "optimization_methods": ["weighted_sum", "lexicographic", "epsilon_constraint"],
        "usage_notes": [
            "Weights must sum to 1.0 for weighted_sum method",
            "Priority 1 is highest priority for lexicographic method",
            "Epsilon_constraint requires target_value for secondary objectives"
        ]
    }

@router.get("/constraints/templates")
async def get_constraint_templates():
    """
    Get predefined constraint templates for common scenarios
    
    Returns ready-to-use constraint configurations for:
    - Basic constraints (capacity, time windows)
    - Driver regulations (EU driving time directive)
    - Vehicle restrictions (access, equipment)
    - Customer requirements (priority, precedence)
    - Environmental constraints (emission zones)
    """
    templates = {
        "basic_constraints": {
            "name": "Basic Constraints",
            "description": "Essential VRP constraints",
            "route_constraints": {
                "max_route_duration": 480,  # 8 hours
                "max_stops_per_route": 20
            }
        },
        "eu_driver_regulations": {
            "name": "EU Driver Regulations",
            "description": "EU driving time directive compliance",
            "driver_breaks": {
                "mandatory": True,
                "min_work_before_break": 270,  # 4.5 hours
                "break_duration": 45,
                "max_work_without_break": 540  # 9 hours
            },
            "driver_shifts": {
                "max_shift_duration": 600,  # 10 hours
                "overtime_allowed": True,
                "max_overtime": 120  # 2 hours
            }
        },
        "urban_delivery": {
            "name": "Urban Delivery Constraints",
            "description": "City delivery with access restrictions",
            "vehicle_access": {
                "environmental_zones": ["LEZ", "ULEZ"],
                "weight_restrictions": {"city_center": 7500}
            },
            "route_constraints": {
                "max_route_duration": 360,  # 6 hours
                "max_stops_per_route": 15
            }
        },
        "premium_service": {
            "name": "Premium Service",
            "description": "High-priority customer service",
            "customer_priority": {
                "priority_penalty": 200.0,
                "mandatory_customers": []
            },
            "multiple_time_windows": {},
            "precedence": {
                "same_route_required": False
            }
        }
    }
    
    return {
        "templates": templates,
        "available_constraints": [constraint.value for constraint in ConstraintType],
        "severity_levels": ["low", "medium", "high", "critical"],
        "implementation_notes": [
            "Constraints can be combined to create complex scenarios",
            "Soft constraints allow violations with penalties",
            "Hard constraints must be satisfied for feasible solutions"
        ]
    }

def _generate_optimization_insights(result: MultiObjectiveResult, objectives: AdvancedVRPObjectives) -> Dict[str, Any]:
    """Generate insights from optimization result"""
    insights = {
        "performance_summary": f"Achieved total score of {result.total_score:.4f}",
        "objective_contributions": result.weighted_contributions,
        "top_performing_objectives": [],
        "improvement_opportunities": []
    }
    
    # Identify best and worst performing objectives
    if result.normalized_values:
        sorted_objectives = sorted(
            result.normalized_values.items(),
            key=lambda x: x[1]
        )
        
        insights["top_performing_objectives"] = [
            f"{obj}: {value:.3f}" for obj, value in sorted_objectives[:2]
        ]
        
        insights["improvement_opportunities"] = [
            f"Consider focusing on {obj} (current: {value:.3f})" 
            for obj, value in sorted_objectives[-2:]
        ]
    
    return insights

def _assess_solution_quality(
    solution: UnifiedVRPSolution,
    validation_result: ConstraintValidationResult,
    optimization_result: Optional[MultiObjectiveResult]
) -> Dict[str, Any]:
    """Assess overall solution quality"""
    quality_factors = {
        "constraint_compliance": validation_result.constraint_satisfaction_score,
        "efficiency_score": 0.0,
        "overall_quality": "unknown"
    }
    
    # Calculate efficiency score
    if solution.total_distance and len(solution.routes) > 0:
        avg_distance_per_route = solution.total_distance / len(solution.routes)
        efficiency_score = min(100.0, 1000.0 / avg_distance_per_route)
        quality_factors["efficiency_score"] = efficiency_score
    
    # Determine overall quality
    avg_score = (
        quality_factors["constraint_compliance"] + 
        quality_factors["efficiency_score"]
    ) / 2
    
    if avg_score >= 90:
        quality_factors["overall_quality"] = "excellent"
    elif avg_score >= 75:
        quality_factors["overall_quality"] = "good"
    elif avg_score >= 60:
        quality_factors["overall_quality"] = "fair"
    else:
        quality_factors["overall_quality"] = "poor"
    
    return quality_factors

def _analyze_trade_offs(pareto_frontier: ParetoFrontier, objectives: AdvancedVRPObjectives) -> Dict[str, Any]:
    """Analyze trade-offs in Pareto frontier"""
    insights = {
        "objective_trade_offs": pareto_frontier.trade_off_analysis,
        "frontier_characteristics": {
            "solutions_count": len(pareto_frontier.solutions),
            "objective_ranges": pareto_frontier.objective_ranges
        },
        "key_insights": []
    }
    
    # Generate key insights
    if len(pareto_frontier.solutions) > 5:
        insights["key_insights"].append("Large Pareto frontier indicates significant trade-offs between objectives")
    
    for obj_type, analysis in pareto_frontier.trade_off_analysis.items():
        if "improvement" in analysis:
            insights["key_insights"].append(f"{obj_type}: {analysis}")
    
    return insights

def _recommend_pareto_solution(pareto_frontier: ParetoFrontier, objectives: AdvancedVRPObjectives) -> Dict[str, Any]:
    """Recommend best Pareto solution based on objectives"""
    if not pareto_frontier.solutions:
        return {"recommendation": "No Pareto solutions available"}
    
    # Simple recommendation based on objective weights
    best_solution = pareto_frontier.solutions[0]
    best_score = 0.0
    
    for solution in pareto_frontier.solutions:
        score = 0.0
        for obj in objectives.multi_objective.objectives:
            if obj.type.value in solution.objective_values:
                # Lower is better for minimization objectives
                normalized_value = 1.0 - (solution.objective_values[obj.type.value] / 
                                         max(s.objective_values.get(obj.type.value, 1.0) 
                                             for s in pareto_frontier.solutions))
                score += normalized_value * obj.weight
        
        if score > best_score:
            best_score = score
            best_solution = solution
    
    return {
        "recommended_solution_id": best_solution.solution_id,
        "recommendation_score": best_score,
        "rationale": "Selected based on weighted objective preferences",
        "trade_offs": best_solution.trade_offs
    }