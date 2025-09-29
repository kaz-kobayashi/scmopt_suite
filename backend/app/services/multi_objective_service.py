"""
Multi-objective optimization service for VRP
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import math
from copy import deepcopy

from app.models.multi_objective_models import (
    MultiObjectiveConfig, ObjectiveType, MultiObjectiveResult,
    ParetoSolution, ParetoFrontier, CostStructure, ServiceLevelConfig,
    EnvironmentalConfig, WorkloadBalanceConfig
)
from app.models.vrp_unified_models import UnifiedVRPSolution

logger = logging.getLogger(__name__)

class MultiObjectiveOptimizer:
    """Multi-objective optimization for VRP problems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_multi_objective(
        self,
        solutions: List[UnifiedVRPSolution],
        config: MultiObjectiveConfig,
        cost_structure: Optional[CostStructure] = None,
        service_level: Optional[ServiceLevelConfig] = None,
        environmental: Optional[EnvironmentalConfig] = None,
        workload_balance: Optional[WorkloadBalanceConfig] = None
    ) -> Tuple[UnifiedVRPSolution, MultiObjectiveResult, Optional[ParetoFrontier]]:
        """
        Perform multi-objective optimization on VRP solutions
        
        Args:
            solutions: List of VRP solutions to evaluate
            config: Multi-objective configuration
            cost_structure: Cost structure configuration
            service_level: Service level configuration
            environmental: Environmental configuration
            workload_balance: Workload balance configuration
            
        Returns:
            Best solution, optimization result, and Pareto frontier if requested
        """
        try:
            self.logger.info(f"Starting multi-objective optimization with {len(solutions)} solutions")
            
            # Calculate objective values for all solutions
            objective_values = []
            for i, solution in enumerate(solutions):
                values = self._calculate_objective_values(
                    solution, config.objectives, cost_structure, 
                    service_level, environmental, workload_balance
                )
                objective_values.append(values)
            
            # Normalize objectives if requested
            if config.normalize_objectives:
                objective_values = self._normalize_objectives(objective_values, config.objectives)
            
            # Select best solution based on method
            if config.method == "weighted_sum":
                best_idx, best_score, contributions = self._weighted_sum_method(
                    objective_values, config.objectives
                )
            elif config.method == "lexicographic":
                best_idx, best_score, contributions = self._lexicographic_method(
                    objective_values, config.objectives
                )
            elif config.method == "epsilon_constraint":
                best_idx, best_score, contributions = self._epsilon_constraint_method(
                    objective_values, config.objectives
                )
            else:
                raise ValueError(f"Unknown multi-objective method: {config.method}")
            
            best_solution = solutions[best_idx]
            
            # Create result
            result = MultiObjectiveResult(
                total_score=best_score,
                objective_values=objective_values[best_idx],
                normalized_values=objective_values[best_idx] if config.normalize_objectives else {},
                weighted_contributions=contributions,
                pareto_rank=None,
                dominated_solutions=None,
                is_pareto_optimal=None
            )
            
            # Generate Pareto frontier if requested
            pareto_frontier = None
            if config.pareto_solutions:
                pareto_frontier = self._generate_pareto_frontier(
                    solutions, objective_values, config
                )
                
                # Update result with Pareto information
                pareto_rank = self._calculate_pareto_rank(objective_values[best_idx], objective_values)
                result.pareto_rank = pareto_rank
                result.is_pareto_optimal = pareto_rank == 1
            
            self.logger.info(f"Multi-objective optimization completed. Best score: {best_score:.4f}")
            return best_solution, result, pareto_frontier
            
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed: {str(e)}")
            raise
    
    def _calculate_objective_values(
        self,
        solution: UnifiedVRPSolution,
        objectives: List,
        cost_structure: Optional[CostStructure],
        service_level: Optional[ServiceLevelConfig],
        environmental: Optional[EnvironmentalConfig],
        workload_balance: Optional[WorkloadBalanceConfig]
    ) -> Dict[str, float]:
        """Calculate objective function values for a solution"""
        values = {}
        
        for obj in objectives:
            if obj.type == ObjectiveType.MINIMIZE_DISTANCE:
                values[obj.type.value] = solution.total_distance or 0.0
            
            elif obj.type == ObjectiveType.MINIMIZE_TIME:
                values[obj.type.value] = solution.total_duration or 0.0
            
            elif obj.type == ObjectiveType.MINIMIZE_VEHICLES:
                values[obj.type.value] = len(solution.routes)
            
            elif obj.type == ObjectiveType.MINIMIZE_COST:
                values[obj.type.value] = self._calculate_total_cost(
                    solution, cost_structure or CostStructure()
                )
            
            elif obj.type == ObjectiveType.MINIMIZE_CO2:
                values[obj.type.value] = self._calculate_co2_emissions(
                    solution, environmental or EnvironmentalConfig()
                )
            
            elif obj.type == ObjectiveType.MAXIMIZE_SERVICE_LEVEL:
                values[obj.type.value] = -self._calculate_service_level(
                    solution, service_level or ServiceLevelConfig()
                )  # Negative for minimization
            
            elif obj.type == ObjectiveType.MINIMIZE_WAITING_TIME:
                values[obj.type.value] = self._calculate_waiting_time(solution)
            
            elif obj.type == ObjectiveType.BALANCE_WORKLOAD:
                values[obj.type.value] = self._calculate_workload_imbalance(
                    solution, workload_balance or WorkloadBalanceConfig()
                )
            
            elif obj.type == ObjectiveType.MINIMIZE_FUEL_CONSUMPTION:
                values[obj.type.value] = self._calculate_fuel_consumption(
                    solution, environmental or EnvironmentalConfig()
                )
            
            elif obj.type == ObjectiveType.MAXIMIZE_PROFIT:
                values[obj.type.value] = -self._calculate_profit(
                    solution, cost_structure or CostStructure()
                )  # Negative for minimization
            
            else:
                self.logger.warning(f"Unknown objective type: {obj.type}")
                values[obj.type.value] = 0.0
        
        return values
    
    def _calculate_total_cost(self, solution: UnifiedVRPSolution, cost_structure: CostStructure) -> float:
        """Calculate total cost of solution"""
        total_cost = 0.0
        
        # Vehicle fixed costs
        total_cost += len(solution.routes) * cost_structure.vehicle_fixed_cost
        
        # Distance-based costs
        total_cost += (solution.total_distance or 0) * cost_structure.distance_cost_per_km
        
        # Time-based costs
        total_cost += (solution.total_duration or 0) / 3600 * cost_structure.time_cost_per_hour
        
        # Fuel costs
        fuel_consumption = (solution.total_distance or 0) * cost_structure.fuel_consumption_per_km
        total_cost += fuel_consumption * cost_structure.fuel_cost_per_liter
        
        return total_cost
    
    def _calculate_co2_emissions(self, solution: UnifiedVRPSolution, env_config: EnvironmentalConfig) -> float:
        """Calculate CO2 emissions"""
        fuel_consumption = (solution.total_distance or 0) * 0.08  # Default fuel consumption
        fuel_consumption *= env_config.fuel_efficiency_factor
        
        # Apply eco-driving bonus
        if env_config.eco_driving_bonus > 0:
            fuel_consumption *= (1 - env_config.eco_driving_bonus)
        
        co2_emissions = fuel_consumption * env_config.co2_emission_factor
        
        # Electric vehicle adjustment
        if env_config.electric_vehicle_factor > 0:
            co2_emissions *= env_config.electric_vehicle_factor
        
        return co2_emissions
    
    def _calculate_service_level(self, solution: UnifiedVRPSolution, service_config: ServiceLevelConfig) -> float:
        """Calculate service level score (higher is better)"""
        service_score = 0.0
        total_customers = 0
        
        for route in solution.routes:
            for stop in route.stops:
                if stop.client_id != 0:  # Skip depot
                    total_customers += 1
                    
                    # On-time delivery score
                    if hasattr(stop, 'arrival_time') and hasattr(stop, 'time_window'):
                        if stop.time_window and len(stop.time_window) >= 2:
                            early_time, late_time = stop.time_window[0], stop.time_window[1]
                            if early_time <= stop.arrival_time <= late_time:
                                service_score += service_config.on_time_delivery_weight
        
        # Normalize by number of customers
        if total_customers > 0:
            service_score /= total_customers
        
        return service_score * 100  # Convert to percentage
    
    def _calculate_waiting_time(self, solution: UnifiedVRPSolution) -> float:
        """Calculate total waiting time"""
        total_waiting = 0.0
        
        for route in solution.routes:
            for stop in route.stops:
                if hasattr(stop, 'waiting_time'):
                    total_waiting += getattr(stop, 'waiting_time', 0)
        
        return total_waiting
    
    def _calculate_workload_imbalance(self, solution: UnifiedVRPSolution, balance_config: WorkloadBalanceConfig) -> float:
        """Calculate workload imbalance penalty"""
        if len(solution.routes) <= 1:
            return 0.0
        
        workloads = []
        for route in solution.routes:
            if balance_config.balance_metric == "distance":
                workload = route.distance
            elif balance_config.balance_metric == "time":
                workload = route.duration / 3600  # Convert to hours
            else:  # stops
                workload = len(route.stops) - 2  # Exclude depot stops
            
            workloads.append(workload)
        
        if not workloads:
            return 0.0
        
        mean_workload = np.mean(workloads)
        max_deviation = np.max(np.abs(np.array(workloads) - mean_workload))
        
        if mean_workload > 0:
            relative_deviation = max_deviation / mean_workload
            if relative_deviation > balance_config.max_workload_deviation:
                excess_deviation = relative_deviation - balance_config.max_workload_deviation
                return excess_deviation * balance_config.penalty_per_deviation
        
        return 0.0
    
    def _calculate_fuel_consumption(self, solution: UnifiedVRPSolution, env_config: EnvironmentalConfig) -> float:
        """Calculate fuel consumption"""
        fuel_consumption = (solution.total_distance or 0) * 0.08  # Default consumption
        fuel_consumption *= env_config.fuel_efficiency_factor
        
        if env_config.eco_driving_bonus > 0:
            fuel_consumption *= (1 - env_config.eco_driving_bonus)
        
        return fuel_consumption
    
    def _calculate_profit(self, solution: UnifiedVRPSolution, cost_structure: CostStructure) -> float:
        """Calculate profit (revenue - costs)"""
        # Simplified profit calculation
        revenue = len([stop for route in solution.routes for stop in route.stops if stop.client_id != 0]) * 50
        costs = self._calculate_total_cost(solution, cost_structure)
        return revenue - costs
    
    def _normalize_objectives(self, objective_values: List[Dict[str, float]], objectives: List) -> List[Dict[str, float]]:
        """Normalize objective values to [0, 1] range"""
        if not objective_values:
            return objective_values
        
        # Find min/max for each objective
        obj_ranges = {}
        for obj in objectives:
            obj_type = obj.type.value
            values = [sol_values[obj_type] for sol_values in objective_values if obj_type in sol_values]
            if values:
                obj_ranges[obj_type] = {
                    'min': min(values),
                    'max': max(values)
                }
        
        # Normalize values
        normalized_values = []
        for sol_values in objective_values:
            normalized = {}
            for obj_type, value in sol_values.items():
                if obj_type in obj_ranges:
                    min_val = obj_ranges[obj_type]['min']
                    max_val = obj_ranges[obj_type]['max']
                    
                    if max_val > min_val:
                        normalized[obj_type] = (value - min_val) / (max_val - min_val)
                    else:
                        normalized[obj_type] = 0.0
                else:
                    normalized[obj_type] = value
            
            normalized_values.append(normalized)
        
        return normalized_values
    
    def _weighted_sum_method(self, objective_values: List[Dict[str, float]], objectives: List) -> Tuple[int, float, Dict[str, float]]:
        """Weighted sum multi-objective method"""
        best_idx = 0
        best_score = float('inf')
        best_contributions = {}
        
        for i, sol_values in enumerate(objective_values):
            score = 0.0
            contributions = {}
            
            for obj in objectives:
                obj_type = obj.type.value
                if obj_type in sol_values:
                    contribution = sol_values[obj_type] * obj.weight
                    score += contribution
                    contributions[obj_type] = contribution
            
            if score < best_score:
                best_score = score
                best_idx = i
                best_contributions = contributions
        
        return best_idx, best_score, best_contributions
    
    def _lexicographic_method(self, objective_values: List[Dict[str, float]], objectives: List) -> Tuple[int, float, Dict[str, float]]:
        """Lexicographic multi-objective method"""
        # Sort objectives by priority
        sorted_objectives = sorted(objectives, key=lambda x: x.priority)
        
        candidates = list(range(len(objective_values)))
        
        for obj in sorted_objectives:
            obj_type = obj.type.value
            if len(candidates) == 1:
                break
            
            # Find minimum value among candidates
            min_value = min(objective_values[i][obj_type] for i in candidates if obj_type in objective_values[i])
            
            # Filter candidates to those with minimum value
            candidates = [i for i in candidates 
                         if obj_type in objective_values[i] and 
                         abs(objective_values[i][obj_type] - min_value) < 1e-6]
        
        best_idx = candidates[0] if candidates else 0
        best_score = sum(objective_values[best_idx][obj.type.value] * obj.weight 
                        for obj in objectives if obj.type.value in objective_values[best_idx])
        
        contributions = {obj.type.value: objective_values[best_idx][obj.type.value] * obj.weight
                        for obj in objectives if obj.type.value in objective_values[best_idx]}
        
        return best_idx, best_score, contributions
    
    def _epsilon_constraint_method(self, objective_values: List[Dict[str, float]], objectives: List) -> Tuple[int, float, Dict[str, float]]:
        """Epsilon-constraint multi-objective method"""
        # Use first objective as primary, others as constraints
        primary_obj = objectives[0]
        constraint_objs = objectives[1:]
        
        candidates = []
        
        for i, sol_values in enumerate(objective_values):
            # Check if solution satisfies all epsilon constraints
            satisfies_constraints = True
            
            for obj in constraint_objs:
                obj_type = obj.type.value
                if obj_type in sol_values:
                    if obj.target_value is not None:
                        tolerance = obj.tolerance or (obj.target_value * 0.1)
                        if sol_values[obj_type] > obj.target_value + tolerance:
                            satisfies_constraints = False
                            break
            
            if satisfies_constraints:
                candidates.append(i)
        
        if not candidates:
            # No solution satisfies constraints, return best for primary objective
            candidates = list(range(len(objective_values)))
        
        # Select best candidate for primary objective
        primary_type = primary_obj.type.value
        best_idx = min(candidates, key=lambda i: objective_values[i].get(primary_type, float('inf')))
        
        best_score = sum(objective_values[best_idx][obj.type.value] * obj.weight 
                        for obj in objectives if obj.type.value in objective_values[best_idx])
        
        contributions = {obj.type.value: objective_values[best_idx][obj.type.value] * obj.weight
                        for obj in objectives if obj.type.value in objective_values[best_idx]}
        
        return best_idx, best_score, contributions
    
    def _generate_pareto_frontier(
        self, 
        solutions: List[UnifiedVRPSolution], 
        objective_values: List[Dict[str, float]], 
        config: MultiObjectiveConfig
    ) -> ParetoFrontier:
        """Generate Pareto frontier from solutions"""
        # Find Pareto-optimal solutions
        pareto_indices = []
        
        for i in range(len(objective_values)):
            is_dominated = False
            
            for j in range(len(objective_values)):
                if i != j and self._dominates(objective_values[j], objective_values[i], config.objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        # Limit number of solutions
        if len(pareto_indices) > config.max_pareto_solutions:
            pareto_indices = pareto_indices[:config.max_pareto_solutions]
        
        # Create Pareto solutions
        pareto_solutions = []
        for idx in pareto_indices:
            solution = ParetoSolution(
                solution_id=f"pareto_{idx}",
                objective_values=objective_values[idx],
                routes=[route.dict() for route in solutions[idx].routes],
                performance_metrics={
                    "total_distance": solutions[idx].total_distance,
                    "total_duration": solutions[idx].total_duration,
                    "num_vehicles": len(solutions[idx].routes)
                },
                trade_offs=self._analyze_trade_offs(objective_values[idx], config.objectives)
            )
            pareto_solutions.append(solution)
        
        # Calculate objective ranges
        objective_ranges = {}
        for obj in config.objectives:
            obj_type = obj.type.value
            values = [sol_values[obj_type] for sol_values in objective_values if obj_type in sol_values]
            if values:
                objective_ranges[obj_type] = {
                    "min": min(values),
                    "max": max(values)
                }
        
        return ParetoFrontier(
            solutions=pareto_solutions,
            dominated_solutions_count=len(solutions) - len(pareto_indices),
            frontier_size=len(pareto_indices),
            objective_ranges=objective_ranges,
            trade_off_analysis=self._generate_trade_off_analysis(objective_values, pareto_indices, config.objectives)
        )
    
    def _dominates(self, solution_a: Dict[str, float], solution_b: Dict[str, float], objectives: List) -> bool:
        """Check if solution A dominates solution B"""
        better_in_at_least_one = False
        
        for obj in objectives:
            obj_type = obj.type.value
            if obj_type in solution_a and obj_type in solution_b:
                if solution_a[obj_type] > solution_b[obj_type]:  # A is worse
                    return False
                elif solution_a[obj_type] < solution_b[obj_type]:  # A is better
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _calculate_pareto_rank(self, solution: Dict[str, float], all_solutions: List[Dict[str, float]]) -> int:
        """Calculate Pareto rank of a solution"""
        rank = 1
        for other_solution in all_solutions:
            if other_solution != solution:
                # Count how many solutions dominate this solution
                dominates = True
                better_in_one = False
                
                for key in solution.keys():
                    if key in other_solution:
                        if other_solution[key] > solution[key]:
                            dominates = False
                            break
                        elif other_solution[key] < solution[key]:
                            better_in_one = True
                
                if dominates and better_in_one:
                    rank += 1
        
        return rank
    
    def _analyze_trade_offs(self, objective_values: Dict[str, float], objectives: List) -> Dict[str, str]:
        """Analyze trade-offs for a solution"""
        trade_offs = {}
        
        # Simple trade-off analysis
        for obj in objectives:
            obj_type = obj.type.value
            if obj_type in objective_values:
                value = objective_values[obj_type]
                if value < 0.3:  # Normalized value threshold
                    trade_offs[obj_type] = "Excellent performance"
                elif value < 0.6:
                    trade_offs[obj_type] = "Good performance"
                elif value < 0.8:
                    trade_offs[obj_type] = "Average performance"
                else:
                    trade_offs[obj_type] = "Poor performance, room for improvement"
        
        return trade_offs
    
    def _generate_trade_off_analysis(
        self, 
        objective_values: List[Dict[str, float]], 
        pareto_indices: List[int], 
        objectives: List
    ) -> Dict[str, str]:
        """Generate overall trade-off analysis"""
        analysis = {}
        
        for obj in objectives:
            obj_type = obj.type.value
            pareto_values = [objective_values[i][obj_type] for i in pareto_indices if obj_type in objective_values[i]]
            
            if pareto_values:
                min_val = min(pareto_values)
                max_val = max(pareto_values)
                
                if max_val > min_val:
                    improvement_potential = ((max_val - min_val) / max_val) * 100
                    analysis[obj_type] = f"Up to {improvement_potential:.1f}% improvement possible"
                else:
                    analysis[obj_type] = "Consistent performance across all solutions"
        
        return analysis