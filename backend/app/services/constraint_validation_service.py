"""
Constraint validation service for advanced VRP constraints
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from app.models.advanced_constraints_models import (
    AdvancedConstraints, ConstraintViolation, ConstraintValidationResult,
    ConstraintType, DriverBreakConstraint, DriverShiftConstraint,
    DriverSkillConstraint, VehicleCompatibilityConstraint, CustomerPriorityConstraint,
    RouteConstraint, MultipleTimeWindowConstraint, PrecedenceConstraint
)
from app.models.vrp_unified_models import UnifiedVRPSolution, UnifiedRouteModel

logger = logging.getLogger(__name__)

class ConstraintValidator:
    """Validates VRP solutions against advanced constraints"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_solution(
        self,
        solution: UnifiedVRPSolution,
        constraints: AdvancedConstraints,
        problem_data: Optional[Dict[str, Any]] = None
    ) -> ConstraintValidationResult:
        """
        Validate a VRP solution against all defined constraints
        
        Args:
            solution: VRP solution to validate
            constraints: Advanced constraints to check
            problem_data: Original problem data for reference
            
        Returns:
            Validation result with violations and penalties
        """
        try:
            self.logger.info("Starting constraint validation")
            
            violations = []
            total_penalty = 0.0
            
            # Validate driver constraints
            if constraints.driver_breaks:
                break_violations = self._validate_driver_breaks(solution, constraints.driver_breaks)
                violations.extend(break_violations)
            
            if constraints.driver_shifts:
                shift_violations = self._validate_driver_shifts(solution, constraints.driver_shifts)
                violations.extend(shift_violations)
            
            if constraints.driver_skills:
                skill_violations = self._validate_driver_skills(solution, constraints.driver_skills)
                violations.extend(skill_violations)
            
            # Validate vehicle constraints
            if constraints.vehicle_compatibility:
                vehicle_violations = self._validate_vehicle_compatibility(solution, constraints.vehicle_compatibility)
                violations.extend(vehicle_violations)
            
            if constraints.vehicle_access:
                access_violations = self._validate_vehicle_access(solution, constraints.vehicle_access)
                violations.extend(access_violations)
            
            # Validate customer constraints
            if constraints.customer_priority:
                priority_violations = self._validate_customer_priority(solution, constraints.customer_priority)
                violations.extend(priority_violations)
            
            if constraints.precedence:
                precedence_violations = self._validate_precedence(solution, constraints.precedence)
                violations.extend(precedence_violations)
            
            # Validate temporal constraints
            if constraints.multiple_time_windows:
                time_violations = self._validate_multiple_time_windows(solution, constraints.multiple_time_windows)
                violations.extend(time_violations)
            
            # Validate route constraints
            if constraints.route_constraints:
                route_violations = self._validate_route_constraints(solution, constraints.route_constraints)
                violations.extend(route_violations)
            
            if constraints.forbidden_sequences:
                sequence_violations = self._validate_forbidden_sequences(solution, constraints.forbidden_sequences)
                violations.extend(sequence_violations)
            
            # Calculate total penalty
            total_penalty = sum(v.penalty_cost for v in violations)
            
            # Calculate constraint satisfaction score
            satisfaction_score = self._calculate_satisfaction_score(violations)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(violations)
            
            result = ConstraintValidationResult(
                is_valid=len(violations) == 0,
                violations=violations,
                total_penalty=total_penalty,
                constraint_satisfaction_score=satisfaction_score,
                recommendations=recommendations
            )
            
            self.logger.info(f"Constraint validation completed. Violations: {len(violations)}, Penalty: {total_penalty:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Constraint validation failed: {str(e)}")
            raise
    
    def _validate_driver_breaks(self, solution: UnifiedVRPSolution, constraint: DriverBreakConstraint) -> List[ConstraintViolation]:
        """Validate driver break constraints"""
        violations = []
        
        for route_idx, route in enumerate(solution.routes):
            if not constraint.mandatory:
                continue
            
            # Calculate total work time
            work_time = route.duration / 60  # Convert to minutes
            
            # Check if breaks are needed
            if work_time > constraint.max_work_without_break:
                required_breaks = int(work_time / constraint.min_work_before_break)
                
                # Count existing breaks (simplified - would need break information in route data)
                existing_breaks = 0  # This would be calculated from actual route data
                
                if existing_breaks < required_breaks:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.DRIVER_BREAK,
                        violation_description=f"Route {route_idx} requires {required_breaks} breaks but has {existing_breaks}",
                        severity="high",
                        penalty_cost=50.0 * (required_breaks - existing_breaks),
                        affected_entities=[f"route_{route_idx}"],
                        suggested_resolution=f"Add {required_breaks - existing_breaks} driver breaks to route"
                    ))
        
        return violations
    
    def _validate_driver_shifts(self, solution: UnifiedVRPSolution, constraint: DriverShiftConstraint) -> List[ConstraintViolation]:
        """Validate driver shift constraints"""
        violations = []
        
        for route_idx, route in enumerate(solution.routes):
            shift_duration = route.duration / 60  # Convert to minutes
            
            # Check maximum shift duration
            if shift_duration > constraint.max_shift_duration:
                overtime = shift_duration - constraint.max_shift_duration
                
                if not constraint.overtime_allowed:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.DRIVER_SHIFT,
                        violation_description=f"Route {route_idx} exceeds maximum shift duration ({shift_duration:.1f} > {constraint.max_shift_duration})",
                        severity="critical",
                        penalty_cost=100.0,
                        affected_entities=[f"route_{route_idx}"],
                        suggested_resolution="Reduce route duration or split route"
                    ))
                elif overtime > constraint.max_overtime:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.DRIVER_OVERTIME,
                        violation_description=f"Route {route_idx} exceeds maximum overtime ({overtime:.1f} > {constraint.max_overtime})",
                        severity="high",
                        penalty_cost=constraint.overtime_penalty * overtime,
                        affected_entities=[f"route_{route_idx}"],
                        suggested_resolution="Reduce overtime or redistribute customers"
                    ))
            
            # Check minimum shift duration
            if shift_duration < constraint.min_shift_duration:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.DRIVER_SHIFT,
                    violation_description=f"Route {route_idx} below minimum shift duration ({shift_duration:.1f} < {constraint.min_shift_duration})",
                    severity="low",
                    penalty_cost=10.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution="Consolidate with other short routes"
                ))
        
        return violations
    
    def _validate_driver_skills(self, solution: UnifiedVRPSolution, constraint: DriverSkillConstraint) -> List[ConstraintViolation]:
        """Validate driver skill constraints"""
        violations = []
        
        # This would require driver-to-route assignment data
        # For now, create placeholder validation
        for route_idx, route in enumerate(solution.routes):
            # Check if route requires specific skills (based on customer requirements)
            required_skills = constraint.required_skills
            
            if required_skills:
                # Simulate skill validation
                missing_skills = []  # This would be calculated from actual driver data
                
                if missing_skills:
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.DRIVER_SKILLS,
                        violation_description=f"Route {route_idx} requires skills: {missing_skills}",
                        severity="high",
                        penalty_cost=75.0 * len(missing_skills),
                        affected_entities=[f"route_{route_idx}"],
                        suggested_resolution="Assign driver with required skills"
                    ))
        
        return violations
    
    def _validate_vehicle_compatibility(self, solution: UnifiedVRPSolution, constraint: VehicleCompatibilityConstraint) -> List[ConstraintViolation]:
        """Validate vehicle compatibility constraints"""
        violations = []
        
        for route_idx, route in enumerate(solution.routes):
            # Check vehicle type compatibility (would need vehicle assignment data)
            vehicle_type = getattr(route, 'vehicle_type', 'unknown')
            
            if constraint.allowed_vehicle_types and vehicle_type not in constraint.allowed_vehicle_types:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.VEHICLE_COMPATIBILITY,
                    violation_description=f"Route {route_idx} uses forbidden vehicle type: {vehicle_type}",
                    severity="high",
                    penalty_cost=100.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution=f"Use allowed vehicle type: {constraint.allowed_vehicle_types}"
                ))
            
            if constraint.forbidden_vehicle_types and vehicle_type in constraint.forbidden_vehicle_types:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.VEHICLE_COMPATIBILITY,
                    violation_description=f"Route {route_idx} uses forbidden vehicle type: {vehicle_type}",
                    severity="high",
                    penalty_cost=100.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution="Change to allowed vehicle type"
                ))
            
            # Check capacity constraints
            total_demand = sum(getattr(stop, 'demand', 0) for stop in route.stops)
            vehicle_capacity = getattr(route, 'capacity', float('inf'))
            
            if total_demand > constraint.max_vehicle_capacity:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.VEHICLE_COMPATIBILITY,
                    violation_description=f"Route {route_idx} demand ({total_demand}) exceeds maximum vehicle capacity ({constraint.max_vehicle_capacity})",
                    severity="critical",
                    penalty_cost=200.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution="Use larger vehicle or redistribute load"
                ))
        
        return violations
    
    def _validate_vehicle_access(self, solution: UnifiedVRPSolution, constraint) -> List[ConstraintViolation]:
        """Validate vehicle access constraints"""
        violations = []
        
        # Placeholder for access restriction validation
        # This would require detailed road network and vehicle specification data
        
        return violations
    
    def _validate_customer_priority(self, solution: UnifiedVRPSolution, constraint: CustomerPriorityConstraint) -> List[ConstraintViolation]:
        """Validate customer priority constraints"""
        violations = []
        
        # Get all served customers
        served_customers = set()
        for route in solution.routes:
            for stop in route.stops:
                if hasattr(stop, 'client_id') and stop.client_id != 0:
                    served_customers.add(str(stop.client_id))
        
        # Check mandatory customers
        for customer in constraint.mandatory_customers:
            if customer not in served_customers:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CUSTOMER_PRIORITY,
                    violation_description=f"Mandatory customer {customer} not served",
                    severity="critical",
                    penalty_cost=500.0,
                    affected_entities=[customer],
                    suggested_resolution="Add mandatory customer to a route"
                ))
        
        # Check VIP customers
        for customer in constraint.vip_customers:
            if customer not in served_customers:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CUSTOMER_PRIORITY,
                    violation_description=f"VIP customer {customer} not served",
                    severity="high",
                    penalty_cost=constraint.priority_penalty,
                    affected_entities=[customer],
                    suggested_resolution="Prioritize VIP customer service"
                ))
        
        return violations
    
    def _validate_precedence(self, solution: UnifiedVRPSolution, constraint: PrecedenceConstraint) -> List[ConstraintViolation]:
        """Validate precedence constraints"""
        violations = []
        
        for before_customer, after_customer in constraint.precedence_pairs:
            before_found = None
            after_found = None
            same_route = False
            
            # Find positions of both customers
            for route_idx, route in enumerate(solution.routes):
                for stop_idx, stop in enumerate(route.stops):
                    if hasattr(stop, 'client_id'):
                        if str(stop.client_id) == before_customer:
                            before_found = (route_idx, stop_idx)
                        elif str(stop.client_id) == after_customer:
                            after_found = (route_idx, stop_idx)
            
            if before_found and after_found:
                # Check if on same route
                if constraint.same_route_required:
                    if before_found[0] == after_found[0]:
                        same_route = True
                        # Check order
                        if before_found[1] >= after_found[1]:
                            violations.append(ConstraintViolation(
                                constraint_type=ConstraintType.PRECEDENCE,
                                violation_description=f"Customer {before_customer} must come before {after_customer} on same route",
                                severity="high",
                                penalty_cost=150.0,
                                affected_entities=[before_customer, after_customer],
                                suggested_resolution="Reorder customers on route"
                            ))
                    else:
                        violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.PRECEDENCE,
                            violation_description=f"Customers {before_customer} and {after_customer} must be on same route",
                            severity="medium",
                            penalty_cost=100.0,
                            affected_entities=[before_customer, after_customer],
                            suggested_resolution="Move customers to same route"
                        ))
        
        return violations
    
    def _validate_multiple_time_windows(self, solution: UnifiedVRPSolution, constraints: Dict[str, MultipleTimeWindowConstraint]) -> List[ConstraintViolation]:
        """Validate multiple time window constraints"""
        violations = []
        
        for route in solution.routes:
            for stop in route.stops:
                if hasattr(stop, 'client_id') and str(stop.client_id) in constraints:
                    customer_constraint = constraints[str(stop.client_id)]
                    arrival_time = getattr(stop, 'arrival_time', 0)
                    
                    # Check if arrival time falls within any allowed window
                    in_window = False
                    for start_time, end_time in customer_constraint.time_windows:
                        if start_time <= arrival_time <= end_time:
                            in_window = True
                            break
                    
                    if not in_window:
                        violations.append(ConstraintViolation(
                            constraint_type=ConstraintType.MULTIPLE_TIME_WINDOWS,
                            violation_description=f"Customer {stop.client_id} arrival time {arrival_time} outside allowed windows",
                            severity="high",
                            penalty_cost=customer_constraint.window_violation_penalty,
                            affected_entities=[str(stop.client_id)],
                            suggested_resolution="Adjust route timing or move customer"
                        ))
        
        return violations
    
    def _validate_route_constraints(self, solution: UnifiedVRPSolution, constraint: RouteConstraint) -> List[ConstraintViolation]:
        """Validate route-level constraints"""
        violations = []
        
        for route_idx, route in enumerate(solution.routes):
            # Check maximum route distance
            if constraint.max_route_distance and route.distance > constraint.max_route_distance:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.ROUTE_LENGTH_LIMIT,
                    violation_description=f"Route {route_idx} distance ({route.distance:.1f}) exceeds limit ({constraint.max_route_distance})",
                    severity="medium",
                    penalty_cost=50.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution="Reduce route distance or split route"
                ))
            
            # Check maximum route duration
            if constraint.max_route_duration and route.duration > constraint.max_route_duration * 60:  # Convert to seconds
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.ROUTE_DURATION_LIMIT,
                    violation_description=f"Route {route_idx} duration ({route.duration/3600:.1f}h) exceeds limit ({constraint.max_route_duration/60:.1f}h)",
                    severity="medium",
                    penalty_cost=75.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution="Reduce route duration or split route"
                ))
            
            # Check maximum stops per route
            if constraint.max_stops_per_route and len(route.stops) > constraint.max_stops_per_route:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.ROUTE_LENGTH_LIMIT,
                    violation_description=f"Route {route_idx} has {len(route.stops)} stops, exceeds limit ({constraint.max_stops_per_route})",
                    severity="low",
                    penalty_cost=25.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution="Reduce number of stops per route"
                ))
            
            # Check minimum stops per route
            customer_stops = len([stop for stop in route.stops if getattr(stop, 'client_id', 0) != 0])
            if customer_stops < constraint.min_stops_per_route:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.ROUTE_LENGTH_LIMIT,
                    violation_description=f"Route {route_idx} has {customer_stops} customer stops, below minimum ({constraint.min_stops_per_route})",
                    severity="low",
                    penalty_cost=10.0,
                    affected_entities=[f"route_{route_idx}"],
                    suggested_resolution="Add more stops or eliminate route"
                ))
        
        return violations
    
    def _validate_forbidden_sequences(self, solution: UnifiedVRPSolution, constraint) -> List[ConstraintViolation]:
        """Validate forbidden sequence constraints"""
        violations = []
        
        # This would require detailed sequence analysis
        # Placeholder implementation
        
        return violations
    
    def _calculate_satisfaction_score(self, violations: List[ConstraintViolation]) -> float:
        """Calculate constraint satisfaction score (0-100)"""
        if not violations:
            return 100.0
        
        # Weight violations by severity
        severity_weights = {
            'low': 1.0,
            'medium': 2.0,
            'high': 4.0,
            'critical': 8.0
        }
        
        total_weighted_violations = sum(
            severity_weights.get(v.severity, 2.0) for v in violations
        )
        
        # Calculate score (higher violations = lower score)
        max_possible_score = 100.0
        penalty = min(total_weighted_violations * 5, max_possible_score)
        
        return max(0.0, max_possible_score - penalty)
    
    def _generate_recommendations(self, violations: List[ConstraintViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        if not violations:
            recommendations.append("Solution satisfies all constraints")
            return recommendations
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.constraint_type not in violation_types:
                violation_types[violation.constraint_type] = []
            violation_types[violation.constraint_type].append(violation)
        
        # Generate type-specific recommendations
        for constraint_type, type_violations in violation_types.items():
            if constraint_type == ConstraintType.DRIVER_BREAK:
                recommendations.append("Consider adding driver break scheduling to route optimization")
            elif constraint_type == ConstraintType.ROUTE_LENGTH_LIMIT:
                recommendations.append("Routes may be too long - consider increasing vehicle capacity or number of vehicles")
            elif constraint_type == ConstraintType.CUSTOMER_PRIORITY:
                recommendations.append("Prioritize high-value customers in route assignments")
            elif constraint_type == ConstraintType.VEHICLE_COMPATIBILITY:
                recommendations.append("Review vehicle-customer compatibility requirements")
        
        # General recommendations based on severity
        critical_violations = [v for v in violations if v.severity == 'critical']
        if critical_violations:
            recommendations.append("Address critical constraint violations immediately")
        
        high_violations = [v for v in violations if v.severity == 'high']
        if len(high_violations) > 3:
            recommendations.append("Consider relaxing some constraints or increasing fleet capacity")
        
        return recommendations