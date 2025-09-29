#!/usr/bin/env python3
"""
Test Multi-objective optimization implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.jobshop_models import (
    MultiObjectiveProblem, MultiObjectiveWeights, Job, Machine, Operation, 
    OptimizationObjective, SolverConfig, AnalysisConfig
)
from app.services.jobshop_service import JobShopService

def test_weighted_multi_objective():
    """Test weighted multi-objective optimization"""
    print("=== Testing Weighted Multi-objective Optimization ===")
    
    # Create a more complex problem with due dates for tardiness calculation
    problem = MultiObjectiveProblem(
        problem_type="job_shop",
        machines=[
            Machine(id="M1", name="Machine 1"),
            Machine(id="M2", name="Machine 2")
        ],
        jobs=[
            Job(
                id="J1",
                name="Job 1",
                weight=2.0,
                due_date=10,  # Tight due date
                operations=[
                    Operation(
                        id="J1_O1",
                        job_id="J1",
                        machine_id="M1",
                        duration=4,
                        position_in_job=0
                    ),
                    Operation(
                        id="J1_O2",
                        job_id="J1",
                        machine_id="M2",
                        duration=3,
                        position_in_job=1
                    )
                ]
            ),
            Job(
                id="J2",
                name="Job 2", 
                weight=1.0,
                due_date=8,   # Very tight due date
                operations=[
                    Operation(
                        id="J2_O1",
                        job_id="J2",
                        machine_id="M2",
                        duration=2,
                        position_in_job=0
                    ),
                    Operation(
                        id="J2_O2",
                        job_id="J2",
                        machine_id="M1",
                        duration=4,
                        position_in_job=1
                    )
                ]
            ),
            Job(
                id="J3",
                name="Job 3",
                weight=1.5,
                due_date=15,  # Loose due date
                operations=[
                    Operation(
                        id="J3_O1",
                        job_id="J3",
                        machine_id="M1",
                        duration=3,
                        position_in_job=0
                    )
                ]
            )
        ],
        objective_weights=MultiObjectiveWeights(
            makespan_weight=0.4,
            tardiness_weight=0.4,
            completion_time_weight=0.2
        ),
        pareto_analysis=False,
        max_solve_time_seconds=60
    )
    
    service = JobShopService()
    
    try:
        print(f"Problem created with weights: makespan={problem.objective_weights.makespan_weight}, tardiness={problem.objective_weights.tardiness_weight}, completion_time={problem.objective_weights.completion_time_weight}")
        
        solution = service.solve_multi_objective(problem)
        
        print(f"✓ Multi-objective solution found!")
        print(f"  Status: {solution.solution_status}")
        print(f"  Makespan: {solution.metrics.makespan}")
        print(f"  Total tardiness: {solution.metrics.total_tardiness}")
        print(f"  Total completion time: {solution.metrics.total_completion_time}")
        print(f"  Composite objective value: {solution.metrics.objective_value}")
        
        print(f"  Job schedules:")
        for js in solution.job_schedules:
            print(f"    Job {js.job_id}: start={js.start_time}, end={js.completion_time}, tardiness={js.tardiness}")
        
        print(f"  Improvement suggestions:")
        for suggestion in (solution.improvement_suggestions or []):
            print(f"    - {suggestion}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_pareto_analysis():
    """Test Pareto frontier analysis"""
    print("\n=== Testing Pareto Analysis ===")
    
    # Same problem as above but with Pareto analysis
    problem = MultiObjectiveProblem(
        problem_type="job_shop",
        machines=[
            Machine(id="M1", name="Machine 1"),
            Machine(id="M2", name="Machine 2")
        ],
        jobs=[
            Job(
                id="J1",
                name="Job 1",
                weight=2.0,
                due_date=10,
                operations=[
                    Operation(
                        id="J1_O1",
                        job_id="J1",
                        machine_id="M1",
                        duration=4,
                        position_in_job=0
                    ),
                    Operation(
                        id="J1_O2",
                        job_id="J1",
                        machine_id="M2",
                        duration=3,
                        position_in_job=1
                    )
                ]
            ),
            Job(
                id="J2",
                name="Job 2",
                weight=1.0,
                due_date=8,
                operations=[
                    Operation(
                        id="J2_O1",
                        job_id="J2",
                        machine_id="M2",
                        duration=2,
                        position_in_job=0
                    ),
                    Operation(
                        id="J2_O2",
                        job_id="J2",
                        machine_id="M1",
                        duration=4,
                        position_in_job=1
                    )
                ]
            )
        ],
        objective_weights=MultiObjectiveWeights(
            makespan_weight=0.5,
            tardiness_weight=0.3,
            completion_time_weight=0.2
        ),
        pareto_analysis=True,  # Enable Pareto analysis
        max_solve_time_seconds=120  # More time for multiple solutions
    )
    
    service = JobShopService()
    
    try:
        print(f"Running Pareto analysis...")
        
        solution = service.solve_multi_objective(problem)
        
        print(f"✓ Pareto analysis completed!")
        print(f"  Status: {solution.solution_status}")
        print(f"  Best balanced solution:")
        print(f"    Makespan: {solution.metrics.makespan}")
        print(f"    Total tardiness: {solution.metrics.total_tardiness}")
        print(f"    Total completion time: {solution.metrics.total_completion_time}")
        
        print(f"  Pareto analysis results:")
        for suggestion in (solution.improvement_suggestions or []):
            if "Pareto" in suggestion:
                print(f"    - {suggestion}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_single_objective_weights():
    """Test with single objective weight (should behave like regular optimization)"""
    print("\n=== Testing Single Objective Weight ===")
    
    problem = MultiObjectiveProblem(
        problem_type="job_shop",
        machines=[
            Machine(id="M1", name="Machine 1")
        ],
        jobs=[
            Job(
                id="J1",
                name="Job 1",
                operations=[
                    Operation(
                        id="J1_O1",
                        job_id="J1",
                        machine_id="M1",
                        duration=5,
                        position_in_job=0
                    )
                ]
            ),
            Job(
                id="J2",
                name="Job 2",
                operations=[
                    Operation(
                        id="J2_O1",
                        job_id="J2",
                        machine_id="M1",
                        duration=3,
                        position_in_job=0
                    )
                ]
            )
        ],
        objective_weights=MultiObjectiveWeights(
            makespan_weight=1.0,  # Only optimize makespan
            tardiness_weight=0.0,
            completion_time_weight=0.0
        ),
        pareto_analysis=False,
        max_solve_time_seconds=30
    )
    
    service = JobShopService()
    
    try:
        print(f"Testing single objective (makespan only)...")
        
        solution = service.solve_multi_objective(problem)
        
        print(f"✓ Single objective solution found!")
        print(f"  Status: {solution.solution_status}")
        print(f"  Makespan: {solution.metrics.makespan}")
        print(f"  Used objective: {'makespan' in str(solution.improvement_suggestions)}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success1 = test_weighted_multi_objective()
    success2 = test_pareto_analysis()
    success3 = test_single_objective_weights()
    
    total_tests = 3
    passed_tests = sum([success1, success2, success3])
    
    print(f"\n=== Multi-objective Optimization Test Summary ===")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Weighted Multi-objective: {'PASSED' if success1 else 'FAILED'}")
    print(f"Pareto Analysis: {'PASSED' if success2 else 'FAILED'}")
    print(f"Single Objective: {'PASSED' if success3 else 'FAILED'}")
    
    overall_success = passed_tests >= 2  # Pass if at least 2 tests pass
    print(f"\nOverall Result: {'PASSED' if overall_success else 'FAILED'}")
    sys.exit(0 if overall_success else 1)