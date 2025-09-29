#!/usr/bin/env python3
"""
Debug weighted tardiness issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.jobshop_models import JobShopProblem, Job, Machine, Operation, OptimizationObjective, JobShopSolveRequest, SolverConfig
from app.services.jobshop_service import JobShopService

def test_weighted_tardiness():
    """Test weighted tardiness directly with the exact same data from test_advanced_constraints.py"""
    print("=== Testing Weighted Tardiness Directly ===")
    
    # Create the exact same problem as in test_advanced_constraints.py
    problem = JobShopProblem(
        problem_type="job_shop",
        machines=[
            Machine(id="M1", name="Machine 1")
        ],
        jobs=[
            Job(
                id="J1",
                name="Low Priority Job",
                priority=1,
                weight=1.0,
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
                name="High Priority Job",
                priority=10,
                weight=5.0,
                due_date=8,
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
        optimization_objective=OptimizationObjective.weighted_tardiness,
        max_solve_time_seconds=60
    )
    
    service = JobShopService()
    
    try:
        print(f"Problem created with objective: {problem.optimization_objective}")
        print(f"Job weights: J1={problem.jobs[0].weight}, J2={problem.jobs[1].weight}")
        print(f"Job due dates: J1={problem.jobs[0].due_date}, J2={problem.jobs[1].due_date}")
        
        solution = service.solve_job_shop(problem)
        
        print(f"✓ Solution status: {solution.solution_status}")
        print(f"  Makespan: {solution.metrics.makespan}")
        print(f"  Total weighted tardiness: {solution.metrics.total_weighted_tardiness}")
        print(f"  Objective value: {solution.metrics.objective_value}")
        
        for js in solution.job_schedules:
            print(f"  Job {js.job_id}: start={js.start_time}, end={js.completion_time}, tardiness={js.tardiness}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_api_wrapper():
    """Test the API wrapper with JobShopSolveRequest"""
    print("\n=== Testing API Wrapper ===")
    
    # Create the same problem but wrapped in JobShopSolveRequest
    problem = JobShopProblem(
        problem_type="job_shop",
        machines=[
            Machine(id="M1", name="Machine 1")
        ],
        jobs=[
            Job(
                id="J1",
                name="Low Priority Job",
                priority=1,
                weight=1.0,
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
                name="High Priority Job",
                priority=10,
                weight=5.0,
                due_date=8,
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
        optimization_objective=OptimizationObjective.weighted_tardiness,
        max_solve_time_seconds=60
    )
    
    request = JobShopSolveRequest(
        problem=problem,
        solver_config=SolverConfig(time_limit_seconds=60),
        analysis_config=None
    )
    
    service = JobShopService()
    
    try:
        print(f"Testing wrapped request...")
        solution = service.solve_job_shop(request.problem, request.solver_config, request.analysis_config)
        
        print(f"✓ Wrapper solution status: {solution.solution_status}")
        print(f"  Makespan: {solution.metrics.makespan}")
        print(f"  Total weighted tardiness: {solution.metrics.total_weighted_tardiness}")
        print(f"  Objective value: {solution.metrics.objective_value}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ Wrapper Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success1 = test_weighted_tardiness()
    success2 = test_api_wrapper()
    print(f"\nResults: Direct={success1}, Wrapper={success2}")