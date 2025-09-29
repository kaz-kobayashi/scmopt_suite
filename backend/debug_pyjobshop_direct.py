#!/usr/bin/env python3
"""
Direct PyJobShop debugging without API
"""

import sys
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from pyjobshop import Model, SolveStatus
    print("✓ PyJobShop imports successful")
except Exception as e:
    print(f"✗ PyJobShop import failed: {e}")
    sys.exit(1)

# Import our service directly
try:
    sys.path.append('/Users/kazuhiro/Documents/2509/scmopt_suite/backend')
    from app.services.jobshop_service import JobShopService
    from app.models.jobshop_models import JobShopProblem, Job, Machine, Operation, OptimizationObjective
    print("✓ Service imports successful")
except Exception as e:
    print(f"✗ Service import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_direct_pyjobshop():
    """Test PyJobShop directly"""
    print("\n=== Testing PyJobShop Directly ===")
    try:
        model = Model()
        
        # Add machines
        m1 = model.add_machine(name="Machine 1")
        print(f"✓ Machine added: {m1}")
        
        # Add job
        job = model.add_job(name="Job 1", weight=1)
        print(f"✓ Job added: {job}")
        
        # Add task
        task = model.add_task(job=job, name="Task 1")
        print(f"✓ Task added: {task}")
        
        # Add mode
        mode = model.add_mode(task=task, resources=m1, duration=5)
        print(f"✓ Mode added: {mode}")
        
        # Set objective
        model.set_objective(weight_makespan=1)
        print("✓ Objective set")
        
        # Solve
        result = model.solve(time_limit=10, display=False)
        print(f"✓ Solved with status: {result.status}")
        
        if result.best is not None:
            print(f"✓ Has solution")
            print(f"  Solution type: {type(result.best)}")
            print(f"  Available attributes: {[attr for attr in dir(result.best) if not attr.startswith('_')]}")
            
            # Try to access tasks
            if hasattr(result.best, 'tasks'):
                print(f"  Tasks: {result.best.tasks}")
            elif hasattr(result.best, 'scheduled_tasks'):
                print(f"  Scheduled tasks: {result.best.scheduled_tasks}")
            else:
                print("  No tasks attribute found")
        else:
            print("✗ No solution found")
            
        return True
        
    except Exception as e:
        print(f"✗ Direct PyJobShop test failed: {e}")
        traceback.print_exc()
        return False

def test_service_creation():
    """Test service creation and basic methods"""
    print("\n=== Testing Service Creation ===")
    try:
        service = JobShopService()
        print(f"✓ Service created")
        print(f"  PyJobShop available: {service.pyjobshop_available}")
        print(f"  OR-Tools available: {service.ortools_available}")
        
        # Test sample problem generation
        sample = service.generate_sample_problem("job_shop")
        print(f"✓ Sample problem generated")
        print(f"  Jobs: {len(sample.jobs)}")
        print(f"  Machines: {len(sample.machines)}")
        print(f"  Problem type: {sample.problem_type}")
        
        return sample
        
    except Exception as e:
        print(f"✗ Service creation failed: {e}")
        traceback.print_exc()
        return None

def test_service_solve(problem):
    """Test service solve method"""
    print("\n=== Testing Service Solve ===")
    try:
        service = JobShopService()
        
        # Add debug logging to understand the problem structure
        print(f"Problem details:")
        print(f"  Type: {problem.problem_type}")
        print(f"  Jobs: {len(problem.jobs)}")
        for job in problem.jobs:
            print(f"    Job {job.id}: {len(job.operations)} operations")
            for op in job.operations:
                print(f"      Op {op.id}: machine {op.machine_id}, duration {op.duration}")
        
        print(f"  Machines: {len(problem.machines)}")
        for machine in problem.machines:
            print(f"    Machine {machine.id}: {machine.name}")
        
        # Try to solve
        print("Attempting to solve...")
        solution = service.solve_job_shop(problem)
        
        print(f"✓ Service solve completed")
        print(f"  Status: {solution.solution_status}")
        print(f"  Feasible: {solution.metrics.feasible}")
        
        if solution.metrics.feasible:
            print(f"  Makespan: {solution.metrics.makespan}")
            print(f"  Job schedules: {len(solution.job_schedules)}")
            print(f"  Machine schedules: {len(solution.machine_schedules)}")
        
        return solution
        
    except Exception as e:
        print(f"✗ Service solve failed: {e}")
        traceback.print_exc()
        return None

def main():
    print("=== PyJobShop Debug Session ===")
    
    # Test 1: Direct PyJobShop
    if not test_direct_pyjobshop():
        print("Direct PyJobShop test failed - aborting")
        return
    
    # Test 2: Service creation
    sample_problem = test_service_creation()
    if sample_problem is None:
        print("Service creation failed - aborting")
        return
    
    # Test 3: Service solve
    solution = test_service_solve(sample_problem)
    
    if solution and solution.metrics.feasible:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")

if __name__ == "__main__":
    main()