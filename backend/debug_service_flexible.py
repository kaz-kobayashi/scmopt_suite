#!/usr/bin/env python3
"""
Debug service flexible job shop directly
"""

import sys
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import our service directly
try:
    sys.path.append('/Users/kazuhiro/Documents/2509/scmopt_suite/backend')
    from app.services.jobshop_service import JobShopService
    from app.models.jobshop_models import JobShopProblem, Job, Machine, Operation, OptimizationObjective, ProblemType
    print("✓ Service imports successful")
except Exception as e:
    print(f"✗ Service import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_service_flexible_directly():
    """Test service flexible job shop creation and solving"""
    print("=== Testing Service Flexible Job Shop ===")
    
    try:
        service = JobShopService()
        
        # Test flexible job shop sample generation
        flexible_problem = service.generate_sample_problem("flexible_job_shop")
        print(f"✓ Flexible sample problem generated")
        print(f"  Type: {flexible_problem.problem_type}")
        print(f"  Jobs: {len(flexible_problem.jobs)}")
        print(f"  Machines: {len(flexible_problem.machines)}")
        
        # Check operations setup
        print("  Operation eligibility:")
        for job in flexible_problem.jobs:
            print(f"    Job {job.id}:")
            for op in job.operations:
                print(f"      {op.id}: machine_id={op.machine_id}, eligible_machines={op.eligible_machines}")
        
        # Test solving
        print("\nSolving flexible job shop problem...")
        solution = service.solve_job_shop(flexible_problem)
        
        print(f"✓ Solution generated")
        print(f"  Status: {solution.solution_status}")
        print(f"  Makespan: {solution.metrics.makespan}")
        print(f"  Feasible: {solution.metrics.feasible}")
        
        # Check machine assignments
        print("  Machine assignments:")
        for machine_schedule in solution.machine_schedules:
            machine_id = machine_schedule.machine_id
            ops = machine_schedule.operations
            utilization = machine_schedule.utilization
            print(f"    {machine_id}: {len(ops)} operations, utilization {utilization:.2%}")
            for op in ops[:3]:  # Show first 3 operations
                print(f"      {op.operation_id}: {op.start_time}-{op.end_time}")
            if len(ops) > 3:
                print(f"      ... and {len(ops)-3} more")
        
        return solution
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        traceback.print_exc()
        return None

def compare_job_shop_vs_flexible():
    """Compare regular job shop vs flexible"""
    print("\n=== Comparing Job Shop vs Flexible ===")
    
    try:
        service = JobShopService()
        
        # Regular job shop
        regular_problem = service.generate_sample_problem("job_shop")
        regular_solution = service.solve_job_shop(regular_problem)
        
        print(f"Regular Job Shop: makespan={regular_solution.metrics.makespan}")
        
        # Flexible job shop
        flexible_problem = service.generate_sample_problem("flexible_job_shop")
        flexible_solution = service.solve_job_shop(flexible_problem)
        
        print(f"Flexible Job Shop: makespan={flexible_solution.metrics.makespan}")
        
        if flexible_solution.metrics.makespan < regular_solution.metrics.makespan:
            improvement = ((regular_solution.metrics.makespan - flexible_solution.metrics.makespan) / regular_solution.metrics.makespan) * 100
            print(f"✓ Flexible achieved {improvement:.1f}% improvement")
        elif flexible_solution.metrics.makespan == regular_solution.metrics.makespan:
            print("= Same makespan achieved")
        else:
            print("⚠️  Flexible makespan is higher than regular")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    solution = test_service_flexible_directly()
    if solution:
        compare_job_shop_vs_flexible()
        print("\n✓ Service test completed successfully")
    else:
        print("\n❌ Service test failed")
        sys.exit(1)