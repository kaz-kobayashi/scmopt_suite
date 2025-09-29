#!/usr/bin/env python3
"""
Test Flexible Job Shop implementation
"""

import requests
import json
import sys

API_BASE_URL = "http://127.0.0.1:8000/api"

def test_flexible_jobshop():
    """Test flexible job shop functionality"""
    print("=== Testing Flexible Job Shop ===")
    
    # 1. Get sample flexible job shop problem
    try:
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/flexible_job_shop", timeout=10)
        if response.status_code != 200:
            print(f"✗ Failed to get sample flexible job shop problem: {response.status_code}")
            return False
        
        problem = response.json()
        print(f"✓ Sample flexible job shop problem: {len(problem['jobs'])} jobs, {len(problem['machines'])} machines")
        print(f"  Problem type: {problem['problem_type']}")
        
        # Print machine eligibility for verification
        if 'machine_eligibility' in problem:
            print("  Machine eligibility:")
            for op_id, machines in problem['machine_eligibility'].items():
                print(f"    {op_id}: {machines}")
        
        # Print operation details
        print("  Operations:")
        for job in problem['jobs']:
            print(f"    Job {job['id']}:")
            for op in job['operations']:
                eligible_machines = op.get('eligible_machines', [])
                print(f"      Op {op['id']}: duration={op['duration']}, eligible_machines={eligible_machines}")
        
    except Exception as e:
        print(f"✗ Failed to get sample problem: {str(e)}")
        return False
    
    # 2. Test API endpoint for flexible job shop
    try:
        solve_request = {
            "problem": problem,
            "solver_config": {"time_limit_seconds": 60},
            "analysis_config": {"include_critical_path": True}
        }
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-flexible", 
                               json=solve_request, timeout=120)
        
        if response.status_code != 200:
            print(f"✗ Failed to solve flexible job shop: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        solution = response.json()
        print(f"✓ Flexible job shop solved!")
        print(f"  Status: {solution['solution_status']}")
        print(f"  Makespan: {solution['metrics']['makespan']}")
        print(f"  Utilization: {solution['metrics']['average_machine_utilization']:.2%}")
        
        # Analyze machine assignments
        print("  Machine assignments:")
        for machine_schedule in solution['machine_schedules']:
            machine_id = machine_schedule['machine_id']
            ops = machine_schedule['operations']
            print(f"    {machine_id}: {len(ops)} operations")
            for op in ops:
                print(f"      {op['operation_id']} (Job {op['job_id']}): {op['start_time']}-{op['end_time']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to solve flexible job shop: {str(e)}")
        return False

def test_regular_vs_flexible():
    """Compare regular job shop vs flexible job shop"""
    print("\n=== Comparing Regular vs Flexible Job Shop ===")
    
    try:
        # Test regular job shop
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/job_shop", timeout=10)
        regular_problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve", 
                               json={"problem": regular_problem}, timeout=60)
        regular_solution = response.json()
        regular_makespan = regular_solution['metrics']['makespan']
        
        print(f"Regular Job Shop Makespan: {regular_makespan}")
        
        # Test flexible job shop
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/flexible_job_shop", timeout=10)
        flexible_problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-flexible", 
                               json={"problem": flexible_problem}, timeout=60)
        flexible_solution = response.json()
        flexible_makespan = flexible_solution['metrics']['makespan']
        
        print(f"Flexible Job Shop Makespan: {flexible_makespan}")
        
        if flexible_makespan <= regular_makespan:
            print(f"✓ Flexible job shop achieved better or equal makespan")
            improvement = ((regular_makespan - flexible_makespan) / regular_makespan) * 100
            print(f"  Improvement: {improvement:.1f}%")
        else:
            print(f"⚠️  Flexible job shop makespan is higher (may be different problem instance)")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison failed: {str(e)}")
        return False

if __name__ == "__main__":
    success1 = test_flexible_jobshop()
    success2 = test_regular_vs_flexible()
    
    overall_success = success1 and success2
    print(f"\nFlexible Job Shop Test Result: {'PASSED' if overall_success else 'FAILED'}")
    sys.exit(0 if overall_success else 1)