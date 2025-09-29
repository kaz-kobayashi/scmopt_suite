#!/usr/bin/env python3
"""
Test Flow Shop implementation
"""

import requests
import json
import sys

API_BASE_URL = "http://127.0.0.1:8000/api"

def test_flow_shop():
    """Test flow shop functionality"""
    print("=== Testing Flow Shop ===")
    
    # 1. Get sample flow shop problem
    try:
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/flow_shop", timeout=10)
        if response.status_code != 200:
            print(f"✗ Failed to get sample flow shop problem: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        problem = response.json()
        print(f"✓ Sample flow shop problem: {len(problem['jobs'])} jobs, {len(problem['machines'])} machines")
        print(f"  Problem type: {problem['problem_type']}")
        
        # Print machine sequence for verification
        if 'machine_sequence' in problem:
            print(f"  Machine sequence: {problem['machine_sequence']}")
        
        # Print operation details to verify flow shop structure
        print("  Operations (should follow same machine sequence):")
        for job in problem['jobs']:
            print(f"    Job {job['id']}:")
            for op in job['operations']:
                print(f"      Op {op['id']}: machine={op['machine_id']}, duration={op['duration']}, position={op['position_in_job']}")
        
    except Exception as e:
        print(f"✗ Failed to get sample problem: {str(e)}")
        return False
    
    # 2. Test API endpoint for flow shop
    try:
        solve_request = {
            "problem": problem,
            "solver_config": {"time_limit_seconds": 60},
            "analysis_config": {"include_critical_path": True}
        }
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-flow", 
                               json=solve_request, timeout=120)
        
        if response.status_code != 200:
            print(f"✗ Failed to solve flow shop: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        solution = response.json()
        print(f"✓ Flow shop solved!")
        print(f"  Status: {solution['solution_status']}")
        print(f"  Makespan: {solution['metrics']['makespan']}")
        print(f"  Utilization: {solution['metrics']['average_machine_utilization']:.2%}")
        
        # Analyze machine assignments
        print("  Machine assignments:")
        for machine_schedule in solution['machine_schedules']:
            machine_id = machine_schedule['machine_id']
            ops = machine_schedule['operations']
            utilization = machine_schedule['utilization']
            print(f"    {machine_id}: {len(ops)} operations, utilization {utilization:.2%}")
            for op in ops:
                print(f"      {op['operation_id']} (Job {op['job_id']}): {op['start_time']}-{op['end_time']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to solve flow shop: {str(e)}")
        return False

def test_flow_shop_vs_job_shop():
    """Compare flow shop vs job shop performance"""
    print("\n=== Comparing Flow Shop vs Job Shop ===")
    
    try:
        # Test regular job shop
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/job_shop", timeout=10)
        job_shop_problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve", 
                               json={"problem": job_shop_problem}, timeout=60)
        job_shop_solution = response.json()
        job_shop_makespan = job_shop_solution['metrics']['makespan']
        
        print(f"Job Shop Makespan: {job_shop_makespan}")
        
        # Test flow shop
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/flow_shop", timeout=10)
        flow_shop_problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-flow", 
                               json={"problem": flow_shop_problem}, timeout=60)
        flow_shop_solution = response.json()
        flow_shop_makespan = flow_shop_solution['metrics']['makespan']
        
        print(f"Flow Shop Makespan: {flow_shop_makespan}")
        
        # Analyze difference
        if flow_shop_makespan < job_shop_makespan:
            improvement = ((job_shop_makespan - flow_shop_makespan) / job_shop_makespan) * 100
            print(f"✓ Flow shop achieved {improvement:.1f}% better makespan")
        elif flow_shop_makespan == job_shop_makespan:
            print("= Same makespan achieved")
        else:
            difference = ((flow_shop_makespan - job_shop_makespan) / job_shop_makespan) * 100
            print(f"⚠️  Flow shop makespan is {difference:.1f}% higher (due to flow constraints)")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison failed: {str(e)}")
        return False

def test_flow_shop_constraints():
    """Verify that flow shop respects machine sequence constraints"""
    print("\n=== Verifying Flow Shop Constraints ===")
    
    try:
        # Get flow shop problem and solution
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/flow_shop", timeout=10)
        problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-flow", 
                               json={"problem": problem}, timeout=60)
        solution = response.json()
        
        machine_sequence = problem.get('machine_sequence', ['M1', 'M2', 'M3'])
        print(f"Expected machine sequence: {machine_sequence}")
        
        # Verify each job follows the machine sequence
        constraint_violations = 0
        
        for job_schedule in solution['job_schedules']:
            job_id = job_schedule['job_id']
            operations = sorted(job_schedule['operations'], key=lambda x: x['start_time'])
            
            print(f"  Job {job_id} sequence:")
            for i, op in enumerate(operations):
                expected_machine = machine_sequence[i] if i < len(machine_sequence) else None
                actual_machine = op['machine_id']
                
                if expected_machine and actual_machine != expected_machine:
                    print(f"    ❌ Operation {op['operation_id']}: expected {expected_machine}, got {actual_machine}")
                    constraint_violations += 1
                else:
                    print(f"    ✓ Operation {op['operation_id']}: {actual_machine} at {op['start_time']}-{op['end_time']}")
        
        if constraint_violations == 0:
            print("✓ All flow shop constraints satisfied")
            return True
        else:
            print(f"❌ {constraint_violations} constraint violations found")
            return False
        
    except Exception as e:
        print(f"✗ Constraint verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    success1 = test_flow_shop()
    success2 = test_flow_shop_vs_job_shop()
    success3 = test_flow_shop_constraints()
    
    overall_success = success1 and success2 and success3
    print(f"\nFlow Shop Test Result: {'PASSED' if overall_success else 'FAILED'}")
    sys.exit(0 if overall_success else 1)