#!/usr/bin/env python3
"""
Test PyJobShop detailed schedule to verify proper optimization
"""

import requests
import json

API_BASE_URL = "http://127.0.0.1:8000/api"

def test_detailed_schedule():
    """Test detailed schedule to verify optimization"""
    print("=== Testing Detailed PyJobShop Schedule ===")
    
    # Get sample problem
    response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/job_shop", timeout=10)
    if response.status_code != 200:
        print(f"Failed to get sample problem: {response.status_code}")
        return False
    
    problem = response.json()
    print(f"Sample problem: {len(problem['jobs'])} jobs, {len(problem['machines'])} machines")
    
    # Print problem details
    for job in problem['jobs']:
        print(f"Job {job['id']}:")
        for op in job['operations']:
            print(f"  Op {op['id']}: machine {op['machine_id']}, duration {op['duration']}")
    
    # Solve the problem
    solve_request = {"problem": problem}
    response = requests.post(f"{API_BASE_URL}/jobshop/solve", json=solve_request, timeout=30)
    
    if response.status_code != 200:
        print(f"Failed to solve: {response.status_code}")
        print(f"Error: {response.text}")
        return False
    
    solution = response.json()
    print(f"\nSolution Status: {solution['solution_status']}")
    print(f"Makespan: {solution['metrics']['makespan']}")
    print(f"Feasible: {solution['metrics']['feasible']}")
    
    # Print detailed schedule
    print("\n=== Job Schedules ===")
    for job_schedule in solution['job_schedules']:
        print(f"Job {job_schedule['job_id']}:")
        print(f"  Start: {job_schedule['start_time']}, Completion: {job_schedule['completion_time']}")
        for op in job_schedule['operations']:
            print(f"    Op {op['operation_id']}: machine {op['machine_id']}, start={op['start_time']}, end={op['end_time']}, duration={op['duration']}")
    
    print("\n=== Machine Schedules ===")
    for machine_schedule in solution['machine_schedules']:
        print(f"Machine {machine_schedule['machine_id']} (utilization: {machine_schedule['utilization']:.2%}):")
        for op in machine_schedule['operations']:
            print(f"  Op {op['operation_id']} (Job {op['job_id']}): start={op['start_time']}, end={op['end_time']}")
    
    # Verify optimization: check if operations are scheduled in parallel where possible
    print("\n=== Optimization Analysis ===")
    
    # Check for overlapping operations on different machines
    all_operations = []
    for job_schedule in solution['job_schedules']:
        for op in job_schedule['operations']:
            all_operations.append({
                'job_id': op['job_id'],
                'operation_id': op['operation_id'],
                'machine_id': op['machine_id'],
                'start_time': op['start_time'],
                'end_time': op['end_time']
            })
    
    # Sort by start time
    all_operations.sort(key=lambda x: x['start_time'])
    
    # Check for parallel execution
    overlaps_found = 0
    for i, op1 in enumerate(all_operations):
        for j, op2 in enumerate(all_operations[i+1:], i+1):
            if op1['machine_id'] != op2['machine_id']:  # Different machines
                if op1['start_time'] < op2['end_time'] and op2['start_time'] < op1['end_time']:  # Time overlap
                    overlaps_found += 1
                    print(f"  Parallel execution detected: {op1['operation_id']} on {op1['machine_id']} and {op2['operation_id']} on {op2['machine_id']}")
    
    if overlaps_found > 0:
        print(f"✓ Optimization working: Found {overlaps_found} parallel executions")
        return True
    else:
        print("⚠️  Warning: No parallel executions detected. May be using sequential scheduling.")
        
        # Check if makespan equals sum of all operation durations (sequential scheduling)
        total_duration = sum([op['end_time'] - op['start_time'] for op in all_operations])
        if solution['metrics']['makespan'] == total_duration:
            print("❌ Sequential scheduling detected - optimization not working properly")
            return False
        else:
            print("✓ Makespan is less than total duration - some optimization occurred")
            return True

if __name__ == "__main__":
    success = test_detailed_schedule()
    print(f"\nDetailed Test Result: {'PASSED' if success else 'FAILED'}")