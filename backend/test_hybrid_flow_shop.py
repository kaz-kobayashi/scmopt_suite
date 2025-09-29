#!/usr/bin/env python3
"""
Test Hybrid Flow Shop implementation
"""

import requests
import json
import sys

API_BASE_URL = "http://127.0.0.1:8000/api"

def test_hybrid_flow_shop():
    """Test hybrid flow shop functionality"""
    print("=== Testing Hybrid Flow Shop ===")
    
    # 1. Get sample hybrid flow shop problem
    try:
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/hybrid_flow_shop", timeout=10)
        if response.status_code != 200:
            print(f"✗ Failed to get sample hybrid flow shop problem: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        problem = response.json()
        print(f"✓ Sample hybrid flow shop problem: {len(problem['jobs'])} jobs, {len(problem['machines'])} machines")
        print(f"  Problem type: {problem['problem_type']}")
        
        # Print stage information
        if 'stages' in problem:
            print("  Stages:")
            for stage in problem['stages']:
                print(f"    {stage['id']}: {stage['machines']} (capacity: {stage['capacity']})")
        
        if 'stage_sequence' in problem:
            print(f"  Stage sequence: {problem['stage_sequence']}")
        
        # Print operation details to verify hybrid flow shop structure
        print("  Operations (should have eligible machines for parallel stages):")
        for job in problem['jobs']:
            print(f"    Job {job['id']}:")
            for op in job['operations']:
                eligible_machines = op.get('eligible_machines', [op.get('machine_id')])
                print(f"      Op {op['id']}: duration={op['duration']}, eligible_machines={eligible_machines}")
        
    except Exception as e:
        print(f"✗ Failed to get sample problem: {str(e)}")
        return False
    
    # 2. Test API endpoint for hybrid flow shop
    try:
        solve_request = {
            "problem": problem,
            "solver_config": {"time_limit_seconds": 60},
            "analysis_config": {"include_critical_path": True}
        }
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-hybrid-flow", 
                               json=solve_request, timeout=120)
        
        if response.status_code != 200:
            print(f"✗ Failed to solve hybrid flow shop: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
        
        solution = response.json()
        print(f"✓ Hybrid flow shop solved!")
        print(f"  Status: {solution['solution_status']}")
        print(f"  Makespan: {solution['metrics']['makespan']}")
        print(f"  Utilization: {solution['metrics']['average_machine_utilization']:.2%}")
        
        # Analyze machine assignments by stage
        print("  Machine assignments by stage:")
        stage_machines = {
            "Stage1": ["M1_1", "M1_2"],
            "Stage2": ["M2_1"],
            "Stage3": ["M3_1", "M3_2"]
        }
        
        for stage, machines in stage_machines.items():
            print(f"    {stage}:")
            for machine_id in machines:
                machine_schedule = next((ms for ms in solution['machine_schedules'] 
                                       if ms['machine_id'] == machine_id), None)
                if machine_schedule:
                    ops = machine_schedule['operations']
                    utilization = machine_schedule['utilization']
                    print(f"      {machine_id}: {len(ops)} operations, utilization {utilization:.2%}")
                    for op in ops:
                        print(f"        {op['operation_id']} (Job {op['job_id']}): {op['start_time']}-{op['end_time']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to solve hybrid flow shop: {str(e)}")
        return False

def test_hybrid_vs_flow_shop():
    """Compare hybrid flow shop vs regular flow shop performance"""
    print("\n=== Comparing Hybrid Flow Shop vs Flow Shop ===")
    
    try:
        # Test regular flow shop
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/flow_shop", timeout=10)
        flow_shop_problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-flow", 
                               json={"problem": flow_shop_problem}, timeout=60)
        flow_shop_solution = response.json()
        flow_shop_makespan = flow_shop_solution['metrics']['makespan']
        
        print(f"Regular Flow Shop Makespan: {flow_shop_makespan}")
        
        # Test hybrid flow shop
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/hybrid_flow_shop", timeout=10)
        hybrid_flow_shop_problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-hybrid-flow", 
                               json={"problem": hybrid_flow_shop_problem}, timeout=60)
        hybrid_flow_shop_solution = response.json()
        hybrid_flow_shop_makespan = hybrid_flow_shop_solution['metrics']['makespan']
        
        print(f"Hybrid Flow Shop Makespan: {hybrid_flow_shop_makespan}")
        
        # Analyze improvement
        if hybrid_flow_shop_makespan < flow_shop_makespan:
            improvement = ((flow_shop_makespan - hybrid_flow_shop_makespan) / flow_shop_makespan) * 100
            print(f"✓ Hybrid flow shop achieved {improvement:.1f}% improvement")
        elif hybrid_flow_shop_makespan == flow_shop_makespan:
            print("= Same makespan achieved")
        else:
            print("⚠️  Results may differ due to different problem instances")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison failed: {str(e)}")
        return False

def test_hybrid_flow_shop_constraints():
    """Verify that hybrid flow shop respects stage sequence and parallel machine constraints"""
    print("\n=== Verifying Hybrid Flow Shop Constraints ===")
    
    try:
        # Get hybrid flow shop problem and solution
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/hybrid_flow_shop", timeout=10)
        problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-hybrid-flow", 
                               json={"problem": problem}, timeout=60)
        solution = response.json()
        
        stage_sequence = problem.get('stage_sequence', ['Stage1', 'Stage2', 'Stage3'])
        stages = {stage['id']: stage['machines'] for stage in problem.get('stages', [])}
        
        print(f"Expected stage sequence: {stage_sequence}")
        print(f"Stage machines: {stages}")
        
        # Verify each job follows the stage sequence
        constraint_violations = 0
        
        for job_schedule in solution['job_schedules']:
            job_id = job_schedule['job_id']
            operations = sorted(job_schedule['operations'], key=lambda x: x['start_time'])
            
            print(f"  Job {job_id} sequence:")
            for i, op in enumerate(operations):
                expected_stage = stage_sequence[i] if i < len(stage_sequence) else None
                actual_machine = op['machine_id']
                
                # Check if machine belongs to expected stage
                if expected_stage and expected_stage in stages:
                    expected_machines = stages[expected_stage]
                    if actual_machine not in expected_machines:
                        print(f"    ❌ Operation {op['operation_id']}: expected stage {expected_stage} machines {expected_machines}, got {actual_machine}")
                        constraint_violations += 1
                    else:
                        print(f"    ✓ Operation {op['operation_id']}: stage {expected_stage}, machine {actual_machine} at {op['start_time']}-{op['end_time']}")
                else:
                    print(f"    ? Operation {op['operation_id']}: machine {actual_machine} at {op['start_time']}-{op['end_time']}")
        
        if constraint_violations == 0:
            print("✓ All hybrid flow shop constraints satisfied")
            return True
        else:
            print(f"❌ {constraint_violations} constraint violations found")
            return False
        
    except Exception as e:
        print(f"✗ Constraint verification failed: {str(e)}")
        return False

def test_parallel_machine_utilization():
    """Test that parallel machines in each stage are utilized effectively"""
    print("\n=== Testing Parallel Machine Utilization ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/hybrid_flow_shop", timeout=10)
        problem = response.json()
        
        response = requests.post(f"{API_BASE_URL}/jobshop/solve-hybrid-flow", 
                               json={"problem": problem}, timeout=60)
        solution = response.json()
        
        # Check utilization of parallel machines in each stage
        stages = {
            "Stage1": ["M1_1", "M1_2"],
            "Stage3": ["M3_1", "M3_2"]
        }
        
        for stage_name, machines in stages.items():
            print(f"  {stage_name} parallel machines:")
            utilizations = []
            
            for machine_id in machines:
                machine_schedule = next((ms for ms in solution['machine_schedules'] 
                                       if ms['machine_id'] == machine_id), None)
                if machine_schedule:
                    utilization = machine_schedule['utilization']
                    ops_count = len(machine_schedule['operations'])
                    utilizations.append(utilization)
                    print(f"    {machine_id}: {utilization:.2%} utilization, {ops_count} operations")
            
            if len(utilizations) > 1:
                avg_utilization = sum(utilizations) / len(utilizations)
                utilization_diff = max(utilizations) - min(utilizations)
                print(f"    Average: {avg_utilization:.2%}, Difference: {utilization_diff:.2%}")
                
                if utilization_diff < 0.3:  # Less than 30% difference
                    print(f"    ✓ Good load balancing between parallel machines")
                else:
                    print(f"    ⚠️  Significant utilization difference between parallel machines")
        
        return True
        
    except Exception as e:
        print(f"✗ Parallel machine utilization test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success1 = test_hybrid_flow_shop()
    success2 = test_hybrid_vs_flow_shop()
    success3 = test_hybrid_flow_shop_constraints()
    success4 = test_parallel_machine_utilization()
    
    overall_success = success1 and success2 and success3 and success4
    print(f"\nHybrid Flow Shop Test Result: {'PASSED' if overall_success else 'FAILED'}")
    sys.exit(0 if overall_success else 1)