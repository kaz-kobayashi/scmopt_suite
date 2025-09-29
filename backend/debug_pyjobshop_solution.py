#!/usr/bin/env python3
"""
Debug PyJobShop solution structure in detail
"""

import sys
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from pyjobshop import Model, SolveStatus
    print("✓ PyJobShop imports successful")
except Exception as e:
    print(f"✗ PyJobShop import failed: {e}")
    sys.exit(1)

def debug_pyjobshop_solution():
    """Create a simple problem and debug the solution structure"""
    print("=== Creating Simple 3-Job, 3-Machine Problem ===")
    
    model = Model()
    
    # Add machines
    m1 = model.add_machine(name="M1")
    m2 = model.add_machine(name="M2")
    m3 = model.add_machine(name="M3")
    machines = [m1, m2, m3]
    
    print(f"✓ Added {len(machines)} machines")
    
    # Add jobs with operations
    jobs_data = [
        {"name": "J1", "ops": [(m1, 3), (m2, 2), (m3, 2)]},  # Job 1: M1(3) -> M2(2) -> M3(2)
        {"name": "J2", "ops": [(m1, 2), (m3, 1), (m2, 4)]},  # Job 2: M1(2) -> M3(1) -> M2(4)
        {"name": "J3", "ops": [(m2, 4), (m1, 3), (m3, 1)]},  # Job 3: M2(4) -> M1(3) -> M3(1)
    ]
    
    jobs = []
    all_tasks = []
    task_to_job_op = {}  # task -> (job_name, op_index)
    
    for job_data in jobs_data:
        job = model.add_job(name=job_data["name"], weight=1)
        jobs.append(job)
        
        prev_task = None
        for op_idx, (machine, duration) in enumerate(job_data["ops"]):
            task = model.add_task(job=job, name=f"{job_data['name']}_T{op_idx}")
            model.add_mode(task=task, resources=machine, duration=duration)
            
            # Store task mapping
            task_to_job_op[task] = (job_data["name"], op_idx, machine, duration)
            all_tasks.append(task)
            
            if prev_task is not None:
                model.add_end_before_start(prev_task, task, delay=0)
            
            prev_task = task
    
    print(f"✓ Added {len(jobs)} jobs with {len(all_tasks)} tasks total")
    
    # Set objective
    model.set_objective(weight_makespan=1)
    print("✓ Set makespan objective")
    
    # Solve
    result = model.solve(time_limit=60, display=False)
    print(f"✓ Solved with status: {result.status}")
    
    if result.best is not None:
        solution = result.best
        print(f"Solution makespan: {solution.makespan}")
        print(f"Solution tasks: {len(solution.tasks)}")
        
        print("\n=== Task Details ===")
        for i, task_data in enumerate(solution.tasks):
            if i < len(all_tasks):
                task = all_tasks[i]
                job_name, op_idx, machine, duration = task_to_job_op[task]
                
                print(f"Task {i} ({job_name}_T{op_idx}):")
                print(f"  TaskData: {task_data}")
                print(f"  Mode: {task_data.mode}")
                print(f"  Resources: {task_data.resources}")
                print(f"  Start: {task_data.start}")
                print(f"  End: {task_data.end}")
                print(f"  Expected machine: {machine} (index ?)")
                print(f"  Expected duration: {duration}")
                print(f"  Actual duration: {task_data.end - task_data.start}")
                print()
        
        # Analyze machine assignments
        print("=== Machine Assignment Analysis ===")
        machine_schedules = {0: [], 1: [], 2: []}
        
        for i, task_data in enumerate(solution.tasks):
            if i < len(all_tasks):
                task = all_tasks[i]
                job_name, op_idx, expected_machine, duration = task_to_job_op[task]
                
                machine_idx = task_data.resources[0] if task_data.resources else None
                if machine_idx is not None:
                    machine_schedules[machine_idx].append({
                        'job_op': f"{job_name}_T{op_idx}",
                        'start': task_data.start,
                        'end': task_data.end,
                        'duration': task_data.end - task_data.start
                    })
        
        for machine_idx, ops in machine_schedules.items():
            print(f"Machine {machine_idx} (M{machine_idx+1}):")
            ops.sort(key=lambda x: x['start'])
            for op in ops:
                print(f"  {op['job_op']}: start={op['start']}, end={op['end']}, duration={op['duration']}")
            print()
        
        # Check for overlaps (should be none on same machine)
        print("=== Overlap Detection ===")
        for machine_idx, ops in machine_schedules.items():
            for i in range(len(ops)):
                for j in range(i+1, len(ops)):
                    op1, op2 = ops[i], ops[j]
                    if op1['start'] < op2['end'] and op2['start'] < op1['end']:
                        print(f"❌ OVERLAP on Machine {machine_idx}: {op1['job_op']} and {op2['job_op']}")
                    else:
                        print(f"✓ No overlap on Machine {machine_idx}: {op1['job_op']} and {op2['job_op']}")
        
        return True
    else:
        print("❌ No solution found")
        return False

if __name__ == "__main__":
    success = debug_pyjobshop_solution()
    print(f"\nDebug Result: {'SUCCESS' if success else 'FAILED'}")