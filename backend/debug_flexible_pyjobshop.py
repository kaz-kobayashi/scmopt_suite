#!/usr/bin/env python3
"""
Debug PyJobShop flexible job shop implementation
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

def debug_flexible_pyjobshop():
    """Create a simple flexible job shop problem and debug"""
    print("=== Creating Flexible Job Shop Problem ===")
    
    model = Model()
    
    # Add machines
    m1 = model.add_machine(name="M1")
    m2 = model.add_machine(name="M2")
    m3 = model.add_machine(name="M3")
    machines = [m1, m2, m3]
    
    print(f"✓ Added {len(machines)} machines")
    
    # Add jobs with operations that can be processed on multiple machines
    jobs_data = [
        {"name": "J1", "ops": [
            {"eligible_machines": [m1, m2], "duration": 3},  # J1_O1: M1 or M2
            {"eligible_machines": [m2, m3], "duration": 2},  # J1_O2: M2 or M3
            {"eligible_machines": [m1, m3], "duration": 2}   # J1_O3: M1 or M3
        ]},
        {"name": "J2", "ops": [
            {"eligible_machines": [m1, m2], "duration": 2},  # J2_O1: M1 or M2
            {"eligible_machines": [m2, m3], "duration": 1},  # J2_O2: M2 or M3
            {"eligible_machines": [m1, m3], "duration": 4}   # J2_O3: M1 or M3
        ]},
        {"name": "J3", "ops": [
            {"eligible_machines": [m1, m2], "duration": 4},  # J3_O1: M1 or M2
            {"eligible_machines": [m2, m3], "duration": 3},  # J3_O2: M2 or M3
            {"eligible_machines": [m1, m3], "duration": 1}   # J3_O3: M1 or M3
        ]}
    ]
    
    jobs = []
    all_tasks = []
    task_to_info = {}  # task -> (job_name, op_index, eligible_machines, duration)
    
    for job_data in jobs_data:
        job = model.add_job(name=job_data["name"], weight=1)
        jobs.append(job)
        
        prev_task = None
        for op_idx, op_data in enumerate(job_data["ops"]):
            task = model.add_task(job=job, name=f"{job_data['name']}_T{op_idx}")
            
            # Store task info
            task_to_info[task] = (job_data["name"], op_idx, op_data["eligible_machines"], op_data["duration"])
            all_tasks.append(task)
            
            # Add modes for each eligible machine
            for machine in op_data["eligible_machines"]:
                model.add_mode(task=task, resources=machine, duration=op_data["duration"])
                print(f"    Added mode: {job_data['name']}_T{op_idx} on {machine.name} with duration {op_data['duration']}")
            
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
                job_name, op_idx, eligible_machines, duration = task_to_info[task]
                
                machine_idx = task_data.resources[0] if task_data.resources else None
                machine_name = machines[machine_idx].name if machine_idx is not None and machine_idx < len(machines) else "Unknown"
                
                print(f"Task {i} ({job_name}_T{op_idx}):")
                print(f"  Assigned to machine: {machine_name} (index {machine_idx})")
                print(f"  Eligible machines: {[m.name for m in eligible_machines]}")
                print(f"  Start: {task_data.start}, End: {task_data.end}, Duration: {task_data.end - task_data.start}")
                print()
        
        # Analyze machine schedules
        print("=== Machine Schedule Analysis ===")
        machine_schedules = {i: [] for i in range(len(machines))}
        
        for i, task_data in enumerate(solution.tasks):
            if i < len(all_tasks):
                task = all_tasks[i]
                job_name, op_idx, eligible_machines, duration = task_to_info[task]
                
                machine_idx = task_data.resources[0] if task_data.resources else None
                if machine_idx is not None and machine_idx < len(machines):
                    machine_schedules[machine_idx].append({
                        'job_op': f"{job_name}_T{op_idx}",
                        'start': task_data.start,
                        'end': task_data.end,
                        'duration': task_data.end - task_data.start
                    })
        
        for machine_idx, ops in machine_schedules.items():
            machine_name = machines[machine_idx].name
            ops.sort(key=lambda x: x['start'])
            total_time = max([op['end'] for op in ops]) if ops else 0
            busy_time = sum([op['duration'] for op in ops])
            utilization = (busy_time / total_time) * 100 if total_time > 0 else 0
            
            print(f"Machine {machine_name} (utilization: {utilization:.1f}%):")
            for op in ops:
                print(f"  {op['job_op']}: start={op['start']}, end={op['end']}, duration={op['duration']}")
            print()
        
        return True
    else:
        print("❌ No solution found")
        return False

if __name__ == "__main__":
    success = debug_flexible_pyjobshop()
    print(f"\nDebug Result: {'SUCCESS' if success else 'FAILED'}")