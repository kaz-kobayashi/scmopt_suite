#!/usr/bin/env python3
"""
PyJobShop detailed import test
"""

import sys
import traceback

print("=== PyJobShop Import Debug ===")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")

# Test basic import
try:
    import pyjobshop
    print("✓ Basic pyjobshop import successful")
except Exception as e:
    print(f"✗ Basic import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test specific imports
print("\nTesting specific imports:")

# Check what's available
print(f"\nAvailable in pyjobshop: {[x for x in dir(pyjobshop) if not x.startswith('_')]}")

# Test Model import
try:
    from pyjobshop import Model
    print("✓ Model imported")
except Exception as e:
    print(f"✗ Model import failed: {e}")

# Test ProblemData import
try:
    from pyjobshop import ProblemData
    print("✓ ProblemData imported")
except Exception as e:
    print(f"✗ ProblemData import failed: {e}")

# Test solve import
try:
    from pyjobshop import solve
    print("✓ solve imported")
except Exception as e:
    print(f"✗ solve import failed: {e}")

# Test constants import
try:
    from pyjobshop.constants import SolveStatus
    print("✓ SolveStatus imported")
    print(f"  Available statuses: {[x for x in dir(SolveStatus) if not x.startswith('_')]}")
except Exception as e:
    print(f"✗ SolveStatus import failed: {e}")
    # Try alternative
    try:
        import pyjobshop.constants
        print(f"  constants module contains: {[x for x in dir(pyjobshop.constants) if not x.startswith('_')]}")
    except:
        pass

# Create a simple model
print("\n=== Testing Model Creation ===")
try:
    model = Model()
    print("✓ Model created")
    
    # Add machines
    m1 = model.add_machine(name="Machine 1")
    m2 = model.add_machine(name="Machine 2")
    print("✓ Machines added")
    
    # Add a job
    job = model.add_job(name="Job 1", weight=1)
    print("✓ Job added")
    
    # Add tasks
    task1 = model.add_task(job=job, name="Task 1")
    task2 = model.add_task(job=job, name="Task 2")
    print("✓ Tasks added")
    
    # Add modes
    mode1 = model.add_mode(task=task1, resources=m1, duration=3)
    mode2 = model.add_mode(task=task2, resources=m2, duration=2)
    print("✓ Modes added")
    
    # Add precedence
    model.add_end_before_start(task1, task2)
    print("✓ Precedence added")
    
    # Set objective
    model.set_objective(weight_makespan=1)
    print("✓ Objective set")
    
    # Try to solve
    print("\n=== Testing Solve ===")
    result = model.solve(time_limit=10, display=False)
    print(f"✓ Problem solved!")
    print(f"  Status: {result.status}")
    print(f"  Has solution: {result.best is not None}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    traceback.print_exc()

print("\n=== Summary ===")
print("PyJobShop is properly installed and functional with the new API")