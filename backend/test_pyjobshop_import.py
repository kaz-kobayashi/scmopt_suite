#!/usr/bin/env python3
"""
PyJobShop import test
"""

import sys
print(f"Python version: {sys.version}")

try:
    import pyjobshop
    print(f"✓ PyJobShop imported successfully")
    print(f"  Version info: {dir(pyjobshop)}")
    
    # Test basic functionality
    from pyjobshop import JobShopInstance, solve
    print("✓ Can import JobShopInstance and solve")
    
    # Try creating a simple instance
    jobs_data = [
        [(0, 3), (1, 2), (2, 2)],  # job 0
        [(0, 2), (2, 1), (1, 4)],  # job 1
        [(1, 4), (2, 3)]           # job 2
    ]
    
    instance = JobShopInstance(jobs_data)
    print("✓ Created JobShopInstance successfully")
    
except ImportError as e:
    print(f"✗ PyJobShop import failed: {e}")
    print(f"  Error details: {type(e).__name__}: {str(e)}")
    
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    print(f"  Error type: {type(e).__name__}")

# Check OR-Tools
try:
    import ortools
    from ortools.sat.python import cp_model
    print(f"✓ OR-Tools imported successfully")
    print(f"  Version: {ortools.__version__}")
except ImportError as e:
    print(f"✗ OR-Tools import failed: {e}")