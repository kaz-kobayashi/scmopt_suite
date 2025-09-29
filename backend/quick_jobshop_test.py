#!/usr/bin/env python3
"""
Quick JobShop API test
"""

import requests
import json
import sys

API_BASE_URL = "http://127.0.0.1:8000/api"

def test_basic_api():
    """基本的なAPIテスト"""
    print("Testing JobShop API...")
    
    # 1. Status check
    try:
        response = requests.get(f"{API_BASE_URL}/jobshop/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✓ API Status: {response.status_code}")
            print(f"  PyJobShop Available: {status['pyjobshop_available']}")
            print(f"  OR-Tools Available: {status['ortools_available']}")
        else:
            print(f"✗ Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Status check failed: {str(e)}")
        return False
    
    # 2. Sample problem
    try:
        response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/job_shop", timeout=10)
        if response.status_code == 200:
            problem = response.json()
            print(f"✓ Sample problem generated: {len(problem['jobs'])} jobs, {len(problem['machines'])} machines")
            
            # 3. Solve the sample problem
            try:
                solve_request = {"problem": problem}
                response = requests.post(f"{API_BASE_URL}/jobshop/solve", json=solve_request, timeout=30)
                if response.status_code == 200:
                    solution = response.json()
                    print(f"✓ Problem solved!")
                    print(f"  Makespan: {solution['metrics']['makespan']}")
                    print(f"  Utilization: {solution['metrics']['average_machine_utilization']:.2%}")
                    print(f"  Status: {solution['solution_status']}")
                    return True
                else:
                    print(f"✗ Solving failed: {response.status_code}")
                    print(f"  Error: {response.text}")
                    return False
            except Exception as e:
                print(f"✗ Solving failed: {str(e)}")
                return False
        else:
            print(f"✗ Sample problem failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Sample problem failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_basic_api()
    print(f"\nTest Result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)