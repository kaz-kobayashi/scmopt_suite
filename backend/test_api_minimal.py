#!/usr/bin/env python3
"""
Minimal test to debug API issues
"""

import requests
import json
import traceback

API_BASE_URL = "http://127.0.0.1:8000/api"

def test_minimal():
    print("=== Minimal JobShop API Test ===")
    
    # Create a very simple problem
    simple_problem = {
        "problem": {
            "problem_type": "job_shop",
            "jobs": [
                {
                    "id": "J1",
                    "name": "Job 1",
                    "operations": [
                        {
                            "id": "J1_O1",
                            "job_id": "J1", 
                            "machine_id": "M1",
                            "duration": 1,
                            "position_in_job": 0
                        }
                    ],
                    "priority": 1,
                    "weight": 1.0,
                    "release_time": 0
                }
            ],
            "machines": [
                {
                    "id": "M1",
                    "name": "Machine 1",
                    "capacity": 1,
                    "available_from": 0
                }
            ],
            "optimization_objective": "makespan",
            "max_solve_time_seconds": 10
        }
    }
    
    print(f"Solving simple problem...")
    print(f"Request: {json.dumps(simple_problem, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/jobshop/solve", json=simple_problem, timeout=30)
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Response Text: {response.text[:500]}...")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(f"Makespan: {result['metrics']['makespan']}")
        else:
            print("FAILED!")
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()