#!/usr/bin/env python3
"""
Test API with weighted tardiness via direct HTTP request
"""

import requests
import json

def test_api_weighted_tardiness():
    """Test the actual API endpoint"""
    print("=== Testing API with Weighted Tardiness ===")
    
    url = "http://127.0.0.1:8000/api/jobshop/solve"
    
    problem_data = {
        "problem": {
            "problem_type": "job_shop",
            "machines": [
                {"id": "M1", "name": "Machine 1"}
            ],
            "jobs": [
                {
                    "id": "J1",
                    "name": "Low Priority Job",
                    "priority": 1,
                    "weight": 1.0,
                    "operations": [
                        {
                            "id": "J1_O1",
                            "job_id": "J1",
                            "machine_id": "M1",
                            "duration": 5,
                            "position_in_job": 0
                        }
                    ]
                },
                {
                    "id": "J2",
                    "name": "High Priority Job",
                    "priority": 10,
                    "weight": 5.0,
                    "due_date": 8,
                    "operations": [
                        {
                            "id": "J2_O1",
                            "job_id": "J2",
                            "machine_id": "M1",
                            "duration": 3,
                            "position_in_job": 0
                        }
                    ]
                }
            ],
            "optimization_objective": "weighted_tardiness",
            "max_solve_time_seconds": 60
        },
        "solver_config": {
            "time_limit_seconds": 60
        },
        "analysis_config": {
            "include_critical_path": True
        }
    }
    
    try:
        print("Making request to API...")
        
        response = requests.post(
            url,
            json=problem_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            solution = response.json()
            print("✓ API request successful!")
            print(f"  Status: {solution['solution_status']}")
            print(f"  Makespan: {solution['metrics']['makespan']}")
            print(f"  Total weighted tardiness: {solution['metrics']['total_weighted_tardiness']}")
            print(f"  Objective value: {solution['metrics']['objective_value']}")
            
            for job_schedule in solution['job_schedules']:
                print(f"  Job {job_schedule['job_id']}: start={job_schedule['start_time']}, end={job_schedule['completion_time']}, tardiness={job_schedule['tardiness']}")
            
            return True
        else:
            print(f"✗ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_api_weighted_tardiness()
    print(f"\nAPI Test Result: {'PASSED' if success else 'FAILED'}")