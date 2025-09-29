#!/usr/bin/env python3
"""
Debug JobShop API request format
"""

import requests
import json

API_BASE_URL = "http://127.0.0.1:8000/api"

# Get sample problem
response = requests.get(f"{API_BASE_URL}/jobshop/sample-problem/job_shop")
print(f"Sample problem response status: {response.status_code}")

if response.status_code == 200:
    problem = response.json()
    print(f"Sample problem structure:")
    print(json.dumps(problem, indent=2)[:1000] + "...")
    
    # Try to solve it
    print(f"\nTrying to solve the problem...")
    solve_response = requests.post(f"{API_BASE_URL}/jobshop/solve", json=problem)
    print(f"Solve response status: {solve_response.status_code}")
    if solve_response.status_code != 200:
        print(f"Error response: {solve_response.text}")
    else:
        solution = solve_response.json()
        print(f"Success! Makespan: {solution['metrics']['makespan']}")
else:
    print(f"Failed to get sample problem: {response.text}")