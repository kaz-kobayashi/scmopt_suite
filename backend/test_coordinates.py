#!/usr/bin/env python3
"""
Test script to verify coordinate system fix for VRP solver
"""
import requests
import json

def test_coordinate_system():
    """Test VRP solving with both scaled and actual coordinates"""
    
    # Test 1: Scaled coordinates (old format)
    print("=== Test 1: Scaled Coordinates (1394000, 357000) ===")
    test_data_scaled = {
        "clients": [
            {"x": 1394000, "y": 357000, "delivery": 5, "service_duration": 10},
            {"x": 1396000, "y": 356500, "delivery": 7, "service_duration": 12}
        ],
        "depots": [
            {"x": 1394500, "y": 357500}
        ],
        "vehicle_types": [
            {
                "num_available": 2,
                "capacity": 100,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 100,
                "tw_early": 480,
                "tw_late": 1080
            }
        ],
        "max_runtime": 30
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/api/pyvrp/solve", 
                               json=test_data_scaled, 
                               headers={"Content-Type": "application/json"})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result.get('routes'):
                route = result['routes'][0]
                duration = route.get('duration', 'N/A')
                print(f"Route duration: {duration} minutes")
                print(f"Travel from depot to first client should be reasonable (5-15 minutes)")
            else:
                print("No routes found")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    
    # Test 2: Small integer coordinates (actual lat/lon * 10000)
    print("=== Test 2: Actual Coordinates (139, 35) ===")
    test_data_actual = {
        "clients": [
            {"x": 139, "y": 35, "delivery": 5, "service_duration": 10},
            {"x": 140, "y": 36, "delivery": 7, "service_duration": 12}
        ],
        "depots": [
            {"x": 139, "y": 35}
        ],
        "vehicle_types": [
            {
                "num_available": 2,
                "capacity": 100,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 100,
                "tw_early": 480,
                "tw_late": 1080
            }
        ],
        "max_runtime": 30
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/api/pyvrp/solve", 
                               json=test_data_actual, 
                               headers={"Content-Type": "application/json"})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result.get('routes'):
                route = result['routes'][0]
                duration = route.get('duration', 'N/A')
                print(f"Route duration: {duration} minutes")
                print(f"Travel between Tokyo locations should be reasonable (5-15 minutes)")
            else:
                print("No routes found")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_coordinate_system()