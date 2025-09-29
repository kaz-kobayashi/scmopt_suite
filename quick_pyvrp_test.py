#!/usr/bin/env python3
"""
Quick PyVRP endpoint test script

Tests basic functionality of the PyVRP endpoint with smaller problem instances.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/api/pyvrp/solve"

def create_simple_cvrp():
    """Create a very simple CVRP problem for quick testing"""
    return {
        "clients": [
            {"x": 10, "y": 10, "delivery": 20, "pickup": 0, "service_duration": 5, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 20, "y": 15, "delivery": 15, "pickup": 0, "service_duration": 5, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 5, "y": 20, "delivery": 25, "pickup": 0, "service_duration": 5, "tw_early": 0, "tw_late": 1440, "required": True}
        ],
        "depots": [
            {"x": 0, "y": 0, "tw_early": 0, "tw_late": 1440}
        ],
        "vehicle_types": [
            {
                "num_available": 2,
                "capacity": 100,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 50,
                "tw_early": 480,
                "tw_late": 1080,
                "max_duration": 600,
                "max_distance": 100000
            }
        ],
        "max_runtime": 10
    }

def create_simple_vrptw():
    """Create a simple VRPTW problem"""
    return {
        "clients": [
            {"x": 10, "y": 10, "delivery": 10, "pickup": 0, "service_duration": 10, "tw_early": 500, "tw_late": 600, "required": True},
            {"x": 20, "y": 20, "delivery": 15, "pickup": 0, "service_duration": 10, "tw_early": 700, "tw_late": 800, "required": True}
        ],
        "depots": [
            {"x": 0, "y": 0, "tw_early": 0, "tw_late": 1440}
        ],
        "vehicle_types": [
            {
                "num_available": 1,
                "capacity": 50,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 100,
                "tw_early": 480,
                "tw_late": 1080,
                "max_duration": 600,
                "max_distance": 100000
            }
        ],
        "max_runtime": 10
    }

def test_endpoint(problem_data, test_name):
    """Test the endpoint with given problem data"""
    print(f"\n🧪 Testing {test_name}...")
    
    start_time = time.time()
    
    try:
        response = requests.post(ENDPOINT, json=problem_data, timeout=30)
        response_time = time.time() - start_time
        
        print(f"   ⏱️  Response time: {response_time:.2f}s")
        print(f"   📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Status: {result.get('status', 'unknown')}")
            print(f"   🎯 Objective value: {result.get('objective_value', 'N/A')}")
            print(f"   🚚 Number of routes: {len(result.get('routes', []))}")
            print(f"   ✔️  Feasible: {result.get('is_feasible', False)}")
            
            # Show route details
            routes = result.get('routes', [])
            for i, route in enumerate(routes):
                clients = route.get('clients', [])
                distance = route.get('distance', 0)
                print(f"     Route {i}: visits {len(clients)} clients, distance={distance}")
            
            return True
        else:
            print(f"   ❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main():
    print("Quick PyVRP Endpoint Test")
    print("=" * 40)
    print(f"Testing endpoint: {ENDPOINT}")
    
    # Test 1: Simple CVRP
    cvrp_problem = create_simple_cvrp()
    cvrp_success = test_endpoint(cvrp_problem, "Simple CVRP")
    
    # Test 2: Simple VRPTW
    vrptw_problem = create_simple_vrptw()
    vrptw_success = test_endpoint(vrptw_problem, "Simple VRPTW")
    
    # Summary
    print("\n" + "=" * 40)
    print("QUICK TEST SUMMARY")
    print("=" * 40)
    
    total_tests = 2
    passed_tests = sum([cvrp_success, vrptw_success])
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ All quick tests passed! Endpoint is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the endpoint implementation.")
    
    print("\n📝 Test problems used:")
    print("   - Simple CVRP: 3 clients, 1 depot, 2 vehicles")
    print("   - Simple VRPTW: 2 clients with time windows, 1 depot, 1 vehicle")

if __name__ == "__main__":
    main()