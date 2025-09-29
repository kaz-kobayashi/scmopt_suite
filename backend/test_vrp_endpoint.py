#!/usr/bin/env python3
"""
Test script to verify VRP endpoint with properly formatted data
"""
import requests
import json

def test_vrp_endpoint():
    """Test VRP endpoint with valid integer coordinates"""
    
    # Test data with integer coordinates (scaled lat/lon)
    test_data = {
        "clients": [
            {"x": 1394000, "y": 357000, "delivery": 5, "service_duration": 10},
            {"x": 1396000, "y": 356500, "delivery": 7, "service_duration": 12},
            {"x": 1393000, "y": 358000, "delivery": 4, "service_duration": 8},
            {"x": 1395000, "y": 355000, "delivery": 6, "service_duration": 15},
            {"x": 1392000, "y": 359000, "delivery": 5, "service_duration": 11}
        ],
        "depots": [
            {"x": 1394500, "y": 357500}
        ],
        "vehicle_types": [
            {
                "num_available": 2,
                "capacity": 100,
                "start_depot": 0,
                "fixed_cost": 100,
                "tw_early": 480,
                "tw_late": 1080
            }
        ],
        "max_runtime": 30
    }
    
    print("=== Testing VRP Endpoint ===")
    print(f"Sending request to http://127.0.0.1:8000/api/pyvrp/solve")
    print(f"Number of clients: {len(test_data['clients'])}")
    print(f"Total demand: {sum(c['delivery'] for c in test_data['clients'])}")
    print(f"Total capacity: {sum(vt['capacity'] * vt['num_available'] for vt in test_data['vehicle_types'])}")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/pyvrp/solve", 
            json=test_data, 
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Solution Found ===")
            print(f"Status: {result.get('status', 'N/A')}")
            print(f"Objective value: {result.get('objective_value', 'N/A')}")
            print(f"Computation time: {result.get('computation_time', 'N/A')} seconds")
            print(f"Number of routes: {len(result.get('routes', []))}")
            
            if result.get('routes'):
                print("\n=== Route Details ===")
                for i, route in enumerate(result['routes']):
                    print(f"\nRoute {i+1}:")
                    print(f"  Vehicle type: {route.get('vehicle_type', 'N/A')}")
                    print(f"  Clients: {route.get('clients', [])}")
                    print(f"  Distance: {route.get('distance', 0)/1000:.1f} km")
                    print(f"  Duration: {route.get('duration', 0)} minutes")
                    print(f"  Start time: {route.get('start_time', 'N/A')} min")
                    print(f"  End time: {route.get('end_time', 'N/A')} min")
                    
                    # Check for reasonable travel times
                    if route.get('duration'):
                        if route['duration'] > 100:
                            print(f"  ⚠️ WARNING: Long route duration ({route['duration']} min)")
                        else:
                            print(f"  ✓ Reasonable route duration")
        else:
            print(f"Error response: {response.text}")
            
            # Try to parse error details
            try:
                error_data = response.json()
                if 'detail' in error_data:
                    print("\nError details:")
                    if isinstance(error_data['detail'], list):
                        for err in error_data['detail']:
                            print(f"  - {err.get('msg', err)}")
                    else:
                        print(f"  - {error_data['detail']}")
            except:
                pass
                
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to backend server")
        print("Make sure the backend is running: cd backend && uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test_vrp_endpoint()