import requests
import json

# Minimal valid VRP problem
data = {
    "clients": [
        {"x": 1394000, "y": 357000, "delivery": 5, "service_duration": 10}
    ],
    "depots": [
        {"x": 1394500, "y": 357500}
    ],
    "vehicle_types": [
        {
            "num_available": 1,
            "capacity": 100,
            "start_depot": 0,
            "fixed_cost": 0
        }
    ],
    "max_runtime": 30
}

print("Sending minimal VRP problem...")
response = requests.post("http://127.0.0.1:8000/api/pyvrp/solve", json=data)
print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text}")
else:
    result = response.json()
    print(f"Success! Routes: {len(result.get('routes', []))}")