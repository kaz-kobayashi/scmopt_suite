#!/bin/bash
# Test VRP endpoint with properly formatted data

echo "Testing VRP endpoint with scaled integer coordinates..."

curl -X POST http://127.0.0.1:8000/api/pyvrp/solve \
  -H "Content-Type: application/json" \
  -d '{
    "clients": [
      {"x": 1394000, "y": 357000, "delivery": 5, "service_duration": 10},
      {"x": 1396000, "y": 356500, "delivery": 7, "service_duration": 12}
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
  }' | python -m json.tool