import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import math
from datetime import datetime, timedelta

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def calculate_distance_matrix(locations: List[Dict[str, Any]]) -> np.ndarray:
    """
    Calculate distance matrix for all locations
    """
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = haversine_distance(
                    locations[i]['lat'], locations[i]['lon'],
                    locations[j]['lat'], locations[j]['lon']
                )
                distance_matrix[i][j] = dist
    
    return distance_matrix

def nearest_neighbor_tsp(distance_matrix: np.ndarray, start_city: int = 0) -> Tuple[List[int], float]:
    """
    Solve TSP using nearest neighbor heuristic
    """
    n = len(distance_matrix)
    unvisited = set(range(n))
    current_city = start_city
    tour = [current_city]
    unvisited.remove(current_city)
    total_distance = 0
    
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: distance_matrix[current_city][city])
        total_distance += distance_matrix[current_city][nearest_city]
        current_city = nearest_city
        tour.append(current_city)
        unvisited.remove(current_city)
    
    # Return to start
    total_distance += distance_matrix[current_city][start_city]
    tour.append(start_city)
    
    return tour, total_distance

def optimize_vehicle_routing(locations_df: pd.DataFrame, 
                           vehicle_capacity: float,
                           max_routes: int = 5,
                           depot_name: str = "Depot") -> Dict[str, Any]:
    """
    Simple Vehicle Routing Problem (VRP) solver
    """
    # Find depot
    depot_idx = None
    for idx, row in locations_df.iterrows():
        if row['name'] == depot_name:
            depot_idx = idx
            break
    
    if depot_idx is None:
        # Use first location as depot
        depot_idx = 0
    
    # Prepare locations
    locations = []
    demands = []
    for idx, row in locations_df.iterrows():
        locations.append({
            'name': row['name'],
            'lat': row['lat'],
            'lon': row['lon']
        })
        demands.append(row.get('demand', 0))
    
    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(locations)
    
    # Simple capacity-based clustering
    routes = []
    remaining_customers = list(range(len(locations)))
    remaining_customers.remove(depot_idx)  # Remove depot
    
    route_id = 0
    while remaining_customers and route_id < max_routes:
        current_route = [depot_idx]  # Start at depot
        current_capacity = 0
        
        while remaining_customers:
            # Find nearest customer that fits capacity
            best_customer = None
            best_distance = float('inf')
            
            for customer in remaining_customers:
                if current_capacity + demands[customer] <= vehicle_capacity:
                    distance = distance_matrix[current_route[-1]][customer]
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer
            
            if best_customer is None:
                break  # No more customers fit
            
            current_route.append(best_customer)
            current_capacity += demands[best_customer]
            remaining_customers.remove(best_customer)
        
        current_route.append(depot_idx)  # Return to depot
        
        # Calculate route statistics
        route_distance = 0
        for i in range(len(current_route) - 1):
            route_distance += distance_matrix[current_route[i]][current_route[i + 1]]
        
        route_info = {
            'route_id': int(route_id),
            'sequence': [int(x) for x in current_route],
            'locations': [str(locations[i]['name']) for i in current_route],
            'total_distance': float(route_distance),
            'total_demand': float(current_capacity),
            'capacity_utilization': float(current_capacity / vehicle_capacity) if vehicle_capacity > 0 else 0.0
        }
        
        routes.append(route_info)
        route_id += 1
    
    # Calculate summary statistics
    total_distance = sum(route['total_distance'] for route in routes)
    total_demand_served = sum(route['total_demand'] for route in routes)
    avg_capacity_utilization = float(np.mean([route['capacity_utilization'] for route in routes])) if routes else 0.0
    
    return {
        'routes': routes,
        'summary': {
            'total_routes': int(len(routes)),
            'total_distance': float(total_distance),
            'total_demand_served': float(total_demand_served),
            'avg_capacity_utilization': float(avg_capacity_utilization),
            'customers_served': int(len(locations) - len(remaining_customers) - 1),  # Exclude depot
            'customers_unserved': int(len(remaining_customers))
        },
        'parameters': {
            'vehicle_capacity': float(vehicle_capacity),
            'max_routes': int(max_routes),
            'depot': str(depot_name)
        }
    }

def calculate_co2_emissions(capacity_kg: float, loading_rate: float, distance_km: float, 
                          fuel_type: str = "gasoline") -> Dict[str, Any]:
    """
    Calculate CO2 emissions for transportation
    """
    # Fuel consumption calculation (L/ton-km)
    if fuel_type.lower() == "diesel":
        fuel_per_ton_km = math.exp(2.67 - 0.927 * math.log(loading_rate) - 0.648 * math.log(capacity_kg/1000))
    else:  # gasoline
        fuel_per_ton_km = math.exp(2.71 - 0.812 * math.log(loading_rate) - 0.654 * math.log(capacity_kg/1000))
    
    # Total fuel consumption
    load_weight_tons = (capacity_kg / 1000) * loading_rate
    total_fuel_liters = fuel_per_ton_km * load_weight_tons * distance_km
    
    # CO2 emissions (kg)
    co2_emissions_kg = total_fuel_liters * 2.322  # kg CO2 per liter
    
    return {
        'fuel_consumption_liters': total_fuel_liters,
        'co2_emissions_kg': co2_emissions_kg,
        'fuel_per_ton_km': fuel_per_ton_km,
        'load_weight_tons': load_weight_tons,
        'distance_km': distance_km,
        'parameters': {
            'capacity_kg': capacity_kg,
            'loading_rate': loading_rate,
            'fuel_type': fuel_type
        }
    }

def optimize_delivery_schedule(orders_df: pd.DataFrame, 
                             working_hours: Tuple[int, int] = (8, 18),
                             service_time_minutes: int = 30) -> Dict[str, Any]:
    """
    Optimize delivery schedule within working hours
    """
    # Sort orders by priority or delivery time window
    if 'priority' in orders_df.columns:
        orders_df = orders_df.sort_values('priority', ascending=False)
    
    schedule = []
    current_time = datetime.now().replace(hour=working_hours[0], minute=0, second=0, microsecond=0)
    end_time = datetime.now().replace(hour=working_hours[1], minute=0, second=0, microsecond=0)
    
    total_travel_time = 0
    for idx, order in orders_df.iterrows():
        # Add travel time if specified
        travel_time = order.get('travel_time_minutes', 0)
        service_start = current_time + timedelta(minutes=travel_time)
        service_end = service_start + timedelta(minutes=service_time_minutes)
        
        if service_end <= end_time:
            schedule.append({
                'order_id': order.get('order_id', idx),
                'customer': order.get('customer', f'Customer_{idx}'),
                'scheduled_start': service_start.strftime('%H:%M'),
                'scheduled_end': service_end.strftime('%H:%M'),
                'travel_time_minutes': travel_time,
                'service_time_minutes': service_time_minutes,
                'status': 'scheduled'
            })
            current_time = service_end
            total_travel_time += travel_time
        else:
            schedule.append({
                'order_id': order.get('order_id', idx),
                'customer': order.get('customer', f'Customer_{idx}'),
                'scheduled_start': None,
                'scheduled_end': None,
                'travel_time_minutes': travel_time,
                'service_time_minutes': service_time_minutes,
                'status': 'unscheduled'
            })
    
    scheduled_orders = len([s for s in schedule if s['status'] == 'scheduled'])
    
    return {
        'schedule': schedule,
        'summary': {
            'total_orders': len(orders_df),
            'scheduled_orders': scheduled_orders,
            'unscheduled_orders': len(orders_df) - scheduled_orders,
            'total_service_time_hours': (scheduled_orders * service_time_minutes) / 60,
            'total_travel_time_hours': total_travel_time / 60,
            'schedule_efficiency': scheduled_orders / len(orders_df) if len(orders_df) > 0 else 0,
            'working_hours': f"{working_hours[0]:02d}:00-{working_hours[1]:02d}:00"
        }
    }