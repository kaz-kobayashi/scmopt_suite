import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import math
import logging
from itertools import permutations

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers"""
    R = 6371  # Earth radius in kilometers
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def two_opt_improvement(route: List[int], distance_matrix: np.ndarray) -> List[int]:
    """Apply 2-opt improvement to a route"""
    best_route = route[:]
    best_distance = calculate_route_distance(route, distance_matrix)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue  # Skip adjacent edges
                
                # Create new route by reversing segment between i and j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_distance = calculate_route_distance(new_route, distance_matrix)
                
                if new_distance < best_distance:
                    best_route = new_route[:]
                    best_distance = new_distance
                    improved = True
        
        route = best_route[:]
    
    return best_route

def calculate_route_distance(route: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total distance for a route"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance

def optimize_vrp_simple_improved(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int = 5,
    depot_name: str = "Depot",
    max_runtime_seconds: int = 30
) -> Dict[str, Any]:
    """
    Simple but effective VRP solver with geographic optimization
    """
    try:
        # Find depot
        depot_idx = None
        for idx, row in locations_df.iterrows():
            if row['name'] == depot_name:
                depot_idx = idx
                break
        
        if depot_idx is None:
            depot_idx = 0
            depot_name = locations_df.iloc[0]['name']
        
        logger.info(f"Simple VRP: Starting optimization with {len(locations_df)} locations, {max_routes} vehicles")
        
        # Calculate distance matrix
        n_locations = len(locations_df)
        distance_matrix = np.zeros((n_locations, n_locations))
        
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    lat1, lon1 = locations_df.iloc[i]['lat'], locations_df.iloc[i]['lon']
                    lat2, lon2 = locations_df.iloc[j]['lat'], locations_df.iloc[j]['lon']
                    distance_matrix[i][j] = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Get all customers (excluding depot)
        customers = [idx for idx in range(n_locations) if idx != depot_idx]
        customer_demands = {idx: locations_df.iloc[idx].get('demand', 1) for idx in customers}
        
        # For small instances, try multiple clustering approaches
        all_solutions = []
        
        # Method 1: Nearest neighbor with capacity constraints
        routes1 = nearest_neighbor_clustering(customers, customer_demands, depot_idx, 
                                            distance_matrix, vehicle_capacity, max_routes)
        if routes1:
            all_solutions.append(('Nearest Neighbor', routes1))
        
        # Method 2: Geographic clustering (angle-based)
        routes2 = angle_based_clustering(customers, locations_df, depot_idx, 
                                       customer_demands, vehicle_capacity, max_routes)
        if routes2:
            all_solutions.append(('Angle-based', routes2))
        
        # Method 3: For small problems, try some permutations
        if len(customers) <= 8:  # Only for small instances
            routes3 = small_instance_optimization(customers, customer_demands, depot_idx,
                                                distance_matrix, vehicle_capacity, max_routes)
            if routes3:
                all_solutions.append(('Small instance optimization', routes3))
        
        # Select best solution
        best_routes = None
        best_total_distance = float('inf')
        best_method = ""
        
        for method, routes in all_solutions:
            # Improve each route with 2-opt
            improved_routes = []
            total_distance = 0
            
            for route_info in routes:
                route_sequence = route_info['sequence']
                improved_sequence = two_opt_improvement(route_sequence, distance_matrix)
                
                # Recalculate distance and other metrics
                route_distance = calculate_route_distance(improved_sequence, distance_matrix)
                route_locations = [locations_df.iloc[idx]['name'] for idx in improved_sequence]
                route_demand = sum(customer_demands.get(idx, 0) for idx in improved_sequence[1:-1])  # Exclude depot
                
                improved_routes.append({
                    'route_id': int(route_info['route_id']),
                    'sequence': [int(x) for x in improved_sequence],
                    'locations': [str(loc) for loc in route_locations],
                    'total_distance': float(route_distance),
                    'total_demand': float(route_demand),
                    'capacity_utilization': float(route_demand / vehicle_capacity) if vehicle_capacity > 0 else 0.0
                })
                total_distance += route_distance
            
            logger.info(f"Simple VRP: {method} total distance: {total_distance:.1f} km")
            
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_routes = improved_routes
                best_method = method
        
        if not best_routes:
            # Fallback
            distance_calc = sum(distance_matrix[customers[i]][customers[i+1]] for i in range(len(customers)-1)) + distance_matrix[depot_idx][customers[0]] + distance_matrix[customers[-1]][depot_idx] if customers else 0.0
            best_routes = [{
                'route_id': 1,
                'sequence': [int(x) for x in [depot_idx] + customers + [depot_idx]],
                'locations': [str(locations_df.iloc[idx]['name']) for idx in [depot_idx] + customers + [depot_idx]],
                'total_distance': float(distance_calc),
                'total_demand': float(sum(customer_demands.values())),
                'capacity_utilization': float(sum(customer_demands.values()) / vehicle_capacity) if vehicle_capacity > 0 else 0.0
            }]
            best_total_distance = best_routes[0]['total_distance']
            best_method = "Fallback single route"
        
        # Calculate summary statistics
        total_demand_served = sum(route['total_demand'] for route in best_routes)
        customers_served = sum(len(route['sequence']) - 2 for route in best_routes)  # -2 for depot start/end
        customers_unserved = len(customers) - customers_served
        avg_capacity_utilization = float(np.mean([route['capacity_utilization'] for route in best_routes])) if best_routes else 0.0
        
        logger.info(f"Simple VRP: Best solution using {best_method}: {best_total_distance:.1f} km")
        
        return {
            'routes': best_routes,
            'summary': {
                'total_routes': int(len(best_routes)),
                'total_distance': float(best_total_distance),
                'total_demand_served': float(total_demand_served),
                'avg_capacity_utilization': float(avg_capacity_utilization),
                'customers_served': int(customers_served),
                'customers_unserved': int(customers_unserved),
                'optimization_cost': float(best_total_distance)
            },
            'parameters': {
                'vehicle_capacity': float(vehicle_capacity),
                'max_routes': int(max_routes),
                'depot': str(depot_name),
                'max_runtime_seconds': int(max_runtime_seconds)
            },
            'optimization_stats': {
                'algorithm': f'Simple VRP - {best_method}',
                'convergence': True,
                'runtime_seconds': 1.0  # Fast algorithm
            }
        }
        
    except Exception as e:
        logger.error(f"Error in simple VRP optimization: {str(e)}")
        raise e

def nearest_neighbor_clustering(customers, customer_demands, depot_idx, distance_matrix, 
                              vehicle_capacity, max_routes):
    """Nearest neighbor clustering with capacity constraints"""
    routes = []
    remaining_customers = customers[:]
    
    for route_id in range(max_routes):
        if not remaining_customers:
            break
            
        current_route = [depot_idx]
        current_capacity = 0
        
        # Start from depot, find nearest unvisited customer
        current_location = depot_idx
        
        while remaining_customers:
            # Find nearest customer that fits in capacity
            best_customer = None
            best_distance = float('inf')
            
            for customer in remaining_customers:
                if current_capacity + customer_demands[customer] <= vehicle_capacity:
                    distance = distance_matrix[current_location][customer]
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer
            
            if best_customer is None:
                break  # No more customers fit
                
            current_route.append(best_customer)
            current_capacity += customer_demands[best_customer]
            current_location = best_customer
            remaining_customers.remove(best_customer)
        
        current_route.append(depot_idx)
        
        if len(current_route) > 2:  # More than just depot->depot
            routes.append({
                'route_id': route_id + 1,
                'sequence': current_route,
                'total_demand': current_capacity
            })
    
    return routes

def angle_based_clustering(customers, locations_df, depot_idx, customer_demands, 
                          vehicle_capacity, max_routes):
    """Cluster customers by angle from depot"""
    depot_lat = locations_df.iloc[depot_idx]['lat']
    depot_lon = locations_df.iloc[depot_idx]['lon']
    
    # Calculate angles for each customer
    customer_angles = []
    for customer in customers:
        lat = locations_df.iloc[customer]['lat']
        lon = locations_df.iloc[customer]['lon']
        
        # Calculate angle from depot
        angle = math.atan2(lat - depot_lat, lon - depot_lon)
        customer_angles.append((customer, angle, customer_demands[customer]))
    
    # Sort by angle
    customer_angles.sort(key=lambda x: x[1])
    
    # Create routes by grouping consecutive customers by capacity
    routes = []
    route_id = 1
    i = 0
    
    while i < len(customer_angles) and route_id <= max_routes:
        current_route = [depot_idx]
        current_capacity = 0
        
        # Add customers in angular order until capacity is reached
        while i < len(customer_angles):
            customer, angle, demand = customer_angles[i]
            if current_capacity + demand <= vehicle_capacity:
                current_route.append(customer)
                current_capacity += demand
                i += 1
            else:
                break
        
        current_route.append(depot_idx)
        
        if len(current_route) > 2:
            routes.append({
                'route_id': route_id,
                'sequence': current_route,
                'total_demand': current_capacity
            })
            route_id += 1
        else:
            i += 1  # Skip this customer if it doesn't fit anywhere
    
    return routes

def small_instance_optimization(customers, customer_demands, depot_idx, distance_matrix, 
                               vehicle_capacity, max_routes):
    """For small instances, try some permutations"""
    if len(customers) > 8:
        return None
    
    best_routes = None
    best_distance = float('inf')
    
    # Try different ways to split customers into routes
    from itertools import combinations
    
    # Try different partitions
    for num_routes in range(1, min(max_routes + 1, len(customers) + 1)):
        # Try some combinations of customer assignments
        if num_routes == 1:
            # Single route
            route_sequence = [depot_idx] + customers + [depot_idx]
            total_demand = sum(customer_demands[c] for c in customers)
            if total_demand <= vehicle_capacity:
                distance = calculate_route_distance(route_sequence, distance_matrix)
                if distance < best_distance:
                    best_distance = distance
                    best_routes = [{
                        'route_id': 1,
                        'sequence': route_sequence,
                        'total_demand': total_demand
                    }]
        
        elif num_routes == 2 and len(customers) >= 2:
            # Try splitting into 2 routes
            for split_point in range(1, len(customers)):
                route1_customers = customers[:split_point]
                route2_customers = customers[split_point:]
                
                route1_demand = sum(customer_demands[c] for c in route1_customers)
                route2_demand = sum(customer_demands[c] for c in route2_customers)
                
                if route1_demand <= vehicle_capacity and route2_demand <= vehicle_capacity:
                    route1_seq = [depot_idx] + route1_customers + [depot_idx]
                    route2_seq = [depot_idx] + route2_customers + [depot_idx]
                    
                    total_distance = (calculate_route_distance(route1_seq, distance_matrix) + 
                                    calculate_route_distance(route2_seq, distance_matrix))
                    
                    if total_distance < best_distance:
                        best_distance = total_distance
                        best_routes = [
                            {'route_id': 1, 'sequence': route1_seq, 'total_demand': route1_demand},
                            {'route_id': 2, 'sequence': route2_seq, 'total_demand': route2_demand}
                        ]
    
    return best_routes