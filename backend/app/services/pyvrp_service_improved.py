import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import math
import pyvrp
import logging

logger = logging.getLogger(__name__)

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

def optimize_vrp_with_pyvrp_improved(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int = 5,
    depot_name: str = "Depot",
    max_runtime_seconds: int = 30
) -> Dict[str, Any]:
    """
    Improved PyVRP optimization using coordinates only (no manual distance matrix)
    """
    try:
        # Find depot
        depot_idx = None
        for idx, row in locations_df.iterrows():
            if row['name'] == depot_name:
                depot_idx = idx
                break
        
        if depot_idx is None:
            depot_idx = 0  # Use first location as depot
            depot_name = locations_df.iloc[0]['name']
        
        logger.info(f"PyVRP Improved: Starting optimization with {len(locations_df)} locations, {max_routes} vehicles, capacity {vehicle_capacity}")
        
        # Prepare data for PyVRP using only coordinates and demands
        clients = []
        for idx, row in locations_df.iterrows():
            if idx != depot_idx:
                clients.append((
                    int(row['lon'] * 1000000),  # Convert to integers with high precision
                    int(row['lat'] * 1000000),
                    int(row.get('demand', 1))
                ))
        
        depot_coords = (
            int(locations_df.iloc[depot_idx]['lon'] * 1000000),
            int(locations_df.iloc[depot_idx]['lat'] * 1000000)
        )
        
        # Create PyVRP problem data directly from coordinates
        from pyvrp.data import ProblemData
        
        # Create distance matrix using Euclidean distance (approximation)
        n_locations = len(locations_df)
        distance_matrix = np.zeros((n_locations, n_locations), dtype=int)
        
        coords = []
        coords.append(depot_coords)  # Depot first
        for idx, row in locations_df.iterrows():
            if idx != depot_idx:
                coords.append((int(row['lon'] * 1000000), int(row['lat'] * 1000000)))
        
        # Calculate distances using actual haversine formula
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    if i == 0:  # Depot
                        lat1, lon1 = locations_df.iloc[depot_idx]['lat'], locations_df.iloc[depot_idx]['lon']
                        other_idx = [idx for idx in locations_df.index if idx != depot_idx][j-1]
                        lat2, lon2 = locations_df.iloc[other_idx]['lat'], locations_df.iloc[other_idx]['lon']
                    elif j == 0:  # To depot
                        other_idx = [idx for idx in locations_df.index if idx != depot_idx][i-1]
                        lat1, lon1 = locations_df.iloc[other_idx]['lat'], locations_df.iloc[other_idx]['lon']
                        lat2, lon2 = locations_df.iloc[depot_idx]['lat'], locations_df.iloc[depot_idx]['lon']
                    else:  # Between clients
                        client_indices = [idx for idx in locations_df.index if idx != depot_idx]
                        idx1 = client_indices[i-1]
                        idx2 = client_indices[j-1]
                        lat1, lon1 = locations_df.iloc[idx1]['lat'], locations_df.iloc[idx1]['lon']
                        lat2, lon2 = locations_df.iloc[idx2]['lat'], locations_df.iloc[idx2]['lon']
                    
                    dist_km = haversine_distance(lat1, lon1, lat2, lon2)
                    distance_matrix[i][j] = int(dist_km * 1000)  # Convert to meters
        
        # Create demands array
        demands = [0]  # Depot has no demand
        for idx, row in locations_df.iterrows():
            if idx != depot_idx:
                demands.append(int(row.get('demand', 1)))
        
        # Create ProblemData
        problem_data = ProblemData(
            clients=coords[1:],  # Exclude depot
            depots=[coords[0]],  # Only depot
            vehicles=[int(vehicle_capacity)] * max_routes,
            demands=demands[1:],  # Exclude depot demand
            distance_matrix=distance_matrix
        )
        
        # Solve with multiple random seeds for better solutions
        best_solution = None
        best_cost = float('inf')
        
        seeds = [42, 123, 456, 789, 999, 1337, 2468, 3691]
        time_per_seed = max(max_runtime_seconds // len(seeds), 5)
        
        for seed in seeds:
            logger.info(f"PyVRP: Trying seed {seed} with {time_per_seed}s runtime")
            result = pyvrp.solve(problem_data, stop=pyvrp.stop.MaxRuntime(time_per_seed), seed=seed)
            
            if result.best and result.cost() < best_cost:
                best_cost = result.cost()
                best_solution = result
                logger.info(f"PyVRP: New best cost {best_cost}")
        
        if not best_solution or not best_solution.best:
            # Fallback to longer single run
            logger.warning("PyVRP: Multi-seed failed, trying single long run")
            result = pyvrp.solve(problem_data, stop=pyvrp.stop.MaxRuntime(max_runtime_seconds), seed=42)
            best_solution = result
        
        # Process results
        routes = []
        total_distance = 0
        total_demand_served = 0
        
        if best_solution.best and best_solution.best.routes():
            client_indices = [idx for idx in locations_df.index if idx != depot_idx]
            
            for route_idx, route in enumerate(best_solution.best.routes()):
                if not list(route):
                    continue
                
                # Build route sequence
                route_sequence = [depot_idx]
                route_locations = [depot_name]
                route_demand = 0
                
                for client_idx in route:
                    if client_idx < len(client_indices):
                        original_idx = client_indices[client_idx]
                        route_sequence.append(original_idx)
                        route_locations.append(locations_df.iloc[original_idx]['name'])
                        route_demand += locations_df.iloc[original_idx].get('demand', 1)
                
                route_sequence.append(depot_idx)
                route_locations.append(depot_name)
                
                # Calculate actual route distance
                route_distance = 0
                for i in range(len(route_sequence) - 1):
                    idx1, idx2 = route_sequence[i], route_sequence[i + 1]
                    lat1, lon1 = locations_df.iloc[idx1]['lat'], locations_df.iloc[idx1]['lon']
                    lat2, lon2 = locations_df.iloc[idx2]['lat'], locations_df.iloc[idx2]['lon']
                    route_distance += haversine_distance(lat1, lon1, lat2, lon2)
                
                routes.append({
                    'route_id': int(route_idx + 1),
                    'sequence': [int(x) for x in route_sequence],
                    'locations': [str(x) for x in route_locations],
                    'total_distance': float(route_distance),
                    'total_demand': float(route_demand),
                    'capacity_utilization': float(route_demand / vehicle_capacity if vehicle_capacity > 0 else 0)
                })
                
                total_distance += route_distance
                total_demand_served += route_demand
        
        customers_served = len([idx for idx in locations_df.index if idx != depot_idx])
        served_in_routes = sum(len(route['sequence']) - 2 for route in routes) if routes else 0
        customers_unserved = customers_served - served_in_routes
        
        avg_capacity_utilization = np.mean([route['capacity_utilization'] for route in routes]) if routes else 0
        
        optimization_stats = {
            'best_cost': float(best_solution.cost()) if best_solution.best else 0.0,
            'runtime_seconds': float(sum(time_per_seed for _ in seeds) if len(seeds) > 1 else max_runtime_seconds),
            'algorithm': 'PyVRP Improved (Multi-seed with coordinates)',
            'convergence': bool(best_solution.best is not None and best_solution.best.is_feasible())
        }
        
        return {
            'routes': routes,
            'summary': {
                'total_routes': int(len(routes)),
                'total_distance': float(total_distance),
                'total_demand_served': float(total_demand_served),
                'avg_capacity_utilization': float(avg_capacity_utilization),
                'customers_served': int(served_in_routes),
                'customers_unserved': int(customers_unserved),
                'optimization_cost': float(optimization_stats['best_cost'])
            },
            'parameters': {
                'vehicle_capacity': float(vehicle_capacity),
                'max_routes': int(max_routes),
                'depot': str(depot_name),
                'max_runtime_seconds': int(max_runtime_seconds)
            },
            'optimization_stats': optimization_stats
        }
    
    except Exception as e:
        logger.error(f"Error in improved PyVRP optimization: {str(e)}")
        # Return fallback using original service
        from app.services.pyvrp_service import fallback_nearest_neighbor
        return fallback_nearest_neighbor(locations_df, vehicle_capacity, max_routes, depot_name)