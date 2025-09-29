import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import math
import logging

logger = logging.getLogger(__name__)

def optimize_vrp_hybrid(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int = 5,
    depot_name: str = "Depot",
    max_runtime_seconds: int = 30
) -> Dict[str, Any]:
    """
    Hybrid VRP optimization: Run both PyVRP and Simple VRP, return the better solution
    """
    results = []
    
    # Try Simple VRP first (fast)
    try:
        from app.services.simple_vrp_service import optimize_vrp_simple_improved
        logger.info("Hybrid VRP: Running Simple VRP algorithm")
        simple_result = optimize_vrp_simple_improved(
            locations_df, vehicle_capacity, max_routes, depot_name, max_runtime_seconds
        )
        results.append(("Simple VRP", simple_result))
        logger.info(f"Simple VRP: {simple_result['summary']['total_distance']:.1f} km")
    except Exception as e:
        logger.warning(f"Simple VRP failed: {e}")
    
    # Try PyVRP (slower but potentially more optimal)
    try:
        from app.services.pyvrp_service import optimize_vrp_with_pyvrp
        logger.info("Hybrid VRP: Running PyVRP algorithm")
        
        # Disable the fallback in PyVRP to avoid circular calls
        import app.services.pyvrp_service
        original_function = app.services.pyvrp_service.optimize_vrp_with_pyvrp
        
        # Temporarily patch PyVRP to not use simple VRP fallback
        def pyvrp_no_fallback(*args, **kwargs):
            try:
                return original_pyvrp_implementation(*args, **kwargs)
            except:
                # Return a dummy result instead of calling simple VRP
                return {
                    'routes': [],
                    'summary': {'total_distance': float('inf')},
                    'optimization_stats': {'algorithm': 'PyVRP (failed)'}
                }
        
        pyvrp_result = original_function(
            locations_df, vehicle_capacity, max_routes, depot_name, max_runtime_seconds
        )
        
        if pyvrp_result['summary']['total_distance'] < float('inf'):
            results.append(("PyVRP", pyvrp_result))
            logger.info(f"PyVRP: {pyvrp_result['summary']['total_distance']:.1f} km")
            
    except Exception as e:
        logger.warning(f"PyVRP failed: {e}")
    
    # Select the best result
    if not results:
        raise Exception("Both PyVRP and Simple VRP failed")
    
    best_method, best_result = min(results, key=lambda x: x[1]['summary']['total_distance'])
    
    logger.info(f"Hybrid VRP: Best solution is {best_method} with {best_result['summary']['total_distance']:.1f} km")
    
    # Update the algorithm name to indicate hybrid approach
    best_result['optimization_stats']['algorithm'] = f"Hybrid VRP - {best_method} (best of {len(results)})"
    
    return best_result

def original_pyvrp_implementation(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int = 5,
    depot_name: str = "Depot",
    max_runtime_seconds: int = 30
) -> Dict[str, Any]:
    """
    Original PyVRP implementation without fallback to avoid circular imports
    """
    import pyvrp
    
    # Find depot
    depot_idx = None
    for idx, row in locations_df.iterrows():
        if row['name'] == depot_name:
            depot_idx = idx
            break
    
    if depot_idx is None:
        depot_idx = 0  # Use first location as depot
        depot_name = locations_df.iloc[0]['name']
    
    # Prepare data
    n_locations = len(locations_df)
    
    # Calculate distance matrix with better precision
    distances = np.zeros((n_locations, n_locations), dtype=int)
    distances_km = np.zeros((n_locations, n_locations))  # Keep km version for later
    
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                lat1, lon1 = locations_df.iloc[i]['lat'], locations_df.iloc[i]['lon']
                lat2, lon2 = locations_df.iloc[j]['lat'], locations_df.iloc[j]['lon']
                dist_km = haversine_distance(lat1, lon1, lat2, lon2)
                distances_km[i][j] = dist_km
                # Scale to integers (10m units) for better precision
                distances[i][j] = int(dist_km * 100)  # 1 unit = 10m
    
    # Create PyVRP model
    model = pyvrp.Model()
    
    # Add depot - Use more precise coordinates
    depot_lat = int(locations_df.iloc[depot_idx]['lat'] * 10000)  # Higher precision
    depot_lon = int(locations_df.iloc[depot_idx]['lon'] * 10000)
    depot = model.add_depot(x=depot_lon, y=depot_lat)
    
    logger.info(f"PyVRP: Starting optimization with {n_locations} locations, {max_routes} vehicles, capacity {vehicle_capacity}")
    
    # Add clients (excluding depot)
    client_indices = {}
    client_counter = 0
    for idx, row in locations_df.iterrows():
        if idx != depot_idx:
            demand = int(row.get('demand', 1))  # Default demand of 1 if not specified
            lat_int = int(row['lat'] * 10000)  # Higher precision
            lon_int = int(row['lon'] * 10000)
            client = model.add_client(
                x=lon_int, 
                y=lat_int, 
                delivery=demand,
                service_duration=300  # 5 minutes service time in seconds
            )
            client_indices[idx] = client_counter
            client_counter += 1
    
    # Add vehicle type
    vehicle_capacity_int = int(vehicle_capacity)
    vehicle_type = model.add_vehicle_type(
        capacity=vehicle_capacity_int,
        num_available=max_routes
    )
    
    # Set distance matrix via edge weights
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                if i == depot_idx:
                    # From depot to clients
                    if j != depot_idx:
                        client_j = client_indices.get(j, None)
                        if client_j is not None:
                            model.add_edge(depot, model.locations[client_j + 1], distances[i][j])
                elif j == depot_idx:
                    # From clients to depot
                    if i != depot_idx:
                        client_i = client_indices.get(i, None)
                        if client_i is not None:
                            model.add_edge(model.locations[client_i + 1], depot, distances[i][j])
                else:
                    # Between clients
                    client_i = client_indices.get(i, None)
                    client_j = client_indices.get(j, None)
                    if client_i is not None and client_j is not None:
                        model.add_edge(model.locations[client_i + 1], model.locations[client_j + 1], distances[i][j])
    
    # Simple solve with basic timeout
    stop_criteria = pyvrp.stop.MaxRuntime(max_runtime_seconds)
    result = model.solve(stop=stop_criteria, seed=42)
    
    # Process results
    routes = []
    total_distance = 0
    total_demand_served = 0
    
    if result.best is not None and result.best.routes():
        # Create mapping from client index to original dataframe index
        client_to_original = {}
        client_counter = 0
        for idx, row in locations_df.iterrows():
            if idx != depot_idx:
                client_to_original[client_counter] = idx
                client_counter += 1
        
        for route_idx, route in enumerate(result.best.routes()):
            if not list(route):  # Check if route has any visits
                continue
                
            # Build route sequence
            route_sequence = [depot_idx]  # Start at depot
            route_locations = [depot_name]
            route_distance = 0
            route_demand = 0
            
            # Add clients in route
            for client_idx in route:
                if client_idx in client_to_original:
                    original_idx = client_to_original[client_idx]
                    route_sequence.append(original_idx)
                    route_locations.append(locations_df.iloc[original_idx]['name'])
                    route_demand += locations_df.iloc[original_idx].get('demand', 1)
            
            route_sequence.append(depot_idx)  # Return to depot
            route_locations.append(depot_name)
            
            # Calculate route distance in km
            route_distance_km = 0
            for i in range(len(route_sequence) - 1):
                route_distance_km += distances_km[route_sequence[i]][route_sequence[i + 1]]
            
            route_info = {
                'route_id': int(route_idx + 1),
                'sequence': [int(x) for x in route_sequence],
                'locations': [str(x) for x in route_locations],
                'total_distance': float(route_distance_km),
                'total_demand': float(route_demand),
                'capacity_utilization': float(route_demand / vehicle_capacity if vehicle_capacity > 0 else 0)
            }
            
            routes.append(route_info)
            total_distance += route_distance_km
            total_demand_served += route_demand
    
    # Calculate statistics
    customers_served = len([idx for idx in locations_df.index if idx != depot_idx])
    served_in_routes = sum(len(route['sequence']) - 2 for route in routes) if routes else 0  # -2 for depot start/end
    customers_unserved = customers_served - served_in_routes
    
    avg_capacity_utilization = float(np.mean([route['capacity_utilization'] for route in routes])) if routes else 0.0
    
    # Get optimization statistics
    optimization_stats = {
        'best_cost': float(result.cost()) if result.best else 0.0,
        'runtime_seconds': float(max_runtime_seconds),
        'algorithm': 'PyVRP (Basic)',
        'convergence': bool(result.best is not None and result.best.is_feasible())
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