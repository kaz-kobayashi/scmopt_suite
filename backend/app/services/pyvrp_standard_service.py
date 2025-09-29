import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import math
import pyvrp
from pyvrp.stop import MaxIterations, MaxRuntime
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

def convert_vrplib_to_kanto_coordinates(vrp_file_path: str, instance) -> List[Dict[str, Any]]:
    """
    Convert VRPLIB coordinates to Kanto region (Japan) coordinates for visualization
    
    Returns:
        List of location dictionaries with name, lat, lon, demand
    """
    try:
        # Parse coordinates directly from VRPLIB file
        coordinates = []
        demands = []
        
        with open(vrp_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse NODE_COORD_SECTION
        coord_section = False
        for line in lines:
            line = line.strip()
            if 'NODE_COORD_SECTION' in line:
                coord_section = True
                continue
            if 'DEMAND_SECTION' in line:
                coord_section = False
                continue
            if coord_section and line and not line.startswith('DEPOT'):
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                    coordinates.append((idx, x, y))
        
        # Parse DEMAND_SECTION
        demand_section = False
        for line in lines:
            line = line.strip()
            if 'DEMAND_SECTION' in line:
                demand_section = True
                continue
            if 'DEPOT_SECTION' in line:
                demand_section = False
                continue
            if demand_section and line and not line.startswith('EOF'):
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    idx, demand = int(parts[0]), float(parts[1])
                    demands.append((idx, demand))
        
        # Create demand lookup
        demand_map = {idx: demand for idx, demand in demands}
        
        # Find coordinate bounds for scaling
        x_coords = [coord[1] for coord in coordinates]
        y_coords = [coord[2] for coord in coordinates]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Kanto region bounds (Tokyo, Kanagawa, Saitama, Chiba)
        kanto_lat_min, kanto_lat_max = 35.0, 36.5    # Latitude range
        kanto_lon_min, kanto_lon_max = 139.0, 140.5  # Longitude range
        
        # Convert coordinates to Kanto region
        locations = []
        
        for idx, x, y in coordinates:
            # Normalize to 0-1 range
            norm_x = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
            norm_y = (y - min_y) / (max_y - min_y) if max_y != min_y else 0.5
            
            # Scale to Kanto region
            lat = kanto_lat_min + norm_y * (kanto_lat_max - kanto_lat_min)
            lon = kanto_lon_min + norm_x * (kanto_lon_max - kanto_lon_min)
            
            demand = demand_map.get(idx, 0)
            
            if idx == 1:  # Depot is location 1 in VRPLIB
                locations.append({
                    'name': 'Tokyo_DC',
                    'lat': 35.6812,  # Tokyo Station (fixed position)
                    'lon': 139.7671,
                    'demand': 0
                })
            else:
                locations.append({
                    'name': f'Customer_{idx}',
                    'lat': lat,
                    'lon': lon,
                    'demand': float(demand)
                })
        
        return locations
        
    except Exception as e:
        logger.warning(f"Could not convert coordinates from {vrp_file_path}: {e}")
        # Fallback to Tokyo area with random coordinates
        locations = [{'name': 'Tokyo_DC', 'lat': 35.6812, 'lon': 139.7671, 'demand': 0}]
        
        # Add random customer locations around Tokyo
        for i in range(min(100, instance.num_clients)):  # Limit to 100 for performance
            lat = 35.5 + np.random.random() * 1.0  # Random lat around Tokyo
            lon = 139.3 + np.random.random() * 1.0  # Random lon around Tokyo
            demand = 1
            locations.append({
                'name': f'Customer_{i+1}',
                'lat': lat,
                'lon': lon,
                'demand': float(demand)
            })
        
        return locations

def solve_vrplib_instance(
    vrp_file_path: str,
    max_runtime_seconds: int = 60,
    max_iterations: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Solve a VRPLIB format instance using standard PyVRP
    
    Args:
        vrp_file_path: Path to .vrp file in VRPLIB format
        max_runtime_seconds: Maximum runtime in seconds
        max_iterations: Maximum iterations
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info(f"PyVRP Standard: Reading VRPLIB instance from {vrp_file_path}")
        
        # Read the VRP instance using PyVRP's standard read function
        instance = pyvrp.read(vrp_file_path, round_func="round")
        
        logger.info(f"PyVRP Standard: Instance loaded - {instance.num_clients} clients, {instance.num_depots} depot(s)")
        
        # Set up stopping criteria - use time limit
        stop_criteria = MaxRuntime(max_runtime_seconds)
        
        logger.info(f"PyVRP Standard: Starting optimization with max runtime {max_runtime_seconds}s, max iterations {max_iterations}")
        
        # Convert coordinates for map visualization
        map_locations = convert_vrplib_to_kanto_coordinates(vrp_file_path, instance)
        
        # Multi-seed optimization for better results
        best_result = None
        best_cost = float('inf')
        
        # Try multiple seeds for better solutions
        seeds = [seed, 1, 456, 789, 123] if seed != 456 else [456, 1, 42, 789, 123]
        runtime_per_seed = max(max_runtime_seconds // len(seeds), 10)
        
        for current_seed in seeds:
            try:
                current_stop = MaxRuntime(runtime_per_seed)
                current_result = pyvrp.solve(
                    instance,
                    stop=current_stop,
                    seed=current_seed,
                    display=False
                )
                
                if current_result.best and current_result.cost() < best_cost:
                    best_cost = current_result.cost()
                    best_result = current_result
                    logger.info(f"PyVRP: New best cost {best_cost} with seed {current_seed}")
            except Exception as e:
                logger.warning(f"PyVRP: Seed {current_seed} failed: {str(e)}")
                continue
        
        # Use best result or fallback to single run
        result = best_result if best_result else pyvrp.solve(
            instance,
            stop=stop_criteria,
            seed=seed,
            display=False
        )
        
        logger.info(f"PyVRP Standard: Optimization complete")
        logger.info(f"Objective: {result.cost()}")
        logger.info(f"Routes: {len(result.best.routes())}")
        logger.info(f"Runtime: {result.runtime:.2f}s")
        logger.info(f"Iterations: {result.num_iterations}")
        
        # Process results into our standard format
        routes = []
        total_distance = float(result.cost())
        total_demand_served = 0
        
        if result.best and result.best.routes():
            for route_idx, route in enumerate(result.best.routes()):
                route_clients = list(route)
                if not route_clients:
                    continue
                
                # Build route sequence (include depot at start/end)
                route_sequence = [0]  # Depot is always 0
                route_demand = 0
                
                # Add client locations and build PyVRP sequence
                pyvrp_sequence = [0]  # Start at depot (PyVRP index 0)
                for client_idx in route_clients:
                    route_sequence.append(client_idx + 1)  # For display (1-based)
                    pyvrp_sequence.append(client_idx + 1)  # For PyVRP (1-based for clients)
                    # Get client demand from instance
                    client_demand = instance.demands[client_idx] if hasattr(instance, 'demands') else 1
                    route_demand += client_demand
                
                route_sequence.append(0)  # Return to depot (for display)
                pyvrp_sequence.append(0)  # Return to depot (for PyVRP)
                
                # Calculate individual route distance using PyVRP distance matrix
                route_distance = 0.0
                try:
                    # Get distance matrix (profile 0 for default)
                    dist_matrix = instance.distance_matrix(0)
                    
                    # Validate indices and calculate distance
                    for i in range(len(pyvrp_sequence) - 1):
                        from_loc = pyvrp_sequence[i]
                        to_loc = pyvrp_sequence[i + 1]
                        
                        # Check bounds (PyVRP uses 0-based indexing for all locations)
                        if from_loc >= dist_matrix.shape[0] or to_loc >= dist_matrix.shape[0]:
                            logger.warning(f"Index out of bounds: from={from_loc}, to={to_loc}, matrix_size={dist_matrix.shape}")
                            continue
                            
                        distance = dist_matrix[from_loc, to_loc]
                        route_distance += distance
                    
                    # Scale distance to reasonable units (PyVRP often uses different scaling)
                    route_distance = route_distance / 1000.0  # Convert to more readable scale
                
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Could not calculate route {route_idx + 1} distance: {e}")
                    route_distance = 0.0
                
                # Create location names for route (matching map_locations)
                route_location_names = []
                for idx in route_sequence:
                    if idx == 0:
                        route_location_names.append('Tokyo_DC')  # Depot
                    else:
                        route_location_names.append(f'Customer_{idx}')
                
                routes.append({
                    'route_id': route_idx + 1,
                    'sequence': route_sequence,
                    'locations': route_location_names,
                    'total_distance': float(route_distance),
                    'total_demand': float(route_demand),
                    'capacity_utilization': float(route_demand / instance.vehicle_capacity[0]) if hasattr(instance, 'vehicle_capacity') and len(instance.vehicle_capacity) > 0 else 0
                })
                
                total_demand_served += route_demand
        
        # Calculate summary statistics
        customers_served = sum(len(route['sequence']) - 2 for route in routes)  # -2 for depot start/end
        customers_unserved = instance.num_clients - customers_served
        
        # Calculate actual total distance from individual routes
        calculated_total_distance = sum(route['total_distance'] for route in routes)
        avg_capacity_utilization = np.mean([route['capacity_utilization'] for route in routes]) if routes else 0
        
        optimization_stats = {
            'best_cost': float(result.cost()),
            'iterations': int(result.num_iterations),
            'runtime_seconds': float(result.runtime),
            'algorithm': 'PyVRP Standard (VRPLIB)',
            'convergence': bool(result.best is not None and result.best.is_feasible()),
            'instance_info': {
                'num_clients': int(instance.num_clients),
                'num_depots': int(instance.num_depots),
                'instance_name': getattr(instance, 'name', vrp_file_path.split('/')[-1].replace('.vrp', '') if '/' in vrp_file_path else vrp_file_path.replace('.vrp', ''))
            }
        }
        
        return {
            'routes': routes,
            'summary': {
                'total_routes': int(len(routes)),
                'total_distance': float(calculated_total_distance),  # Use calculated distance
                'total_demand_served': float(total_demand_served),
                'avg_capacity_utilization': float(avg_capacity_utilization),
                'customers_served': int(customers_served),
                'customers_unserved': int(customers_unserved),
                'optimization_cost': float(result.cost())
            },
            'parameters': {
                'max_runtime_seconds': int(max_runtime_seconds),
                'max_iterations': int(max_iterations),
                'seed': int(seed)
            },
            'optimization_stats': optimization_stats,
            'map_locations': map_locations  # Add coordinates for map visualization
        }
    
    except Exception as e:
        logger.error(f"Error solving VRPLIB instance: {str(e)}")
        raise e

def create_vrp_instance_from_coordinates(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int = 5,
    depot_name: str = "Depot"
) -> Dict[str, Any]:
    """
    Create a VRP instance from coordinates using PyVRP's ProblemData
    
    Args:
        locations_df: DataFrame with columns ['name', 'lat', 'lon', 'demand']
        vehicle_capacity: Maximum capacity per vehicle
        max_routes: Maximum number of vehicles
        depot_name: Name of depot location
        
    Returns:
        ProblemData instance for PyVRP
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
        
        logger.info(f"PyVRP ProblemData: Creating instance with {len(locations_df)} locations")
        
        # Prepare coordinates (use original lat/lon values)
        depots = [(
            float(locations_df.iloc[depot_idx]['lon']),  # x-coordinate
            float(locations_df.iloc[depot_idx]['lat'])   # y-coordinate
        )]
        
        clients = []
        demands = []
        
        for idx, row in locations_df.iterrows():
            if idx != depot_idx:
                clients.append((
                    float(row['lon']),  # x-coordinate
                    float(row['lat'])   # y-coordinate
                ))
                demands.append(int(row.get('demand', 1)))
        
        # Create vehicle types
        vehicle_types = [int(vehicle_capacity)] * max_routes
        
        # Create ProblemData using Model approach
        model = pyvrp.Model()
        
        # Add depot
        depot = model.add_depot(x=depots[0][0], y=depots[0][1])
        
        # Add vehicle type
        vehicle_type = model.add_vehicle_type(
            capacity=vehicle_types[0],
            num_available=max_routes
        )
        
        # Add clients
        for i, (x, y) in enumerate(clients):
            model.add_client(
                x=x, 
                y=y, 
                delivery=demands[i]
            )
        
        problem_data = model
        
        return problem_data
    
    except Exception as e:
        logger.error(f"Error creating VRP instance from coordinates: {str(e)}")
        raise e

def solve_vrp_from_coordinates_standard(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int = 5,
    depot_name: str = "Depot",
    max_runtime_seconds: int = 30
) -> Dict[str, Any]:
    """
    Solve VRP from coordinates using standard PyVRP approach
    
    Args:
        locations_df: DataFrame with columns ['name', 'lat', 'lon', 'demand']
        vehicle_capacity: Maximum capacity per vehicle
        max_routes: Maximum number of vehicles
        depot_name: Name of depot location
        max_runtime_seconds: Maximum optimization runtime
    
    Returns:
        Dictionary with optimization results
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
        
        logger.info(f"PyVRP Standard Coords: Starting optimization with {len(locations_df)} locations")
        
        # Create problem data
        problem_data = create_vrp_instance_from_coordinates(
            locations_df, vehicle_capacity, max_routes, depot_name
        )
        
        # Solve using model.solve instead of pyvrp.solve
        result = problem_data.solve(
            stop=MaxRuntime(max_runtime_seconds),
            seed=42
        )
        
        logger.info(f"PyVRP Standard Coords: Optimization complete")
        logger.info(f"Objective: {result.cost()}")
        logger.info(f"Routes: {len(result.best.routes())}")
        
        # Process results
        routes = []
        total_demand_served = 0
        
        if result.best and result.best.routes():
            # Create mapping from client index to original dataframe index
            client_to_original = {}
            client_counter = 0
            for idx, row in locations_df.iterrows():
                if idx != depot_idx:
                    client_to_original[client_counter] = idx
                    client_counter += 1
            
            for route_idx, route in enumerate(result.best.routes()):
                route_clients = list(route)
                if not route_clients:
                    continue
                
                # Build route sequence
                route_sequence = [depot_idx]  # Start at depot
                route_locations = [depot_name]
                route_demand = 0
                
                # Add clients in route
                for client_idx in route_clients:
                    if client_idx in client_to_original:
                        original_idx = client_to_original[client_idx]
                        route_sequence.append(original_idx)
                        route_locations.append(locations_df.iloc[original_idx]['name'])
                        route_demand += locations_df.iloc[original_idx].get('demand', 1)
                
                route_sequence.append(depot_idx)  # Return to depot
                route_locations.append(depot_name)
                
                # Calculate route distance using haversine
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
                
                total_demand_served += route_demand
        
        # Calculate statistics
        customers_served = len([idx for idx in locations_df.index if idx != depot_idx])
        served_in_routes = sum(len(route['sequence']) - 2 for route in routes) if routes else 0
        customers_unserved = customers_served - served_in_routes
        
        total_distance = sum(route['total_distance'] for route in routes)
        avg_capacity_utilization = np.mean([route['capacity_utilization'] for route in routes]) if routes else 0
        
        optimization_stats = {
            'best_cost': float(result.cost()),
            'iterations': int(result.num_iterations),
            'runtime_seconds': float(result.runtime),
            'algorithm': 'PyVRP Standard (Coordinates)',
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
    
    except Exception as e:
        logger.error(f"Error in standard PyVRP coordinate optimization: {str(e)}")
        raise e

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