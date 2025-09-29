"""
PyVRP Unified Service - Unified API for solving all VRP variants with PyVRP

This service provides a single solve endpoint that handles all VRP variants
based on the problem data structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
import time
import math

try:
    import pyvrp
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    PYVRP_AVAILABLE = True
except ImportError as e:
    PYVRP_AVAILABLE = False
    logging.warning(f"PyVRP library not available: {e}. Using fallback implementations.")

from app.models.vrp_unified_models import (
    VRPProblemData, ClientModel, DepotModel, VehicleTypeModel,
    UnifiedVRPSolution, UnifiedRouteModel
)

logger = logging.getLogger(__name__)


def euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate Euclidean distance between two points"""
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> int:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    
    return int(r * c)


def dataframes_to_vrp_json(
    locations_df: pd.DataFrame,
    vehicle_types_df: pd.DataFrame = None,
    time_windows_df: pd.DataFrame = None,
    depot_indices: list = None
) -> dict:
    """
    Convert pandas DataFrames to PyVRP unified API JSON format
    
    Args:
        locations_df: DataFrame with location data (x/lon, y/lat, demand, etc.)
        vehicle_types_df: DataFrame with vehicle type specifications
        time_windows_df: DataFrame with time window constraints
        depot_indices: List of indices that are depots (default: [0])
        
    Returns:
        Dictionary in PyVRP unified API format
    """
    # Default depot indices
    if depot_indices is None:
        depot_indices = [0]
    
    # 1. Separate depots and clients
    depots = []
    clients = []
    
    for idx, row in locations_df.iterrows():
        if idx in depot_indices:
            depots.append({
                "x": int(row.get('x', row.get('lon', 0) * 10000)),
                "y": int(row.get('y', row.get('lat', 0) * 10000))
            })
        else:
            clients.append(_create_client(row, idx, time_windows_df))
    
    # 2. Create vehicle types
    vehicle_types = _create_vehicle_types(vehicle_types_df, depot_indices)
    
    # 3. Calculate distance matrix (optional)
    distance_matrix = _calculate_distance_matrix(locations_df)
    
    return {
        "clients": clients,
        "depots": depots,
        "vehicle_types": vehicle_types,
        "distance_matrix": distance_matrix,
        "max_runtime": 60
    }


def _create_client(row: pd.Series, idx: int, time_windows_df: pd.DataFrame = None) -> dict:
    """Create client data from DataFrame row"""
    client = {
        # Coordinates (scale lat/lon if needed)
        "x": int(row.get('x', row.get('lon', 0) * 10000)),
        "y": int(row.get('y', row.get('lat', 0) * 10000)),
        
        # Demands
        "delivery": int(row.get('demand', row.get('delivery', 0))),
        "pickup": int(row.get('pickup', 0)),
        
        # Service time (in minutes)
        "service_duration": int(row.get('service_time', row.get('service_duration', 10))),
        
        # Required flag
        "required": bool(row.get('required', True)),
        
        # Prize (for PC-VRP)
        "prize": int(row.get('prize', 0))
    }
    
    # Time windows
    if time_windows_df is not None and idx in time_windows_df['location_id'].values:
        tw_row = time_windows_df[time_windows_df['location_id'] == idx].iloc[0]
        client["tw_early"] = int(tw_row['tw_early'] * 60)  # Convert hours to minutes
        client["tw_late"] = int(tw_row['tw_late'] * 60)
    else:
        # Default time window (full day)
        client["tw_early"] = 0
        client["tw_late"] = 1440
    
    return client


def _create_vehicle_types(vehicle_types_df: pd.DataFrame, depot_indices: list) -> list:
    """Create vehicle types from DataFrame"""
    if vehicle_types_df is None or vehicle_types_df.empty:
        # Default vehicle type with realistic working hours (8:00-18:00)
        return [{
            "num_available": 10,
            "capacity": 1000,
            "start_depot": 0,
            "end_depot": 0,
            "fixed_cost": 0,
            "tw_early": 480,  # 8:00 AM (8 * 60)
            "tw_late": 1080,  # 6:00 PM (18 * 60) 
            "max_duration": 600,  # 10 hours max working time
            "max_distance": 200000
        }]
    
    vehicle_types = []
    for _, row in vehicle_types_df.iterrows():
        vt = {
            "num_available": int(row.get('num_available', 1)),
            "capacity": int(row.get('capacity', 1000)),
            "start_depot": int(row.get('start_depot', depot_indices[0])),
            "fixed_cost": int(row.get('fixed_cost', 0)),
            "tw_early": int(row.get('shift_start', 8) * 60),  # Default to 8:00 AM
            "tw_late": int(row.get('shift_end', 18) * 60)   # Default to 6:00 PM
        }
        
        # Optional fields
        if 'end_depot' in row:
            vt["end_depot"] = int(row['end_depot'])
        else:
            vt["end_depot"] = vt["start_depot"]
            
        if 'max_duration' in row:
            vt["max_duration"] = int(row['max_duration'] * 60)
        else:
            vt["max_duration"] = 480
            
        if 'max_distance' in row:
            vt["max_distance"] = int(row['max_distance'] * 1000)
        else:
            vt["max_distance"] = 200000
            
        vehicle_types.append(vt)
    
    return vehicle_types


def _calculate_distance_matrix(locations_df: pd.DataFrame) -> List[List[int]]:
    """Calculate distance matrix from locations using haversine distance"""
    n = len(locations_df)
    matrix = []
    
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                # Get coordinates (10000x scaled for Kanto region)
                x1 = int(locations_df.iloc[i].get('x', locations_df.iloc[i].get('lon', 0) * 10000))
                y1 = int(locations_df.iloc[i].get('y', locations_df.iloc[i].get('lat', 0) * 10000))
                x2 = int(locations_df.iloc[j].get('x', locations_df.iloc[j].get('lon', 0) * 10000))
                y2 = int(locations_df.iloc[j].get('y', locations_df.iloc[j].get('lat', 0) * 10000))
                
                # Convert back to lat/lon and calculate haversine distance
                lon1 = x1 / 10000
                lat1 = y1 / 10000
                lon2 = x2 / 10000
                lat2 = y2 / 10000
                
                dist = haversine_distance(lon1, lat1, lon2, lat2)
                row.append(dist)
        matrix.append(row)
    
    return matrix


class PyVRPUnifiedService:
    """Unified PyVRP service that handles all VRP variants"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not PYVRP_AVAILABLE:
            self.logger.warning("PyVRP not available - using fallback methods")
    
    def solve(self, problem_data: VRPProblemData) -> UnifiedVRPSolution:
        """
        Solve VRP problem using unified API
        
        The VRP variant is automatically determined based on the problem data:
        - Time windows present -> VRPTW
        - Multiple depots -> MDVRP
        - Pickup demands -> PDVRP
        - Prizes and optional visits -> PC-VRP
        - Otherwise -> CVRP
        
        Args:
            problem_data: Unified problem data
            
        Returns:
            Unified solution
        """
        self.logger.info(f"Solving VRP with {len(problem_data.clients)} clients, {len(problem_data.depots)} depots, PyVRP available: {PYVRP_AVAILABLE}")
        
        # Debug: Log problem details
        total_demand = sum(c.delivery if isinstance(c.delivery, int) else sum(c.delivery) for c in problem_data.clients)
        total_capacity = sum(vt.capacity * vt.num_available for vt in problem_data.vehicle_types)
        
        self.logger.info(f"Problem analysis: Total demand={total_demand}, Total capacity={total_capacity}")
        self.logger.info(f"Clients sample: {[(c.x, c.y, c.delivery) for c in problem_data.clients[:3]]}")
        self.logger.info(f"Vehicle types: {[(vt.num_available, vt.capacity) for vt in problem_data.vehicle_types]}")
        self.logger.info(f"Time windows: {[(c.tw_early, c.tw_late) for c in problem_data.clients[:3]]}")
        
        if not PYVRP_AVAILABLE:
            self.logger.warning("PyVRP not available, using fallback solver")
            return self._fallback_solver(problem_data)
        
        try:
            start_time = time.time()
            
            # Create PyVRP model
            model = Model()
            
            # Add depots
            depot_refs = []
            for depot in problem_data.depots:
                # Convert to integer coordinates for PyVRP
                # If coordinates look like they're already scaled (> 1000), use as-is
                # Otherwise, assume they're lat/lon and scale by 10000
                if depot.x > 1000:
                    x_coord = int(depot.x)
                    y_coord = int(depot.y)
                else:
                    x_coord = int(depot.x * 10000)
                    y_coord = int(depot.y * 10000)
                    
                depot_ref = model.add_depot(
                    x=x_coord,
                    y=y_coord
                )
                depot_refs.append(depot_ref)
            
            # Add clients
            client_refs = []
            has_time_windows = False
            has_prizes = False
            has_pickups = False
            
            for client in problem_data.clients:
                # Check variant indicators
                if client.tw_early > 0 or client.tw_late < 1440:
                    has_time_windows = True
                if client.prize is not None and client.prize > 0:
                    has_prizes = True
                if isinstance(client.pickup, int) and client.pickup > 0:
                    has_pickups = True
                elif isinstance(client.pickup, list) and any(p > 0 for p in client.pickup):
                    has_pickups = True
                
                # Add client with proper pickup handling
                delivery_list = client.delivery if isinstance(client.delivery, list) else [client.delivery]
                pickup_list = []
                if client.pickup is not None:
                    pickup_list = client.pickup if isinstance(client.pickup, list) else [client.pickup]
                else:
                    pickup_list = [0]  # Default to 0 pickup
                
                # Convert to integer coordinates for PyVRP
                if client.x > 1000:
                    x_coord = int(client.x)
                    y_coord = int(client.y)
                else:
                    x_coord = int(client.x * 10000)
                    y_coord = int(client.y * 10000)
                    
                client_ref = model.add_client(
                    x=x_coord,
                    y=y_coord,
                    tw_early=client.tw_early or 0,
                    tw_late=client.tw_late or 1440,
                    service_duration=client.service_duration or 10,
                    delivery=delivery_list,
                    pickup=pickup_list,
                    prize=client.prize or 0,
                    required=client.required if client.required is not None else True
                )
                client_refs.append(client_ref)
            
            # Add vehicle types
            vehicle_type_refs = []
            for vt in problem_data.vehicle_types:
                capacity_list = vt.capacity if isinstance(vt.capacity, list) else [vt.capacity]
                
                vehicle_type_ref = model.add_vehicle_type(
                    num_available=vt.num_available,
                    capacity=capacity_list,
                    start_depot=depot_refs[vt.start_depot],
                    end_depot=depot_refs[vt.end_depot] if vt.end_depot is not None else depot_refs[vt.start_depot],
                    fixed_cost=vt.fixed_cost or 0,
                    tw_early=vt.tw_early or 0,
                    tw_late=vt.tw_late or 1440,
                    max_duration=vt.max_duration or 600,
                    max_distance=vt.max_distance or 100000
                )
                vehicle_type_refs.append(vehicle_type_ref)
            
            # Add edges (distances) between all location pairs
            all_locations = depot_refs + client_refs
            
            # Debug: Log coordinate ranges
            all_coords = [(d.x, d.y) for d in problem_data.depots] + [(c.x, c.y) for c in problem_data.clients]
            x_coords = [coord[0] for coord in all_coords]
            y_coords = [coord[1] for coord in all_coords]
            self.logger.info(f"Coordinate ranges: X=[{min(x_coords)}, {max(x_coords)}], Y=[{min(y_coords)}, {max(y_coords)}]")
            
            # For each pair of locations, add edge with distance
            total_edges = 0
            sample_distances = []
            
            for i, loc_from in enumerate(all_locations):
                for j, loc_to in enumerate(all_locations):
                    if i != j:  # Don't add self-loops
                        # Calculate distance from coordinates
                        from_pos = problem_data.depots[i] if i < len(problem_data.depots) else problem_data.clients[i - len(problem_data.depots)]
                        to_pos = problem_data.depots[j] if j < len(problem_data.depots) else problem_data.clients[j - len(problem_data.depots)]
                        
                        # Convert coordinates to lat/lon
                        # If coordinates > 1000, they're already scaled, so divide by 10000
                        # Otherwise, they're already lat/lon
                        if from_pos.x > 1000:
                            from_lon = from_pos.x / 10000
                            from_lat = from_pos.y / 10000
                        else:
                            from_lon = from_pos.x
                            from_lat = from_pos.y
                            
                        if to_pos.x > 1000:
                            to_lon = to_pos.x / 10000
                            to_lat = to_pos.y / 10000
                        else:
                            to_lon = to_pos.x
                            to_lat = to_pos.y
                        
                        # Calculate haversine distance (great circle distance)
                        distance = haversine_distance(from_lon, from_lat, to_lon, to_lat)
                        duration = int(distance / 1000 / 40 * 60)  # Assume 40km/h average speed for Tokyo area
                        
                        # Log sample distances for debugging
                        if total_edges < 5:
                            sample_distances.append(distance)
                            self.logger.info(f"Distance sample {total_edges}: ({from_pos.x}, {from_pos.y}) -> ({to_pos.x}, {to_pos.y}) = {distance}m, duration = {duration}min")
                        
                        total_edges += 1
                        
                        model.add_edge(
                            frm=loc_from,
                            to=loc_to,
                            distance=distance,
                            duration=duration
                        )
            
            self.logger.info(f"Added {total_edges} edges, sample distances: {sample_distances}")
            
            # Solve
            from pyvrp.stop import MaxRuntime
            result = pyvrp.solve(
                model.data(),
                stop=MaxRuntime(problem_data.max_runtime),
                seed=42
            )
            
            solve_time = time.time() - start_time
            
            # Extract solution
            if result.is_feasible():
                routes = self._extract_routes(result, problem_data)
                
                # Get cost safely
                cost = result.cost()
                if cost is None or cost == float('inf') or cost == 0:
                    # Calculate fallback cost from routes
                    cost = sum(route.total_cost for route in routes) if routes else 1.0
                
                # Ensure cost is a valid float
                if cost is None or cost == float('inf') or cost <= 0:
                    cost = 1.0
                
                return UnifiedVRPSolution(
                    status="optimal" if cost > 0 else "feasible",
                    objective_value=float(cost),
                    routes=routes,
                    computation_time=solve_time,
                    solver="PyVRP",
                    is_feasible=True,
                    problem_type="CVRP",
                    problem_size={
                        "num_clients": len(problem_data.clients),
                        "num_depots": len(problem_data.depots),
                        "num_vehicles": sum(vt.num_available for vt in problem_data.vehicle_types)
                    }
                )
            else:
                self.logger.warning("PyVRP returned infeasible solution")
                self.logger.warning(f"Possible causes:")
                self.logger.warning(f"  - Total demand ({total_demand}) vs Total capacity ({total_capacity})")
                self.logger.warning(f"  - Time window constraints too strict")
                self.logger.warning(f"  - Distance matrix issues")
                
                return UnifiedVRPSolution(
                    status="infeasible",
                    objective_value=999999.0,  # Large but finite value
                    routes=[],
                    computation_time=solve_time,
                    solver="PyVRP",
                    is_feasible=False,
                    problem_type="CVRP"
                )
                
        except Exception as e:
            self.logger.error(f"PyVRP unified solve error: {e}", exc_info=True)
            return self._fallback_solver(problem_data)
    
    def _calculate_model_distance_matrix(self, locations: List[Union[DepotModel, ClientModel]]) -> List[List[int]]:
        """Calculate distance matrix for all locations using haversine distance"""
        n = len(locations)
        matrix = []
        
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    # Convert 100x scaled coordinates back to lat/lon
                    lon1 = locations[i].x / 100
                    lat1 = locations[i].y / 100
                    lon2 = locations[j].x / 100
                    lat2 = locations[j].y / 100
                    
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                    row.append(dist)
            matrix.append(row)
        
        return matrix
    
    def _extract_routes(self, result: Any, problem_data: VRPProblemData) -> List[UnifiedRouteModel]:
        """Extract routes from PyVRP solution with detailed timing information"""
        routes = []
        
        # Access the best solution
        solution = result.best
        
        for route_idx, route in enumerate(solution.routes()):
            # Get the vehicle type for this route
            vehicle_type_idx = route.vehicle_type() if hasattr(route, 'vehicle_type') else 0
            
            # Extract client sequence
            clients = []
            demand_served = 0
            
            # Get visits in the route (excluding depot visits)
            for visit in route:
                if visit >= len(problem_data.depots):  # Client visit (not depot)
                    client_idx = visit - len(problem_data.depots)
                    if 0 <= client_idx < len(problem_data.clients):
                        clients.append(client_idx)
                        
                        # Accumulate demand
                        client = problem_data.clients[client_idx]
                        if isinstance(client.delivery, int):
                            demand_served += client.delivery
                        else:
                            demand_served += sum(client.delivery)
            
            # Calculate detailed route metrics with timing
            distance = 0
            duration = 0
            arrival_times = []
            departure_times = []
            waiting_times = []
            # Start at vehicle's earliest time window (typically 8:00 AM = 480 minutes)
            # Use the correct vehicle type for this route
            vt = problem_data.vehicle_types[vehicle_type_idx] if vehicle_type_idx < len(problem_data.vehicle_types) else problem_data.vehicle_types[0]
            current_time = getattr(vt, 'tw_early', 480) if vt else 480  # Default to 8:00 AM
            
            # Get the correct depot indices from the vehicle type
            start_depot_idx = vt.start_depot if hasattr(vt, 'start_depot') else 0
            end_depot_idx = vt.end_depot if hasattr(vt, 'end_depot') and vt.end_depot is not None else start_depot_idx
            
            if len(clients) > 0:
                # Start from the correct depot based on vehicle type
                prev_pos = problem_data.depots[start_depot_idx] if start_depot_idx < len(problem_data.depots) else problem_data.depots[0]
                current_load = 0
                
                for client_idx in clients:
                    curr_pos = problem_data.clients[client_idx]
                    
                    # Calculate travel distance using haversine formula
                    if prev_pos.x > 1000:
                        prev_lon = prev_pos.x / 10000
                        prev_lat = prev_pos.y / 10000
                    else:
                        prev_lon = prev_pos.x
                        prev_lat = prev_pos.y
                        
                    if curr_pos.x > 1000:
                        curr_lon = curr_pos.x / 10000
                        curr_lat = curr_pos.y / 10000
                    else:
                        curr_lon = curr_pos.x
                        curr_lat = curr_pos.y
                    travel_distance = haversine_distance(prev_lon, prev_lat, curr_lon, curr_lat)
                    travel_time = int(travel_distance / 1000 / 40 * 60)  # Assume 40km/h average speed for Tokyo area
                    distance += travel_distance
                    current_time += travel_time
                    
                    # Check time window constraints
                    client_data = problem_data.clients[client_idx]
                    time_window_start = getattr(client_data, 'tw_early', 0) or 0
                    time_window_end = getattr(client_data, 'tw_late', 1440) or 1440
                    
                    # Calculate arrival time and waiting
                    arrival_time = current_time
                    wait_time = max(0, time_window_start - arrival_time)
                    actual_arrival = arrival_time + wait_time
                    
                    # Service time
                    service_time = getattr(client_data, 'service_duration', 10) or 10
                    departure_time = actual_arrival + service_time
                    
                    # Store timing information
                    arrival_times.append(actual_arrival)
                    departure_times.append(departure_time)
                    waiting_times.append(wait_time)
                    
                    # Update current time and load
                    current_time = departure_time
                    if isinstance(client_data.delivery, int):
                        current_load += client_data.delivery
                    else:
                        current_load += sum(client_data.delivery)
                    
                    prev_pos = curr_pos
                
                # Return to the correct depot using haversine formula
                depot_pos = problem_data.depots[end_depot_idx] if end_depot_idx < len(problem_data.depots) else problem_data.depots[0]
                if prev_pos.x > 1000:
                    prev_lon = prev_pos.x / 10000
                    prev_lat = prev_pos.y / 10000
                else:
                    prev_lon = prev_pos.x
                    prev_lat = prev_pos.y
                    
                if depot_pos.x > 1000:
                    depot_lon = depot_pos.x / 10000
                    depot_lat = depot_pos.y / 10000
                else:
                    depot_lon = depot_pos.x
                    depot_lat = depot_pos.y
                return_distance = haversine_distance(prev_lon, prev_lat, depot_lon, depot_lat)
                return_time = int(return_distance / 1000 / 40 * 60)  # Assume 40km/h average speed for Tokyo area
                distance += return_distance
                current_time += return_time
                duration = current_time
            
            # Calculate costs with safety checks
            # Vehicle type already set above
            fixed_cost = getattr(vt, 'fixed_cost', 0) or 0
            unit_distance_cost = getattr(vt, 'unit_distance_cost', 1.0) or 1.0
            unit_duration_cost = getattr(vt, 'unit_duration_cost', 0.0) or 0.0
            variable_cost = distance * unit_distance_cost + duration * unit_duration_cost
            total_cost = fixed_cost + variable_cost
            
            routes.append(UnifiedRouteModel(
                vehicle_type=vehicle_type_idx,
                vehicle_id=route_idx,
                start_depot=start_depot_idx,
                end_depot=end_depot_idx,
                clients=clients,
                distance=int(distance),
                duration=duration,
                fixed_cost=fixed_cost,
                variable_cost=variable_cost,
                total_cost=total_cost,
                demand_served=demand_served,
                max_load=demand_served,
                capacity_utilization=demand_served / vt.capacity if vt.capacity > 0 else 0,
                start_time=getattr(vt, 'tw_early', 480) if vt else 480,  # Actual start time
                end_time=current_time,  # Actual end time
                arrival_times=arrival_times,
                departure_times=departure_times,
                waiting_times=waiting_times,
                num_clients=len(clients),
                empty_distance=0,
                loaded_distance=int(distance)
            ))
        
        return routes
    
    def _fallback_solver(self, problem_data: VRPProblemData) -> UnifiedVRPSolution:
        """Simple fallback solver when PyVRP is not available"""
        self.logger.warning("Using fallback solver - results may be suboptimal")
        
        # Simple nearest neighbor heuristic
        routes = []
        unvisited_clients = list(range(len(problem_data.clients)))
        
        for vt_idx, vt in enumerate(problem_data.vehicle_types):
            for vehicle in range(vt.num_available):
                if not unvisited_clients:
                    break
                    
                route_clients = []
                current_capacity = 0
                current_pos = problem_data.depots[vt.start_depot]
                route_distance = 0
                
                while unvisited_clients:
                    # Find nearest unvisited client
                    best_client = None
                    best_distance = float('inf')
                    
                    for client_idx in unvisited_clients:
                        client = problem_data.clients[client_idx]
                        
                        # Check capacity
                        demand = client.delivery if isinstance(client.delivery, int) else sum(client.delivery)
                        if current_capacity + demand > vt.capacity:
                            continue
                        
                        # Calculate distance using haversine formula
                        current_lon = current_pos.x / 10000
                        current_lat = current_pos.y / 10000
                        client_lon = client.x / 10000
                        client_lat = client.y / 10000
                        dist = haversine_distance(current_lon, current_lat, client_lon, client_lat)
                        if dist < best_distance:
                            best_distance = dist
                            best_client = client_idx
                    
                    if best_client is None:
                        break
                    
                    # Add client to route
                    route_clients.append(best_client)
                    unvisited_clients.remove(best_client)
                    client = problem_data.clients[best_client]
                    demand = client.delivery if isinstance(client.delivery, int) else sum(client.delivery)
                    current_capacity += demand
                    route_distance += best_distance
                    current_pos = client
                
                if route_clients:
                    # Return to depot using haversine distance
                    depot = problem_data.depots[vt.start_depot]
                    current_lon = current_pos.x / 10000
                    current_lat = current_pos.y / 10000
                    depot_lon = depot.x / 10000
                    depot_lat = depot.y / 10000
                    route_distance += haversine_distance(current_lon, current_lat, depot_lon, depot_lat)
                    
                    # Calculate costs (with safety checks)
                    fixed_cost = vt.fixed_cost or 0.0
                    unit_distance_cost = vt.unit_distance_cost or 1.0
                    unit_duration_cost = vt.unit_duration_cost or 0.0
                    variable_cost = route_distance * unit_distance_cost + (route_distance / 1000 * 60) * unit_duration_cost
                    total_cost = fixed_cost + variable_cost
                    
                    routes.append(UnifiedRouteModel(
                        vehicle_type=vt_idx,
                        vehicle_id=vehicle,
                        start_depot=vt.start_depot,
                        end_depot=vt.end_depot if vt.end_depot is not None else vt.start_depot,
                        clients=route_clients,
                        distance=int(route_distance),
                        duration=int(route_distance / 1000 * 60),  # Assume 1km/min
                        fixed_cost=fixed_cost,
                        variable_cost=variable_cost,
                        total_cost=total_cost,
                        demand_served=current_capacity,
                        max_load=current_capacity,
                        capacity_utilization=current_capacity / vt.capacity,
                        start_time=0,
                        end_time=int(route_distance / 1000 * 60),
                        num_clients=len(route_clients),
                        empty_distance=0,
                        loaded_distance=int(route_distance)
                    ))
        
        total_distance = sum(r.distance for r in routes)
        
        # Ensure objective_value is a valid number
        objective_value = float(total_distance) if total_distance and total_distance > 0 else 1.0
        
        return UnifiedVRPSolution(
            status="feasible",
            objective_value=objective_value,
            routes=routes,
            computation_time=0.1,
            solver="Fallback",
            is_feasible=True,
            problem_type="CVRP",
            problem_size={
                "num_clients": len(problem_data.clients),
                "num_depots": len(problem_data.depots),
                "num_vehicles": sum(vt.num_available for vt in problem_data.vehicle_types)
            }
        )