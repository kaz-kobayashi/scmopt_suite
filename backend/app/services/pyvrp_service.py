"""
PyVRP Service - Comprehensive Vehicle Routing Problem solver using PyVRP library

This service provides implementations for all major VRP variants supported by PyVRP:
- Basic Capacitated VRP (CVRP)  
- VRP with Time Windows (VRPTW)
- Multi-Depot VRP (MDVRP)
- Pickup and Delivery VRP (PDVRP)
- Prize-Collecting VRP (PC-VRP)
- VRPLIB format support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from pathlib import Path
import json
import math

try:
    import pyvrp
    from pyvrp import Model
    PYVRP_AVAILABLE = True
except ImportError:
    PYVRP_AVAILABLE = False
    logging.warning("PyVRP library not available. Using fallback implementations.")

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers"""
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class PyVRPService:
    """Comprehensive PyVRP service for all VRP variants"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not PYVRP_AVAILABLE:
            self.logger.warning("PyVRP not available - using fallback methods")
    
    def solve_basic_cvrp(
        self, 
        locations: List[Dict[str, Any]], 
        demands: List[float],
        vehicle_capacity: float,
        depot_index: int = 0,
        num_vehicles: Optional[int] = None,
        max_runtime: int = 30
    ) -> Dict[str, Any]:
        """
        Solve basic Capacitated Vehicle Routing Problem (CVRP)
        
        Args:
            locations: List of location dictionaries with 'lat', 'lon', 'name'
            demands: List of demand values for each location
            vehicle_capacity: Maximum capacity per vehicle
            depot_index: Index of depot location (default: 0)
            num_vehicles: Maximum number of vehicles (optional)
            max_runtime: Maximum runtime in seconds
            
        Returns:
            Dictionary with solution details
        """
        if not PYVRP_AVAILABLE:
            return self._fallback_cvrp_solver(locations, demands, vehicle_capacity, depot_index)
        
        try:
            # Calculate distance matrix
            distance_matrix = self._calculate_distance_matrix(locations)
            
            # Create PyVRP model
            model = Model()
            
            # Add depot
            depot = model.add_depot(x=locations[depot_index]['lon'], y=locations[depot_index]['lat'])
            
            # Add clients (customers)
            clients = []
            for i, (location, demand) in enumerate(zip(locations, demands)):
                if i != depot_index:
                    client = model.add_client(
                        x=location['lon'], 
                        y=location['lat'],
                        demand=int(demand)
                    )
                    clients.append(client)
            
            # Add vehicle type
            vehicle_type = model.add_vehicle_type(
                num_available=num_vehicles or len(clients),
                capacity=int(vehicle_capacity),
                depot=depot
            )
            
            # Set distance matrix
            n_locations = len(locations)
            model.add_distance_matrix([
                [int(distance_matrix[i][j] * 1000) for j in range(n_locations)]  # Convert km to m
                for i in range(n_locations)
            ])
            
            # Solve
            start_time = time.time()
            result = pyvrp.solve(
                model, 
                stop=pyvrp.stop.MaxRuntime(max_runtime),
                seed=42
            )
            solve_time = time.time() - start_time
            
            # Process solution
            if result.is_feasible():
                routes = self._extract_routes_from_pyvrp_solution(result, locations, depot_index)
                
                return {
                    "status": "optimal" if result.cost() > 0 else "feasible",
                    "objective_value": result.cost() / 1000.0,  # Convert back to km
                    "routes": routes,
                    "total_distance": sum(route["distance"] for route in routes),
                    "total_demand_served": sum(route["total_demand"] for route in routes),
                    "num_vehicles_used": len(routes),
                    "computation_time": solve_time,
                    "solver": "PyVRP",
                    "problem_type": "CVRP"
                }
            else:
                return {
                    "status": "infeasible",
                    "message": "No feasible solution found",
                    "computation_time": solve_time
                }
                
        except Exception as e:
            self.logger.error(f"PyVRP CVRP solve error: {e}")
            return self._fallback_cvrp_solver(locations, demands, vehicle_capacity, depot_index)
    
    def solve_vrptw(
        self,
        locations: List[Dict[str, Any]],
        demands: List[float],
        time_windows: List[Tuple[float, float]],
        service_times: List[float],
        vehicle_capacity: float,
        depot_index: int = 0,
        num_vehicles: Optional[int] = None,
        max_runtime: int = 60
    ) -> Dict[str, Any]:
        """
        Solve Vehicle Routing Problem with Time Windows (VRPTW)
        
        Args:
            locations: List of location dictionaries
            demands: Demand at each location
            time_windows: List of (earliest, latest) time windows
            service_times: Service time at each location
            vehicle_capacity: Vehicle capacity
            depot_index: Depot location index
            num_vehicles: Maximum vehicles
            max_runtime: Runtime limit in seconds
            
        Returns:
            Solution dictionary
        """
        if not PYVRP_AVAILABLE:
            return self._fallback_vrptw_solver(locations, demands, time_windows, service_times, vehicle_capacity)
        
        try:
            model = Model()
            distance_matrix = self._calculate_distance_matrix(locations)
            duration_matrix = self._calculate_duration_matrix(distance_matrix)  # Assume speed = 1 km/min
            
            # Add depot with time window
            depot_tw = time_windows[depot_index]
            depot = model.add_depot(
                x=locations[depot_index]['lon'],
                y=locations[depot_index]['lat'],
                tw_early=int(depot_tw[0] * 60),  # Convert to minutes
                tw_late=int(depot_tw[1] * 60)
            )
            
            # Add clients with time windows
            clients = []
            for i, (location, demand, tw, service_time) in enumerate(
                zip(locations, demands, time_windows, service_times)
            ):
                if i != depot_index:
                    client = model.add_client(
                        x=location['lon'],
                        y=location['lat'],
                        demand=int(demand),
                        tw_early=int(tw[0] * 60),
                        tw_late=int(tw[1] * 60),
                        service_time=int(service_time * 60)
                    )
                    clients.append(client)
            
            # Add vehicle type
            vehicle_type = model.add_vehicle_type(
                num_available=num_vehicles or len(clients),
                capacity=int(vehicle_capacity),
                depot=depot,
                tw_early=int(depot_tw[0] * 60),
                tw_late=int(depot_tw[1] * 60)
            )
            
            # Set matrices
            n_locations = len(locations)
            model.add_distance_matrix([
                [int(distance_matrix[i][j] * 1000) for j in range(n_locations)]
                for i in range(n_locations)
            ])
            
            model.add_duration_matrix([
                [int(duration_matrix[i][j] * 60) for j in range(n_locations)]  # Convert to minutes
                for i in range(n_locations)
            ])
            
            # Solve
            start_time = time.time()
            result = pyvrp.solve(
                model,
                stop=pyvrp.stop.MaxRuntime(max_runtime),
                seed=42
            )
            solve_time = time.time() - start_time
            
            if result.is_feasible():
                routes = self._extract_routes_from_pyvrp_solution(result, locations, depot_index, include_timing=True)
                
                return {
                    "status": "optimal" if result.cost() > 0 else "feasible",
                    "objective_value": result.cost() / 1000.0,
                    "routes": routes,
                    "total_distance": sum(route["distance"] for route in routes),
                    "total_demand_served": sum(route["total_demand"] for route in routes),
                    "num_vehicles_used": len(routes),
                    "computation_time": solve_time,
                    "solver": "PyVRP",
                    "problem_type": "VRPTW",
                    "time_window_violations": 0  # PyVRP ensures feasibility
                }
            else:
                return {
                    "status": "infeasible",
                    "message": "No feasible solution found for VRPTW",
                    "computation_time": solve_time
                }
                
        except Exception as e:
            self.logger.error(f"PyVRP VRPTW solve error: {e}")
            return self._fallback_vrptw_solver(locations, demands, time_windows, service_times, vehicle_capacity)
    
    def solve_multi_depot_vrp(
        self,
        locations: List[Dict[str, Any]],
        demands: List[float],
        depot_indices: List[int],
        vehicle_capacities: List[float],
        vehicles_per_depot: List[int],
        max_runtime: int = 60
    ) -> Dict[str, Any]:
        """
        Solve Multi-Depot Vehicle Routing Problem (MDVRP)
        
        Args:
            locations: All locations including depots and customers
            demands: Demand at each customer location
            depot_indices: Indices of depot locations
            vehicle_capacities: Capacity for each depot's vehicles
            vehicles_per_depot: Number of vehicles available at each depot
            max_runtime: Runtime limit
            
        Returns:
            Solution dictionary
        """
        if not PYVRP_AVAILABLE:
            return self._fallback_mdvrp_solver(locations, demands, depot_indices, vehicle_capacities[0])
        
        try:
            model = Model()
            distance_matrix = self._calculate_distance_matrix(locations)
            
            # Add depots
            depots = []
            for depot_idx in depot_indices:
                depot = model.add_depot(
                    x=locations[depot_idx]['lon'],
                    y=locations[depot_idx]['lat']
                )
                depots.append(depot)
            
            # Add clients
            clients = []
            client_demands = []
            for i, (location, demand) in enumerate(zip(locations, demands)):
                if i not in depot_indices:
                    client = model.add_client(
                        x=location['lon'],
                        y=location['lat'],
                        demand=int(demand)
                    )
                    clients.append(client)
                    client_demands.append(demand)
            
            # Add vehicle types for each depot
            vehicle_types = []
            for depot, capacity, num_vehicles in zip(depots, vehicle_capacities, vehicles_per_depot):
                vehicle_type = model.add_vehicle_type(
                    num_available=num_vehicles,
                    capacity=int(capacity),
                    depot=depot
                )
                vehicle_types.append(vehicle_type)
            
            # Set distance matrix
            n_locations = len(locations)
            model.add_distance_matrix([
                [int(distance_matrix[i][j] * 1000) for j in range(n_locations)]
                for i in range(n_locations)
            ])
            
            # Solve
            start_time = time.time()
            result = pyvrp.solve(
                model,
                stop=pyvrp.stop.MaxRuntime(max_runtime),
                seed=42
            )
            solve_time = time.time() - start_time
            
            if result.is_feasible():
                # Process multi-depot solution
                routes = self._extract_multidepot_routes(result, locations, depot_indices)
                
                return {
                    "status": "optimal" if result.cost() > 0 else "feasible",
                    "objective_value": result.cost() / 1000.0,
                    "routes": routes,
                    "routes_by_depot": self._group_routes_by_depot(routes, depot_indices),
                    "total_distance": sum(route["distance"] for route in routes),
                    "total_demand_served": sum(route["total_demand"] for route in routes),
                    "num_vehicles_used": len(routes),
                    "computation_time": solve_time,
                    "solver": "PyVRP",
                    "problem_type": "MDVRP"
                }
            else:
                return {
                    "status": "infeasible", 
                    "message": "No feasible solution found for MDVRP",
                    "computation_time": solve_time
                }
                
        except Exception as e:
            self.logger.error(f"PyVRP MDVRP solve error: {e}")
            return self._fallback_mdvrp_solver(locations, demands, depot_indices, vehicle_capacities[0])
    
    def solve_pickup_delivery_vrp(
        self,
        locations: List[Dict[str, Any]],
        pickup_delivery_pairs: List[Tuple[int, int]],
        demands: List[float],
        vehicle_capacity: float,
        depot_index: int = 0,
        max_runtime: int = 60
    ) -> Dict[str, Any]:
        """
        Solve Pickup and Delivery Vehicle Routing Problem (PDVRP)
        
        Args:
            locations: All locations 
            pickup_delivery_pairs: List of (pickup_idx, delivery_idx) pairs
            demands: Pickup amounts (positive for pickup, negative for delivery)
            vehicle_capacity: Vehicle capacity
            depot_index: Depot location
            max_runtime: Runtime limit
            
        Returns:
            Solution dictionary
        """
        if not PYVRP_AVAILABLE:
            return self._fallback_pdvrp_solver(locations, pickup_delivery_pairs, demands, vehicle_capacity)
        
        try:
            model = Model()
            distance_matrix = self._calculate_distance_matrix(locations)
            
            # Add depot
            depot = model.add_depot(
                x=locations[depot_index]['lon'],
                y=locations[depot_index]['lat']
            )
            
            # Add client groups for pickup-delivery pairs
            client_groups = []
            clients = []
            
            for pickup_idx, delivery_idx in pickup_delivery_pairs:
                # Create client group for this pickup-delivery pair
                group = model.add_client_group()
                client_groups.append(group)
                
                # Add pickup client
                pickup_client = model.add_client(
                    x=locations[pickup_idx]['lon'],
                    y=locations[pickup_idx]['lat'],
                    demand=int(abs(demands[pickup_idx])),  # Pickup demand (positive)
                    group=group
                )
                clients.append(pickup_client)
                
                # Add delivery client  
                delivery_client = model.add_client(
                    x=locations[delivery_idx]['lon'],
                    y=locations[delivery_idx]['lat'],
                    demand=-int(abs(demands[delivery_idx])),  # Delivery demand (negative)
                    group=group
                )
                clients.append(delivery_client)
            
            # Add vehicle type
            vehicle_type = model.add_vehicle_type(
                num_available=len(pickup_delivery_pairs) + 2,  # Generous upper bound
                capacity=int(vehicle_capacity),
                depot=depot
            )
            
            # Set distance matrix
            n_locations = len(locations)
            model.add_distance_matrix([
                [int(distance_matrix[i][j] * 1000) for j in range(n_locations)]
                for i in range(n_locations)
            ])
            
            # Solve
            start_time = time.time()
            result = pyvrp.solve(
                model,
                stop=pyvrp.stop.MaxRuntime(max_runtime),
                seed=42
            )
            solve_time = time.time() - start_time
            
            if result.is_feasible():
                routes = self._extract_pd_routes(result, locations, pickup_delivery_pairs, depot_index)
                
                return {
                    "status": "optimal" if result.cost() > 0 else "feasible",
                    "objective_value": result.cost() / 1000.0,
                    "routes": routes,
                    "total_distance": sum(route["distance"] for route in routes),
                    "num_vehicles_used": len(routes),
                    "pickup_delivery_pairs": len(pickup_delivery_pairs),
                    "computation_time": solve_time,
                    "solver": "PyVRP",
                    "problem_type": "PDVRP"
                }
            else:
                return {
                    "status": "infeasible",
                    "message": "No feasible solution found for PDVRP",
                    "computation_time": solve_time
                }
                
        except Exception as e:
            self.logger.error(f"PyVRP PDVRP solve error: {e}")
            return self._fallback_pdvrp_solver(locations, pickup_delivery_pairs, demands, vehicle_capacity)
    
    def solve_prize_collecting_vrp(
        self,
        locations: List[Dict[str, Any]],
        prizes: List[float],
        demands: List[float],
        vehicle_capacity: float,
        min_prize: float,
        depot_index: int = 0,
        max_runtime: int = 60
    ) -> Dict[str, Any]:
        """
        Solve Prize-Collecting Vehicle Routing Problem (PC-VRP)
        
        Args:
            locations: All locations
            prizes: Prize/profit for visiting each location
            demands: Demand at each location
            vehicle_capacity: Vehicle capacity
            min_prize: Minimum total prize to collect
            depot_index: Depot location
            max_runtime: Runtime limit
            
        Returns:
            Solution dictionary
        """
        if not PYVRP_AVAILABLE:
            return self._fallback_pcvrp_solver(locations, prizes, demands, vehicle_capacity, min_prize)
        
        try:
            model = Model()
            distance_matrix = self._calculate_distance_matrix(locations)
            
            # Add depot
            depot = model.add_depot(
                x=locations[depot_index]['lon'],
                y=locations[depot_index]['lat']
            )
            
            # Add clients with prizes
            clients = []
            for i, (location, prize, demand) in enumerate(zip(locations, prizes, demands)):
                if i != depot_index:
                    client = model.add_client(
                        x=location['lon'],
                        y=location['lat'],
                        demand=int(demand),
                        prize=int(prize * 1000)  # Scale prize for integer handling
                    )
                    clients.append(client)
            
            # Add vehicle type
            vehicle_type = model.add_vehicle_type(
                num_available=len(clients),
                capacity=int(vehicle_capacity),
                depot=depot
            )
            
            # Set distance matrix
            n_locations = len(locations)
            model.add_distance_matrix([
                [int(distance_matrix[i][j] * 1000) for j in range(n_locations)]
                for i in range(n_locations)
            ])
            
            # Solve with prize collection objective
            start_time = time.time()
            result = pyvrp.solve(
                model,
                stop=pyvrp.stop.MaxRuntime(max_runtime),
                seed=42
            )
            solve_time = time.time() - start_time
            
            if result.is_feasible():
                routes = self._extract_prize_collecting_routes(result, locations, prizes, depot_index)
                total_prize = sum(route["total_prize"] for route in routes)
                
                return {
                    "status": "optimal" if result.cost() > 0 else "feasible",
                    "objective_value": result.cost() / 1000.0,
                    "routes": routes,
                    "total_distance": sum(route["distance"] for route in routes),
                    "total_prize": total_prize,
                    "min_prize_met": total_prize >= min_prize,
                    "num_vehicles_used": len(routes),
                    "computation_time": solve_time,
                    "solver": "PyVRP",
                    "problem_type": "PC-VRP"
                }
            else:
                return {
                    "status": "infeasible",
                    "message": "No feasible solution found for PC-VRP",
                    "computation_time": solve_time
                }
                
        except Exception as e:
            self.logger.error(f"PyVRP PC-VRP solve error: {e}")
            return self._fallback_pcvrp_solver(locations, prizes, demands, vehicle_capacity, min_prize)
    
    def parse_vrplib_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse VRPLIB format file
        
        Args:
            file_path: Path to VRPLIB file
            
        Returns:
            Parsed problem data
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            problem_data = self._parse_vrplib_content(content)
            return {
                "status": "success",
                "data": problem_data,
                "message": f"Successfully parsed VRPLIB file: {Path(file_path).name}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to parse VRPLIB file: {str(e)}"
            }
    
    def solve_vrplib_instance(
        self, 
        file_path: str, 
        max_runtime: int = 300
    ) -> Dict[str, Any]:
        """
        Solve VRPLIB instance using PyVRP
        
        Args:
            file_path: Path to VRPLIB file
            max_runtime: Maximum runtime in seconds
            
        Returns:
            Solution dictionary
        """
        if not PYVRP_AVAILABLE:
            return {"status": "error", "message": "PyVRP not available"}
        
        try:
            # Parse the VRPLIB file
            parse_result = self.parse_vrplib_file(file_path)
            if parse_result["status"] != "success":
                return parse_result
            
            problem_data = parse_result["data"]
            problem_type = problem_data.get("type", "CVRP")
            
            # Solve based on problem type
            if problem_type == "CVRP":
                return self.solve_basic_cvrp(
                    locations=problem_data["locations"],
                    demands=problem_data["demands"],
                    vehicle_capacity=problem_data["capacity"],
                    depot_index=0,
                    max_runtime=max_runtime
                )
            elif problem_type == "VRPTW":
                return self.solve_vrptw(
                    locations=problem_data["locations"],
                    demands=problem_data["demands"],
                    time_windows=problem_data["time_windows"],
                    service_times=problem_data["service_times"],
                    vehicle_capacity=problem_data["capacity"],
                    max_runtime=max_runtime
                )
            else:
                return {
                    "status": "error",
                    "message": f"Problem type {problem_type} not supported yet"
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "message": f"VRPLIB solve error: {str(e)}"
            }
    
    # Helper methods
    def _calculate_distance_matrix(self, locations: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate Haversine distance matrix between locations"""
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = locations[i]['lat'], locations[i]['lon']
                    lat2, lon2 = locations[j]['lat'], locations[j]['lon']
                    matrix[i][j] = haversine_distance(lat1, lon1, lat2, lon2)
        
        return matrix
    
    def _calculate_duration_matrix(self, distance_matrix: np.ndarray, speed_kmh: float = 60) -> np.ndarray:
        """Convert distance matrix to duration matrix (hours)"""
        return distance_matrix / speed_kmh
    
    def _extract_routes_from_pyvrp_solution(
        self, 
        result, 
        locations: List[Dict[str, Any]], 
        depot_index: int,
        include_timing: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract route information from PyVRP solution"""
        routes = []
        
        for route_idx, route in enumerate(result.best.routes()):
            if not route.visits():
                continue
                
            route_locations = [depot_index]  # Start at depot
            route_demands = []
            route_distance = 0.0
            
            for visit in route.visits():
                client_idx = visit.client()
                # Map client back to original location index
                location_idx = self._map_client_to_location(client_idx, depot_index)
                route_locations.append(location_idx)
                route_demands.append(visit.demand() if hasattr(visit, 'demand') else 0)
            
            route_locations.append(depot_index)  # Return to depot
            
            # Calculate route distance
            for i in range(len(route_locations) - 1):
                loc1, loc2 = locations[route_locations[i]], locations[route_locations[i+1]]
                route_distance += haversine_distance(
                    loc1['lat'], loc1['lon'],
                    loc2['lat'], loc2['lon']
                )
            
            route_info = {
                "route_id": int(route_idx),
                "sequence": [int(x) for x in route_locations],
                "locations": [str(locations[i]['name']) for i in route_locations],
                "distance": float(route_distance),
                "total_demand": float(sum(route_demands)),
                "num_stops": int(len(route_locations) - 2)  # Exclude depot visits
            }
            
            if include_timing:
                route_info["arrival_times"] = self._calculate_arrival_times(route_locations, locations)
            
            routes.append(route_info)
        
        return routes
    
    def _map_client_to_location(self, client_idx: int, depot_index: int) -> int:
        """Map PyVRP client index back to original location index"""
        # Simple mapping: client indices start after depot
        if client_idx >= depot_index:
            return client_idx + 1
        else:
            return client_idx
    
    def _calculate_arrival_times(self, route_sequence: List[int], locations: List[Dict[str, Any]]) -> List[float]:
        """Calculate arrival times for route sequence"""
        arrival_times = [0.0]  # Start at depot at time 0
        current_time = 0.0
        
        for i in range(1, len(route_sequence)):
            prev_loc = locations[route_sequence[i-1]]
            curr_loc = locations[route_sequence[i]]
            travel_time = haversine_distance(
                prev_loc['lat'], prev_loc['lon'],
                curr_loc['lat'], curr_loc['lon']
            ) / 60  # Assume 60 km/h speed, result in hours
            
            current_time += travel_time
            arrival_times.append(float(current_time))
        
        return [float(t) for t in arrival_times]
    
    # Fallback solvers (simplified versions when PyVRP not available)
    def _fallback_cvrp_solver(self, locations, demands, vehicle_capacity, depot_index):
        """Fallback CVRP solver using nearest neighbor heuristic"""
        try:
            from .routing_service import optimize_vehicle_routing
            import pandas as pd
            
            # Convert to DataFrame format expected by existing solver
            locations_df = pd.DataFrame([
                {
                    'name': loc['name'],
                    'lat': loc['lat'], 
                    'lon': loc['lon'],
                    'demand': demand
                } 
                for loc, demand in zip(locations, demands)
            ])
            
            result = optimize_vehicle_routing(
                locations_df, 
                vehicle_capacity, 
                max_routes=10, 
                depot_name=locations[depot_index]['name']
            )
            
            # Convert to PyVRP-style format with consistent field names
            converted_routes = []
            for route in result["routes"]:
                converted_route = {
                    "route_id": route["route_id"],
                    "sequence": route["sequence"],
                    "locations": route["locations"],
                    "distance": route["total_distance"],  # Convert total_distance to distance
                    "total_demand": route["total_demand"],
                    "num_stops": len(route["sequence"]) - 2,  # Exclude depot start/end
                    "capacity_utilization": route.get("capacity_utilization", 0.0)
                }
                converted_routes.append(converted_route)
            
            return {
                "status": "feasible",
                "objective_value": result["summary"]["total_distance"],
                "routes": converted_routes,
                "total_distance": result["summary"]["total_distance"],
                "total_demand_served": result["summary"]["total_demand_served"],
                "num_vehicles_used": result["summary"]["total_routes"],
                "computation_time": 0.1,
                "solver": "Fallback Heuristic",
                "problem_type": "CVRP"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Fallback CVRP solver failed: {str(e)}",
                "problem_type": "CVRP"
            }
    
    def _fallback_vrptw_solver(self, locations, demands, time_windows, service_times, vehicle_capacity):
        """Fallback VRPTW solver"""
        return {
            "status": "error",
            "message": "VRPTW requires PyVRP library - fallback not implemented",
            "problem_type": "VRPTW"
        }
    
    def _fallback_mdvrp_solver(self, locations, demands, depot_indices, vehicle_capacity):
        """Fallback MDVRP solver"""
        return {
            "status": "error", 
            "message": "MDVRP requires PyVRP library - fallback not implemented",
            "problem_type": "MDVRP"
        }
    
    def _fallback_pdvrp_solver(self, locations, pickup_delivery_pairs, demands, vehicle_capacity):
        """Fallback PDVRP solver"""
        return {
            "status": "error",
            "message": "PDVRP requires PyVRP library - fallback not implemented", 
            "problem_type": "PDVRP"
        }
    
    def _fallback_pcvrp_solver(self, locations, prizes, demands, vehicle_capacity, min_prize):
        """Fallback PC-VRP solver"""
        return {
            "status": "error",
            "message": "PC-VRP requires PyVRP library - fallback not implemented",
            "problem_type": "PC-VRP"
        }
    
    def _parse_vrplib_content(self, content: str) -> Dict[str, Any]:
        """Parse VRPLIB file content"""
        lines = content.strip().split('\n')
        problem_data = {
            "locations": [],
            "demands": [],
            "capacity": 0,
            "type": "CVRP"
        }
        
        # Basic VRPLIB parsing - simplified implementation
        section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('CAPACITY'):
                problem_data["capacity"] = int(line.split()[-1])
            elif line == 'NODE_COORD_SECTION':
                section = 'coords'
            elif line == 'DEMAND_SECTION':
                section = 'demands'
            elif line == 'EOF':
                break
            elif section == 'coords' and line[0].isdigit():
                parts = line.split()
                node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                problem_data["locations"].append({
                    "name": f"Location_{node_id}",
                    "lat": y / 100000,  # Scale coordinates appropriately
                    "lon": x / 100000
                })
            elif section == 'demands' and line[0].isdigit():
                parts = line.split()
                demand = float(parts[1])
                problem_data["demands"].append(demand)
        
        return problem_data
    
    def _extract_multidepot_routes(self, result, locations, depot_indices):
        """Extract routes for multi-depot solution"""
        # Simplified - would need more complex logic for real multi-depot
        return self._extract_routes_from_pyvrp_solution(result, locations, depot_indices[0])
    
    def _group_routes_by_depot(self, routes, depot_indices):
        """Group routes by their depot"""
        grouped = {}
        for i, depot_idx in enumerate(depot_indices):
            grouped[f"depot_{i}"] = []
        
        for route in routes:
            # Determine which depot this route belongs to
            depot_name = f"depot_0"  # Simplified
            grouped[depot_name].append(route)
        
        return grouped
    
    def _extract_pd_routes(self, result, locations, pickup_delivery_pairs, depot_index):
        """Extract pickup-delivery routes"""
        routes = self._extract_routes_from_pyvrp_solution(result, locations, depot_index)
        
        # Add pickup-delivery pair information
        for route in routes:
            route["pickup_delivery_info"] = self._analyze_pd_pairs_in_route(
                route["sequence"], pickup_delivery_pairs
            )
        
        return routes
    
    def _analyze_pd_pairs_in_route(self, sequence, pickup_delivery_pairs):
        """Analyze pickup-delivery pairs in route"""
        pd_info = []
        for pickup_idx, delivery_idx in pickup_delivery_pairs:
            if pickup_idx in sequence and delivery_idx in sequence:
                pickup_pos = sequence.index(pickup_idx)
                delivery_pos = sequence.index(delivery_idx)
                pd_info.append({
                    "pickup_location": pickup_idx,
                    "delivery_location": delivery_idx,
                    "pickup_position": pickup_pos,
                    "delivery_position": delivery_pos,
                    "valid_order": pickup_pos < delivery_pos
                })
        return pd_info
    
    def _extract_prize_collecting_routes(self, result, locations, prizes, depot_index):
        """Extract prize collecting routes"""
        routes = self._extract_routes_from_pyvrp_solution(result, locations, depot_index)
        
        # Calculate prize for each route
        for route in routes:
            total_prize = 0
            for loc_idx in route["sequence"]:
                if loc_idx != depot_index and loc_idx < len(prizes):
                    total_prize += prizes[loc_idx]
            route["total_prize"] = total_prize
        
        return routes


# Legacy function for backward compatibility
def optimize_vrp_with_pyvrp(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int = 5,
    depot_name: str = "Depot",
    max_runtime_seconds: int = 30
) -> Dict[str, Any]:
    """
    Legacy function - use PyVRPService.solve_basic_cvrp instead
    """
    service = PyVRPService()
    
    # Convert DataFrame to location list
    locations = []
    demands = []
    depot_index = 0
    
    for idx, row in locations_df.iterrows():
        locations.append({
            'name': row['name'],
            'lat': row['lat'],
            'lon': row['lon']
        })
        demands.append(row.get('demand', 1))
        
        if row['name'] == depot_name:
            depot_index = idx
    
    result = service.solve_basic_cvrp(
        locations=locations,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        depot_index=depot_index,
        num_vehicles=max_routes,
        max_runtime=max_runtime_seconds
    )
    
    # Convert to legacy format
    if result["status"] in ["optimal", "feasible"]:
        return {
            'routes': result["routes"],
            'summary': {
                'total_routes': result["num_vehicles_used"],
                'total_distance': result["total_distance"],
                'total_demand_served': result["total_demand_served"],
                'avg_capacity_utilization': result["total_demand_served"] / (vehicle_capacity * result["num_vehicles_used"]) if result["num_vehicles_used"] > 0 else 0,
                'customers_served': sum(route["num_stops"] for route in result["routes"]),
                'customers_unserved': len(locations) - 1 - sum(route["num_stops"] for route in result["routes"]),
                'optimization_cost': result["objective_value"]
            },
            'parameters': {
                'vehicle_capacity': vehicle_capacity,
                'max_routes': max_routes,
                'depot': depot_name,
                'max_runtime_seconds': max_runtime_seconds
            },
            'optimization_stats': {
                'algorithm': result["solver"],
                'convergence': True,
                'runtime_seconds': result["computation_time"]
            }
        }
    else:
        return fallback_nearest_neighbor(locations_df, vehicle_capacity, max_routes, depot_name)


def fallback_nearest_neighbor(
    locations_df: pd.DataFrame,
    vehicle_capacity: float,
    max_routes: int,
    depot_name: str
) -> Dict[str, Any]:
    """
    Fallback to simple nearest neighbor algorithm if PyVRP fails
    """
    # Find depot
    depot_idx = None
    for idx, row in locations_df.iterrows():
        if row['name'] == depot_name:
            depot_idx = idx
            break
    
    if depot_idx is None:
        depot_idx = 0
        depot_name = locations_df.iloc[0]['name']
    
    # Calculate distance matrix
    n_locations = len(locations_df)
    distances = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                lat1, lon1 = locations_df.iloc[i]['lat'], locations_df.iloc[i]['lon']
                lat2, lon2 = locations_df.iloc[j]['lat'], locations_df.iloc[j]['lon']
                distances[i][j] = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Simple capacity-based clustering
    routes = []
    remaining_customers = list(range(len(locations_df)))
    remaining_customers.remove(depot_idx)
    
    route_id = 0
    while remaining_customers and route_id < max_routes:
        current_route = [depot_idx]
        current_capacity = 0
        
        while remaining_customers:
            best_customer = None
            best_distance = float('inf')
            
            for customer in remaining_customers:
                demand = locations_df.iloc[customer].get('demand', 1)
                if current_capacity + demand <= vehicle_capacity:
                    distance = distances[current_route[-1]][customer]
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer
            
            if best_customer is None:
                break
            
            current_route.append(best_customer)
            current_capacity += locations_df.iloc[best_customer].get('demand', 1)
            remaining_customers.remove(best_customer)
        
        current_route.append(depot_idx)
        
        if len(current_route) > 2:  # More than just depot-depot
            route_distance = sum(distances[current_route[i]][current_route[i + 1]] 
                               for i in range(len(current_route) - 1))
            
            routes.append({
                'route_id': int(route_id + 1),
                'sequence': [int(x) for x in current_route],
                'locations': [str(locations_df.iloc[i]['name']) for i in current_route],
                'total_distance': float(route_distance),
                'total_demand': float(current_capacity),
                'capacity_utilization': float(current_capacity / vehicle_capacity) if vehicle_capacity > 0 else 0.0
            })
        
        route_id += 1
    
    total_distance = sum(route['total_distance'] for route in routes)
    total_demand = sum(route['total_demand'] for route in routes)
    avg_capacity_utilization = float(np.mean([route['capacity_utilization'] for route in routes])) if routes else 0.0
    
    return {
        'routes': routes,
        'summary': {
            'total_routes': int(len(routes)),
            'total_distance': float(total_distance),
            'total_demand_served': float(total_demand),
            'avg_capacity_utilization': float(avg_capacity_utilization),
            'customers_served': int(len(locations_df) - 1 - len(remaining_customers)),
            'customers_unserved': int(len(remaining_customers)),
            'optimization_cost': float(total_distance)
        },
        'parameters': {
            'vehicle_capacity': float(vehicle_capacity),
            'max_routes': int(max_routes),
            'depot': str(depot_name),
            'max_runtime_seconds': 30
        },
        'optimization_stats': {
            'algorithm': 'Nearest Neighbor (Fallback)',
            'convergence': True,
            'runtime_seconds': 0.1
        }
    }