import pandas as pd
import numpy as np
import networkx as nx
import math
import time as time_module
from typing import Dict, List, Tuple, Any, Set, Optional
from collections import defaultdict, OrderedDict
import warnings

warnings.filterwarnings('ignore')

# Import optimization libraries
try:
    from gurobipy import Model, GRB, quicksum
    USE_GUROBI = True
except ImportError:
    USE_GUROBI = False

# Always import PuLP
try:
    from pulp import *
    USE_PULP = True
except ImportError:
    USE_PULP = False

from geopy.distance import great_circle

from app.models.snd import SolverError


class SNDService:
    """
    Service Network Design Optimizer (SENDO)
    Exact implementation from 12snd.ipynb notebook
    
    Implements:
    - k-th shortest path algorithm
    - Path-based service network design model
    - Gradient scaling method with column generation
    - Geographic distance calculations
    - Network visualization data generation
    """
    
    def __init__(self):
        self.use_gurobi = USE_GUROBI
        self.use_pulp = USE_PULP
        if not self.use_gurobi and not self.use_pulp:
            warnings.warn("Neither Gurobi nor PuLP is available. Optimization features will be limited.")
    
    def k_th_sp(self, G: nx.DiGraph, source: int, sink: int, k: int, weight: str = "weight") -> Tuple[List[float], List[Tuple]]:
        """
        Find k-th shortest paths and returns path and its cost
        Exact implementation from notebook
        
        Args:
            G: NetworkX directed graph
            source: Source node
            sink: Sink node
            k: Number of shortest paths to find
            weight: Edge attribute name for weights
            
        Returns:
            Tuple of (cost_list, path_list)
        """
        cost_list, path_list = [], []
        try:
            for i, p in enumerate(nx.shortest_simple_paths(G, source, sink, weight=weight)):
                if i >= k:
                    break
                v = p[0]
                cost = 0.0
                for w in p[1:]:
                    cost += G[v][w][weight]
                    v = w
                cost_list.append(cost)
                path_list.append(tuple(p))
        except nx.NetworkXNoPath:
            # No path exists between source and sink
            pass
        return cost_list, path_list
    
    def add_shortest_paths(self, G: nx.DiGraph, K: List[Tuple], all_paths: Set, paths: Dict, 
                          path_id: Dict, path_od: Dict, paths_through: Dict, k: int = 1):
        """
        Add k-shortest paths between all OD pairs
        Exact implementation from notebook
        """
        def add_path(p, o, d):
            """
            Add path with tree constraint consideration
            """
            if len(p) <= 2 or p in paths[o, d]:
                return
            paths[o, d].add(p)
            path_od[p] = (o, d)
            all_paths.add(p)
            path_id[p] = len(all_paths)
            v = p[0]
            for w in p[1:]:
                paths_through[(v, w)].add(p)
                v = w
            # Recursively add successor paths for tree constraint
            add_path(p[1:], p[1], d)
        
        for (o, d) in K:
            cost, path = self.k_th_sp(G, o, d, k)
            for p in path:
                add_path(p, o, d)
    
    def sndp(self, K: List[Tuple], all_paths: Set, paths: Dict, path_id: Dict, 
             path_od: Dict, paths_through: Dict, Demand: np.ndarray, Distance: np.ndarray,
             transfer_cost: Dict, capacity: float, relax: bool = False, 
             cpu: float = 10.0) -> Any:
        """
        Service Network Design Problem path-based model
        Exact implementation from notebook
        
        Args:
            K: List of (origin, destination) pairs
            all_paths: Set of all paths
            paths: Dictionary mapping (o,d) to set of paths
            path_id: Dictionary mapping path to ID
            path_od: Dictionary mapping path to (origin, destination)
            paths_through: Dictionary mapping edge to set of paths
            Demand: Demand matrix
            Distance: Distance matrix
            transfer_cost: Transfer cost at each node
            capacity: Vehicle capacity
            relax: Whether to relax integer constraints
            cpu: CPU time limit
            
        Returns:
            Optimized model with solution data
        """
        if self.use_pulp:
            # PuLP implementation
            m = LpProblem("SNDP", LpMinimize)
            
            # Variables
            x, y = {}, {}
            for (i, j) in K:
                if relax:
                    y[i, j] = LpVariable(f"y[{i},{j}]", lowBound=0, cat='Continuous')
                else:
                    y[i, j] = LpVariable(f"y[{i},{j}]", lowBound=0, cat='Integer')
            
            for p, i in path_id.items():
                x[i] = LpVariable(f"x[{i}]", cat='Binary')
            
            total_transfer_cost = LpVariable("total_transfer_cost", cat='Continuous')
            total_vehicle_cost = LpVariable("total_vehicle_cost", cat='Continuous')
            
            # Path transfer costs
            trans_cost = {}
            for (o, d) in K:
                for p in paths[o, d]:
                    sum_tc = 0.0
                    for node in p:  # Include all nodes in path
                        sum_tc += transfer_cost.get(node, 0.0)
                    trans_cost[p] = sum_tc
            
            # Objective function
            m += total_transfer_cost + total_vehicle_cost
            
            # Transfer cost constraint
            m += total_transfer_cost == lpSum(Demand[o, d] * trans_cost[p] * x[path_id[p]] 
                                             for (o, d) in K for p in paths[o, d])
            
            # Vehicle cost constraint
            m += total_vehicle_cost == lpSum(Distance[i, j] * y[i, j] for (i, j) in y)
            
            # Path selection constraints
            for (o, d) in K:
                m += lpSum(x[path_id[p]] for p in paths[o, d]) == 1
            
            # Capacity constraints
            for (i, j) in K:
                m += lpSum(Demand[path_od[p][0], path_od[p][1]] * x[path_id[p]] 
                          for p in paths_through[i, j]) <= capacity * y[i, j]
            
            # Tree constraints (successor path constraints)
            for p in all_paths:
                if len(p) >= 3:
                    successor_path = p[1:]
                    if successor_path in path_id:
                        m += x[path_id[p]] <= x[path_id[successor_path]]
            
            # Solve
            solver = PULP_CBC_CMD(timeLimit=cpu, msg=0)
            m.solve(solver)
            
            # Store solution data
            m.__data = x, y, total_transfer_cost, total_vehicle_cost
            return m
        else:
            raise SolverError("No optimization solver available")
    
    def make_result(self, Demand: np.ndarray, dc_df: pd.DataFrame, K: List[Tuple], 
                   paths: Dict, path_id: Dict, path_od: Dict, paths_through: Dict, 
                   x: Dict, y: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate result DataFrames from optimization solution
        Exact implementation from notebook
        """
        n = len(dc_df)
        dc_names = dc_df['name'].values
        
        # Path results
        origin, destin, od_path = [], [], []
        for o in range(n):
            for d in range(n):
                if d == o:
                    continue
                origin.append(dc_names[o])
                destin.append(dc_names[d])
                
                path_list = []
                for p in paths[o, d]:
                    if hasattr(x[path_id[p]], 'varValue'):
                        x_val = x[path_id[p]].varValue or 0
                    else:
                        x_val = getattr(x[path_id[p]], 'X', 0)
                    
                    if x_val > 0.1:
                        path_list = [dc_names[v] for v in p]
                        break
                od_path.append(path_list)
        
        path_df = pd.DataFrame({
            "origin": origin, 
            "destination": destin, 
            "path": od_path
        })
        
        # Vehicle results
        head, tail, flow, number = [], [], [], []
        head_id, tail_id = [], []
        
        for (v, w) in K:
            if hasattr(y[v, w], 'varValue'):
                y_val = y[v, w].varValue or 0
            else:
                y_val = getattr(y[v, w], 'X', 0)
            
            if y_val > 0.1:
                head_id.append(v)
                tail_id.append(w)
                head.append(dc_names[v])
                tail.append(dc_names[w])
                number.append(y_val)
                
                # Calculate flow
                total_flow = 0
                for p in paths_through[v, w]:
                    if hasattr(x[path_id[p]], 'varValue'):
                        x_val = x[path_id[p]].varValue or 0
                    else:
                        x_val = getattr(x[path_id[p]], 'X', 0)
                    total_flow += Demand[path_od[p][0], path_od[p][1]] * x_val
                flow.append(total_flow)
        
        vehicle_df = pd.DataFrame({
            "from_id": head_id,
            "to_id": tail_id, 
            "from": head,
            "to": tail,
            "flow": flow,
            "number": number
        })
        
        return path_df, vehicle_df
    
    def solve_sndp(self, dc_df: pd.DataFrame, od_df: pd.DataFrame, 
                   cost_per_dis: float, cost_per_time: float, capacity: float,
                   max_cpu: float = 10, scaling: bool = True, k: int = 10, 
                   alpha: float = 0.5, max_iter: int = 100, use_osrm: bool = False) -> Dict[str, Any]:
        """
        Solve Service Network Design Problem
        Exact implementation from notebook
        
        Args:
            dc_df: Distribution center DataFrame
            od_df: Origin-destination demand DataFrame
            cost_per_dis: Cost per kilometer
            cost_per_time: Cost per hour
            capacity: Vehicle capacity
            max_cpu: Maximum CPU time
            scaling: Use gradient scaling method
            k: Number of k-shortest paths
            alpha: Gradient scaling parameter
            max_iter: Maximum iterations for scaling
            use_osrm: Use OSRM for real distances
            
        Returns:
            Dictionary with solution results
        """
        start_time = time_module.time()
        
        n = len(dc_df)
        Demand = od_df.values
        
        # Transfer costs and capacities
        transfer_cost = {}
        for i in range(len(dc_df)):
            transfer_cost[i] = dc_df.iloc[i].get('vc', 0.0)
        
        # Distance and duration calculations
        if use_osrm:
            # TODO: Implement OSRM integration
            raise NotImplementedError("OSRM integration not yet implemented")
        else:
            distances = np.zeros((n, n))
            durations = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    row1 = dc_df.iloc[i]
                    row2 = dc_df.iloc[j]
                    dist = great_circle((row1['lat'], row1['lon']), 
                                      (row2['lat'], row2['lon'])).km
                    distances[i, j] = dist
                    durations[i, j] = dist / 50  # Assume 50 km/h
        
        # Create OD pairs
        K = [(i, j) for i in range(n) for j in range(n) if i != j]
        
        # Calculate costs
        cost = np.zeros((n, n))
        for (i, j) in K:
            cost[i, j] = distances[i, j] * cost_per_dis + durations[i, j] * cost_per_time
        
        # Build graph
        G = nx.DiGraph()
        pos = {}
        for i in range(len(dc_df)):
            row = dc_df.iloc[i]
            G.add_node(i, lat=row['lat'], lon=row['lon'])
            pos[i] = (row['lon'], row['lat'])
        
        for (i, j) in K:
            G.add_edge(i, j, weight=cost[i, j])
        
        # Initialize path data structures
        all_paths = set()
        paths = {}
        path_od = {}
        
        for (o, d) in K:
            cost_list, path_list = self.k_th_sp(G, o, d, k)
            paths[o, d] = set(path_list)
            for p in path_list:
                path_od[p] = (o, d)
                all_paths.add(p)
        
        # Create path mappings
        path_id = {}
        for i, p in enumerate(all_paths):
            path_id[p] = i
        
        # Set of paths through each edge
        paths_through = defaultdict(set)
        for p in path_id:
            v = p[0]
            for w in p[1:]:
                paths_through[(v, w)].add(p)
                v = w
        
        # Gradient scaling + column generation
        cost_prime = cost.copy()
        iterations = 0
        
        if scaling:
            for iter_ in range(max_iter):
                iterations += 1
                model = self.sndp(K, all_paths, paths, path_id, path_od, paths_through,
                                Demand, cost_prime, transfer_cost, capacity, 
                                relax=True, cpu=max_cpu)
                
                if model.status != LpStatusOptimal:
                    raise SolverError(f"Failed to solve relaxed problem! Status: {model.status}")
                
                x, y, _, _ = model.__data
                
                # Update costs using gradient scaling
                for (i, j) in K:
                    if hasattr(y[i, j], 'varValue'):
                        ybar = y[i, j].varValue or 0
                    else:
                        ybar = getattr(y[i, j], 'X', 0)
                    
                    if ybar >= 0.000001:
                        ratio = math.ceil(ybar) / ybar
                        cost_prime[i, j] = alpha * cost[i, j] * ratio + (1 - alpha) * cost_prime[i, j]
                        G[i][j]["weight"] = cost_prime[i, j]
                
                # Add new shortest paths
                before_count = len(all_paths)
                self.add_shortest_paths(G, K, all_paths, paths, path_id, path_od, paths_through, k=k)
                after_count = len(all_paths)
                
                if before_count == after_count:
                    break
        
        # Solve integer problem with original costs
        model = self.sndp(K, all_paths, paths, path_id, path_od, paths_through,
                         Demand, cost, transfer_cost, capacity, 
                         relax=False, cpu=max_cpu)
        
        if model.status != LpStatusOptimal:
            raise SolverError(f"Failed to solve integer problem! Status: {model.status}")
        
        x, y, total_transfer_cost, total_vehicle_cost = model.__data
        
        # Extract solution values
        if hasattr(total_transfer_cost, 'varValue'):
            transfer_val = total_transfer_cost.varValue or 0
            vehicle_val = total_vehicle_cost.varValue or 0
        else:
            transfer_val = getattr(total_transfer_cost, 'X', 0)
            vehicle_val = getattr(total_vehicle_cost, 'X', 0)
        
        # Generate result DataFrames
        path_df, vehicle_df = self.make_result(Demand, dc_df, K, paths, path_id, path_od, paths_through, x, y)
        
        # Cost breakdown
        cost_breakdown = {
            "transfer_cost": transfer_val,
            "vehicle_cost": vehicle_val,
            "total_cost": transfer_val + vehicle_val
        }
        
        computation_time = time_module.time() - start_time
        
        return {
            "status": "optimal",
            "path_df": path_df,
            "vehicle_df": vehicle_df,
            "cost_breakdown": cost_breakdown,
            "computation_time": computation_time,
            "iterations": iterations,
            "paths_generated": len(all_paths),
            "pos": pos
        }
    
    def generate_visualization_data(self, dc_df: pd.DataFrame, path_df: pd.DataFrame, 
                                   vehicle_df: pd.DataFrame, pos: Dict, 
                                   destination_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate data for network visualization
        """
        nodes = []
        edges = []
        
        # Add DC nodes
        for i, row in dc_df.iterrows():
            nodes.append({
                "id": str(i),
                "name": row['name'],
                "lat": row['lat'],
                "lon": row['lon'],
                "node_type": "dc"
            })
        
        # Add vehicle edges (base network)
        for _, row in vehicle_df.iterrows():
            edges.append({
                "from_node": str(row['from_id']),
                "to_node": str(row['to_id']),
                "edge_type": "vehicle",
                "weight": row['number'],
                "color": "yellow",
                "width": max(2, min(10, row['number']))
            })
        
        # Add path edges
        if destination_filter:
            # Filter paths by destination
            filtered_paths = path_df[path_df['destination'] == destination_filter]
        else:
            filtered_paths = path_df
        
        for _, row in filtered_paths.iterrows():
            if row['path']:
                path_nodes = row['path']
                for i in range(len(path_nodes) - 1):
                    # Find node IDs
                    from_id = dc_df[dc_df['name'] == path_nodes[i]].index[0]
                    to_id = dc_df[dc_df['name'] == path_nodes[i + 1]].index[0]
                    
                    edges.append({
                        "from_node": str(from_id),
                        "to_node": str(to_id),
                        "edge_type": "path",
                        "weight": 1.0,
                        "color": "red",
                        "width": 3
                    })
        
        # Calculate center and zoom
        center_lat = dc_df['lat'].mean()
        center_lon = dc_df['lon'].mean()
        
        return {
            "nodes": nodes,
            "edges": edges,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": 6
        }