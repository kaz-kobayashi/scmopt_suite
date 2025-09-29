import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import optimization libraries
USE_GUROBI = False
USE_PULP = False

try:
    from gurobipy import Model, GRB, quicksum
    USE_GUROBI = True
except ImportError:
    pass

try:
    from pulp import *
    USE_PULP = True
except ImportError:
    pass

class TransportationService:
    """
    Advanced transportation problem solver service.
    Implements various transportation problem variants to match notebook computational procedures.
    """
    
    def __init__(self):
        self.solver_backend = "networkx"  # Default to NetworkX for reliability
        
    def transportation(self, C: np.ndarray, capacity: Optional[int] = None) -> Tuple[float, Dict]:
        """
        Solve transportation problem using NetworkX network simplex
        Exact implementation from notebook k-median solver
        
        Args:
            C: Cost matrix (n_customers x n_facilities)
            capacity: Optional capacity constraint per facility
            
        Returns:
            Tuple: (optimal_cost, flow_dict)
        """
        if capacity is None:
            # Standard transportation problem
            return self._solve_standard_transportation(C)
        else:
            # Capacitated transportation problem
            return self._solve_capacitated_transportation(C, capacity)
    
    def _solve_standard_transportation(self, C: np.ndarray) -> Tuple[float, Dict]:
        """
        Solve standard transportation problem (assignment problem)
        """
        n, m = C.shape
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add supply nodes (customers with demand=1)
        for i in range(n):
            G.add_node(f"cust_{i}", demand=1)
            
        # Add demand nodes (facilities with supply=1) 
        for j in range(m):
            G.add_node(f"fac_{j}", demand=-1)
            
        # Add edges with costs
        for i in range(n):
            for j in range(m):
                G.add_edge(f"cust_{i}", f"fac_{j}", weight=C[i, j])
                
        # Solve using network simplex
        cost, flow = nx.network_simplex(G)
        
        return cost, flow
    
    def _solve_capacitated_transportation(self, C: np.ndarray, capacity: int) -> Tuple[float, Dict]:
        """
        Solve capacitated transportation problem
        Matches the exact implementation from lnd_service.py
        """
        M = capacity
        n, m = C.shape
        C_ceil = np.ceil(C)  # For network simplex integer requirements
        
        G = nx.DiGraph()
        sum_demand = 0
        
        # Add facility nodes (supply)
        for j in range(m):
            sum_demand -= M
            G.add_node(f"plant{j}", demand=-M)
            
        # Add customer nodes (demand)
        for i in range(n):
            sum_demand += 1
            G.add_node(i, demand=1)
            
        # Add dummy customer node for excess capacity
        G.add_node("dummy", demand=-sum_demand)
        
        # Add arcs with costs
        for i in range(n):
            for j in range(m):
                G.add_edge(f"plant{j}", i, weight=C_ceil[i, j])
                
        # Add arcs to dummy node (zero cost)
        for j in range(m):
            G.add_edge(f"plant{j}", "dummy", weight=0)
            
        cost, flow = nx.network_simplex(G)
        return cost, flow
    
    def multi_commodity_transportation(self, 
                                     supply: Dict[Tuple[str, str], float],
                                     demand: Dict[Tuple[str, str], float], 
                                     costs: Dict[Tuple[str, str, str], float]) -> Dict[str, Any]:
        """
        Solve multi-commodity transportation problem
        
        Args:
            supply: {(supplier, product): supply_amount}
            demand: {(customer, product): demand_amount}  
            costs: {(supplier, customer, product): unit_cost}
            
        Returns:
            Dictionary with solution details
        """
        if not USE_GUROBI and not USE_PULP:
            raise ValueError("Multi-commodity transportation requires Gurobi or PuLP")
            
        suppliers = list(set(s for s, p in supply.keys()))
        customers = list(set(c for c, p in demand.keys()))
        products = list(set(p for s, p in supply.keys()) | set(p for c, p in demand.keys()))
        
        if USE_GUROBI:
            return self._solve_multi_commodity_gurobi(suppliers, customers, products, supply, demand, costs)
        else:
            return self._solve_multi_commodity_pulp(suppliers, customers, products, supply, demand, costs)
    
    def _solve_multi_commodity_gurobi(self, suppliers, customers, products, supply, demand, costs):
        """
        Solve multi-commodity transportation using Gurobi
        """
        model = Model("MultiCommodityTransportation")
        
        # Decision variables: x[i,j,p] = flow from supplier i to customer j for product p
        x = {}
        for i in suppliers:
            for j in customers:
                for p in products:
                    if (i, j, p) in costs:
                        x[i, j, p] = model.addVar(lb=0, name=f"x_{i}_{j}_{p}")
        
        # Objective: minimize total transportation cost
        model.setObjective(
            quicksum(costs[i, j, p] * x[i, j, p] 
                    for i, j, p in costs.keys() 
                    if (i, j, p) in x), 
            GRB.MINIMIZE
        )
        
        # Supply constraints
        for (i, p), supply_val in supply.items():
            model.addConstr(
                quicksum(x.get((i, j, p), 0) for j in customers 
                        if (i, j, p) in x) <= supply_val,
                name=f"supply_{i}_{p}"
            )
        
        # Demand constraints  
        for (j, p), demand_val in demand.items():
            model.addConstr(
                quicksum(x.get((i, j, p), 0) for i in suppliers 
                        if (i, j, p) in x) >= demand_val,
                name=f"demand_{j}_{p}"
            )
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            solution = {}
            for (i, j, p), var in x.items():
                if var.x > 1e-6:
                    solution[(i, j, p)] = var.x
                    
            return {
                "status": "optimal",
                "objective_value": model.objVal,
                "solution": solution,
                "solver": "gurobi"
            }
        else:
            return {
                "status": "infeasible",
                "objective_value": None,
                "solution": {},
                "solver": "gurobi"
            }
    
    def _solve_multi_commodity_pulp(self, suppliers, customers, products, supply, demand, costs):
        """
        Solve multi-commodity transportation using PuLP
        """
        prob = LpProblem("MultiCommodityTransportation", LpMinimize)
        
        # Decision variables
        x = {}
        for i in suppliers:
            for j in customers:
                for p in products:
                    if (i, j, p) in costs:
                        x[i, j, p] = LpVariable(f"x_{i}_{j}_{p}", lowBound=0)
        
        # Objective function
        prob += lpSum([costs[i, j, p] * x[i, j, p] 
                      for i, j, p in costs.keys() 
                      if (i, j, p) in x])
        
        # Supply constraints
        for (i, p), supply_val in supply.items():
            prob += lpSum([x.get((i, j, p), 0) for j in customers 
                          if (i, j, p) in x]) <= supply_val
        
        # Demand constraints
        for (j, p), demand_val in demand.items():
            prob += lpSum([x.get((i, j, p), 0) for i in suppliers 
                          if (i, j, p) in x]) >= demand_val
        
        prob.solve()
        
        if prob.status == LpStatusOptimal:
            solution = {}
            for (i, j, p), var in x.items():
                if var.varValue > 1e-6:
                    solution[(i, j, p)] = var.varValue
                    
            return {
                "status": "optimal", 
                "objective_value": value(prob.objective),
                "solution": solution,
                "solver": "pulp"
            }
        else:
            return {
                "status": "infeasible",
                "objective_value": None,
                "solution": {},
                "solver": "pulp"
            }
    
    def unbalanced_transportation(self, 
                                supply: List[float], 
                                demand: List[float],
                                costs: np.ndarray) -> Dict[str, Any]:
        """
        Solve unbalanced transportation problem by adding dummy supply/demand
        
        Args:
            supply: List of supply amounts
            demand: List of demand amounts  
            costs: Cost matrix (supply_nodes x demand_nodes)
            
        Returns:
            Solution dictionary with flows and costs
        """
        supply = np.array(supply)
        demand = np.array(demand)
        
        total_supply = supply.sum()
        total_demand = demand.sum()
        
        if total_supply > total_demand:
            # Add dummy demand node
            demand = np.append(demand, total_supply - total_demand)
            costs = np.column_stack([costs, np.zeros(len(supply))])
        elif total_demand > total_supply:
            # Add dummy supply node  
            supply = np.append(supply, total_demand - total_supply)
            costs = np.row_stack([costs, np.zeros(len(demand))])
        
        # Now solve balanced problem
        return self._solve_balanced_transportation(supply, demand, costs)
    
    def _solve_balanced_transportation(self, 
                                     supply: np.ndarray, 
                                     demand: np.ndarray, 
                                     costs: np.ndarray) -> Dict[str, Any]:
        """
        Solve balanced transportation problem using NetworkX
        """
        m, n = len(supply), len(demand)
        
        # Create network
        G = nx.DiGraph()
        
        # Add supply nodes
        for i in range(m):
            G.add_node(f"supply_{i}", demand=-supply[i])
            
        # Add demand nodes
        for j in range(n):
            G.add_node(f"demand_{j}", demand=demand[j])
            
        # Add edges
        for i in range(m):
            for j in range(n):
                G.add_edge(f"supply_{i}", f"demand_{j}", weight=costs[i, j])
        
        try:
            cost, flow = nx.network_simplex(G)
            
            # Convert flow to matrix format
            flow_matrix = np.zeros((m, n))
            for (u, v), flow_val in flow.items():
                if u.startswith('supply_') and v.startswith('demand_'):
                    i = int(u.split('_')[1])
                    j = int(v.split('_')[1])
                    flow_matrix[i, j] = flow_val
            
            return {
                "status": "optimal",
                "objective_value": cost,
                "flow_matrix": flow_matrix.tolist(),
                "solution": flow,
                "solver": "networkx"
            }
        except Exception as e:
            return {
                "status": "error",
                "objective_value": None,
                "flow_matrix": None,
                "solution": {},
                "error": str(e),
                "solver": "networkx"
            }
    
    def transshipment_problem(self,
                            supply_nodes: Dict[str, float],
                            demand_nodes: Dict[str, float], 
                            transship_nodes: List[str],
                            costs: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        """
        Solve transshipment problem with intermediate nodes
        
        Args:
            supply_nodes: {node: supply_amount}
            demand_nodes: {node: demand_amount}
            transship_nodes: List of transshipment node names
            costs: {(from_node, to_node): cost}
            
        Returns:
            Solution dictionary
        """
        # Create network graph
        G = nx.DiGraph()
        
        # Add supply nodes
        for node, supply_val in supply_nodes.items():
            G.add_node(node, demand=-supply_val)
            
        # Add demand nodes
        for node, demand_val in demand_nodes.items():
            G.add_node(node, demand=demand_val)
            
        # Add transshipment nodes (balanced)
        for node in transship_nodes:
            G.add_node(node, demand=0)
            
        # Add edges with costs
        for (from_node, to_node), cost in costs.items():
            G.add_edge(from_node, to_node, weight=cost)
        
        try:
            cost, flow = nx.network_simplex(G)
            return {
                "status": "optimal",
                "objective_value": cost,
                "flow": flow,
                "solver": "networkx_transshipment"
            }
        except Exception as e:
            return {
                "status": "error", 
                "objective_value": None,
                "flow": {},
                "error": str(e),
                "solver": "networkx_transshipment"
            }
    
    def analyze_transportation_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transportation problem solution and provide insights
        
        Args:
            solution: Solution dictionary from transportation solver
            
        Returns:
            Analysis results
        """
        analysis = {
            "solution_status": solution.get("status", "unknown"),
            "total_cost": solution.get("objective_value", 0),
            "solver_used": solution.get("solver", "unknown")
        }
        
        if solution.get("status") == "optimal":
            flow = solution.get("flow", {})
            if isinstance(flow, dict) and flow:
                # Calculate flow statistics
                flow_values = [v for v in flow.values() if v > 1e-6]
                analysis.update({
                    "total_flow": sum(flow_values),
                    "num_active_routes": len(flow_values),
                    "avg_flow_per_route": np.mean(flow_values) if flow_values else 0,
                    "max_flow_on_route": max(flow_values) if flow_values else 0,
                    "min_flow_on_route": min(flow_values) if flow_values else 0
                })
            
            # Flow matrix analysis if available
            if "flow_matrix" in solution and solution["flow_matrix"]:
                flow_matrix = np.array(solution["flow_matrix"])
                analysis.update({
                    "matrix_shape": flow_matrix.shape,
                    "utilization_rate": (flow_matrix > 1e-6).sum() / flow_matrix.size,
                    "flow_concentration": np.std(flow_matrix.flatten()) / np.mean(flow_matrix.flatten()) if np.mean(flow_matrix.flatten()) > 0 else 0
                })
        
        return analysis
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about transportation service capabilities
        
        Returns:
            Service information dictionary
        """
        return {
            "transportation_service": {
                "description": "Advanced transportation problem solver with multiple variants",
                "capabilities": [
                    "Standard transportation problems",
                    "Capacitated transportation problems", 
                    "Multi-commodity transportation",
                    "Unbalanced transportation problems",
                    "Transshipment problems with intermediate nodes",
                    "Transportation solution analysis"
                ],
                "solvers": {
                    "networkx": "Network simplex algorithm (default)",
                    "gurobi": "Commercial optimization solver" if USE_GUROBI else "Not available",
                    "pulp": "Open-source optimization" if USE_PULP else "Not available"
                }
            },
            "problem_types": {
                "standard": "Basic transportation/assignment problems",
                "capacitated": "Transportation with facility capacity limits",
                "multi_commodity": "Multiple products transported simultaneously", 
                "unbalanced": "Supply and demand totals don't match",
                "transshipment": "Intermediate transfer nodes allowed"
            },
            "integration": {
                "k_median": "Used in k-median Lagrangian relaxation",
                "lnd_models": "Integrated with logistics network design",
                "facility_location": "Supports facility location optimization"
            }
        }