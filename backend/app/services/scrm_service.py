import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
import random
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict, OrderedDict
import warnings

warnings.filterwarnings('ignore')

# Import optimization libraries
try:
    from gurobipy import Model, GRB, quicksum
    USE_GUROBI = True
except ImportError:
    USE_GUROBI = False

# Always import PuLP as fallback
try:
    from pulp import *
    USE_PULP = True
except ImportError:
    USE_PULP = False

from app.services.inventory_service import read_willems


class SCMGraph(nx.DiGraph):
    """
    Supply Chain Management Graph class with hierarchical layout functionality
    Exact implementation from 09scrm.ipynb notebook
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def layout(self):
        """
        Generate hierarchical layout for supply chain graphs
        """
        try:
            # Try to use depth-based positioning if available
            depth_dic = nx.get_node_attributes(self, "relDepth")
            if depth_dic:
                pos = {}
                depth_groups = defaultdict(list)
                
                for node, depth in depth_dic.items():
                    depth_groups[int(depth)].append(node)
                
                max_depth = max(depth_groups.keys())
                for depth, nodes in depth_groups.items():
                    for i, node in enumerate(nodes):
                        x = max_depth - depth
                        y = i - len(nodes) / 2
                        pos[node] = (x, y)
                
                return pos
            else:
                # Fall back to spring layout if no depth information
                return nx.spring_layout(self)
                
        except Exception:
            # Final fallback to spring layout
            return nx.spring_layout(self)


class SCRMService:
    """
    Supply Chain Risk Management Service
    Exact implementation from 09scrm.ipynb notebook
    
    Implements:
    - Time-to-Survival (TTS) analysis
    - Risk optimization models
    - Data generation from benchmark problems
    - Visualization functions
    - CSV data conversion
    """
    
    def __init__(self):
        self.use_gurobi = USE_GUROBI
        self.use_pulp = USE_PULP
        if not self.use_gurobi and not self.use_pulp:
            warnings.warn("Neither Gurobi nor PuLP is available. Some optimization features may be limited.")
    
    def data_generation_for_scrm(self, BOM: SCMGraph, n_plnts: int = 3, n_flex: int = 2, 
                                 prob: float = 0.5, capacity_factor: float = 1.0,
                                 production_factor: float = 1.0, pipeline_factor: float = 1.0, 
                                 seed: int = 1) -> Tuple[Dict, float, Dict, Dict, Dict, Dict, Dict, SCMGraph, SCMGraph, Dict, Dict]:
        """
        Generate data for SCRM analysis from benchmark problems
        Exact implementation from notebook
        
        Args:
            BOM: Bill of Materials graph from benchmark
            n_plnts: Number of plants per stage (default: 3)
            n_flex: Number of products per plant (flexibility) (default: 2)  
            prob: Edge probability in plant graph (default: 0.5)
            capacity_factor: Plant capacity multiplier (default: 1.0)
            production_factor: Production limit multiplier (default: 1.0)
            pipeline_factor: Pipeline inventory multiplier (default: 1.0)
            seed: Random seed (default: 1)
            
        Returns:
            Demand, total_demand, UB, Capacity, Pipeline, R, Product, G, ProdGraph, pos2, pos3
        """
        random.seed(seed)

        # Get depth and products at each depth
        depth_dic = nx.get_node_attributes(BOM, "relDepth")
        depth_set = set(depth_dic.values())
        depth = len(depth_set)
        ProdInDepth = [[] for d in range(depth)]
        
        for i in depth_dic:
            d = int(depth_dic[i])
            ProdInDepth[d].append(i)

        # Generate plant graph
        G = SCMGraph()
        Product = {}
        pos2 = {}
        NodeInDepth = [[] for d in range(depth)]
        id_ = 0
        
        for d in range(depth):
            added_prod = []
            for i in range(n_plnts):
                G.add_node(id_)
                pos2[id_] = (depth-d, n_plnts-i)
                NodeInDepth[d].append(id_)
                Product[id_] = random.sample(
                    ProdInDepth[d], min(n_flex, len(ProdInDepth[d])))
                added_prod.extend(Product[id_])
                id_ += 1
            
            # Add any unassigned products to last plant
            all_prod = set(ProdInDepth[d])
            added_prod_set = set(added_prod)
            Product[id_ - 1].extend(all_prod-added_prod_set)

        # Add edges in plant graph
        for d in range(1, depth):
            for count1, i in enumerate(NodeInDepth[d]):
                for count2, j in enumerate(NodeInDepth[d-1]):
                    if count1 == count2:
                        G.add_edge(i, j)
                    elif random.random() <= prob:
                        G.add_edge(i, j)

        # Construct product graph
        ProdGraph = nx.tensor_product(G, BOM)
        Temp = ProdGraph.copy()
        for (i, p) in Temp:
            if p not in Product[i]:
                ProdGraph.remove_node((i, p))

        # Convert to SCMGraph for layout
        ProdGraph2 = SCMGraph()
        ProdGraph2.add_nodes_from(ProdGraph.nodes())
        ProdGraph2.add_edges_from(ProdGraph.edges())
        pos3 = ProdGraph2.layout()

        # Generate parameters
        Pipeline, Demand, UB = {}, {}, {}
        Capacity = {}
        total_demand = 0.
        
        for n in ProdGraph:
            if BOM.out_degree(n[1]) == 0:
                Demand[n] = float(BOM.nodes[n[1]]["avgDemand"])
                total_demand += Demand[n]

        for i in G:
            Capacity[i] = total_demand * capacity_factor
            
        for n in ProdGraph:
            UB[n] = total_demand * production_factor
            Pipeline[n] = float(BOM.nodes[n[1]]["stageTime"]) * total_demand * pipeline_factor

        R = {}
        for (u, v) in ProdGraph.edges():
            (i, p) = u
            (j, q) = v
            R[u, v] = 1

        return Demand, total_demand, UB, Capacity, Pipeline, R, Product, G, ProdGraph2, pos2, pos3
    
    def solve_scrm(self, Demand: Dict, UB: Dict, Capacity: Dict, Pipeline: Dict, R: Dict, 
                   Product: Dict, ProdGraph: SCMGraph, BOM: SCMGraph) -> List[float]:
        """
        Solve SCRM analysis problem
        Exact implementation from notebook
        
        Args:
            Demand: Demand at demand points
            UB: Production upper bounds at each point
            Capacity: Plant production capacities  
            Pipeline: Pipeline inventory at each point
            R: Parent-child production ratios
            Product: Products producible at each plant
            ProdGraph: Production graph
            BOM: Bill of Materials graph
            
        Returns:
            survival_time: List of Time-to-Survival for each disruption scenario
        """
        survival_time = []
        tempUB = {}
        
        for s in ProdGraph:
            # Set up scenario s (disrupt node s)
            for n in ProdGraph:
                tempUB[n] = UB[n]
            tempUB[s] = 0.0

            # Create optimization model
            if self.use_gurobi:
                model = Model()
                model.setParam('OutputFlag', 0)
                tn = model.addVar(name='tn', vtype=GRB.CONTINUOUS)
                
                u, y = {}, {}
                for i, j in ProdGraph.edges():
                    y[i, j] = model.addVar(name=f'y({i},{j})')
                for j in ProdGraph:
                    u[j] = model.addVar(name=f'u({j})', ub=tempUB[j])

                model.update()
                model.setObjective(tn, GRB.MAXIMIZE)

                # Constraints
                # Production and inbound flow relationship
                for j in ProdGraph:
                    if ProdGraph.in_degree(j) > 0:
                        (plant, prod) = j
                        for child in BOM.predecessors(prod):
                            model.addConstr(u[j] <= quicksum((1/float(R[i, j])) * y[i, j]
                                                           for i in ProdGraph.predecessors(j)
                                                           if i[1] == child),
                                           name=f"BOM{j}_{child}")

                # Production and outbound flow relationship
                for i in ProdGraph:
                    if ProdGraph.out_degree(i) > 0:
                        model.addConstr(quicksum(y[i, j] for j in ProdGraph.successors(i))
                                       <= u[i] + Pipeline[i], name=f"BOM2_{i}")

                # Demand satisfaction
                for j in Demand:
                    model.addConstr(u[j] >= Demand[j]*tn, name=f"Demand{j}")

                # Plant capacity constraints
                for f in Capacity:
                    model.addConstr(quicksum(u[f, p] for p in Product[f]) <= Capacity[f]*tn,
                                   name=f"Capacity{f}")

                model.optimize()
                if model.status == GRB.OPTIMAL:
                    survival_time.append(tn.X)
                else:
                    survival_time.append(0.0)
                    
            else:
                # PuLP implementation
                model = LpProblem(f"SCRM_Scenario_{s}", LpMaximize)
                tn = LpVariable('tn', cat='Continuous')
                
                u, y = {}, {}
                for i, j in ProdGraph.edges():
                    y[i, j] = LpVariable(f'y({i},{j})', lowBound=0)
                for j in ProdGraph:
                    u[j] = LpVariable(f'u({j})', lowBound=0, upBound=tempUB[j])

                # Objective
                model += tn

                # Constraints
                for j in ProdGraph:
                    if ProdGraph.in_degree(j) > 0:
                        (plant, prod) = j
                        for child in BOM.predecessors(prod):
                            model += u[j] <= lpSum((1/float(R[i, j])) * y[i, j]
                                                  for i in ProdGraph.predecessors(j)
                                                  if i[1] == child)

                for i in ProdGraph:
                    if ProdGraph.out_degree(i) > 0:
                        model += lpSum(y[i, j] for j in ProdGraph.successors(i)) <= u[i] + Pipeline[i]

                for j in Demand:
                    model += u[j] >= Demand[j] * tn

                for f in Capacity:
                    model += lpSum(u[f, p] for p in Product[f]) <= Capacity[f] * tn

                model.solve(PULP_CBC_CMD(msg=0))
                if model.status == LpStatusOptimal:
                    survival_time.append(tn.varValue)
                else:
                    survival_time.append(0.0)

        return survival_time
    
    def solve_expected_value_minimization(self, Demand: Dict, UB: Dict, Capacity: Dict, Pipeline: Dict, 
                                         R: Dict, Product: Dict, ProdGraph: SCMGraph, BOM: SCMGraph,
                                         prob: Dict, TTR: Dict, h: Dict, b: Dict) -> Dict[str, Any]:
        """
        Solve Expected Value Minimization Model
        Exact implementation from notebook
        
        Args:
            Demand: Demand at demand points
            UB: Production upper bounds at each point
            Capacity: Plant production capacities
            Pipeline: Pipeline inventory at each point  
            R: Parent-child production ratios
            Product: Products producible at each plant
            ProdGraph: Production graph
            BOM: Bill of Materials graph
            prob: Disruption probabilities for each scenario
            TTR: Time-To-Recovery for each plant in each scenario
            h: Inventory holding costs
            b: Backorder costs
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate TMAX for each scenario
        TMAX = {}
        for s in prob:
            max_ttr = 0
            if hasattr(s, "__iter__") and not isinstance(s, str):
                for i in s:  # Multiple plant disruption scenario
                    max_ttr = max(TTR.get(i, 0), max_ttr)
                TMAX[s] = max_ttr
            else:  # Single plant disruption
                TMAX[s] = TTR.get(s, 0)
        
        if self.use_gurobi:
            model = Model("Expected_Value_Minimization")
            model.setParam('OutputFlag', 0)
            
            # Variables
            u, y, I, B = {}, {}, {}, {}
            
            # Flow variables for each scenario
            for i, j in ProdGraph.edges():
                for s in prob:
                    y[i, j, s] = model.addVar(name=f'y({i},{j},{s})')
            
            # Inventory variables (first-stage decisions)
            for j in ProdGraph:
                I[j] = model.addVar(name=f'I({j})')
                
                # Production and backorder variables for each scenario
                for s in prob:
                    B[j, s] = model.addVar(name=f"B({j},{s})")
                    
                    # Set disruption bounds
                    try:
                        plant_id = j[0] if isinstance(j, (tuple, list)) and len(j) > 0 else j
                        if hasattr(s, "__iter__") and not isinstance(s, str):
                            ub_val = 0.0 if plant_id in s else UB.get(j, 0.0)
                        else:
                            ub_val = 0.0 if plant_id == s else UB.get(j, 0.0)
                    except (IndexError, KeyError, TypeError) as e:
                        # Fallback: no disruption
                        ub_val = UB.get(j, 0.0)
                    
                    u[j, s] = model.addVar(name=f'u({j},{s})', ub=ub_val)
            
            model.update()
            
            # Objective: minimize inventory costs + expected backorder costs
            obj = quicksum(h.get(i, 1.0) * I[i] for i in ProdGraph) + \
                  quicksum(prob[s] * b.get(i, 1000.0) * B[i, s] 
                          for i in ProdGraph for s in prob)
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Constraints
            for s in prob:
                # Production and inbound flow relationship
                for j in ProdGraph:
                    if ProdGraph.in_degree(j) > 0:
                        (plant, prod) = j
                        for child in BOM.predecessors(prod):
                            model.addConstr(u[j, s] <= quicksum((1/float(R[i, j])) * y[i, j, s]
                                                              for i in ProdGraph.predecessors(j)
                                                              if i[1] == child),
                                           name=f"BOM{j}_{child}_{s}")
                
                # Production and outbound flow relationship
                for i in ProdGraph:
                    if ProdGraph.out_degree(i) > 0:
                        model.addConstr(quicksum(y[i, j, s] for j in ProdGraph.successors(i))
                                       <= u[i, s] + I[i], name=f"BOM2_{i}_{s}")
                
                # Demand satisfaction with backorders
                for j in Demand:
                    model.addConstr(u[j, s] + I[j] + B[j, s] >= Demand[j] * TMAX[s], 
                                   name=f"Demand{j}_{s}")
                
                # Plant capacity constraints with TTR consideration
                for f in Capacity:
                    if hasattr(s, "__iter__") and not isinstance(s, str):
                        recovery_time = max(TTR.get(i, 0) for i in s) if any(i in s for i in [f]) else 0
                    else:
                        recovery_time = TTR.get(f, 0) if f == s else 0
                    
                    available_time = max(TMAX[s] - recovery_time, 0)
                    model.addConstr(quicksum(u[(f, p), s] for p in Product.get(f, [])) <= Capacity.get(f, 0) * available_time,
                                   name=f"Capacity{f}_{s}")
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                result = {
                    "status": "optimal",
                    "objective_value": model.objVal,
                    "inventory_solution": {str(j): I[j].X for j in ProdGraph},
                    "total_inventory_cost": sum(h.get(i, 1.0) * I[i].X for i in ProdGraph),
                    "expected_backorder_cost": sum(prob[s] * b.get(i, 1000.0) * B[i, s].X 
                                                  for i in ProdGraph for s in prob),
                    "scenarios_analyzed": len(prob)
                }
                return result
            else:
                return {"status": "infeasible", "message": "No optimal solution found"}
                
        else:
            # PuLP implementation
            model = LpProblem("Expected_Value_Minimization", LpMinimize)
            
            # Variables
            u, y, I, B = {}, {}, {}, {}
            
            for i, j in ProdGraph.edges():
                for s in prob:
                    y[i, j, s] = LpVariable(f'y({i},{j},{s})', lowBound=0)
            
            for j in ProdGraph:
                I[j] = LpVariable(f'I({j})', lowBound=0)
                for s in prob:
                    B[j, s] = LpVariable(f"B({j},{s})", lowBound=0)
                    
                    try:
                        plant_id = j[0] if isinstance(j, (tuple, list)) and len(j) > 0 else j
                        if hasattr(s, "__iter__") and not isinstance(s, str):
                            ub_val = 0.0 if plant_id in s else UB.get(j, 0.0)
                        else:
                            ub_val = 0.0 if plant_id == s else UB.get(j, 0.0)
                    except (IndexError, KeyError, TypeError) as e:
                        ub_val = UB.get(j, 0.0)
                    
                    u[j, s] = LpVariable(f'u({j},{s})', lowBound=0, upBound=ub_val)
            
            # Objective
            model += lpSum(h.get(i, 1.0) * I[i] for i in ProdGraph) + \
                     lpSum(prob[s] * b.get(i, 1000.0) * B[i, s] 
                           for i in ProdGraph for s in prob)
            
            # Constraints
            for s in prob:
                for j in ProdGraph:
                    if ProdGraph.in_degree(j) > 0:
                        (plant, prod) = j
                        for child in BOM.predecessors(prod):
                            model += u[j, s] <= lpSum((1/float(R[i, j])) * y[i, j, s]
                                                     for i in ProdGraph.predecessors(j)
                                                     if i[1] == child)
                
                for i in ProdGraph:
                    if ProdGraph.out_degree(i) > 0:
                        model += lpSum(y[i, j, s] for j in ProdGraph.successors(i)) <= u[i, s] + I[i]
                
                for j in Demand:
                    model += u[j, s] + I[j] + B[j, s] >= Demand[j] * TMAX[s]
                
                for f in Capacity:
                    if hasattr(s, "__iter__") and not isinstance(s, str):
                        recovery_time = max(TTR.get(i, 0) for i in s) if any(i in s for i in [f]) else 0
                    else:
                        recovery_time = TTR.get(f, 0) if f == s else 0
                    
                    available_time = max(TMAX[s] - recovery_time, 0)
                    model += lpSum(u[f, p, s] for p in Product[f]) <= Capacity[f] * available_time
            
            model.solve(PULP_CBC_CMD(msg=0))
            
            if model.status == LpStatusOptimal:
                result = {
                    "status": "optimal",
                    "objective_value": value(model.objective),
                    "inventory_solution": {str(j): I[j].varValue for j in ProdGraph},
                    "total_inventory_cost": sum(h.get(i, 1.0) * I[i].varValue for i in ProdGraph),
                    "expected_backorder_cost": sum(prob[s] * b.get(i, 1000.0) * B[i, s].varValue 
                                                  for i in ProdGraph for s in prob),
                    "scenarios_analyzed": len(prob)
                }
                return result
            else:
                return {"status": "infeasible", "message": "No optimal solution found"}
    
    def solve_cvar_model(self, Demand: Dict, UB: Dict, Capacity: Dict, Pipeline: Dict,
                        R: Dict, Product: Dict, ProdGraph: SCMGraph, BOM: SCMGraph,
                        prob: Dict, TTR: Dict, h: Dict, b: Dict, beta: float = 0.95) -> Dict[str, Any]:
        """
        Solve CVaR (Conditional Value at Risk) Model
        Exact implementation from notebook
        
        Args:
            Demand: Demand at demand points
            UB: Production upper bounds at each point
            Capacity: Plant production capacities
            Pipeline: Pipeline inventory at each point
            R: Parent-child production ratios
            Product: Products producible at each plant
            ProdGraph: Production graph
            BOM: Bill of Materials graph
            prob: Disruption probabilities for each scenario
            TTR: Time-To-Recovery for each plant in each scenario
            h: Inventory holding costs
            b: Backorder costs
            beta: Risk level for CVaR (default: 0.95)
            
        Returns:
            Dictionary with CVaR analysis results
        """
        # Calculate TMAX for each scenario
        TMAX = {}
        for s in prob:
            max_ttr = 0
            if hasattr(s, "__iter__") and not isinstance(s, str):
                for i in s:
                    max_ttr = max(TTR.get(i, 0), max_ttr)
                TMAX[s] = max_ttr
            else:
                TMAX[s] = TTR.get(s, 0)
        
        if self.use_gurobi:
            model = Model("CVaR_Model")
            model.setParam('OutputFlag', 0)
            
            # Variables
            u, y, I, B = {}, {}, {}, {}
            theta = model.addVar(name='theta', lb=-GRB.INFINITY)  # VaR variable
            z = {}  # Excess variables for CVaR
            
            # Flow variables
            for i, j in ProdGraph.edges():
                for s in prob:
                    y[i, j, s] = model.addVar(name=f'y({i},{j},{s})')
            
            # Inventory and scenario variables
            for j in ProdGraph:
                I[j] = model.addVar(name=f'I({j})')
                for s in prob:
                    B[j, s] = model.addVar(name=f"B({j},{s})")
                    
                    try:
                        plant_id = j[0] if isinstance(j, (tuple, list)) and len(j) > 0 else j
                        if hasattr(s, "__iter__") and not isinstance(s, str):
                            ub_val = 0.0 if plant_id in s else UB.get(j, 0.0)
                        else:
                            ub_val = 0.0 if plant_id == s else UB.get(j, 0.0)
                    except (IndexError, KeyError, TypeError) as e:
                        ub_val = UB.get(j, 0.0)
                    
                    u[j, s] = model.addVar(name=f'u({j},{s})', ub=ub_val)
            
            # CVaR excess variables
            for s in prob:
                z[s] = model.addVar(name=f'z({s})')
            
            model.update()
            
            # CVaR objective
            cvar_obj = theta + (1.0 / (1.0 - beta)) * quicksum(prob[s] * z[s] for s in prob)
            model.setObjective(cvar_obj, GRB.MINIMIZE)
            
            # CVaR constraints - define scenario costs
            for s in prob:
                scenario_cost = quicksum(h.get(i, 1.0) * I[i] for i in ProdGraph) + \
                               quicksum(b.get(i, 1000.0) * B[i, s] for i in ProdGraph)
                
                model.addConstr(z[s] >= scenario_cost - theta, name=f"CVaR_excess_{s}")
                model.addConstr(z[s] >= 0, name=f"CVaR_non_neg_{s}")
            
            # Standard SCRM constraints
            for s in prob:
                for j in ProdGraph:
                    if ProdGraph.in_degree(j) > 0:
                        (plant, prod) = j
                        for child in BOM.predecessors(prod):
                            model.addConstr(u[j, s] <= quicksum((1/float(R[i, j])) * y[i, j, s]
                                                              for i in ProdGraph.predecessors(j)
                                                              if i[1] == child),
                                           name=f"BOM{j}_{child}_{s}")
                
                for i in ProdGraph:
                    if ProdGraph.out_degree(i) > 0:
                        model.addConstr(quicksum(y[i, j, s] for j in ProdGraph.successors(i))
                                       <= u[i, s] + I[i], name=f"BOM2_{i}_{s}")
                
                for j in Demand:
                    model.addConstr(u[j, s] + I[j] + B[j, s] >= Demand[j] * TMAX[s],
                                   name=f"Demand{j}_{s}")
                
                for f in Capacity:
                    if hasattr(s, "__iter__") and not isinstance(s, str):
                        recovery_time = max(TTR.get(i, 0) for i in s) if any(i in s for i in [f]) else 0
                    else:
                        recovery_time = TTR.get(f, 0) if f == s else 0
                    
                    available_time = max(TMAX[s] - recovery_time, 0)
                    model.addConstr(quicksum(u[(f, p), s] for p in Product.get(f, [])) <= Capacity.get(f, 0) * available_time,
                                   name=f"Capacity{f}_{s}")
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                # Calculate scenario costs for analysis
                scenario_costs = {}
                for s in prob:
                    cost = sum(h.get(i, 1.0) * I[i].X for i in ProdGraph) + \
                           sum(b.get(i, 1000.0) * B[i, s].X for i in ProdGraph)
                    scenario_costs[str(s)] = cost
                
                var_value = theta.X
                cvar_value = model.objVal
                
                result = {
                    "status": "optimal",
                    "cvar_value": cvar_value,
                    "var_value": var_value,
                    "beta": beta,
                    "inventory_solution": {str(j): I[j].X for j in ProdGraph},
                    "scenario_costs": scenario_costs,
                    "expected_cost": sum(prob[s] * scenario_costs[str(s)] for s in prob),
                    "scenarios_analyzed": len(prob)
                }
                return result
            else:
                return {"status": "infeasible", "message": "No optimal solution found"}
                
        else:
            # PuLP implementation for CVaR
            model = LpProblem("CVaR_Model", LpMinimize)
            
            # Variables
            u, y, I, B = {}, {}, {}, {}
            theta = LpVariable('theta')  # VaR variable
            z = {}  # Excess variables for CVaR
            
            for i, j in ProdGraph.edges():
                for s in prob:
                    y[i, j, s] = LpVariable(f'y({i},{j},{s})', lowBound=0)
            
            for j in ProdGraph:
                I[j] = LpVariable(f'I({j})', lowBound=0)
                for s in prob:
                    B[j, s] = LpVariable(f"B({j},{s})", lowBound=0)
                    
                    try:
                        plant_id = j[0] if isinstance(j, (tuple, list)) and len(j) > 0 else j
                        if hasattr(s, "__iter__") and not isinstance(s, str):
                            ub_val = 0.0 if plant_id in s else UB.get(j, 0.0)
                        else:
                            ub_val = 0.0 if plant_id == s else UB.get(j, 0.0)
                    except (IndexError, KeyError, TypeError) as e:
                        ub_val = UB.get(j, 0.0)
                    
                    u[j, s] = LpVariable(f'u({j},{s})', lowBound=0, upBound=ub_val)
            
            for s in prob:
                z[s] = LpVariable(f'z({s})', lowBound=0)
            
            # CVaR objective
            model += theta + (1.0 / (1.0 - beta)) * lpSum(prob[s] * z[s] for s in prob)
            
            # CVaR constraints
            for s in prob:
                scenario_cost = lpSum(h.get(i, 1.0) * I[i] for i in ProdGraph) + \
                               lpSum(b.get(i, 1000.0) * B[i, s] for i in ProdGraph)
                model += z[s] >= scenario_cost - theta
            
            # Standard SCRM constraints
            for s in prob:
                for j in ProdGraph:
                    if ProdGraph.in_degree(j) > 0:
                        (plant, prod) = j
                        for child in BOM.predecessors(prod):
                            model += u[j, s] <= lpSum((1/float(R[i, j])) * y[i, j, s]
                                                     for i in ProdGraph.predecessors(j)
                                                     if i[1] == child)
                
                for i in ProdGraph:
                    if ProdGraph.out_degree(i) > 0:
                        model += lpSum(y[i, j, s] for j in ProdGraph.successors(i)) <= u[i, s] + I[i]
                
                for j in Demand:
                    model += u[j, s] + I[j] + B[j, s] >= Demand[j] * TMAX[s]
                
                for f in Capacity:
                    if hasattr(s, "__iter__") and not isinstance(s, str):
                        recovery_time = max(TTR.get(i, 0) for i in s) if any(i in s for i in [f]) else 0
                    else:
                        recovery_time = TTR.get(f, 0) if f == s else 0
                    
                    available_time = max(TMAX[s] - recovery_time, 0)
                    model += lpSum(u[f, p, s] for p in Product[f]) <= Capacity[f] * available_time
            
            model.solve(PULP_CBC_CMD(msg=0))
            
            if model.status == LpStatusOptimal:
                # Calculate scenario costs
                scenario_costs = {}
                for s in prob:
                    cost = sum(h.get(i, 1.0) * I[i].varValue for i in ProdGraph) + \
                           sum(b.get(i, 1000.0) * B[i, s].varValue for i in ProdGraph)
                    scenario_costs[str(s)] = cost
                
                var_value = theta.varValue
                cvar_value = value(model.objective)
                
                result = {
                    "status": "optimal",
                    "cvar_value": cvar_value,
                    "var_value": var_value,
                    "beta": beta,
                    "inventory_solution": {str(j): I[j].varValue for j in ProdGraph},
                    "scenario_costs": scenario_costs,
                    "expected_cost": sum(prob[s] * scenario_costs[str(s)] for s in prob),
                    "scenarios_analyzed": len(prob)
                }
                return result
            else:
                return {"status": "infeasible", "message": "No optimal solution found"}
    
    def make_df_for_scrm(self, G: SCMGraph, Demand: Dict, UB: Dict, Capacity: Dict, 
                         Pipeline: Dict, BOM: SCMGraph, fn: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convert data structures to DataFrames for CSV export
        Exact implementation from notebook
        
        Args:
            G: Plant graph
            Demand: Demand dictionary
            UB: Upper bounds dictionary
            Capacity: Capacity dictionary
            Pipeline: Pipeline inventory dictionary
            BOM: Bill of Materials graph
            fn: File identifier
            
        Returns:
            Tuple of DataFrames: (bom_df, trans_df, plnt_prod_df, plnt_df)
        """
        # BOM DataFrame
        col_name = ["child", "parent", "units"]
        col_list = {i: [] for i in col_name}
        for (i, j) in BOM.edges():
            col_list["child"].append(i)
            col_list["parent"].append(j)
            col_list["units"].append(1)

        bom_df = pd.DataFrame(col_list, columns=col_name)

        # Transportation DataFrame
        col_name = ["from_node", "to_node", "kind"]
        col_list = {i: [] for i in col_name}
        for (i, j) in G.edges():
            col_list["from_node"].append(i)
            col_list["to_node"].append(j)
            col_list["kind"].append("plnt-plnt")

        trans_df = pd.DataFrame(col_list, columns=col_name)

        # Plant-Product DataFrame
        col_name = ["plnt", "prod", "ub", "pipeline", "demand"]
        col_list = {i: [] for i in col_name}
        for i, p in Pipeline:
            col_list["plnt"].append(i)
            col_list["prod"].append(p)
            col_list["ub"].append(UB[i, p])
            col_list["pipeline"].append(Pipeline[i, p])
            if (i, p) in Demand:
                col_list["demand"].append(Demand[i, p])
            else:
                col_list["demand"].append(None)
        
        plnt_prod_df = pd.DataFrame(col_list, columns=col_name)

        # Plant DataFrame
        col_name = ["name", "ub"]
        col_list = {i: [] for i in col_name}
        for i in Capacity:
            col_list["name"].append(i)
            col_list["ub"].append(Capacity[i])
        
        plnt_df = pd.DataFrame(col_list, columns=col_name)

        return bom_df, trans_df, plnt_prod_df, plnt_df
    
    def prepare_from_dataframes(self, bom_df: pd.DataFrame, plnt_df: pd.DataFrame, 
                               plnt_prod_df: pd.DataFrame, trans_df: pd.DataFrame) -> Tuple:
        """
        Reconstruct data structures from DataFrames
        Exact implementation from notebook
        
        Args:
            bom_df: BOM DataFrame
            plnt_df: Plant DataFrame
            plnt_prod_df: Plant-Product DataFrame
            trans_df: Transportation DataFrame
            
        Returns:
            Tuple of reconstructed data structures
        """
        # Reconstruct basic dictionaries
        Demand, UB, Pipeline = {}, {}, {}
        Product = {i: [] for i in list(plnt_df.name)}
        
        for row in plnt_prod_df.itertuples():
            if pd.notna(row.demand):
                Demand[row.plnt, row.prod] = row.demand
            UB[row.plnt, row.prod] = row.ub
            Pipeline[row.plnt, row.prod] = row.pipeline
            Product[row.plnt].append(row.prod)

        Capacity = {}
        for row in plnt_df.itertuples():
            Capacity[row.name] = row.ub

        # Reconstruct BOM
        BOM = SCMGraph()
        for row in bom_df.itertuples():
            BOM.add_edge(row.child, row.parent, weight=row.units)
        pos = BOM.layout()

        # Reconstruct Plant Transportation Graph
        G = SCMGraph()
        for row in trans_df.itertuples():
            G.add_edge(row.from_node, row.to_node)
        pos2 = G.layout()

        # Reconstruct product graph
        ProdGraph = nx.tensor_product(G, BOM)
        Temp = ProdGraph.copy()
        for (i, p) in Temp:
            if p not in Product[i]:
                ProdGraph.remove_node((i, p))

        # Convert to SCMGraph for layout
        ProdGraph2 = SCMGraph()
        ProdGraph2.add_nodes_from(ProdGraph.nodes())
        ProdGraph2.add_edges_from(ProdGraph.edges())
        pos3 = ProdGraph2.layout()

        # Calculate R (conversion ratios)
        R = {}
        for (u, v) in ProdGraph.edges():
            (i, p) = u
            (j, q) = v
            R[u, v] = BOM[p][q]["weight"]

        return Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph2, pos, pos2, pos3
    
    def draw_graph(self, G: SCMGraph, pos: Dict, title: str = "", size: int = 30, 
                   color: str = "Yellow") -> go.Figure:
        """
        Draw graph using Plotly
        Exact implementation from notebook
        
        Args:
            G: Graph to draw
            pos: Node positions
            title: Graph title
            size: Node size
            color: Node color
            
        Returns:
            Plotly figure object
        """
        x_, y_, text_ = [], [], []
        for idx, i in enumerate(G):
            x_.append(pos[i][0])
            y_.append(pos[i][1])
            text_.append(str(i))

        node_trace = go.Scatter(
            x=x_,
            y=y_,
            mode='markers',
            text=text_,
            marker=dict(
                size=size,
                color=color
            ),
            hoverinfo="text",
            name="nodes",
            showlegend=False
        )

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name="edges",
            showlegend=False
        )

        layout = go.Layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=title
        )

        data = [node_trace, edge_trace]
        fig = go.Figure(data, layout)
        return fig
    
    def draw_scrm(self, ProdGraph: SCMGraph, survival_time: List[float], Pipeline: Dict, 
                  UB: Dict, pos3: Dict) -> go.Figure:
        """
        Draw SCRM risk analysis network
        Exact implementation from notebook
        
        Args:
            ProdGraph: Production graph
            survival_time: Time-to-Survival for each node
            Pipeline: Pipeline inventory
            UB: Upper bounds
            pos3: Node positions
            
        Returns:
            Plotly figure with risk analysis visualization
        """
        ST = np.array(survival_time)
        scaledST = np.log(ST + 1.001)
        size_ = scaledST * 30 + 1.

        x_, y_, text_ = [], [], []
        color_ = []
        
        for i, n in enumerate(ProdGraph):
            x_.append(pos3[n][0])
            y_.append(pos3[n][1])
            text_.append(f"{n} 生存期間:{survival_time[i]:.2f} パイプライン在庫:{Pipeline[n]:.2f} 生産量上限:{UB[n]}")
            color_.append(Pipeline[n])

        node_trace = go.Scatter(
            x=x_,
            y=y_,
            mode='markers',
            text=text_,
            hoverinfo="text",
            marker=dict(
                size=size_,
                colorscale="Hot",
                reversescale=True,
                color=color_,
                colorbar=dict(
                    thickness=15,
                    title='Pipeline Inventory',
                    xanchor='left',
                    titleside='right'
                ),
            ),
            name="nodes",
            showlegend=False
        )

        edge_x = []
        edge_y = []
        for edge in ProdGraph.edges():
            x0, y0 = pos3[edge[0]]
            x1, y1 = pos3[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name="edges",
            showlegend=False
        )

        layout = go.Layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title="Risk Analysis Network (点の大きさ: 途絶時の生存期間, 色： パイプライン在庫)"
        )

        data = [node_trace, edge_trace]
        fig = go.Figure(data, layout)
        return fig
    
    def generate_test_data(self, benchmark_id: str = "01", n_plnts: int = 3, 
                          n_flex: int = 2, seed: int = 1) -> Dict[str, Any]:
        """
        Generate test data for SCRM analysis
        
        Args:
            benchmark_id: Willems benchmark problem ID
            n_plnts: Number of plants per stage
            n_flex: Flexibility parameter
            seed: Random seed
            
        Returns:
            Dictionary with all necessary data for SCRM analysis
        """
        try:
            # Read benchmark BOM data  
            willems_data = read_willems(problem_instance="default")
            
            # Create simple BOM structure for testing
            BOM = SCMGraph()
            # Add sample nodes with required attributes
            products = ["Product_A", "Product_B", "Component_A", "Component_B", "Raw_Material"]
            
            for i, product in enumerate(products):
                BOM.add_node(product, 
                           relDepth=str(i // 2),  # Assign depth levels
                           avgDemand=100.0 + i * 20,  # Sample demand
                           stageTime=1.0 + i * 0.5)  # Sample stage time
            
            # Add edges (child -> parent relationships)
            BOM.add_edge("Component_A", "Product_A", weight=2)
            BOM.add_edge("Component_B", "Product_A", weight=1)
            BOM.add_edge("Component_A", "Product_B", weight=1) 
            BOM.add_edge("Raw_Material", "Component_A", weight=3)
            BOM.add_edge("Raw_Material", "Component_B", weight=2)
            
            pos = BOM.layout()
            
            # Generate SCRM data
            result = self.data_generation_for_scrm(
                BOM, n_plnts=n_plnts, n_flex=n_flex, seed=seed
            )
            
            Demand, total_demand, UB, Capacity, Pipeline, R, Product, G, ProdGraph, pos2, pos3 = result
            
            # Convert to DataFrames
            bom_df, trans_df, plnt_prod_df, plnt_df = self.make_df_for_scrm(
                G, Demand, UB, Capacity, Pipeline, BOM, benchmark_id
            )
            
            return {
                "benchmark_id": benchmark_id,
                "total_demand": total_demand,
                "bom_df": bom_df,
                "trans_df": trans_df,
                "plnt_prod_df": plnt_prod_df,
                "plnt_df": plnt_df,
                "demand": Demand,
                "ub": UB,
                "capacity": Capacity,
                "pipeline": Pipeline,
                "R": R,
                "product": Product,
                "pos": pos,
                "pos2": pos2,
                "pos3": pos3
            }
            
        except Exception as e:
            raise Exception(f"Error generating test data: {str(e)}")
    
    def generate_scenario_data(self, V: List[int], K_max: int = 2) -> Tuple[Dict, Dict]:
        """
        Generate disruption scenarios and calculate TTR/TMAX
        Exact implementation from notebook
        
        Args:
            V: List of plant IDs
            K_max: Maximum number of simultaneous disruptions
            
        Returns:
            Tuple of (prob, TTR) dictionaries
        """
        from itertools import combinations
        
        # Basic disruption probabilities for single plants
        prob = {1: 0.0667, 2: 0.1333, 3: 0.2, 4: 0.1667, 5: 0.0667, 
                6: 0.0667, 7: 0.1667, 8: 0.0667, 9: 0.0667}
        
        # Time-To-Recovery for each plant
        TTR0 = {1: 1, 2: 1.2, 3: 1.4, 4: 1.6, 5: 1.8, 
                6: 2, 7: 2.2, 8: 2.4, 9: 2.6}
        
        TTR = defaultdict(int)
        for i in TTR0:
            TTR[i] = TTR0[i]
        
        # Generate combination scenarios
        for k in range(2, K_max + 1):
            for comb in combinations(V, k):
                prob_ = 1.0
                for i in comb:
                    prob_ *= prob.get(i, 0.01)
                prob[comb] = prob_
        
        return prob, TTR
    
    def run_full_analysis(self, data: Dict[str, Any], model_type: str = "tts", 
                         analysis_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run complete SCRM analysis with specified optimization model
        
        Args:
            data: Data dictionary from generate_test_data or uploaded CSV
            model_type: Type of optimization model ("tts", "expected_value", "cvar")
            analysis_params: Additional parameters for optimization models
            
        Returns:
            Analysis results with model-specific outputs
        """
        try:
            # Set default parameters
            if analysis_params is None:
                analysis_params = {}
            
            # Reconstruct data structures if needed
            if "bom_df" in data:
                result = self.prepare_from_dataframes(
                    data["bom_df"], data["plnt_df"], 
                    data["plnt_prod_df"], data["trans_df"]
                )
                Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = result
            else:
                # Assume data structures are already available
                Demand = data["demand"]
                UB = data["ub"]
                Capacity = data["capacity"]
                Pipeline = data["pipeline"]
                R = data["R"]
                Product = data["product"]
                BOM = data.get("BOM")
                ProdGraph = data.get("ProdGraph")
                pos3 = data["pos3"]
            
            # Generate base analysis results
            base_results = {
                "model_type": model_type,
                "total_nodes": len(list(ProdGraph.nodes())),
            }
            
            if model_type == "tts":
                # Time-to-Survival analysis
                survival_time = self.solve_scrm(Demand, UB, Capacity, Pipeline, R, Product, ProdGraph, BOM)
                
                # Generate visualizations
                risk_fig = self.draw_scrm(ProdGraph, survival_time, Pipeline, UB, pos3)
                
                # Find critical nodes
                node_list = list(ProdGraph.nodes())
                survival_data = [(node_list[i], survival_time[i]) for i in range(len(survival_time))]
                survival_data.sort(key=lambda x: x[1])
                
                critical_nodes = survival_data[:5]
                
                base_results.update({
                    "survival_time": survival_time,
                    "critical_nodes": critical_nodes,
                    "risk_visualization": risk_fig,
                    "average_survival_time": np.mean(survival_time),
                    "min_survival_time": min(survival_time),
                    "max_survival_time": max(survival_time)
                })
                
            elif model_type == "expected_value":
                # Generate scenarios
                V = list(Capacity.keys())
                K_max = analysis_params.get("max_disruptions", 2)
                prob, TTR = self.generate_scenario_data(V, K_max)
                
                # Generate cost parameters
                h = {i: analysis_params.get("inventory_cost", 1.0) for i in ProdGraph}
                b = {i: analysis_params.get("backorder_cost", 1000.0) for i in ProdGraph}
                
                # Solve expected value model
                ev_result = self.solve_expected_value_minimization(
                    Demand, UB, Capacity, Pipeline, R, Product, ProdGraph, BOM,
                    prob, TTR, h, b
                )
                
                base_results.update({
                    "expected_value_result": ev_result,
                    "scenarios_analyzed": len(prob),
                    "total_inventory_cost": ev_result.get("total_inventory_cost", 0),
                    "expected_backorder_cost": ev_result.get("expected_backorder_cost", 0),
                    "objective_value": ev_result.get("objective_value", 0)
                })
                
            elif model_type == "cvar":
                # Generate scenarios
                V = list(Capacity.keys())
                K_max = analysis_params.get("max_disruptions", 2)
                prob, TTR = self.generate_scenario_data(V, K_max)
                
                # Generate cost parameters
                h = {i: analysis_params.get("inventory_cost", 1.0) for i in ProdGraph}
                b = {i: analysis_params.get("backorder_cost", 1000.0) for i in ProdGraph}
                beta = analysis_params.get("risk_level", 0.95)
                
                # Solve CVaR model
                cvar_result = self.solve_cvar_model(
                    Demand, UB, Capacity, Pipeline, R, Product, ProdGraph, BOM,
                    prob, TTR, h, b, beta
                )
                
                base_results.update({
                    "cvar_result": cvar_result,
                    "scenarios_analyzed": len(prob),
                    "beta": beta,
                    "cvar_value": cvar_result.get("cvar_value", 0),
                    "var_value": cvar_result.get("var_value", 0),
                    "expected_cost": cvar_result.get("expected_cost", 0)
                })
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return base_results
            
        except Exception as e:
            raise Exception(f"Error running SCRM analysis: {str(e)}")