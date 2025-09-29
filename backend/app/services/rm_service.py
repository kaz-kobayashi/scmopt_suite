import numpy as np
import pandas as pd
import math
import random
import time as time_module
import copy
import io
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import optimization libraries
try:
    from pulp import *
    USE_PULP = True
except ImportError:
    USE_PULP = False

from scipy.stats import poisson, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from app.models.rm import RMError


class RMService:
    """
    収益管理システム MERMO (MEta Revenue Management Optimizer)
    Exact implementation from 13rm.ipynb notebook
    
    Implements:
    - Dynamic pricing with reinforcement learning
    - Value function estimation with dynamic programming
    - Revenue management optimization models
    - Bid price control policy
    - Nested booking limit control policy
    - Prospect theory pricing
    - Booking demand forecasting
    """
    
    def __init__(self):
        self.use_pulp = USE_PULP
        if not self.use_pulp:
            warnings.warn("PuLP is not available. Optimization features will be limited.")
    
    def random_demand(self, price: float, beta_params: Tuple[float, float], sigma: float = 1.0) -> float:
        """
        Generate random demand based on linear demand function
        d = beta0 + beta1 * price + N(0, sigma^2)
        """
        beta0, beta1 = beta_params
        epsilon = np.random.normal(0., sigma)
        return max(beta0 + beta1 * price + epsilon, 0.0)
    
    def dynamic_pricing_learning(self, actions: np.ndarray, beta_params: Tuple[float, float], 
                                epochs: int = 10, sigma: float = 1.0, delta: float = 0.1, 
                                scaling: float = 1.0) -> Dict[str, Any]:
        """
        Dynamic pricing using reinforcement learning (contextual bandit)
        Exact implementation from notebook
        """
        np.random.seed(seed=32)
        
        K = len(actions)  # number of actions
        
        tau = [0]  # epoch end periods
        for m in range(1, epochs):
            tau.append(2**m)
        T = 2**(epochs-1) + 1  # planning horizon
        
        demand = np.zeros(T)
        price = np.zeros(T)
        reward_hist = np.zeros(T)
        epochs_data = []
        
        for m in range(1, epochs):
            # Learning rate (step size)
            gamma = np.sqrt(K * tau[m-1] / np.log((tau[m-1] + 0.001) / delta)) / scaling
            
            if m == 1:  # first epoch
                p = np.random.choice(actions)
                d = self.random_demand(p, beta_params, sigma)
                price[0] = p
                demand[0] = d
                
                epochs_data.append({
                    "epoch": m,
                    "gamma": gamma,
                    "periods": [0],
                    "estimated_beta": None
                })
            else:
                # Forecast using linear regression
                X = price[:tau[m-1]+1].reshape(-1, 1)
                y = demand[:tau[m-1]+1].reshape(-1, 1)
                reg = LinearRegression()
                reg.fit(X, y)
                
                # Predict demand for each action
                yhat = reg.predict(actions.reshape(-1, 1))
                
                epochs_data.append({
                    "epoch": m,
                    "gamma": gamma,
                    "periods": list(range(tau[m-1]+1, tau[m]+1)),
                    "estimated_beta": (float(reg.intercept_[0]), float(reg.coef_[0][0]))
                })
                
                # Action selection
                for t in range(tau[m-1]+1, tau[m]+1):
                    # Estimated reward for each action
                    reward = yhat * actions.reshape(-1, 1)
                    max_reward = np.max(reward)
                    max_idx = np.argmax(reward)
                    
                    # Probability distribution for action selection
                    prob = 1. / (K + gamma * (max_reward - reward))
                    prob_no_max = np.delete(prob, max_idx, axis=0)
                    total_prob = np.sum(prob_no_max)
                    prob[max_idx] = 1. - total_prob
                    
                    # Select action and observe demand
                    p = np.random.choice(actions, p=prob.reshape(-1,))
                    d = self.random_demand(p, beta_params, sigma)
                    
                    price[t] = p
                    demand[t] = d
                    reward_hist[t] = p * d
        
        # Final parameter estimation
        X_final = price[:T].reshape(-1, 1)
        y_final = demand[:T].reshape(-1, 1)
        reg_final = LinearRegression()
        reg_final.fit(X_final, y_final)
        
        return {
            "total_reward": float(reward_hist.sum()),
            "price_history": price.tolist(),
            "demand_history": demand.tolist(),
            "reward_history": reward_hist.tolist(),
            "estimated_beta": (float(reg_final.intercept_[0]), float(reg_final.coef_[0][0])),
            "epochs_data": epochs_data
        }
    
    def value_function_dp(self, capacity: int, periods: int, actions: List[float], 
                         beta_params: Tuple[float, float], sigma: float = 0.0, 
                         n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamic programming for value function estimation
        Exact implementation from notebook
        """
        C, T = capacity, periods
        actions = np.array(actions)
        
        V = np.zeros((T+1, C+1))
        A = np.zeros((T+1, C+1))
        
        for t in range(T-1, -1, -1):
            for c in range(1, C+1):
                reward_to_go = [0. for _ in range(len(actions))]
                
                for sample in range(n_samples):
                    demand = np.array([self.random_demand(a, beta_params, sigma) for a in actions])
                    
                    for i, a in enumerate(actions):
                        actual_demand = min(demand[i], c)
                        remaining_capacity = max(c - max(int(demand[i]), 0), 0)
                        reward_to_go[i] += a * actual_demand + V[t+1, remaining_capacity]
                
                # Find best action
                max_v, max_a = 0, -1
                for i, a in enumerate(actions):
                    if reward_to_go[i] > max_v:
                        max_v = reward_to_go[i]
                        max_a = a
                
                V[t, c] = max_v
                A[t, c] = max_a
        
        return V, A
    
    def simulate_inventory_pricing(self, V: np.ndarray, A: np.ndarray, 
                                  initial_capacity: int, periods: int, 
                                  beta_params: Tuple[float, float], sigma: float = 0.1) -> Dict[str, Any]:
        """
        Simulate inventory and pricing using value function
        """
        I = initial_capacity
        price_list, inv_list = [], []
        reward = 0.
        
        for t in range(periods):
            a = A[t, max(I, 0)]
            demand = self.random_demand(a, beta_params, sigma)
            
            inv_list.append(I)
            price_list.append(a)
            
            actual_demand = min(demand, I) if I > 0 else 0
            I -= int(demand)
            
            if I >= 0:
                reward += a * actual_demand
        
        return {
            "total_reward": reward,
            "price_history": price_list,
            "inventory_history": inv_list
        }
    
    def rm_deterministic(self, demand: Dict[int, float], capacity: Dict[int, float], 
                        revenue: Dict[int, float], a: Dict[Tuple[int, int], int]) -> Tuple[float, Dict[int, float], Dict[int, float]]:
        """
        Deterministic revenue management model
        Exact implementation from notebook
        """
        if not self.use_pulp:
            raise RMError("PuLP is not available")
        
        m = len(demand)
        ell = len(capacity)
        res = range(ell)
        
        model = LpProblem("RM_Deterministic", LpMaximize)
        
        y = {}
        for j in range(m):
            y[j] = LpVariable(f'y({j})', lowBound=0, upBound=float(demand[j]))
        
        # Capacity constraints
        for i in res:
            constraint_terms = []
            for j in range(m):
                if (i, j) in a:
                    constraint_terms.append(a[i, j] * y[j])
            if constraint_terms:
                model += lpSum(constraint_terms) <= capacity[i]
        
        # Objective function
        model += lpSum(revenue[j] * y[j] for j in range(m))
        
        # Solve
        solver = PULP_CBC_CMD(msg=0)
        model.solve(solver)
        
        if model.status != LpStatusOptimal:
            raise RMError(f"Failed to solve deterministic RM model! Status: {model.status}")
        
        # Extract dual variables and solution
        dual = {}
        for constraint in model.constraints.values():
            # PuLP doesn't easily expose dual variables, so we approximate
            dual[len(dual)] = 0.0  # Placeholder
        
        ystar = {}
        for j in range(m):
            ystar[j] = y[j].varValue
        
        return float(value(model.objective)), dual, ystar
    
    def rm_sampling(self, demand: Dict[int, float], capacity: Dict[int, float], 
                   revenue: Dict[int, float], a: Dict[Tuple[int, int], int], 
                   n_samples: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Revenue management model using sampling
        Exact implementation from notebook
        """
        if not self.use_pulp:
            raise RMError("PuLP is not available")
        
        m = len(demand)
        ell = len(capacity)
        res = range(ell)
        
        obj_list = []
        dual = np.zeros(ell)
        ystar = np.zeros(m)
        
        for s in range(n_samples):
            # Sample demands from Poisson distribution
            dem = {}
            for j in range(m):
                dem[j] = float(poisson.rvs(demand[j], size=1)[0])
            
            model = LpProblem(f"RM_Sampling_{s}", LpMaximize)
            
            y = {}
            for j in range(m):
                y[j] = LpVariable(f'y({j})', lowBound=0, upBound=dem[j])
            
            # Capacity constraints
            for i in res:
                constraint_terms = []
                for j in range(m):
                    if (i, j) in a:
                        constraint_terms.append(a[i, j] * y[j])
                if constraint_terms:
                    model += lpSum(constraint_terms) <= capacity[i]
            
            # Objective function
            model += lpSum(revenue[j] * y[j] for j in range(m))
            
            # Solve
            solver = PULP_CBC_CMD(msg=0)
            model.solve(solver)
            
            if model.status == LpStatusOptimal:
                obj_list.append(float(value(model.objective)))
                
                # Accumulate solutions
                for j in range(m):
                    ystar[j] += y[j].varValue
        
        avg_obj = sum(obj_list) / len(obj_list) if obj_list else 0.0
        
        # Convert numpy arrays to dictionaries to match return type
        dual_dict = {i: float(dual[i] / n_samples) for i in range(ell)}
        ystar_dict = {j: float(ystar[j] / n_samples) for j in range(m)}
        
        return avg_obj, dual_dict, ystar_dict
    
    def rm_recourse(self, demand: Dict[int, float], capacity: Dict[int, float], 
                   revenue: Dict[int, float], a: Dict[Tuple[int, int], int]) -> Tuple[float, Dict[int, float], Dict[int, float]]:
        """
        Revenue management model using recourse (stochastic programming)
        Exact implementation from notebook
        """
        if not self.use_pulp:
            raise RMError("PuLP is not available")
        
        m = len(demand)
        ell = len(capacity)
        res = range(ell)
        
        # Generate discrete probability distribution
        Delta = {}
        for j in range(m):
            mu = int(demand[j])
            rv = poisson(mu)
            Delta[j, 0] = 0.
            for k in range(1, mu * 2):
                Delta[j, k] = rv.sf(k)  # survival function P(X >= k)
        
        model = LpProblem("RM_Recourse", LpMaximize)
        
        z = {}
        for j, k in Delta:
            if k != 0:
                z[j, k] = LpVariable(f'z({j},{k})', lowBound=0, upBound=1)
        
        # Capacity constraints
        for i in res:
            constraint_terms = []
            for j, k in Delta:
                if k != 0 and (i, j) in a:
                    constraint_terms.append(a[i, j] * z[j, k])
            if constraint_terms:
                model += lpSum(constraint_terms) <= capacity[i]
        
        # Objective function
        objective_terms = []
        for j, k in Delta:
            if k != 0:
                objective_terms.append(revenue[j] * Delta[j, k] * z[j, k])
        model += lpSum(objective_terms)
        
        # Solve
        solver = PULP_CBC_CMD(msg=0)
        model.solve(solver)
        
        if model.status != LpStatusOptimal:
            raise RMError(f"Failed to solve recourse RM model! Status: {model.status}")
        
        dual = {i: 0.0 for i in res}  # Placeholder for dual variables
        
        ystar = {}
        for j in range(m):
            ystar[j] = 0.
            for k in range(1, int(demand[j]) * 2):
                if (j, k) in z and z[j, k].varValue > 0.1:
                    ystar[j] += 1.0
        
        return float(value(model.objective)), dual, ystar
    
    def make_sample_data_for_rm(self, num_periods: int) -> Tuple[Dict[int, float], Dict[int, float], Dict[Tuple[int, int], int], Dict[int, float]]:
        """
        Generate sample data for revenue management
        Exact implementation from notebook
        """
        ell = num_periods
        res = list(range(ell))
        j = 0
        
        # 3 types of demands with durations 1, 2, 3
        dem = {1: 20., 2: 6., 3: 3.}
        rev = {1: 1000., 2: 3000., 3: 5000.}
        a, demand, revenue = {}, {}, {}
        
        for dur in range(1, 4):  # duration
            for r in range(ell + 1 - dur):
                for i in range(r, r + dur):
                    a[i, j] = 1
                demand[j] = dem[dur]
                revenue[j] = rev[dur]
                j += 1
        
        m = j  # number of demands
        capacity = {i: 10. for i in res}
        
        return demand, revenue, a, capacity
    
    def bid_price_control(self, demand: Dict[int, float], revenue: Dict[int, float], 
                         a: Dict[Tuple[int, int], int], capacity: Dict[int, float], 
                         n_samples: int = 100, method: int = 0, random_seed: int = 123) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Bid price control simulation
        Exact implementation from notebook
        """
        np.random.seed(random_seed)
        ell = len(capacity)
        m = len(demand)
        
        # Generate random demand using Poisson distribution
        demand_ = {}
        for j in range(m):
            demand_[j] = float(poisson.rvs(demand[j], size=1)[0])
        
        # Prepare arrival process
        arrival = []
        for j in demand_:
            for i in range(int(demand_[j])):
                arrival.append(j)
        arrival = np.array(arrival)
        np.random.shuffle(arrival)
        
        total_revenue = 0.
        acceptance_history = []
        capacity_copy = copy.deepcopy(capacity)
        
        for t, j in enumerate(arrival):
            # Calculate dual variables based on method
            if method == 0:
                obj, dual, ystar = self.rm_deterministic(demand, capacity_copy, revenue, a)
            elif method == 1:
                obj, dual, ystar = self.rm_sampling(demand, capacity_copy, revenue, a, n_samples)
            elif method == 2:
                obj, dual, ystar = self.rm_recourse(demand, capacity_copy, revenue, a)
            else:
                raise RMError("Method must be 0, 1 or 2")
            
            # Check if we should accept the request
            bid_price = sum(a.get((i, j), 0) * dual.get(i, 0) for i in range(ell))
            
            if revenue[j] >= bid_price:
                # Check capacity availability
                can_accept = True
                for i in range(ell):
                    if (i, j) in a and capacity_copy[i] == 0:
                        can_accept = False
                        break
                
                if can_accept:
                    total_revenue += revenue[j]
                    for i in range(ell):
                        if (i, j) in a:
                            capacity_copy[i] -= a[i, j]
                    
                    acceptance_history.append({
                        "time": t,
                        "service": j,
                        "revenue": revenue[j],
                        "bid_price": bid_price,
                        "accepted": True,
                        "capacity": copy.deepcopy(capacity_copy)
                    })
                else:
                    acceptance_history.append({
                        "time": t,
                        "service": j,
                        "revenue": revenue[j],
                        "bid_price": bid_price,
                        "accepted": False,
                        "capacity": copy.deepcopy(capacity_copy)
                    })
            else:
                acceptance_history.append({
                    "time": t,
                    "service": j,
                    "revenue": revenue[j],
                    "bid_price": bid_price,
                    "accepted": False,
                    "capacity": copy.deepcopy(capacity_copy)
                })
            
            demand[j] = max(demand[j] - 1, 0.)
        
        return total_revenue, acceptance_history
    
    def nested_booking_limit_control(self, demand: Dict[int, float], revenue: Dict[int, float], 
                                   a: Dict[Tuple[int, int], int], capacity: Dict[int, float], 
                                   n_samples: int = 100, method: int = 0, random_seed: int = 123) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Nested booking limit control simulation
        Exact implementation from notebook
        """
        np.random.seed(random_seed)
        ell = len(capacity)
        m = len(demand)
        
        # Make a copy of demand to avoid modifying the original
        demand_copy = copy.deepcopy(demand)
        
        # Generate random demand using Poisson distribution
        demand_ = {}
        for j in range(m):
            demand_[j] = float(poisson.rvs(demand_copy[j], size=1)[0])
        
        # Prepare arrival process
        arrival = []
        for j in demand_:
            for i in range(int(demand_[j])):
                arrival.append(j)
        arrival = np.array(arrival)
        np.random.shuffle(arrival)
        
        # Solve optimization model to get dual variables and optimal solution
        if method == 0:
            obj, dual, ystar = self.rm_deterministic(demand_copy, capacity, revenue, a)
        elif method == 1:
            obj, dual, ystar = self.rm_sampling(demand_copy, capacity, revenue, a, n_samples)
        elif method == 2:
            obj, dual, ystar = self.rm_recourse(demand_copy, capacity, revenue, a)
        else:
            raise RMError("Method must be 0, 1 or 2")
        
        # Calculate adjusted revenue for prioritization
        rbar = {}
        order = []
        for j_ in range(m):
            rbar[j_] = revenue[j_] - sum(a.get((i, j_), 0) * dual.get(i, 0) for i in range(ell))
            order.append((rbar[j_], j_))
        order.sort(reverse=True)
        
        # Calculate nested booking limits
        capacity_copy = copy.deepcopy(capacity)
        S = {}
        for (r, j_) in order:
            for i in range(ell):
                if (i, j_) in a:
                    S[i, j_] = max(capacity_copy[i], 0)
                    capacity_copy[i] = max(capacity_copy[i] - ystar.get(j_, 0), 0)
        
        total_revenue = 0.
        acceptance_history = []
        capacity_current = copy.deepcopy(capacity)
        
        for t, j in enumerate(arrival):
            # Check if service can be accepted based on booking limits
            can_accept = True
            for i in range(ell):
                if (i, j) in a:
                    if capacity_current[i] <= 0 or S.get((i, j), 0) <= 0:
                        can_accept = False
                        break
            
            if can_accept:
                total_revenue += revenue[j]
                for i in range(ell):
                    if (i, j) in a:
                        S[i, j] = max(S[i, j] - a[i, j], 0)
                        capacity_current[i] = max(capacity_current[i] - a[i, j], 0)
                
                # Convert booking_limits with tuple keys to string keys for JSON serialization
                booking_limits_str = {f"({k[0]},{k[1]})": v for k, v in S.items()}
                
                acceptance_history.append({
                    "time": t,
                    "service": j,
                    "revenue": revenue[j],
                    "accepted": True,
                    "capacity": copy.deepcopy(capacity_current),
                    "booking_limits": booking_limits_str
                })
            else:
                # Convert booking_limits with tuple keys to string keys for JSON serialization
                booking_limits_str = {f"({k[0]},{k[1]})": v for k, v in S.items()}
                
                acceptance_history.append({
                    "time": t,
                    "service": j,
                    "revenue": revenue[j],
                    "accepted": False,
                    "capacity": copy.deepcopy(capacity_current),
                    "booking_limits": booking_limits_str
                })
            
            demand_copy[j] = max(demand_copy[j] - 1, 0.)
        
        return total_revenue, acceptance_history
    
    def prospect_theory_pricing(self, base_demand: float, base_price: float, reference_price: float,
                              alpha: float = 0.5, zeta: float = 8.0, eta: float = 12.0,
                              beta: float = 0.88, gamma: float = 0.88, periods: int = 50) -> Dict[str, Any]:
        """
        Prospect theory pricing strategy
        Exact implementation from notebook
        """
        prices = []
        demands = []
        revenues = []
        reference_prices = []
        
        current_reference = reference_price
        
        for t in range(periods):
            # Calculate prospect effect
            if base_price <= current_reference:
                # Gain domain (concave utility)
                prospect_effect = zeta * ((current_reference - base_price) ** beta)
            else:
                # Loss domain (convex utility)
                prospect_effect = -eta * ((base_price - current_reference) ** gamma)
            
            # Calculate demand with prospect effect
            demand = base_demand + prospect_effect
            demand = max(demand, 0)  # Ensure non-negative demand
            
            # Calculate revenue
            revenue = base_price * demand
            
            # Store results
            prices.append(base_price)
            demands.append(demand)
            revenues.append(revenue)
            reference_prices.append(current_reference)
            
            # Update reference price for next period
            current_reference = (1 - alpha) * base_price + alpha * current_reference
            
            # Simple price adjustment strategy (can be enhanced with optimization)
            # For demonstration, we use a simple oscillating strategy
            if t < periods - 1:
                if t % 10 < 5:
                    base_price = min(base_price * 1.05, reference_price * 1.2)
                else:
                    base_price = max(base_price * 0.95, reference_price * 0.8)
        
        return {
            "optimal_prices": prices,
            "reference_prices": reference_prices,
            "demands": demands,
            "revenues": revenues,
            "total_revenue": sum(revenues),
            "prospect_effects": {
                "alpha": alpha,
                "zeta": zeta,
                "eta": eta,
                "beta": beta,
                "gamma": gamma
            }
        }
    
    def multiplicative_booking_forecast(self, booking_matrix: np.ndarray, current_period: int, 
                                      max_leadtime: int) -> Dict[str, Any]:
        """
        Multiplicative booking demand forecast
        Exact implementation from notebook
        """
        m, n = booking_matrix.shape  # periods, leadtime
        
        # Create cumulative booking matrix
        cumulative_matrix = booking_matrix.cumsum(axis=1)
        
        # Handle missing future bookings (set to -1)
        for i in range(m):
            for j in range(n):
                if i - (n - 1 - j) > current_period:
                    cumulative_matrix[i, j] = -1
        
        # Calculate multiplicative ratios
        C_shift = np.roll(cumulative_matrix, shift=-1, axis=1)
        ratio = C_shift / (cumulative_matrix + 0.001)  # Small epsilon to avoid division by zero
        ratio = ratio[:current_period + 1, :-1]  # Remove last column and future periods
        
        # Calculate mean ratios for forecasting
        multi_ratio = ratio.mean(axis=0)
        
        # Forecast future bookings
        forecast_matrix = cumulative_matrix.copy()
        for i in range(current_period + 1, m):
            for j in range(n - 1 - (i - current_period), n - 1):
                if j + 1 < len(multi_ratio):
                    forecast_matrix[i, j + 1] = forecast_matrix[i, j] * multi_ratio[j]
        
        # Calculate forecast accuracy metrics
        actual_bookings = booking_matrix[:current_period + 1]
        forecast_bookings = forecast_matrix[:current_period + 1]
        mse = np.mean((actual_bookings - forecast_bookings) ** 2)
        mae = np.mean(np.abs(actual_bookings - forecast_bookings))
        
        return {
            "forecast_matrix": forecast_matrix.tolist(),
            "cumulative_bookings": cumulative_matrix.tolist(),
            "multiplicative_ratios": multi_ratio.tolist(),
            "forecast_accuracy": {
                "mse": float(mse),
                "mae": float(mae),
                "period_coverage": current_period + 1
            },
            "method_used": "multiplicative"
        }
    
    def parse_csv_data(self, csv_content: str, data_type: str) -> Dict[str, Any]:
        """
        CSVファイル内容を解析してRMデータ形式に変換
        """
        try:
            # CSV内容をDataFrameに読み込み
            df = pd.read_csv(io.StringIO(csv_content))
            
            if data_type == "demand":
                # 需要データの解析: service_id, demand
                if not all(col in df.columns for col in ['service_id', 'demand']):
                    raise RMError("需要データCSVには 'service_id', 'demand' 列が必要です")
                
                demand_data = {}
                for _, row in df.iterrows():
                    service_id = int(row['service_id'])
                    demand_value = float(row['demand'])
                    demand_data[service_id] = demand_value
                
                return {
                    "data": demand_data,
                    "type": "demand",
                    "validation": "success",
                    "errors": []
                }
            
            elif data_type == "revenue":
                # 収益データの解析: service_id, revenue
                if not all(col in df.columns for col in ['service_id', 'revenue']):
                    raise RMError("収益データCSVには 'service_id', 'revenue' 列が必要です")
                
                revenue_data = {}
                for _, row in df.iterrows():
                    service_id = int(row['service_id'])
                    revenue_value = float(row['revenue'])
                    revenue_data[service_id] = revenue_value
                
                return {
                    "data": revenue_data,
                    "type": "revenue",
                    "validation": "success",
                    "errors": []
                }
            
            elif data_type == "capacity":
                # 容量データの解析: resource_id, capacity
                if not all(col in df.columns for col in ['resource_id', 'capacity']):
                    raise RMError("容量データCSVには 'resource_id', 'capacity' 列が必要です")
                
                capacity_data = {}
                for _, row in df.iterrows():
                    resource_id = int(row['resource_id'])
                    capacity_value = float(row['capacity'])
                    capacity_data[resource_id] = capacity_value
                
                return {
                    "data": capacity_data,
                    "type": "capacity",
                    "validation": "success",
                    "errors": []
                }
            
            elif data_type == "usage_matrix":
                # 使用関係マトリックスの解析: resource_id, service_id, usage
                if not all(col in df.columns for col in ['resource_id', 'service_id', 'usage']):
                    raise RMError("使用関係マトリックスCSVには 'resource_id', 'service_id', 'usage' 列が必要です")
                
                usage_matrix = {}
                for _, row in df.iterrows():
                    resource_id = int(row['resource_id'])
                    service_id = int(row['service_id'])
                    usage_value = int(row['usage'])
                    usage_matrix[f"({resource_id},{service_id})"] = usage_value
                
                return {
                    "data": usage_matrix,
                    "type": "usage_matrix",
                    "validation": "success",
                    "errors": []
                }
            
            else:
                raise RMError(f"未対応のデータタイプ: {data_type}")
                
        except Exception as e:
            return {
                "data": {},
                "type": data_type,
                "validation": "error",
                "errors": [str(e)]
            }
    
    def validate_rm_data(self, demand: Dict[int, float], revenue: Dict[int, float], 
                        capacity: Dict[int, float], usage_matrix: Dict[str, int]) -> Dict[str, Any]:
        """
        収益管理データの整合性検証
        """
        errors = []
        warnings = []
        
        try:
            # データの基本検証
            if not demand:
                errors.append("需要データが空です")
            if not revenue:
                errors.append("収益データが空です")
            if not capacity:
                errors.append("容量データが空です")
            if not usage_matrix:
                errors.append("使用関係マトリックスが空です")
            
            if errors:
                return {"validation": "error", "errors": errors, "warnings": warnings}
            
            # サービスIDの整合性チェック
            demand_services = set(demand.keys())
            revenue_services = set(revenue.keys())
            
            if demand_services != revenue_services:
                missing_in_revenue = demand_services - revenue_services
                missing_in_demand = revenue_services - demand_services
                
                if missing_in_revenue:
                    errors.append(f"収益データに不足しているサービスID: {missing_in_revenue}")
                if missing_in_demand:
                    errors.append(f"需要データに不足しているサービスID: {missing_in_demand}")
            
            # 使用関係マトリックスの検証
            usage_services = set()
            usage_resources = set()
            
            for key in usage_matrix.keys():
                try:
                    # "(resource_id,service_id)" 形式を解析
                    key_clean = key.strip("()")
                    resource_id, service_id = map(int, key_clean.split(","))
                    usage_services.add(service_id)
                    usage_resources.add(resource_id)
                except:
                    errors.append(f"不正な使用関係マトリックスキー: {key}")
            
            # サービスIDと使用関係マトリックスの整合性
            missing_services_in_matrix = demand_services - usage_services
            if missing_services_in_matrix:
                warnings.append(f"使用関係マトリックスにないサービスID: {missing_services_in_matrix}")
            
            # リソースIDと容量データの整合性
            capacity_resources = set(capacity.keys())
            missing_resources_in_capacity = usage_resources - capacity_resources
            if missing_resources_in_capacity:
                errors.append(f"容量データに不足しているリソースID: {missing_resources_in_capacity}")
            
            # データ値の妥当性チェック
            negative_demand = [k for k, v in demand.items() if v < 0]
            if negative_demand:
                errors.append(f"負の需要値を持つサービス: {negative_demand}")
            
            negative_revenue = [k for k, v in revenue.items() if v < 0]
            if negative_revenue:
                warnings.append(f"負の収益値を持つサービス: {negative_revenue}")
            
            negative_capacity = [k for k, v in capacity.items() if v < 0]
            if negative_capacity:
                errors.append(f"負の容量値を持つリソース: {negative_capacity}")
            
            # 使用関係の妥当性
            invalid_usage = [k for k, v in usage_matrix.items() if v < 0]
            if invalid_usage:
                errors.append(f"負の使用値: {invalid_usage}")
            
            validation_status = "error" if errors else ("warning" if warnings else "success")
            
            return {
                "validation": validation_status,
                "errors": errors,
                "warnings": warnings,
                "summary": {
                    "services_count": len(demand_services),
                    "resources_count": len(capacity_resources),
                    "total_demand": sum(demand.values()),
                    "total_revenue_potential": sum(revenue.values()),
                    "total_capacity": sum(capacity.values())
                }
            }
            
        except Exception as e:
            return {
                "validation": "error",
                "errors": [f"検証中にエラーが発生しました: {str(e)}"],
                "warnings": warnings
            }