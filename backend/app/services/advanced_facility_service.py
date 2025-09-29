import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import AgglomerativeClustering
from geopy.distance import great_circle as distance
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class AdvancedFacilityLocationService:
    """
    Advanced facility location algorithms including hierarchical clustering and k-median optimization.
    Extracted from 05lnd.ipynb to provide high-performance facility location solutions.
    """
    
    def __init__(self):
        self.epsilon = 1e-6
        
    def great_circle_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points on Earth using Haversine formula
        Returns distance in kilometers
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        R = 6371
        distance = R * c
        
        return distance

    def compute_distance_matrix(self, customer_df: pd.DataFrame) -> np.ndarray:
        """
        Compute distance matrix for hierarchical clustering
        """
        n = len(customer_df)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1 = customer_df.iloc[i]['lat'], customer_df.iloc[i]['lon']
                lat2, lon2 = customer_df.iloc[j]['lat'], customer_df.iloc[j]['lon']
                dist = self.great_circle_distance(lat1, lon1, lat2, lon2)
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances

    def hierarchical_clustering(
        self, 
        customer_df: pd.DataFrame, 
        weight: List[float], 
        distances: Optional[np.ndarray] = None, 
        num_of_facilities: int = 2, 
        linkage: str = "average"
    ) -> Tuple[List[float], List[float], List[int], float]:
        """
        Hierarchical clustering function for facility location
        
        Args:
            customer_df: Customer dataframe with lat/lon
            weight: Customer weights/demands
            distances: Precomputed distance matrix (optional)
            num_of_facilities: Number of facilities to locate
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            
        Returns:
            X: Facility latitudes
            Y: Facility longitudes  
            partition: Customer assignments
            total_cost: Total weighted distance
        """
        try:
            customer_df = customer_df.reset_index(drop=True)
        except:
            pass
            
        if distances is None:
            distances = self.compute_distance_matrix(customer_df)
            
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=num_of_facilities, 
            metric="precomputed", 
            linkage=linkage
        ).fit(distances)
        
        partition = clustering.labels_
        
        # Create clusters dictionary
        cluster = {i: [] for i in range(num_of_facilities)}
        for i, label in enumerate(partition):
            cluster[label].append(i)

        # Find center of each cluster (median approach)
        X, Y = [], []
        total_cost = 0
        
        for i in range(num_of_facilities):
            if len(cluster[i]) == 0:
                continue
                
            min_cost = float('inf')
            min_j = -1
            
            for j in cluster[i]:
                cost = 0
                for j2 in cluster[i]:
                    if j == j2:
                        continue
                    cost += distances[j][j2] * weight[j2]
                    
                if cost < min_cost:
                    min_cost = cost
                    min_j = j
                    
            total_cost += min_cost
            X.append(customer_df.iloc[min_j]['lat'])
            Y.append(customer_df.iloc[min_j]['lon'])
            
        return X, Y, partition.tolist(), total_cost

    def transportation(self, C: np.ndarray, capacity: int) -> Tuple[float, Dict]:
        """
        Solve transportation problem using network simplex
        Used in k-median optimization for upper bound improvement
        """
        M = capacity
        n, m = C.shape
        C = np.ceil(C)  # for network simplex
        
        G = nx.DiGraph()
        sum_ = 0
        
        # Add plant nodes
        for j in range(m):
            sum_ -= M
            G.add_node(f"plant{j}", demand=-M)
            
        # Add customer nodes
        for i in range(n):
            sum_ += 1
            G.add_node(i, demand=1)
            
        # Add dummy customer node
        G.add_node("dummy", demand=-sum_)
        
        # Add arcs with costs
        for i in range(n):
            for j in range(m):
                G.add_edge(f"plant{j}", i, weight=C[i, j])
                
        for j in range(m):
            G.add_edge(f"plant{j}", "dummy", weight=0)
            
        cost, flow = nx.flow.network_simplex(G)
        return cost, flow

    def find_median(self, C: np.ndarray, flow: Dict, n: int, m: int) -> float:
        """
        Find median centers for clusters to improve upper bound
        """
        total_cost = 0
        
        for j, f in enumerate(flow):
            if j >= m:
                break
                
            cluster = []
            for i in range(n):
                if flow[f][i] == 1:
                    cluster.append(i)
                    
            if len(cluster) > 1:
                nodes = np.array(cluster)
                subC = C[np.ix_(nodes, nodes)]
                best_location = nodes[subC.sum(axis=0).argmin()]
                cost = subC.sum(axis=0).min()
                total_cost += cost
                
        return total_cost

    def solve_k_median(
        self, 
        customer_df: pd.DataFrame, 
        weight: List[float], 
        cost: np.ndarray, 
        num_of_facilities: int,
        max_iter: int = 100, 
        max_lr: float = 0.01, 
        moms: Tuple[float, float] = (0.85, 0.95),
        convergence: float = 1e-5, 
        lr_find: bool = False, 
        adam: bool = False, 
        capacity: Optional[int] = None
    ) -> Tuple[List[float], List[float], List[int], float, List[float], List[float], List[float]]:
        """
        Solve k-median problem using Lagrange relaxation with advanced optimization techniques
        
        Args:
            customer_df: Customer dataframe
            weight: Customer weights
            cost: Cost matrix  
            num_of_facilities: Number of facilities
            max_iter: Maximum iterations
            max_lr: Maximum learning rate
            moms: Momentum bounds for fit-one-cycle
            convergence: Convergence tolerance
            lr_find: Whether to perform learning rate finding
            adam: Whether to use Adam optimizer
            capacity: Capacity constraint per facility
            
        Returns:
            X, Y: Facility coordinates
            partition: Customer assignments
            best_ub: Best upper bound (total cost)
            lb_list: Lower bound history
            ub_list: Upper bound history  
            phi_list: Learning rate history
        """
        m = num_of_facilities
        half_iter = max_iter // 2
        
        # Learning rate scheduling (fit-one-cycle)
        lrs = (max_lr/25., max_lr)
        lr_sche = np.concatenate([
            np.linspace(lrs[0], lrs[1], half_iter), 
            lrs[1]/2 + (lrs[1]/2) * np.cos(np.linspace(0, np.pi, half_iter))
        ])
        
        mom_sche = np.concatenate([
            np.linspace(moms[1], moms[0], half_iter),
            moms[1] - (moms[1]-moms[0])/2 - (moms[1]-moms[0])/2 * np.cos(np.linspace(0, np.pi, half_iter))
        ])

        if lr_find:
            phi = 1e-10
            report_iter = 1
        else:
            report_iter = 100

        n = len(cost)
        u = np.zeros(n)  # Lagrange multipliers
        w = np.array(weight)
        c = np.array(cost)
        C = c * w.reshape((n, 1))
        
        m_t = np.zeros(n)  # Momentum term
        best_ub = np.inf
        best_open = {}
        
        # Adam optimizer variables
        if adam:
            beta_2 = 0.999
            epsilon = 1e-8
            v_t = np.zeros(n)

        lb_list, ub_list, phi_list = [], [], []
        
        for t in range(max_iter):
            # Compute reduced costs
            Cbar = C - u.reshape((n, 1))
            xstar = np.zeros(n)
            ystar = np.zeros(n)
            
            # Solve subproblems
            if capacity is not None:
                # Capacity constrained case
                ystar = np.sort(np.where(Cbar > 0, 0., Cbar), axis=1)[:, :capacity].sum(axis=1)
            else:
                # Uncapacitated case
                ystar = np.where(Cbar < 0, Cbar, 0.).sum(axis=0)
                
            # Select facilities
            idx = np.argsort(ystar)
            open_node = set(idx[:m])
            
            # Compute lower bound
            lb = u.sum() + ystar[idx[:m]].sum()
            lb_list.append(lb)

            # Compute customer assignments
            xstar = np.where(Cbar < 0, 1, 0)[:, idx[:m]].sum(axis=1)

            # Compute upper bound
            if capacity is not None:
                if t % report_iter == 0:
                    ub_cost, flow = self.transportation(C[:, idx[:m]], capacity)
                    median_cost = self.find_median(C, flow, n, m)
                    ub = min(ub_cost, median_cost)
            else:
                ub = C[:, np.array(list(open_node))].min(axis=1).sum()
            
            if ub < best_ub:
                best_ub = ub
                best_open = open_node.copy()
            ub_list.append(best_ub)

            # Compute subgradient
            g_t = 1. - xstar
            norm = np.dot(g_t, g_t)
            
            # Check convergence
            if lb > convergence:
                gap = (best_ub - lb) / lb
                if gap <= convergence:
                    print(f"Converged: gap={gap:.6f}, ub={best_ub:.2f}, lb={lb:.2f}")
                    break
            else:
                gap = 10.
                
            if t % report_iter == 0:
                print(f"Iter {t}: gap={gap:.3f}, lb={lb:.2f}, ub={best_ub:.2f}, norm={norm:.2f}")
                
            # Check for early termination conditions
            if norm <= convergence and capacity is None:
                print("Subgradient norm converged!")
                break
                
            if lb < -1e5 and lr_find:
                break
                
            # Update learning parameters
            if lr_find:
                phi *= 2.
                beta_1 = moms[1]
            else:
                phi = lr_sche[t]
                beta_1 = mom_sche[t]
                
            phi_list.append(phi)
            
            # Update momentum
            m_t = beta_1 * m_t + (1 - beta_1) * g_t
            
            # Update Lagrange multipliers
            if adam:
                v_t = beta_2 * v_t + (1 - beta_2) * (g_t ** 2)
                m_cap = m_t / (1 - beta_1 ** (t + 1))
                v_cap = v_t / (1 - beta_2 ** (t + 1))
                u = u + (phi * m_cap) / (np.sqrt(v_cap) + epsilon)
            else:
                alpha = (1.05 * best_ub - lb) / norm if norm > 0 else 0
                u = u + phi * alpha * m_t

        # Construct solution
        X, Y = [], []
        for i in best_open:
            row = customer_df.iloc[i]
            X.append(row['lat'])
            Y.append(row['lon'])
            
        # Create facility index mapping
        facility_index = {}
        for idx, i in enumerate(best_open):
            facility_index[i] = idx
            
        # Assign customers to facilities
        partition = np.zeros(n, int)
        for i in range(n):
            min_cost = np.inf
            for j in best_open:
                if c[i, j] < min_cost:
                    min_cost = c[i, j]
                    partition[i] = facility_index[j]
                    
        return X, Y, partition.tolist(), best_ub, lb_list, ub_list, phi_list

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about available algorithms
        """
        return {
            "hierarchical_clustering": {
                "description": "Hierarchical clustering for facility location",
                "linkage_methods": ["ward", "complete", "average", "single"],
                "features": [
                    "Real distance matrix support",
                    "Multiple linkage criteria",
                    "Median-based facility positioning"
                ]
            },
            "k_median": {
                "description": "Advanced k-median optimization with Lagrange relaxation",
                "features": [
                    "Fit-one-cycle learning rate scheduling",
                    "Adam optimizer integration", 
                    "Learning rate finder",
                    "Capacity constraints support",
                    "Advanced convergence monitoring"
                ]
            }
        }