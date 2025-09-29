import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, KMeans
import plotly.graph_objects as go
from geopy.distance import great_circle as distance
from openpyxl import Workbook
import networkx as nx
import ast
import warnings
warnings.filterwarnings('ignore')

class CustomerAggregationService:
    """
    Customer aggregation service for scalable logistics network optimization.
    Implements hierarchical clustering with demand weights and road distances.
    """
    
    def __init__(self, osrm_host: str = "localhost"):
        self.osrm_host = osrm_host
        self.mapbox_access_token = None  # Will be configurable
        
    def make_total_demand(self, demand_df: pd.DataFrame, 
                         time_col: str = 'time', 
                         demand_col: str = 'demand',
                         customer_col: str = 'customer') -> pd.DataFrame:
        """
        Calculate annual/planning period total demand from time-series demand data
        Matches the notebook's make_total_demand function exactly
        
        Args:
            demand_df: DataFrame with time-series demand data
            time_col: Column name for time periods
            demand_col: Column name for demand values  
            customer_col: Column name for customer identifiers
            
        Returns:
            DataFrame with aggregated demand by customer
        """
        try:
            # Group by customer and sum demands across all time periods
            total_demand = demand_df.groupby(customer_col)[demand_col].sum().reset_index()
            total_demand = total_demand.rename(columns={demand_col: 'total_demand'})
            
            # Add any missing customers with zero demand
            if 'name' in demand_df.columns and customer_col != 'name':
                customer_names = demand_df.groupby(customer_col)['name'].first().reset_index()
                total_demand = total_demand.merge(customer_names, on=customer_col, how='left')
            
            return total_demand
            
        except Exception as e:
            print(f"Error in make_total_demand: {str(e)}")
            # Fallback: assume demand_df already has aggregated demand
            if demand_col in demand_df.columns:
                return demand_df[[customer_col, demand_col]].rename(columns={demand_col: 'total_demand'})
            else:
                raise ValueError(f"Cannot process demand data: {str(e)}")
        
    def compute_durations(self, cust_df: pd.DataFrame, plnt_df: Optional[pd.DataFrame] = None, 
                         toll: bool = True) -> Tuple[List[List[float]], List[List[float]], pd.DataFrame]:
        """
        Compute road distances and durations using OSRM routing engine
        
        Args:
            cust_df: Customer DataFrame with columns ['name', 'lat', 'lon']
            plnt_df: Plant DataFrame (optional)
            toll: Whether to include toll roads
            
        Returns:
            Tuple: (durations, distances, node_df)
        """
        if plnt_df is not None:
            node_df = pd.concat([
                cust_df[["name", "lat", "lon"]], 
                plnt_df[["name", "lat", "lon"]]
            ])
        else:
            node_df = cust_df.copy()
        
        n = len(node_df)
        ROUTE = []
        for row in node_df.itertuples():
            ROUTE.append([row.lat, row.lon])
        
        # Build route string for OSRM API
        route_str = ""
        for (i, j) in ROUTE:
            route_str += f"{j},{i};"
        
        try:
            # Make API request to OSRM
            if toll:
                url = f'http://{self.osrm_host}:5000/table/v1/driving/{route_str[:-1]}?annotations=distance,duration'
            else:
                url = f'http://{self.osrm_host}:5000/table/v1/driving/{route_str[:-1]}?annotations=distance,duration&exclude=toll'
            
            response = requests.get(url, timeout=30)
            result = response.json()
            
            durations = result["durations"]
            distances = result["distances"]
            
        except (requests.RequestException, KeyError) as e:
            # Fallback to great circle distances if OSRM is not available
            print(f"OSRM not available, using great circle distances: {e}")
            durations = []
            distances = []
            
            for i in range(n):
                duration_row = []
                distance_row = []
                for j in range(n):
                    if i == j:
                        duration_row.append(0)
                        distance_row.append(0)
                    else:
                        lat1, lon1 = ROUTE[i]
                        lat2, lon2 = ROUTE[j]
                        dist_km = distance((lat1, lon1), (lat2, lon2)).kilometers
                        distance_row.append(dist_km * 1000)  # Convert to meters
                        duration_row.append(dist_km * 60)    # Assume 60 km/h average speed
                durations.append(duration_row)
                distances.append(distance_row)
        
        # Handle missing values
        for i in range(n):
            for j in range(n):
                if durations[i][j] is None or durations[i][j] > 3600*24:
                    durations[i][j] = 3600*24  # 24 hours max
                    distances[i][j] = 1000000  # 1000 km max
                    
        return durations, distances, node_df
        
    def hierarchical_clustering(self, cust_df: pd.DataFrame, weight: List[float], 
                              durations: List[List[float]], num_of_facilities: int = 2, 
                              linkage: str = "average") -> Tuple[List[float], List[float], List[int], float]:
        """
        Perform hierarchical clustering with demand weights
        
        Args:
            cust_df: Customer DataFrame
            weight: Customer weights (demand volumes)
            durations: Duration matrix between customers
            num_of_facilities: Number of clusters/facilities
            linkage: Linkage criterion for clustering
            
        Returns:
            Tuple: (X, Y, partition, total_cost)
        """
        try:
            cust_df = cust_df.reset_index(drop=True)
        except:
            pass
            
        # Convert durations to numpy array
        duration_matrix = np.array(durations)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=num_of_facilities, 
            metric="precomputed", 
            linkage=linkage
        ).fit(duration_matrix)
        
        partition = clustering.labels_

        # Create clusters dictionary
        cluster = {i: [] for i in range(num_of_facilities)}
        for i in range(len(cust_df)):
            cluster[partition[i]].append(i)

        # Find center of each cluster (medoid approach)
        X, Y = [], []
        total_cost = 0
        
        for i in range(num_of_facilities):
            if len(cluster[i]) == 0:
                continue
                
            min_cost = float('inf')
            min_j = -1
            
            # Find customer that minimizes weighted distance to all others in cluster
            for j in cluster[i]:
                cost = 0
                for j2 in cluster[i]:
                    if j == j2:
                        continue
                    cost += durations[j][j2] * weight[j2]
                    
                if cost < min_cost:
                    min_cost = cost
                    min_j = j
                    
            total_cost += min_cost
            X.append(float(cust_df.iloc[min_j]['lat']))
            Y.append(float(cust_df.iloc[min_j]['lon']))
            
        return X, Y, partition.tolist(), total_cost
        
    def make_aggregated_cust_df(self, cust_df: pd.DataFrame, X: List[float], Y: List[float], 
                               partition: List[int], weight: List[float]) -> pd.DataFrame:
        """
        Create aggregated customer DataFrame from clustering results
        
        Args:
            cust_df: Original customer DataFrame
            X: Facility latitudes
            Y: Facility longitudes
            partition: Customer cluster assignments
            weight: Customer weights
            
        Returns:
            DataFrame: Aggregated customer data
        """
        cluster = {i: [] for i in range(len(X))}
        for i in range(len(cust_df)):
            cluster[partition[i]].append(i)

        cluster_list, total_demand = [], []
        name_list = []
        
        for i in range(len(X)):
            name_list.append(f"agg_cust_{i}")
            cluster_list.append(cluster[i])
            total = sum(weight[j] for j in cluster[i])
            total_demand.append(total)

        aggregated_cust_df = pd.DataFrame({
            "name": name_list, 
            "lat": X, 
            "lon": Y, 
            "customers": cluster_list, 
            "weight": total_demand
        })
        
        return aggregated_cust_df
        
    def show_optimized_continuous_network(self, cust_df: pd.DataFrame, X: List[float], Y: List[float], 
                                        partition: List[int], weight: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Create visualization data for customer aggregation results
        
        Args:
            cust_df: Customer DataFrame
            X: Facility latitudes
            Y: Facility longitudes
            partition: Customer assignments
            weight: Customer weights (optional)
            
        Returns:
            Dict: Plotly figure data for visualization
        """
        if weight is None:
            weight = [10] * len(cust_df)
            facility_weights = [20] * len(X)
        else:
            weight = np.array(weight)
            max_weight = weight.max()
            weight = 10 + weight/(max_weight+1)*30
            
            # Calculate facility weights
            facility_weights = np.zeros(len(X))
            for i in range(len(cust_df)):
                j = partition[i]
                facility_weights[j] += weight[i]
            max_facility_weight = facility_weights.max()
            facility_weights = 10 + facility_weights/(max_facility_weight+1)*40
        
        # Create edge traces for connections
        edge_trace_lat, edge_trace_lng = [], []
        x, y = cust_df['lat'].values, cust_df['lon'].values
        
        for i in range(len(cust_df)):
            j = partition[i]
            edge_trace_lat += [x[i], X[j], None]
            edge_trace_lng += [y[i], Y[j], None]

        # Prepare data for Plotly
        data = [
            {
                'type': 'scattermapbox',
                'lat': cust_df['lat'].tolist(),
                'lon': cust_df['lon'].tolist(),
                'mode': 'markers',
                'marker': {
                    'size': weight.tolist() if hasattr(weight, 'tolist') else weight,
                    'color': 'pink',
                    'opacity': 0.5
                },
                'text': cust_df['name'].tolist(),
                'name': 'Customers'
            },
            {
                'type': 'scattermapbox',
                'lat': X,
                'lon': Y,
                'mode': 'markers',
                'marker': {
                    'size': facility_weights.tolist() if hasattr(facility_weights, 'tolist') else facility_weights,
                    'color': 'red',
                    'opacity': 0.7
                },
                'name': 'Aggregated Facilities'
            },
            {
                'type': 'scattermapbox',
                'lat': edge_trace_lat,
                'lon': edge_trace_lng,
                'mode': 'lines',
                'line': {'width': 0.5, 'color': 'yellow'},
                'hoverinfo': 'none',
                'name': 'Connections'
            }
        ]

        layout = {
            'autosize': True,
            'hovermode': 'closest',
            'mapbox': {
                'bearing': 0,
                'center': {
                    'lat': np.mean(cust_df['lat']),
                    'lon': np.mean(cust_df['lon'])
                },
                'pitch': 0,
                'zoom': 8,
                'style': 'open-street-map'  # Use free OpenStreetMap style
            },
            'showlegend': True
        }

        return {'data': data, 'layout': layout}
        
    def customer_aggregation(self, cust_df: pd.DataFrame, prod_df: pd.DataFrame, 
                           demand_df: pd.DataFrame, num_of_facilities: int, 
                           linkage: str = "complete", toll: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
        """
        Main customer aggregation function
        
        Args:
            cust_df: Customer DataFrame
            prod_df: Product DataFrame
            demand_df: Demand DataFrame (customers x products)
            num_of_facilities: Number of aggregated facilities
            linkage: Hierarchical clustering linkage method
            toll: Whether to use toll roads in routing
            
        Returns:
            Tuple: (aggregated_cust_df, visualization_figure, aggregated_demand)
        """
        # Calculate weights for aggregation based on product weights and demand
        weight_of_prod = prod_df.iloc[:, 1].values  # Product weights
        demand_values = demand_df.iloc[:, 1:].values  # Customer demand matrix
        
        n_cust = len(cust_df)
        n_prod = len(prod_df)
        weight = np.zeros(n_cust)
        
        for i in range(n_cust):
            for p in range(n_prod):
                weight[i] += demand_values[i][p] * weight_of_prod[p]
        
        # Calculate road distance and time
        durations, distances, node_df = self.compute_durations(cust_df, toll=toll)
        
        # Perform hierarchical clustering
        X, Y, partition, cost = self.hierarchical_clustering(
            cust_df, weight.tolist(), durations, num_of_facilities, linkage=linkage
        )
        
        # Create visualization
        fig_data = self.show_optimized_continuous_network(cust_df, X, Y, partition, weight.tolist())
        
        # Create aggregated customer DataFrame
        aggregated_cust_df = self.make_aggregated_cust_df(cust_df, X, Y, partition, weight.tolist())
        
        # Aggregate demand
        dem_agg = np.zeros(shape=(num_of_facilities, n_prod))
        for i in range(num_of_facilities):
            customers_in_cluster = aggregated_cust_df.iloc[i]['customers']
            for c in customers_in_cluster:
                for p in range(n_prod):
                    dem_agg[i, p] += demand_values[c, p]
        
        return aggregated_cust_df, fig_data, dem_agg
        
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about customer aggregation service capabilities
        
        Returns:
            Dict: Service information
        """
        return {
            "customer_aggregation": {
                "description": "Scalable customer clustering for large-scale logistics optimization",
                "features": [
                    "Demand-weighted hierarchical clustering",
                    "Road distance-based clustering with OSRM integration",
                    "Multiple linkage methods support",
                    "Automatic demand aggregation across clusters",
                    "Interactive visualization with Plotly",
                    "Medoid-based facility positioning"
                ],
                "linkage_methods": ["ward", "complete", "average", "single"],
                "clustering_algorithm": "AgglomerativeClustering with precomputed distances"
            },
            "osrm_integration": {
                "host": self.osrm_host,
                "features": [
                    "Real road distance calculations",
                    "Duration estimates with traffic considerations", 
                    "Toll road inclusion/exclusion options",
                    "Fallback to great circle distances"
                ]
            }
        }

    def kmeans(self, cust_df: pd.DataFrame, weight: List[float], num_of_facilities: int = 1, 
               batch: bool = True) -> Tuple[List[float], List[float], np.ndarray, float]:
        """
        K-means clustering for customer aggregation
        Implements the notebook kmeans function exactly
        
        Args:
            cust_df: Customer DataFrame with lat/lon
            weight: Customer weights (demand volumes)
            num_of_facilities: Number of clusters
            batch: Use MiniBatchKMeans if True, regular KMeans if False
            
        Returns:
            Tuple: (X, Y, partition, cost)
        """
        if batch:
            model = MiniBatchKMeans(n_clusters=num_of_facilities, random_state=42)
        else:
            model = KMeans(n_clusters=num_of_facilities, random_state=42)
        
        model.fit(cust_df[["lat", "lon"]], sample_weight=weight)
        partition = model.labels_
        
        cluster = {i: [] for i in range(num_of_facilities)}
        for i, row in enumerate(cust_df.itertuples()):
            cluster[model.labels_[i]].append(i)

        # Find weighted center of each cluster
        X, Y = [], [] 
        for i in range(num_of_facilities):
            if len(cluster[i]) == 0:
                continue
            lat_sum, lon_sum = 0., 0.
            total_weight = 0.
            for j in cluster[i]:
                lat_sum += cust_df.iloc[j]['lat'] * weight[j]
                lon_sum += cust_df.iloc[j]['lon'] * weight[j]
                total_weight += weight[j]
            X.append(lat_sum / total_weight)
            Y.append(lon_sum / total_weight)
            
        # Calculate total cost using great circle distance
        cost = 0.
        x, y = cust_df['lat'].values, cust_df['lon'].values
        for j in range(len(X)):
            for i in cluster[j]:
                d = distance((x[i], y[i]), (X[j], Y[j])).km
                cost += d * weight[i]    
        
        return X, Y, partition, cost

    def solve_k_median(self, cust_df: pd.DataFrame, weight: List[float], cost: np.ndarray, 
                      num_of_facilities: int, max_iter: int = 100, max_lr: float = 0.01, 
                      moms: Tuple[float, float] = (0.85, 0.95), convergence: float = 1e-5, 
                      lr_find: bool = False, adam: bool = False, 
                      capacity: Optional[float] = None) -> Tuple[List[float], List[float], np.ndarray, float, List[float], List[float], List[float]]:
        """
        K-median problem solver using Lagrange relaxation with fit-one-cycle method
        Implements the notebook solve_k_median function exactly
        
        Args:
            cust_df: Customer DataFrame
            weight: Customer demand weights
            cost: Cost matrix (typically road distances/durations)
            num_of_facilities: Number of facilities to select
            max_iter: Maximum iterations
            max_lr: Maximum learning rate
            moms: Momentum bounds for fit-one-cycle
            convergence: Convergence tolerance
            lr_find: Learning rate finder mode
            adam: Use Adam optimizer
            capacity: Facility capacity constraint
            
        Returns:
            Tuple: (X, Y, partition, best_ub, lb_list, ub_list, phi_list)
        """
        import math
        
        m = num_of_facilities 
        half_iter = max_iter // 2
        lrs = (max_lr / 25., max_lr)
        lr_sche = np.concatenate([
            np.linspace(lrs[0], lrs[1], half_iter), 
            lrs[1]/2 + (lrs[1]/2) * np.cos(np.linspace(0, np.pi, half_iter))
        ])
        mom_sche = np.concatenate([
            np.linspace(moms[1], moms[0], half_iter), 
            moms[1] - (moms[1] - moms[0])/2 - (moms[1] - moms[0])/2 * np.cos(np.linspace(0, np.pi, half_iter))
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
        
        m_t = np.zeros(n)
        best_ub = np.inf 
        best_open = {}
        
        # Adam parameters
        if adam:
            beta_2 = 0.999
            epsilon = 1e-8
            m_t = np.zeros(n)
            v_t = np.zeros(n)

        lb_list, ub_list, phi_list = [], [], []
        
        for t in range(max_iter):
            Cbar = C - u.reshape((n, 1))  # Reduced costs
            xstar = np.zeros(n)
            ystar = np.zeros(n)
            
            # Solve 0-1 knapsack problem (with capacity constraints)
            if capacity is not None:
                ystar = np.sort(np.where(Cbar > 0, 0., Cbar), axis=1)[:, :int(capacity)].sum(axis=1)
            else:    
                ystar = np.where(Cbar < 0, Cbar, 0.).sum(axis=0)
                
            idx = np.argsort(ystar)  # Sort in ascending order
            open_node = set(idx[:m])  # Select m facilities with smallest values
            
            # Calculate lower bound
            lb = u.sum() + ystar[idx[:m]].sum()
            lb_list.append(lb)

            xstar = np.where(Cbar < 0, 1, 0)[:, idx[:m]].sum(axis=1)

            # Calculate upper bound (solve assignment problem if capacity constrained)
            if capacity is not None:
                if t % report_iter == 0:
                    cost_val, flow = self._transportation(C[:, idx[:m]], capacity)
                    cost2 = self._find_median(C, flow, n, m)
                    ub = min(cost_val, cost2)
            else:
                ub = C[:, np.array(list(open_node))].min(axis=1).sum()
            
            if ub < best_ub:
                best_ub = ub
                best_open = open_node.copy()
            ub_list.append(best_ub)

            g_t = 1. - xstar  # Subgradient calculation
            norm = np.dot(g_t, g_t)  # Norm calculation
            
            if lb > convergence:
                gap = (best_ub - lb) / lb
                if gap <= convergence:
                    print("gap=", gap, best_ub, lb)
                    break
            else:
                gap = 10.
                
            if t % report_iter == 0:
                print(f"{t}: {gap:.3f}, {lb:.5f}, {best_ub:.5f}, {norm:.2f}")
                
            if norm <= convergence and capacity is None: 
                print("norm is 0!")
                break    
                
            if lb < -1e5:
                if lr_find:
                    break
                    
            if lr_find:
                phi *= 2. 
                beta_1 = moms[1]
            else:
                # Fit one cycle 
                phi = lr_sche[t]
                beta_1 = mom_sche[t] 
                
            phi_list.append(phi)
                
            # Update moving average
            m_t = beta_1 * m_t + (1 - beta_1) * g_t
            
            if adam:
                # Update second moment estimate
                v_t = beta_2 * v_t + (1 - beta_2) * (g_t ** 2)
                m_cap = m_t / (1 - beta_1 ** (t + 1))
                v_cap = v_t / (1 - beta_2 ** (t + 1))
                u = u + (phi * m_cap) / (np.sqrt(v_cap) + epsilon)
            else:
                alpha = (1.05 * best_ub - lb) / norm 
                u = u + phi * alpha * m_t
        
        # Construct solution
        X, Y = [], []
        for i in best_open:
            row = cust_df.iloc[i]
            X.append(row.lat)
            Y.append(row.lon)
            
        facility_index = {}
        for idx, i in enumerate(best_open):
            facility_index[i] = idx 
            
        partition = np.zeros(n, int)
        for i in range(n):
            min_cost = np.inf
            for j in best_open:
                if c[i, j] < min_cost:
                    min_cost = c[i, j] 
                    partition[i] = facility_index[j]
                
        return X, Y, partition, best_ub, lb_list, ub_list, phi_list

    def _transportation(self, C: np.ndarray, capacity: float) -> Tuple[float, Dict]:
        """
        Solve transportation problem with unit demands using network simplex
        
        Args:
            C: Cost matrix (n x m)
            capacity: Plant capacity (assumed uniform)
            
        Returns:
            Tuple: (cost, flow)
        """
        M = int(capacity)
        n, m = C.shape
        C = np.ceil(C)  # For network simplex 
        G = nx.DiGraph()
        sum_ = 0
        
        for j in range(m):
            sum_ -= M
            G.add_node(f"plant{j}", demand=-M)
            
        for i in range(n):
            sum_ += 1 
            G.add_node(i, demand=1)
            
        # Add dummy customer with demand sum_
        G.add_node("dummy", demand=-sum_)
        
        for i in range(n):
            for j in range(m):
                G.add_edge(f"plant{j}", i, weight=C[i, j])
                
        for j in range(m):
            G.add_edge(f"plant{j}", "dummy", weight=0)
            
        cost, flow = nx.flow.network_simplex(G)
        return cost, flow

    def _find_median(self, C: np.ndarray, flow: Dict, n: int, m: int) -> float:
        """
        Find cluster medians to improve upper bound
        
        Args:
            C: Cost matrix
            flow: Flow solution from transportation
            n: Number of customers
            m: Number of facilities
            
        Returns:
            float: Total cost
        """
        total_cost = 0
        for j, f in enumerate(flow):
            if j >= m:
                break
            cluster = []
            for i in range(n):
                if f in flow and i in flow[f] and flow[f][i] == 1:
                    cluster.append(i)
            if len(cluster) > 1:
                # Find the best location for the cluster
                nodes = np.array(cluster)
                subC = C[np.ix_(nodes, nodes)]
                best_location = nodes[subC.sum(axis=0).argmin()]
                cost = subC.sum(axis=0).min()
                total_cost += cost
        return total_cost

    def make_aggregated_df(self, cust_df: pd.DataFrame, demand_df: pd.DataFrame, 
                          total_demand_df: pd.DataFrame, X: List[float], Y: List[float], 
                          partition: np.ndarray, weight: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create aggregated dataframes for customers and demand
        Implements the notebook make_aggregated_df function exactly
        
        Args:
            cust_df: Original customer DataFrame
            demand_df: Time-series demand DataFrame (unused in current implementation)
            total_demand_df: Aggregated demand by customer and product
            X: Facility latitudes
            Y: Facility longitudes
            partition: Customer cluster assignments
            weight: Customer weights
            
        Returns:
            Tuple: (aggregated_cust_df, aggregated_total_demand_df)
        """
        cluster = {i: [] for i in range(len(X))}
        for i, row in enumerate(cust_df.itertuples()):
            cluster[partition[i]].append(i)

        cluster_list, total_demand = [], [] 
        name_list = []
        for i in range(len(X)):
            name_list.append(f"cust{i}")
            cluster_list.append(cluster[i])
            total = 0
            for j in cluster[i]:
                total += weight[j]
            total_demand.append(total)

        aggregated_cust_df = pd.DataFrame({
            "name": name_list,
            "lat": X, 
            "lon": Y, 
            "customers": cluster_list, 
            "demand": total_demand
        })
        
        # Create customer index mapping
        cust_idx = {}
        for i, row in enumerate(cust_df.itertuples()):
            cust_idx[str(row.name)] = i    

        # Aggregate demand by cluster
        cluster_idx = []
        try:
            total_demand_df.reset_index(inplace=True)
        except ValueError:
            pass
            
        for row in total_demand_df.cust:
            cluster_idx.append("cust" + str(partition[cust_idx[str(row)]]))

        total_demand_df.loc[:, "cluster_idx"] = cluster_idx
        aggregated_total_demand_df = pd.pivot_table(
            total_demand_df, 
            index=["prod", "cluster_idx"], 
            values="demand", 
            aggfunc=sum
        )
        aggregated_total_demand_df.reset_index(inplace=True)
        aggregated_total_demand_df.rename(columns={"cluster_idx": "cust"}, inplace=True)

        return aggregated_cust_df, aggregated_total_demand_df

    def aggregate_demand_by_cluster(self, cust_df: pd.DataFrame, aggregated_cust_df: pd.DataFrame, 
                                   demand_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate demand data by customer clusters
        Implements the notebook aggregate_demand_by_cluster function exactly
        
        Args:
            cust_df: Original customer DataFrame
            aggregated_cust_df: Aggregated customer DataFrame with cluster info
            demand_df: Original demand DataFrame
            
        Returns:
            pd.DataFrame: Aggregated demand by cluster and product
        """
        # Create partition mapping from aggregated customer data
        partition = np.zeros(len(cust_df), int)
        for j, row in enumerate(aggregated_cust_df.itertuples()):
            customers_in_cluster = ast.literal_eval(str(row.customers))
            for i in customers_in_cluster:
                partition[i] = j

        # Create customer index mapping
        cust_idx = {}
        for i, row in enumerate(cust_df.itertuples()):
            cust_idx[row.name] = i    

        # Assign cluster indices to demand data
        cluster_idx = []
        try:
            demand_df.reset_index(inplace=True)
        except ValueError:
            pass
            
        for row in demand_df.cust:
            cluster_idx.append("cust" + str(partition[cust_idx[row]]))
            
        demand_df.loc[:, "cluster_idx"] = cluster_idx
        
        # Aggregate demand by cluster
        aggregated_cluster_df = pd.pivot_table(
            demand_df, 
            index=["date", "prod", "cluster_idx"], 
            values="demand", 
            aggfunc=sum
        )
        aggregated_cluster_df.reset_index(inplace=True)
        aggregated_cluster_df.rename(columns={"cluster_idx": "cust"}, inplace=True)
        
        return aggregated_cluster_df