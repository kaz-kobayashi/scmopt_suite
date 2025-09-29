import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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

def weiszfeld_algorithm(points: List[Tuple[float, float]], 
                       weights: Optional[List[float]] = None,
                       max_iterations: int = 1000,
                       tolerance: float = 1e-6) -> Tuple[float, float]:
    """
    Weiszfeld algorithm to find geometric median (Fermat point) - Single facility version
    
    Args:
        points: List of (x, y) coordinates
        weights: Optional weights for each point
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
    
    Returns:
        Tuple of (x, y) coordinates of the geometric median
    """
    if not points:
        raise ValueError("Points list cannot be empty")
    
    n = len(points)
    if weights is None:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError("Weights length must match points length")
    
    # Convert to numpy arrays for easier computation
    points_array = np.array(points)
    weights_array = np.array(weights)
    
    # Initialize with weighted centroid
    x_current = np.sum(points_array[:, 0] * weights_array) / np.sum(weights_array)
    y_current = np.sum(points_array[:, 1] * weights_array) / np.sum(weights_array)
    
    for iteration in range(max_iterations):
        # Calculate distances from current point to all points
        distances = np.sqrt((points_array[:, 0] - x_current)**2 + 
                           (points_array[:, 1] - y_current)**2)
        
        # Avoid division by zero
        non_zero_distances = distances > tolerance
        
        if not np.any(non_zero_distances):
            # Current point is exactly on one of the input points
            break
        
        # Calculate weights for Weiszfeld iteration
        weiszfeld_weights = weights_array[non_zero_distances] / distances[non_zero_distances]
        filtered_points = points_array[non_zero_distances]
        
        # Update position
        total_weight = np.sum(weiszfeld_weights)
        x_new = np.sum(filtered_points[:, 0] * weiszfeld_weights) / total_weight
        y_new = np.sum(filtered_points[:, 1] * weiszfeld_weights) / total_weight
        
        # Check convergence
        if abs(x_new - x_current) < tolerance and abs(y_new - y_current) < tolerance:
            break
        
        x_current, y_current = x_new, y_new
    
    return (x_current, y_current)


def multi_facility_weiszfeld(customer_df: pd.DataFrame, 
                           num_facilities: int,
                           lat_col: str = 'lat',
                           lon_col: str = 'lon', 
                           demand_col: str = 'demand',
                           max_iterations: int = 1000,
                           tolerance: float = 1e-4,
                           random_state: Optional[int] = None,
                           initial_locations: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Multi-facility Weiszfeld algorithm for weighted geometric median problem
    
    This algorithm extends the single-facility Weiszfeld algorithm to multiple facilities
    by iteratively assigning customers to nearest facilities and updating facility locations.
    
    Args:
        customer_df: DataFrame with customer data
        num_facilities: Number of facilities to locate
        lat_col: Column name for latitude
        lon_col: Column name for longitude  
        demand_col: Column name for demand/weights
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        random_state: Random seed for reproducibility
        initial_locations: Optional initial facility locations
    
    Returns:
        Dictionary with facility locations, assignments, and convergence info
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n = len(customer_df)  # Number of customers
    k = num_facilities    # Number of facilities
    
    if k > n:
        raise ValueError(f"Number of facilities ({k}) cannot exceed number of customers ({n})")
    
    # Extract customer data
    x = customer_df[lat_col].values
    y = customer_df[lon_col].values
    
    # Extract weights (demand)
    if demand_col in customer_df.columns:
        weights = customer_df[demand_col].values
        weights = np.where(pd.isna(weights), 1.0, weights)  # Replace NaN with 1.0
    else:
        weights = np.ones(n)
    
    # Initialize facility locations
    if initial_locations is not None:
        if len(initial_locations) != k:
            raise ValueError(f"Number of initial locations ({len(initial_locations)}) must equal num_facilities ({k})")
        X = np.array([loc[0] for loc in initial_locations])
        Y = np.array([loc[1] for loc in initial_locations])
    else:
        # Random initialization from customer locations
        perm = np.random.permutation(n)
        X = x[perm[:k]].copy()
        Y = y[perm[:k]].copy()
    
    # Store previous positions for convergence check
    prev_X = X.copy()
    prev_Y = Y.copy()
    
    # Main iteration loop
    iteration_count = 0
    for main_iter in range(max_iterations):
        iteration_count += 1
        
        # Step 1: Assign customers to nearest facilities
        distances = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                distances[i, j] = great_circle_distance(x[i], y[i], X[j], Y[j])
        
        # Find nearest facility for each customer
        assignments = distances.argmin(axis=1)
        
        # Create customer lists for each facility
        customer_lists = [[] for _ in range(k)]
        for i in range(n):
            customer_lists[assignments[i]].append(i)
        
        # Step 2: Update facility locations using weighted centroids (initialization)
        for j in range(k):
            if customer_lists[j]:  # If facility j has assigned customers
                customer_indices = customer_lists[j]
                total_weight = weights[customer_indices].sum()
                if total_weight > 0:
                    X[j] = (weights[customer_indices] * x[customer_indices]).sum() / total_weight
                    Y[j] = (weights[customer_indices] * y[customer_indices]).sum() / total_weight
        
        # Step 3: Weiszfeld refinement for each facility
        for weiszfeld_iter in range(max_iterations):
            new_X = X.copy()
            new_Y = Y.copy()
            
            for j in range(k):
                if not customer_lists[j]:  # Skip facilities with no customers
                    continue
                    
                customer_indices = customer_lists[j]
                sum_x = sum_y = sum_weights = 0.0
                
                for i in customer_indices:
                    dist = great_circle_distance(x[i], y[i], X[j], Y[j])
                    if dist > tolerance:  # Avoid division by zero
                        weight_over_dist = weights[i] / dist
                        sum_x += weight_over_dist * x[i]
                        sum_y += weight_over_dist * y[i]
                        sum_weights += weight_over_dist
                    else:
                        # Customer is at facility location - no update needed
                        new_X[j] = x[i]
                        new_Y[j] = y[i]
                        break
                
                if sum_weights > 0:
                    new_X[j] = sum_x / sum_weights
                    new_Y[j] = sum_y / sum_weights
            
            # Check Weiszfeld convergence
            weiszfeld_error = 0.0
            for j in range(k):
                weiszfeld_error += great_circle_distance(X[j], Y[j], new_X[j], new_Y[j])
                
            if weiszfeld_error <= tolerance:
                break
                
            X = new_X
            Y = new_Y
        
        # Check main loop convergence
        main_error = 0.0
        for j in range(k):
            main_error += great_circle_distance(X[j], Y[j], prev_X[j], prev_Y[j])
            
        if main_error <= tolerance:
            break
            
        prev_X = X.copy()
        prev_Y = Y.copy()
    
    # Calculate final assignments and costs
    final_distances = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            final_distances[i, j] = great_circle_distance(x[i], y[i], X[j], Y[j])
    
    final_assignments = final_distances.argmin(axis=1)
    total_cost = 0.0
    
    for i in range(n):
        assigned_facility = final_assignments[i]
        distance = final_distances[i, assigned_facility]
        total_cost += weights[i] * distance
    
    # Calculate facility statistics
    facility_stats = []
    for j in range(k):
        customers_assigned = [i for i in range(n) if final_assignments[i] == j]
        
        if customers_assigned:
            total_demand_served = weights[customers_assigned].sum()
            distances_to_facility = [final_distances[i, j] for i in customers_assigned]
            avg_distance = np.mean(distances_to_facility)
        else:
            total_demand_served = 0.0
            avg_distance = 0.0
        
        facility_stats.append({
            'facility_index': j,
            'location': (float(X[j]), float(Y[j])),
            'customers_assigned': len(customers_assigned),
            'total_demand_served': float(total_demand_served),
            'average_distance': float(avg_distance)
        })
    
    return {
        'facility_locations': [(float(X[j]), float(Y[j])) for j in range(k)],
        'assignments': final_assignments.tolist(),
        'total_cost': float(total_cost),
        'facility_stats': facility_stats,
        'algorithm': 'multi_facility_weiszfeld',
        'iterations': iteration_count,
        'converged': main_error <= tolerance,
        'parameters': {
            'num_facilities': num_facilities,
            'max_iterations': max_iterations,
            'tolerance': tolerance,
            'random_state': random_state
        }
    }


def repeated_multi_facility_weiszfeld(customer_df: pd.DataFrame,
                                    num_facilities: int,
                                    num_runs: int = 10,
                                    lat_col: str = 'lat',
                                    lon_col: str = 'lon',
                                    demand_col: str = 'demand',
                                    max_iterations: int = 1000,
                                    tolerance: float = 1e-4,
                                    base_random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Repeated multi-facility Weiszfeld algorithm with multiple random initializations
    
    This algorithm runs the multi-facility Weiszfeld algorithm multiple times with
    different random initializations to find the global optimum solution.
    
    Args:
        customer_df: DataFrame with customer data
        num_facilities: Number of facilities to locate
        num_runs: Number of runs with different initializations
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        demand_col: Column name for demand/weights
        max_iterations: Maximum number of iterations per run
        tolerance: Convergence tolerance
        base_random_state: Base random seed (if None, uses system random)
    
    Returns:
        Dictionary with best solution and all run results
    """
    if num_runs <= 0:
        raise ValueError("Number of runs must be positive")
        
    n = len(customer_df)  # Number of customers
    k = num_facilities    # Number of facilities
    
    if k > n:
        raise ValueError(f"Number of facilities ({k}) cannot exceed number of customers ({n})")
    
    best_solution = None
    best_cost = float('inf')
    all_runs = []
    
    # Generate different random seeds for each run
    if base_random_state is not None:
        np.random.seed(base_random_state)
        random_seeds = np.random.randint(0, 100000, size=num_runs)
    else:
        random_seeds = [None] * num_runs
    
    print(f"Running repeated multi-facility Weiszfeld algorithm with {num_runs} runs...")
    
    for run_idx in range(num_runs):
        current_seed = random_seeds[run_idx] if random_seeds[run_idx] is not None else None
        
        try:
            # Run multi-facility Weiszfeld with current random seed
            solution = multi_facility_weiszfeld(
                customer_df=customer_df,
                num_facilities=num_facilities,
                lat_col=lat_col,
                lon_col=lon_col,
                demand_col=demand_col,
                max_iterations=max_iterations,
                tolerance=tolerance,
                random_state=current_seed
            )
            
            # Track this run
            run_result = {
                'run_index': run_idx,
                'random_state': current_seed,
                'total_cost': solution['total_cost'],
                'iterations': solution['iterations'],
                'converged': solution['converged'],
                'facility_locations': solution['facility_locations']
            }
            all_runs.append(run_result)
            
            # Check if this is the best solution so far
            if solution['total_cost'] < best_cost:
                best_cost = solution['total_cost']
                best_solution = solution.copy()
                best_solution['best_run_index'] = run_idx
                print(f"Run {run_idx + 1}/{num_runs}: New best cost = {best_cost:.2f}")
            else:
                print(f"Run {run_idx + 1}/{num_runs}: Cost = {solution['total_cost']:.2f}")
                
        except Exception as e:
            print(f"Run {run_idx + 1}/{num_runs} failed: {str(e)}")
            run_result = {
                'run_index': run_idx,
                'random_state': current_seed,
                'total_cost': float('inf'),
                'iterations': 0,
                'converged': False,
                'facility_locations': [],
                'error': str(e)
            }
            all_runs.append(run_result)
            continue
    
    if best_solution is None:
        raise RuntimeError("All runs failed to find a valid solution")
    
    # Calculate statistics across all runs
    valid_costs = [run['total_cost'] for run in all_runs if run['total_cost'] != float('inf')]
    
    if not valid_costs:
        raise RuntimeError("No valid solutions found across all runs")
    
    cost_stats = {
        'best_cost': float(min(valid_costs)),
        'worst_cost': float(max(valid_costs)),
        'mean_cost': float(np.mean(valid_costs)),
        'std_cost': float(np.std(valid_costs)),
        'median_cost': float(np.median(valid_costs)),
        'success_rate': len(valid_costs) / num_runs
    }
    
    # Update algorithm name and add run information
    best_solution['algorithm'] = 'repeated_multi_facility_weiszfeld'
    best_solution['num_runs'] = num_runs
    best_solution['cost_statistics'] = cost_stats
    best_solution['all_runs'] = all_runs
    best_solution['parameters']['num_runs'] = num_runs
    best_solution['parameters']['base_random_state'] = base_random_state
    
    print(f"Completed {num_runs} runs. Best cost: {best_cost:.2f}, Mean cost: {cost_stats['mean_cost']:.2f}")
    
    return best_solution

def cluster_customers_kmeans(customer_df: pd.DataFrame, 
                           n_clusters: int,
                           lat_col: str = 'lat',
                           lon_col: str = 'lon',
                           demand_col: str = 'demand',
                           random_state: int = 42) -> pd.DataFrame:
    """
    Cluster customers using K-means algorithm
    
    Args:
        customer_df: DataFrame with customer data
        n_clusters: Number of clusters to create
        lat_col: Column name for latitude
        lon_col: Column name for longitude  
        demand_col: Column name for demand (used for weighted clustering)
        random_state: Random state for reproducibility
    
    Returns:
        DataFrame with cluster assignments added
    """
    df = customer_df.copy()
    
    # Validate required columns
    required_cols = [lat_col, lon_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare data for clustering
    X = df[[lat_col, lon_col]].values
    
    # Apply K-means clustering
    if n_clusters > len(df):
        n_clusters = len(df)
        print(f"Warning: n_clusters reduced to {n_clusters} (number of data points)")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Calculate cluster centers
    cluster_centers = []
    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        if len(cluster_data) > 0:
            if demand_col in df.columns:
                # Weighted centroid using demand
                total_demand = cluster_data[demand_col].sum()
                if total_demand > 0:
                    center_lat = (cluster_data[lat_col] * cluster_data[demand_col]).sum() / total_demand
                    center_lon = (cluster_data[lon_col] * cluster_data[demand_col]).sum() / total_demand
                else:
                    center_lat = cluster_data[lat_col].mean()
                    center_lon = cluster_data[lon_col].mean()
            else:
                center_lat = cluster_data[lat_col].mean()
                center_lon = cluster_data[lon_col].mean()
                
            cluster_centers.append({
                'cluster': i,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'num_customers': len(cluster_data),
                'total_demand': cluster_data[demand_col].sum() if demand_col in df.columns else len(cluster_data)
            })
    
    # Store cluster centers for later use
    df.attrs['cluster_centers'] = cluster_centers
    
    return df

def cluster_customers_hierarchical(customer_df: pd.DataFrame,
                                 n_clusters: int,
                                 lat_col: str = 'lat',
                                 lon_col: str = 'lon',
                                 linkage_method: str = 'ward') -> pd.DataFrame:
    """
    Cluster customers using hierarchical clustering
    
    Args:
        customer_df: DataFrame with customer data
        n_clusters: Number of clusters to create
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        linkage_method: Linkage method for hierarchical clustering
    
    Returns:
        DataFrame with cluster assignments added
    """
    df = customer_df.copy()
    
    # Validate required columns
    required_cols = [lat_col, lon_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare data for clustering
    X = df[[lat_col, lon_col]].values
    
    # Calculate distance matrix
    distances = pdist(X)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method=linkage_method)
    
    # Get cluster assignments
    if n_clusters > len(df):
        n_clusters = len(df)
        print(f"Warning: n_clusters reduced to {n_clusters} (number of data points)")
    
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df['cluster'] = clusters - 1  # Convert to 0-based indexing
    
    # Store linkage matrix for dendrogram plotting
    df.attrs['linkage_matrix'] = linkage_matrix
    
    return df

def transportation(C, capacity):
    """
    Solve transportation problem for capacity-constrained k-median
    
    Args:
        C: Cost matrix (customers x facilities)  
        capacity: Facility capacity
    
    Returns:
        cost, flow: Optimal cost and flow solution
    """
    M = capacity
    n, m = C.shape
    C = np.ceil(C)  # for network simplex 
    G = nx.DiGraph()
    sum_ = 0
    
    # Add facility nodes (supply nodes)
    for j in range(m):
        sum_ -= M
        G.add_node(f"plant{j}", demand=-M)
    
    # Add customer nodes (demand nodes)
    for i in range(n):
        sum_ += 1 
        G.add_node(i, demand=1)
        
    # Add dummy customer with remaining demand
    G.add_node("dummy", demand=-sum_)
    
    # Add edges from facilities to customers
    for i in range(n):
        for j in range(m):
            G.add_edge(f"plant{j}", i, weight=C[i,j])
    
    # Add edges from facilities to dummy
    for j in range(m):
        G.add_edge(f"plant{j}", "dummy", weight=0)
    
    cost, flow = nx.network_simplex(G)
    return cost, flow


def find_median(C, flow, n, m):
    """
    Find cluster median for improving k-median solution
    
    Args:
        C: Distance matrix
        flow: Flow solution from transportation problem
        n: Number of customers
        m: Number of facilities
    
    Returns:
        Total cost of median-based solution
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
            # Find the best location for the cluster
            nodes = np.array(cluster)
            subC = C[np.ix_(nodes, nodes)]
            best_location = nodes[subC.sum(axis=0).argmin()]
            cost = subC.sum(axis=0).min()
            total_cost += cost
    return total_cost


def solve_k_median_lagrange(customer_df: pd.DataFrame,
                           facility_candidates: List[Tuple[float, float]],
                           k: int,
                           demand_col: str = 'demand',
                           lat_col: str = 'lat', 
                           lon_col: str = 'lon',
                           max_iterations: int = 100,
                           max_lr: float = 0.01,
                           moms: Tuple[float, float] = (0.85, 0.95),
                           convergence: float = 1e-5,
                           lr_find: bool = False,
                           adam: bool = False,
                           capacity: Optional[int] = None) -> Dict[str, Any]:
    """
    Solve k-median problem using Lagrangian relaxation with advanced optimization
    
    This is the full implementation from the notebook that includes:
    - Lagrangian relaxation of assignment constraints
    - Subgradient method with momentum
    - Fit-one-cycle learning rate scheduling
    - Adam optimization (optional)
    - Capacity constraints support
    
    Args:
        customer_df: DataFrame with customer locations and demands
        facility_candidates: List of candidate facility locations (lat, lon)
        k: Number of facilities to select
        demand_col: Column name for demand
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        max_iterations: Maximum number of iterations
        max_lr: Maximum learning rate for fit-one-cycle scheduling
        moms: Tuple of (min_momentum, max_momentum) for momentum scheduling
        convergence: Convergence tolerance
        lr_find: Whether to perform learning rate finding
        adam: Whether to use Adam optimization
        capacity: Optional capacity constraint per facility
    
    Returns:
        Dictionary with solution details including bounds convergence
    """
    # Build cost matrix from customer locations and facility candidates
    n = len(customer_df)  # Number of customers
    m = len(facility_candidates)  # Number of facility candidates
    
    # Create cost matrix using great circle distances
    cost_matrix = np.zeros((n, m))
    weights = np.zeros(n)
    
    for i, (_, customer) in enumerate(customer_df.iterrows()):
        weights[i] = customer[demand_col] if demand_col in customer_df.columns and not pd.isna(customer[demand_col]) else 1.0
        customer_lat, customer_lon = customer[lat_col], customer[lon_col]
        
        for j, (fac_lat, fac_lon) in enumerate(facility_candidates):
            cost_matrix[i, j] = great_circle_distance(customer_lat, customer_lon, fac_lat, fac_lon)
    
    # Apply the full k-median Lagrangian relaxation algorithm
    half_iter = max_iterations // 2
    lrs = (max_lr / 25., max_lr)
    
    # Fit-one-cycle learning rate scheduling
    lr_sche = np.concatenate([
        np.linspace(lrs[0], lrs[1], half_iter),
        lrs[1]/2 + (lrs[1]/2) * np.cos(np.linspace(0, np.pi, half_iter))
    ])
    mom_sche = np.concatenate([
        np.linspace(moms[1], moms[0], half_iter),
        moms[1] - (moms[1] - moms[0])/2 - (moms[1] - moms[0])/2 * np.cos(np.linspace(0, np.pi, half_iter))
    ])

    if lr_find:
        phi = 1e-10  # Step size parameter for learning rate finding
        report_iter = 1
    else:
        report_iter = 100

    # Initialize Lagrange multipliers
    u = np.zeros(n)
    w = weights
    c = cost_matrix
    C = c * w.reshape((n, 1))  # Weighted cost matrix
    
    # Initialize momentum and best solution tracking
    m_t = np.zeros(n)
    best_ub = np.inf
    best_open = {}
    
    # Adam parameters
    if adam:
        beta_2 = 0.999
        epsilon = 1e-8
        v_t = np.zeros(n)

    lb_list, ub_list, phi_list = [], [], []
    
    # Main iteration loop
    for t in range(max_iterations):
        # Calculate reduced costs
        Cbar = C - u.reshape((n, 1))
        
        # Solve Lagrangian relaxation subproblem
        if capacity is not None:
            # Capacity-constrained case: solve 0-1 knapsack for each facility
            ystar = np.sort(np.where(Cbar > 0, 0., Cbar), axis=1)[:, :capacity].sum(axis=1)
        else:
            # Unconstrained case: simple aggregation
            ystar = np.where(Cbar < 0, Cbar, 0.).sum(axis=0)
            
        # Select k best facilities
        idx = np.argsort(ystar)
        open_node = set(idx[:k])
        
        # Calculate lower bound (Lagrangian dual)
        lb = u.sum() + ystar[idx[:k]].sum()
        lb_list.append(lb)

        # Calculate assignments for dual solution
        xstar = np.where(Cbar < 0, 1, 0)[:, idx[:k]].sum(axis=1)

        # Calculate upper bound (primal solution)
        if capacity is not None:
            if t % report_iter == 0:
                trans_cost, flow = transportation(C[:, idx[:k]], capacity)
                median_cost = find_median(C, flow, n, k)
                ub = min(trans_cost, median_cost)
        else:
            ub = C[:, np.array(list(open_node))].min(axis=1).sum()
        
        if ub < best_ub:
            best_ub = ub
            best_open = open_node.copy()
        ub_list.append(best_ub)

        # Calculate subgradient
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
            
        if norm <= convergence and capacity is None:
            print("Subgradient norm is zero - optimal!")
            break    
            
        if lb < -1e5:
            if lr_find:
                break
                
        # Update step size and momentum parameters
        if lr_find:
            phi *= 2. 
            beta_1 = moms[1]
        else:
            phi = lr_sche[t] if t < len(lr_sche) else lr_sche[-1]
            beta_1 = mom_sche[t] if t < len(mom_sche) else mom_sche[-1]
            
        phi_list.append(phi)
            
        # Update momentum
        m_t = beta_1 * m_t + (1 - beta_1) * g_t
        
        # Update Lagrange multipliers
        if adam:
            # Adam optimization
            v_t = beta_2 * v_t + (1 - beta_2) * (g_t**2)
            m_cap = m_t / (1 - beta_1**(t+1))
            v_cap = v_t / (1 - beta_2**(t+1))
            u = u + (phi * m_cap) / (np.sqrt(v_cap) + epsilon)
        else:
            # Classical subgradient with momentum
            alpha = (1.05 * best_ub - lb) / norm if norm > 0 else 0
            u = u + phi * alpha * m_t

    # Construct final solution
    selected_facilities = list(best_open)
    selected_locations = [facility_candidates[i] for i in selected_facilities]
    
    # Assign customers to selected facilities
    assignments = []
    total_cost = 0.0
    
    for i in range(n):
        min_distance = float('inf')
        best_facility_idx = 0
        
        for j, facility_idx in enumerate(selected_facilities):
            distance = cost_matrix[i, facility_idx]
            if distance < min_distance:
                min_distance = distance
                best_facility_idx = j
        
        assignments.append(best_facility_idx)
        total_cost += weights[i] * min_distance

    # Calculate facility statistics
    facility_stats = []
    for i, facility_idx in enumerate(selected_facilities):
        customers_assigned = [j for j, assignment in enumerate(assignments) if assignment == i]
        
        total_demand_served = sum(weights[j] for j in customers_assigned)
        avg_distance = 0.0
        
        if customers_assigned:
            distances = [cost_matrix[j, facility_idx] for j in customers_assigned]
            avg_distance = sum(distances) / len(distances)
        
        facility_stats.append({
            'facility_index': facility_idx,
            'location': selected_locations[i],
            'customers_assigned': len(customers_assigned),
            'total_demand_served': total_demand_served,
            'average_distance': avg_distance
        })

    return {
        'selected_facilities': selected_facilities,
        'assignments': assignments,
        'total_cost': total_cost,
        'facility_locations': selected_locations,
        'facility_stats': facility_stats,
        'algorithm': 'lagrangian_relaxation',
        'iterations': t + 1,
        'convergence_gap': gap,
        'lower_bounds': lb_list,
        'upper_bounds': ub_list,
        'learning_rates': phi_list if lr_find else None,
        'parameters': {
            'max_iterations': max_iterations,
            'max_lr': max_lr,
            'momentum_range': moms,
            'adam': adam,
            'capacity': capacity
        }
    }

def calculate_facility_service_area(customer_df: pd.DataFrame,
                                  facility_location: Tuple[float, float],
                                  lat_col: str = 'lat',
                                  lon_col: str = 'lon',
                                  demand_col: str = 'demand') -> Dict[str, Any]:
    """
    Calculate service area statistics for a facility location
    
    Args:
        customer_df: DataFrame with customer data
        facility_location: (lat, lon) tuple of facility location
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        demand_col: Column name for demand
    
    Returns:
        Dictionary with service area statistics
    """
    facility_lat, facility_lon = facility_location
    
    # Calculate distances to all customers
    distances = []
    demands = []
    
    for _, customer in customer_df.iterrows():
        distance = great_circle_distance(
            customer[lat_col], customer[lon_col],
            facility_lat, facility_lon
        )
        distances.append(distance)
        if demand_col in customer_df.columns:
            demands.append(customer[demand_col])
        else:
            demands.append(1)
    
    distances = np.array(distances)
    demands = np.array(demands)
    
    # Calculate statistics
    total_demand = np.sum(demands)
    weighted_avg_distance = np.sum(distances * demands) / total_demand if total_demand > 0 else 0
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    
    # Calculate service levels at different distance thresholds
    service_levels = {}
    thresholds = [10, 25, 50, 100, 200]  # km
    
    for threshold in thresholds:
        customers_within = np.sum(distances <= threshold)
        demand_within = np.sum(demands[distances <= threshold])
        service_levels[f'{threshold}km'] = {
            'customers_served': int(customers_within),
            'customers_percentage': float(customers_within / len(distances) * 100),
            'demand_served': float(demand_within),
            'demand_percentage': float(demand_within / total_demand * 100) if total_demand > 0 else 0
        }
    
    return {
        'facility_location': facility_location,
        'total_customers': len(customer_df),
        'total_demand': float(total_demand),
        'average_distance': float(np.mean(distances)),
        'weighted_average_distance': float(weighted_avg_distance),
        'max_distance': float(max_distance),
        'min_distance': float(min_distance),
        'median_distance': float(np.median(distances)),
        'service_levels': service_levels
    }

def generate_candidate_facilities(customer_df: pd.DataFrame,
                                method: str = 'grid',
                                n_candidates: int = 20,
                                lat_col: str = 'lat',
                                lon_col: str = 'lon') -> List[Tuple[float, float]]:
    """
    Generate candidate facility locations using various methods
    
    Args:
        customer_df: DataFrame with customer locations
        method: Method for generating candidates ('grid', 'random', 'customer_locations')
        n_candidates: Number of candidate locations to generate
        lat_col: Column name for latitude
        lon_col: Column name for longitude
    
    Returns:
        List of candidate facility locations
    """
    lats = customer_df[lat_col].values
    lons = customer_df[lon_col].values
    
    min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
    min_lon, max_lon = float(np.min(lons)), float(np.max(lons))
    
    candidates = []
    
    if method == 'grid':
        # Generate grid of candidate locations
        grid_size = int(np.ceil(np.sqrt(n_candidates)))
        lat_step = (max_lat - min_lat) / (grid_size - 1) if grid_size > 1 else 0
        lon_step = (max_lon - min_lon) / (grid_size - 1) if grid_size > 1 else 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(candidates) >= n_candidates:
                    break
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                candidates.append((lat, lon))
    
    elif method == 'random':
        # Generate random locations within bounding box
        np.random.seed(42)  # For reproducibility
        for _ in range(n_candidates):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            candidates.append((lat, lon))
    
    elif method == 'customer_locations':
        # Use customer locations as candidates
        candidates = [(lat, lon) for lat, lon in zip(lats, lons)]
        if len(candidates) > n_candidates:
            # Sample random subset
            np.random.seed(42)
            indices = np.random.choice(len(candidates), n_candidates, replace=False)
            candidates = [candidates[i] for i in indices]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return candidates


def solve_multiple_source_lnd(
    customer_df: pd.DataFrame,
    warehouse_df: pd.DataFrame,
    factory_df: pd.DataFrame,
    product_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    factory_capacity_df: pd.DataFrame,
    transportation_cost: float = 1.0,
    delivery_cost: float = 2.0,
    warehouse_fixed_cost: float = 10000.0,
    warehouse_variable_cost: float = 1.0,
    num_warehouses: Optional[int] = None,
    single_sourcing: bool = False,
    max_runtime: int = 300
) -> Dict[str, Any]:
    """
    Solve Multiple Source Logistics Network Design problem using mixed-integer optimization
    
    Simplified implementation for debugging and basic functionality
    """
    try:
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpStatusOptimal, PULP_CBC_CMD
        from pulp import LpBinary, LpContinuous
        import time
    except ImportError:
        return {'status': 'Error', 'message': 'PuLP is required for optimization. Install with: pip install pulp'}
    
    try:
        start_time = time.time()
        
        # Data validation and preparation
        print(f"Input data shapes: customers={len(customer_df)}, warehouses={len(warehouse_df)}, factories={len(factory_df)}")
        print(f"Input data shapes: products={len(product_df)}, demand={len(demand_df)}, factory_capacity={len(factory_capacity_df)}")
        
        # Get ID columns (flexible naming)
        customer_id_col = 'customer_id' if 'customer_id' in customer_df.columns else 'name'
        warehouse_id_col = 'warehouse_id' if 'warehouse_id' in warehouse_df.columns else 'name'  
        factory_id_col = 'factory_id' if 'factory_id' in factory_df.columns else 'name'
        product_id_col = 'product_id' if 'product_id' in product_df.columns else 'name'
        
        # Extract lists
        customers = customer_df[customer_id_col].tolist()
        warehouses = warehouse_df[warehouse_id_col].tolist()
        factories = factory_df[factory_id_col].tolist() 
        products = product_df[product_id_col].tolist()
        
        print(f"Extracted data: customers={len(customers)}, warehouses={len(warehouses)}, factories={len(factories)}, products={len(products)}")
        
        # Build demand dictionary
        demand_dict = {}
        demand_customer_col = 'customer_id' if 'customer_id' in demand_df.columns else 'customer'
        demand_product_col = 'product_id' if 'product_id' in demand_df.columns else 'product'
        
        for _, row in demand_df.iterrows():
            customer_key = row[demand_customer_col]
            product_key = row[demand_product_col] 
            demand_dict[(customer_key, product_key)] = float(row['demand'])
            
        print(f"Built demand dictionary with {len(demand_dict)} entries")
        
        # Build factory capacity dictionary
        factory_capacity_dict = {}
        fc_factory_col = 'factory_id' if 'factory_id' in factory_capacity_df.columns else 'factory'
        fc_product_col = 'product_id' if 'product_id' in factory_capacity_df.columns else 'product'
        
        for _, row in factory_capacity_df.iterrows():
            factory_key = row[fc_factory_col]
            product_key = row[fc_product_col]
            factory_capacity_dict[(factory_key, product_key)] = float(row['capacity'])
            
        print(f"Built factory capacity dictionary with {len(factory_capacity_dict)} entries")
        
        # Calculate distances and costs
        transport_costs = {}  # factory -> warehouse
        delivery_costs = {}   # warehouse -> customer
        
        for f in factories:
            f_lat, f_lon = customer_df[customer_df[customer_id_col] == f][['lat', 'lon']].iloc[0] if f in customers else factory_df[factory_df[factory_id_col] == f][['lat', 'lon']].iloc[0]
            for w in warehouses:
                w_lat, w_lon = warehouse_df[warehouse_df[warehouse_id_col] == w][['lat', 'lon']].iloc[0]
                dist = great_circle_distance(float(f_lat), float(f_lon), float(w_lat), float(w_lon))
                transport_costs[(f, w)] = transportation_cost * dist
                
        for w in warehouses:
            w_lat, w_lon = warehouse_df[warehouse_df[warehouse_id_col] == w][['lat', 'lon']].iloc[0]
            for c in customers:
                c_lat, c_lon = customer_df[customer_df[customer_id_col] == c][['lat', 'lon']].iloc[0]
                dist = great_circle_distance(float(w_lat), float(w_lon), float(c_lat), float(c_lon))
                delivery_costs[(w, c)] = delivery_cost * dist
                
        print(f"Calculated {len(transport_costs)} transport costs and {len(delivery_costs)} delivery costs")
        
        # Create simplified optimization model
        model = LpProblem("MS_LND_Simplified", LpMinimize)
        
        # Decision variables
        # y[w] = 1 if warehouse w is opened
        y_warehouse = LpVariable.dicts("warehouse_open", warehouses, cat=LpBinary)
        
        # x[f,w,p] = flow of product p from factory f to warehouse w  
        x_flow = {}
        for f in factories:
            for w in warehouses:
                for p in products:
                    if (f, p) in factory_capacity_dict:
                        x_flow[(f, w, p)] = LpVariable(f"flow_{f}_{w}_{p}", lowBound=0, cat=LpContinuous)
                        
        # z[w,c,p] = flow of product p from warehouse w to customer c
        z_flow = {}
        for w in warehouses:
            for c in customers:
                for p in products:
                    if (c, p) in demand_dict:
                        z_flow[(w, c, p)] = LpVariable(f"delivery_{w}_{c}_{p}", lowBound=0, cat=LpContinuous)
        
        print(f"Created {len(y_warehouse)} warehouse variables, {len(x_flow)} flow variables, {len(z_flow)} delivery variables")
        
        # Objective function
        fixed_costs = lpSum([warehouse_fixed_cost * y_warehouse[w] for w in warehouses])
        transport_costs_expr = lpSum([transport_costs.get((f, w), 0) * x_flow.get((f, w, p), 0) 
                                    for f in factories for w in warehouses for p in products 
                                    if (f, w, p) in x_flow])
        delivery_costs_expr = lpSum([delivery_costs.get((w, c), 0) * z_flow.get((w, c, p), 0)
                                   for w in warehouses for c in customers for p in products
                                   if (w, c, p) in z_flow])
        
        model += fixed_costs + transport_costs_expr + delivery_costs_expr
        
        # Constraints
        
        # 1. Demand satisfaction
        for c in customers:
            for p in products:
                if (c, p) in demand_dict:
                    model += lpSum([z_flow.get((w, c, p), 0) for w in warehouses 
                                  if (w, c, p) in z_flow]) == demand_dict[(c, p)]
        
        # 2. Flow balance at warehouses  
        for w in warehouses:
            for p in products:
                inflow = lpSum([x_flow.get((f, w, p), 0) for f in factories if (f, w, p) in x_flow])
                outflow = lpSum([z_flow.get((w, c, p), 0) for c in customers if (w, c, p) in z_flow])
                model += inflow == outflow
        
        # 3. Factory capacity
        for f in factories:
            for p in products:
                if (f, p) in factory_capacity_dict:
                    model += lpSum([x_flow.get((f, w, p), 0) for w in warehouses 
                                  if (f, w, p) in x_flow]) <= factory_capacity_dict[(f, p)]
        
        # 4. Warehouse opening constraints
        for w in warehouses:
            for f in factories:
                for p in products:
                    if (f, w, p) in x_flow:
                        # Flow can only happen if warehouse is open
                        model += x_flow[(f, w, p)] <= 999999 * y_warehouse[w]  # Big-M constraint
        
        # 5. Number of warehouses constraint
        if num_warehouses is not None:
            model += lpSum([y_warehouse[w] for w in warehouses]) == num_warehouses
        
        print("Model created, starting optimization...")
        
        # Solve
        solver = PULP_CBC_CMD(timeLimit=max_runtime, msg=True)
        model.solve(solver)
        
        runtime = time.time() - start_time
        status = LpStatus[model.status]
        
        print(f"Optimization completed in {runtime:.2f}s with status: {status}")
        
        if model.status != LpStatusOptimal:
            return {
                'status': status,
                'message': f'Optimization failed with status: {status}',
                'runtime': runtime
            }
        
        # Extract solution
        selected_warehouses = []
        total_cost = model.objective.value()
        
        for w in warehouses:
            if y_warehouse[w].value() > 0.5:
                selected_warehouses.append({
                    'warehouse_id': w,
                    'location': warehouse_df[warehouse_df[warehouse_id_col] == w][['lat', 'lon']].iloc[0].to_dict()
                })
        
        # Cost breakdown
        fixed_cost = sum(warehouse_fixed_cost for w in warehouses if y_warehouse[w].value() > 0.5)
        transport_cost = sum(transport_costs.get((f, w), 0) * x_flow[(f, w, p)].value()
                           for (f, w, p) in x_flow if x_flow[(f, w, p)].value() > 0.001)
        delivery_cost_total = sum(delivery_costs.get((w, c), 0) * z_flow[(w, c, p)].value() 
                                for (w, c, p) in z_flow if z_flow[(w, c, p)].value() > 0.001)
        
        return {
            'status': 'Optimal',
            'runtime': runtime,
            'total_cost': total_cost,
            'selected_warehouses': selected_warehouses,
            'cost_breakdown': {
                'fixed_cost': fixed_cost,
                'transportation_cost': transport_cost,
                'delivery_cost': delivery_cost_total,
                'variable_cost': 0  # Simplified
            },
            'num_warehouses_opened': len(selected_warehouses),
            'message': f'Optimization completed successfully in {runtime:.2f} seconds'
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error in Multiple Source LND: {str(e)}"
        print(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return {
            'status': 'Error',
            'message': error_msg,
            'traceback': traceback.format_exc()
        }

def solve_single_source_lnd(
    customer_df: pd.DataFrame,
    warehouse_df: pd.DataFrame,
    factory_df: pd.DataFrame,
    product_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    factory_capacity_df: pd.DataFrame,
    transportation_cost: float = 1.0,
    delivery_cost: float = 2.0,
    warehouse_fixed_cost: float = 10000.0,
    warehouse_variable_cost: float = 1.0,
    num_warehouses: Optional[int] = None,
    max_runtime: int = 300
) -> Dict[str, Any]:
    """
    Solve Single Source Logistics Network Design problem using mixed-integer optimization
    
    Single source constraint: each customer is served by exactly one warehouse
    This simplifies the model by using assignment variables instead of flow variables
    """
    try:
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpStatusOptimal, PULP_CBC_CMD
        from pulp import LpBinary, LpContinuous
        import time
    except ImportError:
        return {'status': 'Error', 'message': 'PuLP is required for optimization. Install with: pip install pulp'}
    
    try:
        start_time = time.time()
        
        # Data validation and preparation
        print(f"Single Source LND - Input data shapes: customers={len(customer_df)}, warehouses={len(warehouse_df)}, factories={len(factory_df)}")
        print(f"Single Source LND - Input data shapes: products={len(product_df)}, demand={len(demand_df)}, factory_capacity={len(factory_capacity_df)}")
        
        # Get ID columns (flexible naming)
        customer_id_col = 'customer_id' if 'customer_id' in customer_df.columns else 'name'
        warehouse_id_col = 'warehouse_id' if 'warehouse_id' in warehouse_df.columns else 'name'  
        factory_id_col = 'factory_id' if 'factory_id' in factory_df.columns else 'name'
        product_id_col = 'product_id' if 'product_id' in product_df.columns else 'name'
        
        # Extract lists
        customers = customer_df[customer_id_col].tolist()
        warehouses = warehouse_df[warehouse_id_col].tolist()
        factories = factory_df[factory_id_col].tolist() 
        products = product_df[product_id_col].tolist()
        
        print(f"Single Source LND - Extracted data: customers={len(customers)}, warehouses={len(warehouses)}, factories={len(factories)}, products={len(products)}")
        
        # Build demand dictionary
        demand_dict = {}
        demand_keys = set()  # Track all valid (customer, product) pairs
        demand_customer_col = 'customer_id' if 'customer_id' in demand_df.columns else 'customer'
        demand_product_col = 'product_id' if 'product_id' in demand_df.columns else 'product'
        
        # Calculate total demand per customer
        customer_total_demand = {}
        for _, row in demand_df.iterrows():
            customer_key = row[demand_customer_col]
            product_key = row[demand_product_col] 
            demand_value = float(row['demand'])
            
            demand_dict[(customer_key, product_key)] = demand_value
            demand_keys.add((customer_key, product_key))
            customer_total_demand[customer_key] = customer_total_demand.get(customer_key, 0) + demand_value
            
        print(f"Single Source LND - Built demand dictionary with {len(demand_dict)} entries")
        print(f"Single Source LND - Customer total demands: {list(customer_total_demand.items())[:3]}...")
        
        # Build factory capacity dictionary
        factory_capacity_dict = {}
        fc_factory_col = 'factory_id' if 'factory_id' in factory_capacity_df.columns else 'factory'
        fc_product_col = 'product_id' if 'product_id' in factory_capacity_df.columns else 'product'
        
        for _, row in factory_capacity_df.iterrows():
            factory_key = row[fc_factory_col]
            product_key = row[fc_product_col]
            factory_capacity_dict[(factory_key, product_key)] = float(row['capacity'])
            
        print(f"Single Source LND - Built factory capacity dictionary with {len(factory_capacity_dict)} entries")
        
        # Calculate distances and costs
        transport_costs = {}  # factory -> warehouse
        delivery_costs = {}   # warehouse -> customer
        
        for f in factories:
            f_lat, f_lon = factory_df[factory_df[factory_id_col] == f][['lat', 'lon']].iloc[0]
            for w in warehouses:
                w_lat, w_lon = warehouse_df[warehouse_df[warehouse_id_col] == w][['lat', 'lon']].iloc[0]
                dist = great_circle_distance(float(f_lat), float(f_lon), float(w_lat), float(w_lon))
                transport_costs[(f, w)] = transportation_cost * dist
                
        for w in warehouses:
            w_lat, w_lon = warehouse_df[warehouse_df[warehouse_id_col] == w][['lat', 'lon']].iloc[0]
            for c in customers:
                c_lat, c_lon = customer_df[customer_df[customer_id_col] == c][['lat', 'lon']].iloc[0]
                dist = great_circle_distance(float(w_lat), float(w_lon), float(c_lat), float(c_lon))
                delivery_costs[(w, c)] = delivery_cost * dist
                
        print(f"Single Source LND - Calculated {len(transport_costs)} transport costs and {len(delivery_costs)} delivery costs")
        
        # Create Single Source LND optimization model
        model = LpProblem("SS_LND", LpMinimize)
        
        # Decision variables
        
        # y[w] = 1 if warehouse w is opened
        y_warehouse = LpVariable.dicts("warehouse_open", warehouses, cat=LpBinary)
        
        # z[w,c] = 1 if customer c is assigned to warehouse w (single sourcing constraint)
        z_assignment = LpVariable.dicts("customer_assignment", 
                                       [(w, c) for w in warehouses for c in customers], 
                                       cat=LpBinary)
        
        # x[f,w,p] = flow of product p from factory f to warehouse w  
        x_flow = {}
        for f in factories:
            for w in warehouses:
                for p in products:
                    if (f, p) in factory_capacity_dict:
                        x_flow[(f, w, p)] = LpVariable(f"flow_{f}_{w}_{p}", lowBound=0, cat=LpContinuous)
                        
        print(f"Single Source LND - Created {len(y_warehouse)} warehouse variables, {len(z_assignment)} assignment variables, {len(x_flow)} flow variables")
        
        # Objective function: minimize total cost
        
        # 1. Warehouse fixed costs
        fixed_costs = lpSum([warehouse_fixed_cost * y_warehouse[w] for w in warehouses])
        
        # 2. Transportation costs (factory to warehouse)
        transport_costs_expr = lpSum([transport_costs.get((f, w), 0) * x_flow.get((f, w, p), 0) 
                                    for f in factories for w in warehouses for p in products 
                                    if (f, w, p) in x_flow])
        
        # 3. Delivery costs (warehouse to customer) - based on assignment
        delivery_costs_expr = lpSum([delivery_costs.get((w, c), 0) * customer_total_demand.get(c, 0) * z_assignment.get((w, c), 0)
                                   for w in warehouses for c in customers
                                   if (w, c) in z_assignment])
        
        # 4. Warehouse variable costs
        variable_costs_expr = lpSum([warehouse_variable_cost * x_flow.get((f, w, p), 0)
                                   for f in factories for w in warehouses for p in products
                                   if (f, w, p) in x_flow])
        
        model += fixed_costs + transport_costs_expr + delivery_costs_expr + variable_costs_expr
        
        # Constraints
        
        # 1. Single sourcing: each customer assigned to exactly one warehouse
        for c in customers:
            model += lpSum([z_assignment.get((w, c), 0) for w in warehouses if (w, c) in z_assignment]) == 1
        
        # 2. Customer assignment only if warehouse is open
        for w in warehouses:
            for c in customers:
                if (w, c) in z_assignment:
                    model += z_assignment[(w, c)] <= y_warehouse[w]
        
        # 3. Flow balance at warehouses (aggregate demand per customer)
        for w in warehouses:
            for p in products:
                # Inflow from factories
                inflow = lpSum([x_flow.get((f, w, p), 0) for f in factories if (f, w, p) in x_flow])
                
                # Outflow to assigned customers (product-specific demand)
                outflow = lpSum([demand_dict.get((c, p), 0) * z_assignment.get((w, c), 0) 
                               for c in customers if (w, c) in z_assignment])
                
                model += inflow >= outflow  # Allow overproduction
        
        # 4. Factory capacity constraints
        for f in factories:
            for p in products:
                if (f, p) in factory_capacity_dict:
                    model += lpSum([x_flow.get((f, w, p), 0) for w in warehouses 
                                  if (f, w, p) in x_flow]) <= factory_capacity_dict[(f, p)]
        
        # 5. Warehouse capacity constraints (optional upper bounds)
        for w in warehouses:
            warehouse_row = warehouse_df[warehouse_df[warehouse_id_col] == w].iloc[0]
            upper_bound = warehouse_row.get('upper_bound', 99999)
            if upper_bound < 99999:
                total_flow = lpSum([x_flow.get((f, w, p), 0) 
                                  for f in factories for p in products if (f, w, p) in x_flow])
                model += total_flow <= upper_bound * y_warehouse[w]
        
        # 6. Number of warehouses constraint
        if num_warehouses is not None:
            model += lpSum([y_warehouse[w] for w in warehouses]) == num_warehouses
        
        print("Single Source LND - Model created, starting optimization...")
        
        # Solve
        solver = PULP_CBC_CMD(timeLimit=max_runtime, msg=True)
        model.solve(solver)
        
        runtime = time.time() - start_time
        status = LpStatus[model.status]
        
        print(f"Single Source LND - Optimization completed in {runtime:.2f}s with status: {status}")
        
        if model.status != LpStatusOptimal:
            return {
                'status': status,
                'message': f'Optimization failed with status: {status}',
                'runtime': runtime
            }
        
        # Extract solution
        selected_warehouses = []
        customer_assignments = []
        total_cost = model.objective.value()
        
        # Extract opened warehouses with statistics
        warehouse_stats = {}
        for w in warehouses:
            if y_warehouse[w].value() > 0.5:
                warehouse_stats[w] = {
                    'customers_assigned': 0,
                    'total_demand_served': 0
                }
        
        # Extract customer assignments and update warehouse statistics
        for w in warehouses:
            for c in customers:
                if (w, c) in z_assignment and z_assignment[(w, c)].value() > 0.5:
                    customer_row = customer_df[customer_df[customer_id_col] == c].iloc[0]
                    warehouse_row = warehouse_df[warehouse_df[warehouse_id_col] == w].iloc[0]
                    
                    # Sum all product demands for this customer
                    total_customer_demand = 0
                    for p in products:
                        if (c, p) in demand_keys:
                            demand_row = demand_df[(demand_df[customer_id_col] == c) & (demand_df[product_id_col] == p)]
                            if not demand_row.empty:
                                total_customer_demand += float(demand_row.iloc[0]['demand'])
                    
                    customer_assignments.append({
                        'customer_id': str(c),
                        'customer_name': str(c),  # Use ID as name if name column not available
                        'customer_location': [float(customer_row['lat']), float(customer_row['lon'])],
                        'warehouse_index': str(w),
                        'warehouse_name': str(w),  # Use ID as name if name column not available
                        'warehouse_location': [float(warehouse_row['lat']), float(warehouse_row['lon'])],
                        'product_id': 'all',  # Since we're aggregating all products
                        'product_name': 'All Products',
                        'demand': total_customer_demand,
                        'transportation_cost': sum(transport_costs.get((f, w), 0) * x_flow[(f, w, p)].value() 
                                                  for f in factories for p in products 
                                                  if (f, w, p) in x_flow and x_flow[(f, w, p)].value() > 0.001),
                        'delivery_cost': delivery_costs.get((w, c), 0) * total_customer_demand
                    })
                    
                    # Update warehouse statistics
                    if w in warehouse_stats:
                        warehouse_stats[w]['customers_assigned'] += 1
                        warehouse_stats[w]['total_demand_served'] += total_customer_demand
        
        # Build selected warehouses list with statistics
        for w, stats in warehouse_stats.items():
            warehouse_row = warehouse_df[warehouse_df[warehouse_id_col] == w].iloc[0]
            selected_warehouses.append({
                'warehouse_index': str(w),
                'warehouse_name': str(w),  # Use ID as name if name column not available
                'location': [float(warehouse_row['lat']), float(warehouse_row['lon'])],
                'customers_assigned': stats['customers_assigned'],
                'total_demand_served': stats['total_demand_served']
            })
        
        # Cost breakdown
        fixed_cost = sum(warehouse_fixed_cost for w in warehouses if y_warehouse[w].value() > 0.5)
        transport_cost = sum(transport_costs.get((f, w), 0) * x_flow[(f, w, p)].value()
                           for (f, w, p) in x_flow if x_flow[(f, w, p)].value() > 0.001)
        delivery_cost_total = sum(assignment['delivery_cost'] for assignment in customer_assignments)
        variable_cost_total = sum(warehouse_variable_cost * x_flow[(f, w, p)].value()
                                for (f, w, p) in x_flow if x_flow[(f, w, p)].value() > 0.001)
        
        return {
            'status': 'Optimal',
            'runtime': runtime,
            'total_cost': total_cost,
            'selected_warehouses': selected_warehouses,
            'customer_assignments': customer_assignments,
            'cost_breakdown': {
                'fixed_cost': fixed_cost,
                'transportation_cost': transport_cost,
                'delivery_cost': delivery_cost_total,
                'variable_cost': variable_cost_total
            },
            'num_warehouses_opened': len(selected_warehouses),
            'num_customer_assignments': len(customer_assignments),
            'single_sourcing': True,
            'message': f'Single Source LND optimization completed successfully in {runtime:.2f} seconds'
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error in Single Source LND: {str(e)}"
        print(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return {
            'status': 'Error',
            'message': error_msg,
            'traceback': traceback.format_exc()
        }


def elbow_method_analysis(
    customer_df: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    demand_col: Optional[str] = None,
    min_facilities: int = 1,
    max_facilities: int = 10,
    algorithm: str = 'weiszfeld',
    max_iterations: int = 1000,
    tolerance: float = 1e-4,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform elbow method analysis to determine optimal number of facilities
    
    Args:
        customer_df: DataFrame with customer data
        lat_col: Latitude column name
        lon_col: Longitude column name
        demand_col: Optional demand/weight column name
        min_facilities: Minimum number of facilities to test
        max_facilities: Maximum number of facilities to test
        algorithm: Algorithm to use ('weiszfeld', 'kmeans', 'hierarchical')
        max_iterations: Maximum iterations for optimization
        tolerance: Convergence tolerance
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with analysis results including costs and optimal number
    """
    try:
        print(f"Starting elbow method analysis for {min_facilities} to {max_facilities} facilities...")
        
        # Extract customer data
        customers = []
        weights = []
        
        for _, row in customer_df.iterrows():
            customers.append((float(row[lat_col]), float(row[lon_col])))
            if demand_col and demand_col in customer_df.columns:
                weights.append(float(row[demand_col]))
            else:
                weights.append(1.0)
        
        # Run analysis for different number of facilities
        results = []
        costs = []
        
        for num_facilities in range(min_facilities, max_facilities + 1):
            print(f"Testing {num_facilities} facilities...")
            
            if algorithm == 'weiszfeld':
                # Use repeated multi-facility Weiszfeld
                result = repeated_multi_facility_weiszfeld(
                    customer_df=customer_df,
                    num_facilities=num_facilities,
                    num_runs=10,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    demand_col=demand_col,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    base_random_state=random_state
                )
                
                # Extract the solution (result is already the best solution)
                facility_locations = result['facility_locations']
                assignments = result['assignments']
                total_cost = result['total_cost']
                
            elif algorithm == 'kmeans':
                # Use k-means clustering
                clustering_result = cluster_customers_kmeans(
                    customer_df=customer_df,
                    n_clusters=num_facilities,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    demand_col=demand_col,
                    random_state=random_state
                )
                
                # Calculate total cost as sum of distances to cluster centers
                total_cost = 0
                facility_locations = []
                assignments = []
                
                for cluster_stat in clustering_result['cluster_statistics']:
                    facility_locations.append((cluster_stat['center_lat'], cluster_stat['center_lon']))
                
                for i, (lat, lon) in enumerate(customers):
                    cluster_id = clustering_result['clustered_data'][i]['cluster']
                    center_lat = clustering_result['cluster_statistics'][cluster_id]['center_lat']
                    center_lon = clustering_result['cluster_statistics'][cluster_id]['center_lon']
                    distance = great_circle_distance(lat, lon, center_lat, center_lon)
                    total_cost += distance * weights[i]
                    assignments.append(cluster_id)
                    
            elif algorithm == 'hierarchical':
                # Use hierarchical clustering
                clustering_result = cluster_customers_hierarchical(
                    customer_df=customer_df,
                    n_clusters=num_facilities,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    demand_col=demand_col,
                    linkage_method='ward'
                )
                
                # Calculate total cost similar to k-means
                total_cost = 0
                facility_locations = []
                assignments = []
                
                for cluster_stat in clustering_result['cluster_statistics']:
                    facility_locations.append((cluster_stat['center_lat'], cluster_stat['center_lon']))
                
                for i, (lat, lon) in enumerate(customers):
                    cluster_id = clustering_result['clustered_data'][i]['cluster']
                    center_lat = clustering_result['cluster_statistics'][cluster_id]['center_lat']
                    center_lon = clustering_result['cluster_statistics'][cluster_id]['center_lon']
                    distance = great_circle_distance(lat, lon, center_lat, center_lon)
                    total_cost += distance * weights[i]
                    assignments.append(cluster_id)
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Calculate additional metrics
            avg_distance = total_cost / sum(weights)
            
            # Calculate within-cluster sum of squares (WCSS)
            wcss = 0
            for i, (lat, lon) in enumerate(customers):
                facility_idx = assignments[i]
                facility_lat, facility_lon = facility_locations[facility_idx]
                distance = great_circle_distance(lat, lon, facility_lat, facility_lon)
                wcss += (distance ** 2) * weights[i]
            
            results.append({
                'num_facilities': num_facilities,
                'total_cost': total_cost,
                'average_distance': avg_distance,
                'wcss': wcss,
                'facility_locations': facility_locations,
                'assignments': assignments
            })
            
            costs.append(total_cost)
            
        # Find optimal number using elbow method
        # Calculate rate of change (derivative)
        cost_changes = []
        for i in range(1, len(costs)):
            change = costs[i-1] - costs[i]
            cost_changes.append(change)
        
        # Calculate second derivative to find elbow
        if len(cost_changes) > 1:
            second_derivatives = []
            for i in range(1, len(cost_changes)):
                second_derivative = cost_changes[i-1] - cost_changes[i]
                second_derivatives.append(second_derivative)
            
            # Find the elbow point (maximum second derivative)
            elbow_index = second_derivatives.index(max(second_derivatives)) + 1
            optimal_num_facilities = min_facilities + elbow_index
        else:
            # If not enough data points, use middle value
            optimal_num_facilities = (min_facilities + max_facilities) // 2
        
        # Calculate percentage improvements
        improvements = []
        for i in range(1, len(costs)):
            improvement = ((costs[i-1] - costs[i]) / costs[i-1]) * 100
            improvements.append(improvement)
        
        return {
            'analysis_results': results,
            'costs': costs,
            'cost_changes': cost_changes,
            'improvements': improvements,
            'optimal_num_facilities': optimal_num_facilities,
            'algorithm': algorithm,
            'parameters': {
                'min_facilities': min_facilities,
                'max_facilities': max_facilities,
                'max_iterations': max_iterations,
                'tolerance': tolerance,
                'random_state': random_state
            },
            'num_customers': len(customers),
            'total_demand': sum(weights)
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error in elbow method analysis: {str(e)}"
        print(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return {
            'status': 'Error',
            'message': error_msg,
            'traceback': traceback.format_exc()
        }
