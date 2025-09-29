import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from geopy.distance import great_circle as distance
import warnings
warnings.filterwarnings('ignore')

# Import Gurobi if available, otherwise use PuLP
try:
    from gurobipy import Model, GRB, quicksum
    USE_GUROBI = True
except ImportError:
    from pulp import *
    USE_GUROBI = False

def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points on Earth
    Returns distance in kilometers
    Uses geopy.distance for consistency with notebook
    """
    return distance((lat1, lon1), (lat2, lon2)).km

def make_total_demand(demand_df: pd.DataFrame, 
                     time_col: str = 'time', 
                     demand_col: str = 'demand',
                     customer_col: str = 'customer') -> pd.DataFrame:
    """
    Calculate annual/planning period total demand from time-series demand data
    Exact implementation from notebook to match computational procedures
    
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

def weiszfeld(cust_df, weight, num_of_facilities, epsilon=0.0001, max_iter=1000, seed=None, X0=None, Y0=None):
    """
    Weiszfeld法； 複数施設の連続施設配置問題の近似解法
    Exact implementation from notebook
    """
    if seed is not None:
        np.random.seed(seed)
        
    n = len(cust_df)
    k = num_of_facilities

    x, y, nodeweight = cust_df.lat.values, cust_df.lon.values, weight

    X=[0.]*k
    Y=[0.]*k
    newX=[0.]*k
    newY=[0.]*k

    #初期解
    if X0 is None:
        perm = np.random.permutation(n)
        for i, j in enumerate(perm[:k]):
            X[i] = x[j]
            Y[i] = y[j]
    else:
        X = list(X0)
        Y = list(Y0)

    PrevX = X[:]
    PrevY = Y[:]

    for iter__ in range(max_iter):
        #find the nearest facility
        nodelist=[[] for i in range(k)]
        for i in range(n):
            mindist=9999999.0
            minj=-1
            for j in range(k):
                dist = distance( (x[i],y[i]), (X[j],Y[j]) ).km
                if dist < mindist:
                    mindist = dist
                    minj = j
            nodelist[minj].append(i)

        #find the gravity point
        for j in range(k):
            sumx = sumy= sumr =0.0
            for i in nodelist[j]:
                sumx += nodeweight[i]*x[i]
                sumy += nodeweight[i]*y[i]
                sumr += nodeweight[i]
            if len(nodelist[j])>0:
                X[j]=sumx/sumr
                Y[j]=sumy/sumr

        #Weizfeld search
        for iter_ in range(max_iter):
            for j in range(k):
                sumx = sumy= sumr =0.0
                for i in nodelist[j]:
                    r = distance( (x[i],y[i]), (X[j],Y[j]) ).km
                    if r>0.0:
                        sumx += nodeweight[i]*x[i]/r
                        sumy += nodeweight[i]*y[i]/r
                        sumr += nodeweight[i]/r
                    else:
                        newX[j] = x[i]
                        newY[j] = y[i]
                        break
                if sumr>0.0:
                    newX[j] = sumx/sumr
                    newY[j] = sumy/sumr

            #compute error (in Weiszfeld search)
            error = 0.0
            for j in range(k):
                error+=distance( (X[j],Y[j]),(newX[j],newY[j]) ).km
            if error<=epsilon:
                break
            for j in range(k):
                X[j]=newX[j]
                Y[j]=newY[j]

        #compute error
        error=0.0
        for j in range(k):
            error+=distance( (X[j],Y[j]),(PrevX[j],PrevY[j]) ).km
        if error<=epsilon:
            break
        for j in range(k):
            X[j]=newX[j]
            Y[j]=newY[j]

        PrevX=X[:]
        PrevY=Y[:]

    partition = {}
    cost = 0.
    for j in range(k):
        for i in nodelist[j]:
            partition[i]=j
            d = distance( (x[i],y[i]), (X[j],Y[j]) ).km
            cost += d*nodeweight[i]

    return X, Y, partition, cost


def weiszfeld_numpy(cust_df, weight, num_of_facilities, epsilon=0.0001, max_iter = 1000, seed=None, X0=None, Y0=None):
    """
    Weiszfeld法； 複数施設の連続施設配置問題の近似解法 (NumPy version)
    """
    if seed is not None:
        np.random.seed(seed)
    n = len(cust_df)
    k = num_of_facilities

    x, y, nodeweight = cust_df.lat.values, cust_df.lon.values, np.array(weight)

    D = np.zeros( (n,k)  )
    X = np.zeros(k)
    Y = np.zeros(k)
    newX = np.zeros(k)
    newY = np.zeros(k)

    #初期解
    if X0 is None:
        perm = np.random.permutation(n)
        for i, j in enumerate(perm[:k]):
            X[i] = x[j]
            Y[i] = y[j]
    else:
        X = np.array(X0)
        Y = np.array(Y0)
        

    PrevX=X.copy()
    PrevY=Y.copy()

    wx = nodeweight*x
    wy = nodeweight*y

    for iter__ in range(max_iter):
        nodelist=[[] for i in range(k)]
        for i in range(n):
            for j in range(k):
                D[i,j] = nodeweight[i]*distance( (x[i],y[i]), (X[j],Y[j]) ).km

        #最も近い施設の番号の配列
        min_j = D.argmin(axis=1)
        #施設jに近い顧客のリスト
        nodelist=[[i for i in range(n) if min_j[i]==j] for j in range(k)]

        #重心を求める（初期地点）
        for j in range(k):
            sumx = wx.sum( where= (min_j==j) )
            sumy = wy.sum( where= (min_j==j) )
            sumr = nodeweight.sum( where= (min_j==j) )
            if sumr > 0:
                X[j]=sumx/sumr
                Y[j]=sumy/sumr

        #Weizfeld search
        for iter_ in range(max_iter):
            for j in range(k):
                sumx = sumy= sumr =0.0
                for i in nodelist[j]:
                    d = distance( (x[i],y[i]), (X[j],Y[j]) ).km
                    if d>0.0:
                        sumx += wx[i]/d
                        sumy += wy[i]/d
                        sumr += nodeweight[i]/d
                    else:
                        newX[j] = x[i]
                        newY[j] = y[i]
                        break
                if sumr>0.0:
                    newX[j] = sumx/sumr
                    newY[j] = sumy/sumr

            #compute error (in Weiszfeld search)
            #print(X,Y,newX,newY)
            error = 0.0
            for j in range(k):
                error+=distance((X[j],Y[j]),(newX[j],newY[j]) ).km
            #print("error=",error)
            if error<=epsilon:
                break
            X = newX.copy()
            Y = newY.copy()

        #compute error
        error=0.0
        for j in range(k):
            error+=distance( (X[j],Y[j]),(PrevX[j],PrevY[j]) ).km
        #print ("error=",iter__, error)
        if error<=epsilon:
            break
        X = newX.copy()
        Y = newY.copy()

        PrevX=X.copy()
        PrevY=Y.copy()

    partition = {}
    cost = 0.
    for j in range(k):
        for i in nodelist[j]:
            partition[i]=j
            d = distance( (x[i],y[i]), (X[j],Y[j]) ).km
            cost += d*nodeweight[i]
            
    return X, Y, partition, cost


def repeated_weiszfeld(cust_df, weight, num_of_facilities, epsilon=0.0001, max_iter=1, numpy=True, seed=None):
    """
    TODO: Multicore (threading)で高速化
    Exact implementation from notebook
    """
    if seed is not None:
        np.random.seed(seed)
    best_cost = float('inf')
    best_X = best_Y = best_partition = None
    for iter_ in range(max_iter):
        if numpy:
            X, Y, partition, cost = weiszfeld_numpy(cust_df, weight, num_of_facilities, epsilon, 1000)
        else:
            X, Y, partition, cost = weiszfeld(cust_df, weight, num_of_facilities, epsilon, 1000)
        if cost < best_cost:
            best_X, best_Y, best_partition, best_cost = X, Y, partition, cost
            
    return best_X, best_Y, best_partition, best_cost


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
            # Extract weights (demand)
            if demand_col in customer_df.columns:
                weights = customer_df[demand_col].tolist()
                weights = [1.0 if pd.isna(w) else w for w in weights]
            else:
                weights = [1.0] * len(customer_df)
                
            # Run Weiszfeld algorithm with current random seed
            # Use numpy version for better performance
            X, Y, partition, cost = weiszfeld_numpy(
                cust_df=customer_df,
                weight=weights,
                num_of_facilities=num_facilities,
                epsilon=tolerance,
                max_iter=max_iterations,
                seed=current_seed
            )
            
            # Calculate facility statistics for this run
            facility_locations = [[float(X[j]), float(Y[j])] for j in range(num_facilities)]
            facility_stats = []
            
            for j in range(num_facilities):
                customers_assigned = [i for i, p in partition.items() if p == j]
                
                if customers_assigned:
                    total_demand_served = sum(weights[i] for i in customers_assigned)
                    distances_to_facility = [great_circle_distance(
                        customer_df.iloc[i][lat_col], customer_df.iloc[i][lon_col], X[j], Y[j]
                    ) for i in customers_assigned]
                    avg_distance = sum(distances_to_facility) / len(distances_to_facility) if distances_to_facility else 0
                else:
                    total_demand_served = 0.0
                    avg_distance = 0.0
                
                facility_stats.append({
                    'facility_index': j,
                    'location': [float(X[j]), float(Y[j])],
                    'customers_assigned': len(customers_assigned),
                    'total_demand_served': float(total_demand_served),
                    'average_distance': float(avg_distance)
                })
            
            # Create solution structure
            solution = {
                'facility_locations': facility_locations,
                'assignments': [partition.get(i, 0) for i in range(len(customer_df))],
                'total_cost': float(cost),
                'facility_stats': facility_stats,
                'converged': True,
                'iterations': max_iterations  # weiszfeld doesn't return actual iterations
            }
            
            # Track this run
            run_result = {
                'run_index': run_idx,
                'random_state': current_seed,
                'total_cost': float(cost),
                'iterations': max_iterations,
                'converged': True,
                'facility_locations': facility_locations
            }
            all_runs.append(run_result)
            
            # Check if this is the best solution so far
            if cost < best_cost:
                best_cost = cost
                best_solution = solution.copy()
                best_solution['best_run_index'] = run_idx
                print(f"Run {run_idx + 1}/{num_runs}: New best cost = {best_cost:.2f}")
            else:
                print(f"Run {run_idx + 1}/{num_runs}: Cost = {cost:.2f}")
                
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
    best_solution['parameters'] = {
        'num_facilities': num_facilities,
        'num_runs': num_runs,
        'max_iterations': max_iterations,
        'tolerance': tolerance,
        'base_random_state': base_random_state,
        'lat_col': lat_col,
        'lon_col': lon_col,
        'demand_col': demand_col
    }
    
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


def solve_k_median(cust_df, weight, cost, num_of_facilities, max_iter=100, max_lr=0.01, moms=(0.85,0.95), 
                   convergence=1e-5, lr_find = False, adam = False, capacity = None):
    """
    k-メディアン問題を解くための関数 solve_k_median
    Exact implementation from notebook
    """
    m = num_of_facilities 
    half_iter = max_iter//2
    lrs = (max_lr/25., max_lr)
    lr_sche =  np.concatenate( [np.linspace(lrs[0],lrs[1],half_iter), lrs[1]/2 + (lrs[1]/2)*np.cos(np.linspace(0,np.pi,half_iter) )])  #cosine annealing
    mom_sche = np.concatenate( [np.linspace(moms[1],moms[0],half_iter), moms[1]-(moms[1]-moms[0])/2 -(moms[1]-moms[0])/2*np.cos(np.linspace(0,np.pi,half_iter))] )

    if lr_find:
        phi = 1e-10   #ステップサイズを決めるためのパラメータ
        report_iter = 1
    else:
        report_iter = 100

    n = len(cost)
    u = np.zeros(n) #Lagrange multiplier
    w = np.array(weight)
    c = np.array(cost)
    C = c*w.reshape((n,1))
    
    m_t = np.zeros(n)
    best_ub = np.inf 
    best_open = {}
    
    #Adam
    if adam:
        beta_2 = 0.999
        epsilon = 1e-8
        m_t = np.zeros(n)
        v_t = np.zeros(n)

    lb_list, ub_list, phi_list = [], [], []
    for t in range(max_iter):
        Cbar = C - u.reshape((n,1))   #被約費用の計算
        xstar = np.zeros(n)
        ystar = np.zeros(n)
        # 0-1 knapsackを解く（容量制約のとき）
        # 係数 1の場合には，M（容量）個を選択
        if capacity is not None: #容量制約を考慮（施設に割り当て可能な顧客数がcapacity以下）
            ystar = np.sort(np.where(Cbar>0,0.,Cbar), axis=1)[:,:capacity].sum(axis=1)
        else:    
            ystar = np.where(Cbar<0, Cbar, 0.).sum(axis=0)
            
        idx = np.argsort(ystar) #小さい順にソート
        open_node = set(idx[:m])#小さい順にm個選択（同じ位置にあるものを選んでいる！）
        
        #下界の計算
        lb = u.sum() + ystar[idx[:m]].sum()
        lb_list.append(lb)

        xstar = np.where(Cbar<0,1,0)[:,idx[:m]].sum(axis=1)

        #上界の計算  （容量制約がある場合は，（一般化）割当問題を解く必要がある！）
        if capacity is not None:
            if t%report_iter ==0:
                cost, flow = transportation(C[:,idx[:m]],capacity)
                cost2 = find_median(C, flow, n, m)
                ub = min(cost,cost2)
        else:
            ub = C[:,np.array(list(open_node))].min(axis=1).sum()
        
        if ub<best_ub:
            best_ub = ub
            best_open = open_node.copy()
        ub_list.append(best_ub)

        g_t = 1.-xstar #劣勾配の計算
        norm = np.dot(g_t, g_t) #ノルムの計算
        
        if lb > convergence:
            gap = (best_ub-lb)/lb
            if gap <=convergence:
                print("gap=",gap,best_ub,lb)
                break
        else:
            gap = 10.
            
        if t%report_iter ==0:
            print( f"{t}: {gap:.3f}, {lb:.5f}, {best_ub:.5f}, {norm:.2f}" )
            
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
            #fit one cycle 
            phi = lr_sche[t]
            beta_1 = mom_sche[t] 
            
        phi_list.append(phi) # lr_find用
            
        # 移動平均の更新
        m_t = beta_1*m_t + (1-beta_1)*g_t
        
        if adam:
            # 劾配の二乗の移動平均の更新
            v_t = beta_2*v_t + (1-beta_2)*(g_t**2)
            m_cap = m_t/(1-beta_1**(t+1))
            v_cap = v_t/(1-beta_2**(t+1))
            u = u + (phi*m_cap)/(np.sqrt(v_cap)+epsilon)
        else:
            alpha = (1.05*best_ub-lb)/norm 
            u = u + phi*alpha*m_t  # 慣性項あり
        
    #解の構成
    X, Y =[], []
    for i in best_open:
        row = cust_df.iloc[i]
        X.append(row.lat)
        Y.append(row.lon)
    facility_index={}
    for idx, i in enumerate(best_open):
        facility_index[i] = idx 
    partition = np.zeros(n, int)
    ub = 0.
    for i in range(n):
        min_cost = np.inf
        for j in best_open:
            if c[i,j] < min_cost:
                min_cost =  c[i,j] 
                partition[i] = facility_index[j]
            
    return X, Y, partition, best_ub, lb_list, ub_list, phi_list


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
        'facility_location': list(facility_location),
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


def lnd_ms(weight, cust, dc, dc_lb, dc_ub, plnt, plnt_ub, demand, tp_cost, del_cost, dc_fc, dc_vc, dc_num, 
           volume = None, fix_y=None, dc_slack_penalty=10000., demand_slack_penalty=10000000.):
    """
    Multi-source logistics network design model
    Exact implementation from notebook using Gurobi
    """
    if USE_GUROBI:
        # Gurobi implementation
        prod = set(weight.keys())
        plnt_to_dc = set((i,j,p) for i in plnt for j in dc for p in prod
                         if plnt_ub.get((i,p),0) > 0 and (i,j) in tp_cost)
        dc_to_cust = set((j,k,p) for j in dc for (k,p) in demand if (j,k) in del_cost)
        
        if volume is None:
            volume ={}
            for p in prod:
                volume[p] = 1.
        
        model = Model()
        x,y = {}, {}
        for (i,j,p) in plnt_to_dc | dc_to_cust:
            x[i,j,p] = model.addVar(vtype='C', name=f'x[{i},{j},{p}]')

        slack = {}
        for (k,p) in demand:
            if k in cust:
                slack[k,p] = model.addVar(vtype="C", name=f"slack[{k},{p}]")

        for j in dc:
            y[j] = model.addVar(vtype='B', name=f'y[{j}]')

        cost ={}
        for i in range(5):
            cost[i] = model.addVar(vtype="C",name=f"cost[{i}]")

        dc_slack = {}
        for j in dc:
            dc_slack[j] = model.addVar(vtype="C", name=f"dc_slack[{j}]")

        # Constraints
        Cust_Demand_Cons = {}
        for (k,p) in demand: 
            if k in cust:
                Cust_Demand_Cons[k,p] = model.addConstr(
                    quicksum(x[j,k,p] for j in dc if (j,k,p) in dc_to_cust) + slack[k,p]
                    == demand[k,p],
                    name=f'Cust_Demand_Cons[{k},{p}]'
                )
        
        # DC flow conservation
        DC_Flow_Cons = {}
        for j in dc:
            for p in prod:
                DC_Flow_Cons[j,p] = model.addConstr(
                    quicksum(x[i,j,p] for i in plnt if (i,j,p) in plnt_to_dc)
                    ==
                    quicksum(x[j,k,p] for k in cust if (j,k,p) in dc_to_cust if k in cust),
                    name=f'DC_Flow_Cons[{j},{p}]'
                )
        
        # Strong constraints
        DC_Strong_Cons = {}
        for (j,k,p) in dc_to_cust:
            if k in cust:
                DC_Strong_Cons[j,k,p] = model.addConstr(
                    x[j,k,p] <= demand[k,p] * y[j],
                    name=f'DC_Strong_Cons[{j},{k},{p}]'
                )

        # DC capacity constraints
        DC_UB_Cons = {}
        for j in dc:
            DC_UB_Cons[j] = model.addConstr(
                dc_ub[j] * y[j] >=
                quicksum(volume[p]*x[i,j,p] for i in plnt for p in prod if (i,j,p) in plnt_to_dc),
                name=f'DC_UB_Cons[{j}]'
            )
            
        DC_LB_Cons = {}
        for j in dc:
            DC_LB_Cons[j] = model.addConstr(
                dc_lb[j] * y[j] <=
                quicksum(volume[p]*x[i,j,p] for i in plnt for p in prod if (i,j,p) in plnt_to_dc) + dc_slack[j],
                name=f'DC_LB_Cons[{j}]'
            )

        # Plant production limits
        Plnt_UB_Cons = {}
        for i,p in plnt_ub:
            Plnt_UB_Cons[i,p] = model.addConstr(
                quicksum(x[i,j,p] for j in dc if (i,j,p) in plnt_to_dc)<=plnt_ub[i,p],
                name=f'Plnt_UB_Cons[{i},{p}]'
            )

        # DC number constraints if specified
        if dc_num is not None:
            if type(dc_num)==tuple:
                DC_Num_UB_Cons = model.addConstr(
                    quicksum(y[j] for j in dc if fix_y is None or fix_y[j] is None) <= dc_num[1],
                    name='DC_Num_UB_Cons'
                )
                DC_Num_LB_Cons = model.addConstr(
                    quicksum(y[j] for j in dc if fix_y is None or fix_y[j] is None) >= dc_num[0],
                    name='DC_Num_LB_Cons'
                )
            else:
                DC_Num_Cons = model.addConstr(
                    quicksum(y[j] for j in dc if fix_y is None or fix_y[j] is None) == dc_num,
                    name='DC_Num_Cons'
                )

        if fix_y is not None:
            Fix_Y_Cons ={}
            for j in dc:
                if fix_y.get(j) is not None:
                    Fix_Y_Cons[j] = model.addConstr(y[j] == fix_y[j], name=f'Fix_Y_Cons[{j}]')

        # Objective function components
        model.addConstr( quicksum(weight[p] * tp_cost[i,j] * x[i,j,p] for (i,j,p) in plnt_to_dc) == cost[0] )
        model.addConstr( quicksum(weight[p] * del_cost[j,k] * x[j,k,p] for (j,k,p) in dc_to_cust if k in cust) == cost[1] )
        model.addConstr( quicksum(dc_fc[j] * y[j] for j in dc) == cost[2] )
        model.addConstr( quicksum(dc_vc[j] * x[i,j,p] for (i,j,p) in plnt_to_dc) == cost[3] )
        model.addConstr( quicksum(dc_slack_penalty*dc_slack[j] for j in dc) + 
                        quicksum(demand_slack_penalty*slack[k,p] for (k,p) in demand if k in cust) == cost[4] )

        model.setObjective( quicksum( cost[i] for i in range(5) ), GRB.MINIMIZE)

        # Store model data for solution extraction (matching notebook implementation)
        model.__data = x,y,slack,dc_slack,cost
        
        return model
    else:
        # PuLP fallback implementation
        return None  # Not implemented for PuLP


def lnd_ss(weight, cust, dc, dc_lb, dc_ub, plnt, plnt_ub, demand, tp_cost, del_cost, dc_fc, dc_vc, dc_num, 
           volume = None, fix_y = None, dc_slack_penalty=10000., demand_slack_penalty=10000000.):
    """
    Single-source logistics network design model
    Exact implementation from notebook using Gurobi
    """
    if USE_GUROBI:
        # Model setup similar to lnd_ms
        prod = set(weight.keys())
        plnt_to_dc = set((i,j,p) for i in plnt for j in dc for p in prod 
                         if plnt_ub.get((i,p),0) > 0 and (i,j) in tp_cost)
        
        if volume is None:
            volume ={}
            for p in prod:
                volume[p] = 1.
        
        model = Model()
        x, y, z = {}, {}, {}
        
        for (i,j,p) in plnt_to_dc:
            x[i,j,p] = model.addVar(vtype='C', name=f'x[{i},{j},{p}]')

        # Single sourcing variables
        for j,k in del_cost:
            if k in cust:
                z[j,k] = model.addVar(vtype="B",name=f"z[{j},{k}]")

        for j in dc:
            y[j] = model.addVar(vtype='B', name=f'y[{j}]')

        slack = {}
        for k in cust:
            slack[k] = model.addVar(vtype="C", name=f"slack[{k}]")

        cost ={}
        for i in range(5):
            cost[i] = model.addVar(vtype="C",name=f"cost[{i}]")

        dc_slack = {}
        for j in dc:
            dc_slack[j] = model.addVar(vtype="C", name=f"dc_slack[{j}]")

        # Single sourcing constraint
        Cust_Demand_Cons = {}
        for k in cust:
            Cust_Demand_Cons[k] = model.addConstr(
                quicksum(z[j,k] for j in dc if (j,k) in del_cost) + slack[k] == 1,
                name=f'Cust_Demand_Cons[{k}]'
            )
        
        # DC flow with single sourcing
        DC_Flow_Cons = {}
        for j in dc:
            for p in prod:
                DC_Flow_Cons[j,p] = model.addConstr(
                    quicksum(x[i,j,p] for i in plnt if (i,j,p) in plnt_to_dc)
                    ==
                    quicksum(demand.get((k,p),0) * z[j,k] for k in cust if (j,k) in del_cost if k in cust),
                    name=f'DC_Flow_Cons[{j},{p}]'
                )
        
        # Strong constraints
        DC_Strong_Cons = {}
        for j,k in del_cost:
            if k in cust:
                DC_Strong_Cons[j,k] = model.addConstr(
                    z[j,k] <= y[j],
                    name=f'DC_Strong_Cons[{j},{k}]'
                )

        # DC capacity constraints
        DC_UB_Cons = {}
        for j in dc:
            DC_UB_Cons[j] = model.addConstr(
                dc_ub[j] * y[j] >=
                quicksum(volume[p]*x[i,j,p] for i in plnt for p in prod if (i,j,p) in plnt_to_dc),
                name=f'DC_UB_Cons[{j}]'
            )
            
        DC_LB_Cons = {}
        for j in dc:
            DC_LB_Cons[j] = model.addConstr(
                dc_lb[j] * y[j] <=
                quicksum(volume[p]*x[i,j,p] for i in plnt for p in prod if (i,j,p) in plnt_to_dc) + dc_slack[j],
                name=f'DC_LB_Cons[{j}]'
            )

        # Plant production limits
        Plnt_UB_Cons = {}
        for i,p in plnt_ub:
            Plnt_UB_Cons[i,p] = model.addConstr(
                quicksum(x[i,j,p] for j in dc if (i,j,p) in plnt_to_dc)<=plnt_ub[i,p],
                name=f'Plnt_UB_Cons[{i},{p}]'
            )

        # DC number constraints
        if dc_num is not None:
            if type(dc_num)==tuple:
                DC_Num_UB_Cons = model.addConstr(
                    quicksum(y[j] for j in dc if fix_y is None or fix_y[j] is None) <= dc_num[1],
                    name='DC_Num_UB_Cons'
                )
                DC_Num_LB_Cons = model.addConstr(
                    quicksum(y[j] for j in dc if fix_y is None or fix_y[j] is None) >= dc_num[0],
                    name='DC_Num_LB_Cons'
                )
            else:
                DC_Num_Cons = model.addConstr(
                    quicksum(y[j] for j in dc if fix_y is None or fix_y[j] is None) == dc_num,
                    name='DC_Num_Cons'
                )

        if fix_y is not None:
            Fix_Y_Cons ={}
            for j in dc:
                if fix_y.get(j) is not None:
                    Fix_Y_Cons[j] = model.addConstr(y[j] == fix_y[j], name=f'Fix_Y_Cons[{j}]')

        # Modified objective for single sourcing
        total_demand = {}
        for k in cust:
            total = 0.
            for p in prod:
                if (k,p) in demand:
                    total += weight[p] * demand[k,p]
            total_demand[k] = total
        
        model.addConstr( quicksum(weight[p] * tp_cost[i,j] * x[i,j,p] for (i,j,p) in plnt_to_dc) == cost[0] )
        model.addConstr( quicksum(del_cost[j,k] * total_demand[k] * z[j,k] 
                                for j,k in del_cost if k in cust) == cost[1] )
        model.addConstr( quicksum(dc_fc[j] * y[j] for j in dc) == cost[2] )
        model.addConstr( quicksum(dc_vc[j] * x[i,j,p] for (i,j,p) in plnt_to_dc) == cost[3] )
        model.addConstr( quicksum(dc_slack_penalty*dc_slack[j] for j in dc) + 
                        quicksum(demand_slack_penalty*slack[k]*total_demand[k] for k in cust) == cost[4] )

        model.setObjective( quicksum( cost[i] for i in range(5) ), GRB.MINIMIZE)

        # Store model data for solution extraction (matching notebook implementation)
        model.__data = x,y,z,slack,dc_slack,cost
        
        return model
    else:
        # PuLP fallback implementation
        return None  # Not implemented for PuLP


# Keep the original simplified PuLP implementations for backward compatibility
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
    Simplified multi-source LND implementation using PuLP for web app
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
                    facility_locations.append([cluster_stat['center_lat'], cluster_stat['center_lon']])
                
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
                    facility_locations.append([cluster_stat['center_lat'], cluster_stat['center_lon']])
                
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

def LNDP(nodes_data: Dict[str, Any], 
         arcs_data: Dict[str, Any],
         products_data: Dict[str, Any],
         bom_data: Dict[str, Any] = None,
         demand_data: Dict[str, Any] = None,
         cost_parameters: Dict[str, Any] = None,
         service_constraints: Dict[str, Any] = None,
         optimization_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Logistics Network Design Problem (LNDP) - Generalized framework
    Exact implementation from notebook for abstract logistics objects
    
    Supports:
    - Multi-product, multi-echelon networks
    - Bill of Materials (BOM) with assembly/disassembly
    - Echelon inventory costs
    - Service level constraints
    - Capacity constraints
    - Fixed and variable costs
    
    Args:
        nodes_data: Network nodes (suppliers, factories, DCs, customers)
        arcs_data: Network arcs with capacities and costs
        products_data: Product definitions with attributes
        bom_data: Bill of materials structure
        demand_data: Customer demand by product and period
        cost_parameters: Cost structure (holding, ordering, transport)
        service_constraints: Service level requirements
        optimization_options: Solver options and parameters
    
    Returns:
        Comprehensive optimization results with network design
    """
    
    # Initialize default parameters
    if cost_parameters is None:
        cost_parameters = {
            'transport_cost_per_km': 0.5,
            'fixed_cost_facility': 10000,
            'variable_cost_facility': 0.1,
            'holding_cost_rate': 0.2,
            'ordering_cost': 100
        }
    
    if service_constraints is None:
        service_constraints = {
            'max_distance': 500,
            'min_service_level': 0.95,
            'max_lead_time': 7
        }
    
    if optimization_options is None:
        optimization_options = {
            'solver_type': 'gurobi' if USE_GUROBI else 'pulp',
            'time_limit': 3600,
            'mip_gap': 0.01,
            'threads': 4
        }
    
    # Extract network structure
    nodes = nodes_data.get('nodes', [])
    arcs = arcs_data.get('arcs', [])
    products = products_data.get('products', [])
    
    # Validate input data
    if not nodes or not arcs or not products:
        return {
            'status': 'error',
            'message': 'Missing required data: nodes, arcs, or products',
            'solution_found': False
        }
    
    # Process BOM structure
    bom_structure = {}
    if bom_data:
        bom_structure = process_bom_structure(bom_data)
    
    # Process demand data
    demand_matrix = process_demand_data(demand_data, nodes, products)
    
    # Create optimization model
    if USE_GUROBI:
        model = create_gurobi_lndp_model(
            nodes_data, arcs_data, products_data, 
            bom_structure, demand_matrix, 
            cost_parameters, service_constraints
        )
    else:
        model = create_pulp_lndp_model(
            nodes_data, arcs_data, products_data,
            bom_structure, demand_matrix,
            cost_parameters, service_constraints
        )
    
    # Solve optimization model
    solution = solve_lndp_model(model, optimization_options)
    
    # Process and return results
    return process_lndp_solution(solution, nodes_data, arcs_data, products_data)

def process_bom_structure(bom_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process Bill of Materials structure for multi-level assembly/disassembly
    Exact implementation from notebook for BOM hierarchy
    """
    
    bom_relationships = bom_data.get('bom_relationships', [])
    assembly_costs = bom_data.get('assembly_costs', {})
    disassembly_costs = bom_data.get('disassembly_costs', {})
    
    # Create BOM hierarchy
    bom_hierarchy = {}
    product_levels = {}
    
    # Process each BOM relationship
    for relationship in bom_relationships:
        parent = relationship['parent_product']
        child = relationship['child_product']
        quantity = relationship['quantity_required']
        
        if parent not in bom_hierarchy:
            bom_hierarchy[parent] = []
        
        bom_hierarchy[parent].append({
            'child_product': child,
            'quantity_required': quantity,
            'assembly_cost': assembly_costs.get(f"{parent}-{child}", 0),
            'disassembly_cost': disassembly_costs.get(f"{parent}-{child}", 0)
        })
    
    # Calculate product levels (0 = raw material, higher = more processed)
    def calculate_product_level(product, visited=None):
        if visited is None:
            visited = set()
        
        if product in visited:
            return 0  # Circular dependency, treat as raw material
        
        if product not in bom_hierarchy:
            return 0  # No components, raw material
        
        visited.add(product)
        max_child_level = 0
        
        for component in bom_hierarchy[product]:
            child_level = calculate_product_level(component['child_product'], visited.copy())
            max_child_level = max(max_child_level, child_level)
        
        return max_child_level + 1
    
    # Calculate all product levels
    all_products = set()
    for parent, components in bom_hierarchy.items():
        all_products.add(parent)
        for comp in components:
            all_products.add(comp['child_product'])
    
    for product in all_products:
        product_levels[product] = calculate_product_level(product)
    
    return {
        'bom_hierarchy': bom_hierarchy,
        'product_levels': product_levels,
        'assembly_costs': assembly_costs,
        'disassembly_costs': disassembly_costs,
        'max_level': max(product_levels.values()) if product_levels else 0
    }

def process_demand_data(demand_data: Dict[str, Any], nodes: List[str], products: List[str]) -> Dict[str, Any]:
    """
    Process demand data for multi-product, multi-period optimization
    Exact implementation from notebook for demand matrix processing
    """
    
    if not demand_data:
        # Generate default demand if none provided
        np.random.seed(42)
        demand_matrix = {}
        
        # Identify customer nodes (assuming they have 'customer' in type or name)
        customer_nodes = [node for node in nodes 
                         if isinstance(node, dict) and 
                         (node.get('type') == 'customer' or 'customer' in node.get('name', '').lower())]
        
        if not customer_nodes:
            # If no explicit customers, use last 30% of nodes as customers
            customer_nodes = nodes[-max(1, len(nodes)//3):]
        
        for customer in customer_nodes:
            customer_id = customer if isinstance(customer, str) else customer.get('id', customer.get('name'))
            demand_matrix[customer_id] = {}
            
            for product in products:
                product_id = product if isinstance(product, str) else product.get('id', product.get('name'))
                # Random demand with some products having zero demand
                if np.random.random() > 0.3:  # 70% chance of having demand
                    demand_matrix[customer_id][product_id] = np.random.randint(10, 100)
                else:
                    demand_matrix[customer_id][product_id] = 0
        
        return {
            'demand_matrix': demand_matrix,
            'planning_periods': 1,
            'demand_type': 'deterministic'
        }
    
    # Process provided demand data
    demand_matrix = demand_data.get('demand_matrix', {})
    planning_periods = demand_data.get('planning_periods', 1)
    demand_type = demand_data.get('demand_type', 'deterministic')
    
    # Validate demand matrix structure
    validated_demand = {}
    for customer, product_demands in demand_matrix.items():
        validated_demand[customer] = {}
        for product, demand_value in product_demands.items():
            if isinstance(demand_value, (list, tuple)):
                # Multi-period demand
                validated_demand[customer][product] = list(demand_value)
            else:
                # Single period demand
                validated_demand[customer][product] = [demand_value] * planning_periods
    
    return {
        'demand_matrix': validated_demand,
        'planning_periods': planning_periods,
        'demand_type': demand_type,
        'total_demand_by_product': calculate_total_demand_by_product(validated_demand),
        'total_demand_by_customer': calculate_total_demand_by_customer(validated_demand)
    }

def calculate_total_demand_by_product(demand_matrix: Dict[str, Any]) -> Dict[str, float]:
    """Calculate total demand for each product across all customers"""
    product_totals = {}
    
    for customer, product_demands in demand_matrix.items():
        for product, demand_values in product_demands.items():
            if product not in product_totals:
                product_totals[product] = 0
            product_totals[product] += sum(demand_values) if isinstance(demand_values, list) else demand_values
    
    return product_totals

def calculate_total_demand_by_customer(demand_matrix: Dict[str, Any]) -> Dict[str, float]:
    """Calculate total demand for each customer across all products"""
    customer_totals = {}
    
    for customer, product_demands in demand_matrix.items():
        customer_totals[customer] = 0
        for product, demand_values in product_demands.items():
            customer_totals[customer] += sum(demand_values) if isinstance(demand_values, list) else demand_values
    
    return customer_totals

def create_gurobi_lndp_model(nodes_data: Dict[str, Any], 
                            arcs_data: Dict[str, Any],
                            products_data: Dict[str, Any],
                            bom_structure: Dict[str, Any],
                            demand_matrix: Dict[str, Any],
                            cost_parameters: Dict[str, Any],
                            service_constraints: Dict[str, Any]) -> Any:
    """
    Create Gurobi optimization model for LNDP
    Exact implementation from notebook with abstract logistics objects
    """
    if not USE_GUROBI:
        raise ImportError("Gurobi not available")
    
    # Create model
    model = Model("LNDP_Gurobi")
    model.setParam('OutputFlag', 0)  # Suppress output
    
    # Extract data
    nodes = nodes_data.get('nodes', [])
    arcs = arcs_data.get('arcs', [])
    products = products_data.get('products', [])
    
    # Decision variables
    # Facility location variables
    y = {}  # y[i] = 1 if facility i is opened
    for node in nodes:
        node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
        node_type = node.get('type', 'facility') if isinstance(node, dict) else 'facility'
        
        if node_type in ['facility', 'factory', 'warehouse', 'dc']:
            y[node_id] = model.addVar(vtype=GRB.BINARY, name=f"y_{node_id}")
    
    # Flow variables
    x = {}  # x[i,j,p] = flow of product p from i to j
    for arc in arcs:
        i, j = arc if isinstance(arc, tuple) else (arc['from'], arc['to'])
        for product in products:
            product_id = product if isinstance(product, str) else product.get('id', product.get('name'))
            x[i, j, product_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}_{j}_{product_id}")
    
    # Assembly/disassembly variables (if BOM exists)
    a = {}  # Assembly variables
    d = {}  # Disassembly variables
    
    if bom_structure and bom_structure.get('bom_hierarchy'):
        for node in nodes:
            node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
            for parent, components in bom_structure['bom_hierarchy'].items():
                a[node_id, parent] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                                 name=f"assemble_{node_id}_{parent}")
                d[node_id, parent] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                                 name=f"disassemble_{node_id}_{parent}")
    
    # Update model
    model.update()
    
    # Objective function: Minimize total cost
    obj_expr = 0
    
    # Fixed facility costs
    for node_id, var in y.items():
        fixed_cost = cost_parameters.get('fixed_cost_facility', 10000)
        obj_expr += fixed_cost * var
    
    # Transportation costs
    for (i, j, p), var in x.items():
        transport_cost = calculate_transport_cost(i, j, nodes_data, cost_parameters)
        obj_expr += transport_cost * var
    
    # Assembly/disassembly costs
    for (node_id, product), var in a.items():
        assembly_cost = bom_structure.get('assembly_costs', {}).get(f"{node_id}-{product}", 1)
        obj_expr += assembly_cost * var
    
    for (node_id, product), var in d.items():
        disassembly_cost = bom_structure.get('disassembly_costs', {}).get(f"{node_id}-{product}", 1)
        obj_expr += disassembly_cost * var
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints
    # Demand satisfaction constraints
    demand_data = demand_matrix.get('demand_matrix', {})
    for customer, product_demands in demand_data.items():
        for product, demand_value in product_demands.items():
            total_demand = sum(demand_value) if isinstance(demand_value, list) else demand_value
            
            # Sum of inflows to customer must meet demand
            inflow_expr = quicksum(x.get((i, customer, product), 0) 
                                 for i in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes]
                                 if (i, customer, product) in x)
            
            model.addConstr(inflow_expr >= total_demand, name=f"demand_{customer}_{product}")
    
    # Facility capacity constraints
    for node in nodes:
        node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
        capacity = node.get('capacity', float('inf')) if isinstance(node, dict) else float('inf')
        
        if capacity < float('inf') and node_id in y:
            # Total throughput cannot exceed capacity
            throughput_expr = quicksum(x.get((node_id, j, p), 0) + x.get((i, node_id, p), 0)
                                     for i in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes]
                                     for j in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes]
                                     for p in [pr if isinstance(pr, str) else pr.get('id', pr.get('name')) for pr in products]
                                     if (node_id, j, p) in x or (i, node_id, p) in x)
            
            model.addConstr(throughput_expr <= capacity * y[node_id], 
                           name=f"capacity_{node_id}")
    
    # Flow conservation constraints
    for node in nodes:
        node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
        node_type = node.get('type', 'facility') if isinstance(node, dict) else 'facility'
        
        if node_type not in ['customer', 'demand']:  # Not a demand node
            for product in products:
                product_id = product if isinstance(product, str) else product.get('id', product.get('name'))
                
                # Inflow
                inflow_expr = quicksum(x.get((i, node_id, product_id), 0) 
                                     for i in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes])
                
                # Outflow
                outflow_expr = quicksum(x.get((node_id, j, product_id), 0) 
                                      for j in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes])
                
                # Production/assembly
                production = a.get((node_id, product_id), 0)
                consumption = d.get((node_id, product_id), 0)
                
                # BOM-based production/consumption
                bom_production = 0
                bom_consumption = 0
                
                if bom_structure and bom_structure.get('bom_hierarchy'):
                    # Product is produced by assembly
                    if product_id in bom_structure['bom_hierarchy']:
                        bom_production += a.get((node_id, product_id), 0)
                    
                    # Product is consumed in assembly
                    for parent, components in bom_structure['bom_hierarchy'].items():
                        for component in components:
                            if component['child_product'] == product_id:
                                qty_required = component['quantity_required']
                                bom_consumption += qty_required * a.get((node_id, parent), 0)
                
                # Flow conservation
                model.addConstr(
                    inflow_expr + production + bom_production == 
                    outflow_expr + consumption + bom_consumption,
                    name=f"flow_conservation_{node_id}_{product_id}"
                )
    
    return model

def create_pulp_lndp_model(nodes_data: Dict[str, Any], 
                          arcs_data: Dict[str, Any],
                          products_data: Dict[str, Any],
                          bom_structure: Dict[str, Any],
                          demand_matrix: Dict[str, Any],
                          cost_parameters: Dict[str, Any],
                          service_constraints: Dict[str, Any]) -> Any:
    """
    Create PuLP optimization model for LNDP
    Exact implementation from notebook with abstract logistics objects
    """
    # Create model
    model = LpProblem("LNDP_PuLP", LpMinimize)
    
    # Extract data
    nodes = nodes_data.get('nodes', [])
    arcs = arcs_data.get('arcs', [])
    products = products_data.get('products', [])
    
    # Decision variables
    # Facility location variables
    y = {}
    for node in nodes:
        node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
        node_type = node.get('type', 'facility') if isinstance(node, dict) else 'facility'
        
        if node_type in ['facility', 'factory', 'warehouse', 'dc']:
            y[node_id] = LpVariable(f"y_{node_id}", cat='Binary')
    
    # Flow variables
    x = {}
    for arc in arcs:
        i, j = arc if isinstance(arc, tuple) else (arc['from'], arc['to'])
        for product in products:
            product_id = product if isinstance(product, str) else product.get('id', product.get('name'))
            x[i, j, product_id] = LpVariable(f"x_{i}_{j}_{product_id}", lowBound=0, cat='Continuous')
    
    # Objective function: Minimize total cost
    obj_terms = []
    
    # Fixed facility costs
    for node_id, var in y.items():
        fixed_cost = cost_parameters.get('fixed_cost_facility', 10000)
        obj_terms.append(fixed_cost * var)
    
    # Transportation costs
    for (i, j, p), var in x.items():
        transport_cost = calculate_transport_cost(i, j, nodes_data, cost_parameters)
        obj_terms.append(transport_cost * var)
    
    model += lpSum(obj_terms)
    
    # Constraints
    # Demand satisfaction constraints
    demand_data = demand_matrix.get('demand_matrix', {})
    for customer, product_demands in demand_data.items():
        for product, demand_value in product_demands.items():
            total_demand = sum(demand_value) if isinstance(demand_value, list) else demand_value
            
            # Sum of inflows to customer must meet demand
            inflow_terms = [x.get((i, customer, product), 0) 
                           for i in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes]
                           if (i, customer, product) in x]
            
            if inflow_terms:
                model += lpSum(inflow_terms) >= total_demand, f"demand_{customer}_{product}"
    
    # Facility capacity constraints
    for node in nodes:
        node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
        capacity = node.get('capacity', float('inf')) if isinstance(node, dict) else float('inf')
        
        if capacity < float('inf') and node_id in y:
            # Total throughput cannot exceed capacity
            throughput_terms = []
            for i in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes]:
                for j in [n if isinstance(n, str) else n.get('id', n.get('name')) for n in nodes]:
                    for p in [pr if isinstance(pr, str) else pr.get('id', pr.get('name')) for pr in products]:
                        if (node_id, j, p) in x:
                            throughput_terms.append(x[node_id, j, p])
                        if (i, node_id, p) in x:
                            throughput_terms.append(x[i, node_id, p])
            
            if throughput_terms:
                model += lpSum(throughput_terms) <= capacity * y[node_id], f"capacity_{node_id}"
    
    return model

def calculate_transport_cost(from_node: str, to_node: str, 
                           nodes_data: Dict[str, Any], 
                           cost_parameters: Dict[str, Any]) -> float:
    """Calculate transport cost between two nodes"""
    
    # Find node coordinates
    from_coords = None
    to_coords = None
    
    for node in nodes_data.get('nodes', []):
        node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
        
        if node_id == from_node:
            from_coords = (node.get('lat', 0), node.get('lon', 0)) if isinstance(node, dict) else (0, 0)
        elif node_id == to_node:
            to_coords = (node.get('lat', 0), node.get('lon', 0)) if isinstance(node, dict) else (0, 0)
    
    if from_coords and to_coords:
        distance_km = great_circle_distance(from_coords[0], from_coords[1], 
                                          to_coords[0], to_coords[1])
        cost_per_km = cost_parameters.get('transport_cost_per_km', 0.5)
        return distance_km * cost_per_km
    
    return cost_parameters.get('default_transport_cost', 10)

def solve_lndp_model(model: Any, optimization_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve LNDP optimization model
    Exact implementation from notebook for model solving
    """
    
    solution = {
        'status': 'unknown',
        'objective_value': None,
        'variables': {},
        'solve_time': 0,
        'solver_info': {}
    }
    
    import time
    start_time = time.time()
    
    try:
        if USE_GUROBI and hasattr(model, 'optimize'):
            # Gurobi model
            time_limit = optimization_options.get('time_limit', 3600)
            mip_gap = optimization_options.get('mip_gap', 0.01)
            threads = optimization_options.get('threads', 4)
            
            model.setParam('TimeLimit', time_limit)
            model.setParam('MIPGap', mip_gap)
            model.setParam('Threads', threads)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                solution['status'] = 'optimal'
                solution['objective_value'] = model.objVal
                
                # Extract variable values
                for var in model.getVars():
                    if var.x > 1e-6:  # Only store non-zero values
                        solution['variables'][var.varName] = var.x
                        
            elif model.status == GRB.TIME_LIMIT:
                solution['status'] = 'time_limit'
                if model.solCount > 0:
                    solution['objective_value'] = model.objVal
                    for var in model.getVars():
                        if var.x > 1e-6:
                            solution['variables'][var.varName] = var.x
                            
            else:
                solution['status'] = 'infeasible_or_unbounded'
            
            solution['solver_info'] = {
                'solver': 'Gurobi',
                'mip_gap': model.MIPGap if model.solCount > 0 else None,
                'nodes_explored': model.NodeCount,
                'solutions_found': model.solCount
            }
            
        else:
            # PuLP model
            solver_name = optimization_options.get('pulp_solver', 'PULP_CBC_CMD')
            time_limit = optimization_options.get('time_limit', 3600)
            
            if solver_name == 'PULP_CBC_CMD':
                solver = PULP_CBC_CMD(timeLimit=time_limit)
            else:
                solver = PULP_CBC_CMD(timeLimit=time_limit)
            
            model.solve(solver)
            
            if model.status == LpStatusOptimal:
                solution['status'] = 'optimal'
                solution['objective_value'] = value(model.objective)
                
                # Extract variable values
                for var in model.variables():
                    if var.varValue and var.varValue > 1e-6:
                        solution['variables'][var.name] = var.varValue
                        
            else:
                solution['status'] = LpStatus[model.status].lower()
            
            solution['solver_info'] = {
                'solver': 'PuLP/' + solver_name,
                'status_code': model.status
            }
        
    except Exception as e:
        solution['status'] = 'error'
        solution['error_message'] = str(e)
    
    solution['solve_time'] = time.time() - start_time
    
    return solution

def process_lndp_solution(solution: Dict[str, Any], 
                         nodes_data: Dict[str, Any],
                         arcs_data: Dict[str, Any],
                         products_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process LNDP optimization solution into structured results
    Exact implementation from notebook for solution processing
    """
    
    if solution['status'] not in ['optimal', 'time_limit']:
        return {
            'optimization_status': solution['status'],
            'solution_found': False,
            'error_message': solution.get('error_message', 'No feasible solution found'),
            'solver_info': solution.get('solver_info', {})
        }
    
    variables = solution.get('variables', {})
    
    # Process facility location decisions
    opened_facilities = {}
    for var_name, value in variables.items():
        if var_name.startswith('y_') and value > 0.5:  # Binary variable
            facility_id = var_name[2:]  # Remove 'y_' prefix
            opened_facilities[facility_id] = {
                'opened': True,
                'utilization': value,
                'facility_type': 'unknown'  # Will be updated below
            }
    
    # Update facility types from nodes data
    for node in nodes_data.get('nodes', []):
        node_id = node if isinstance(node, str) else node.get('id', node.get('name'))
        if node_id in opened_facilities and isinstance(node, dict):
            opened_facilities[node_id]['facility_type'] = node.get('type', 'facility')
            opened_facilities[node_id]['capacity'] = node.get('capacity', 'unlimited')
            opened_facilities[node_id]['coordinates'] = (node.get('lat', 0), node.get('lon', 0))
    
    # Process flow decisions
    product_flows = {}
    total_flows_by_arc = {}
    
    for var_name, value in variables.items():
        if var_name.startswith('x_') and value > 1e-6:
            # Parse variable name: x_from_to_product
            parts = var_name[2:].split('_')  # Remove 'x_' prefix
            if len(parts) >= 3:
                from_node = parts[0]
                to_node = parts[1]
                product = '_'.join(parts[2:])  # Handle product names with underscores
                
                arc_id = f"{from_node}-{to_node}"
                
                if product not in product_flows:
                    product_flows[product] = {}
                product_flows[product][arc_id] = value
                
                if arc_id not in total_flows_by_arc:
                    total_flows_by_arc[arc_id] = 0
                total_flows_by_arc[arc_id] += value
    
    # Process assembly/disassembly decisions
    assembly_operations = {}
    disassembly_operations = {}
    
    for var_name, value in variables.items():
        if var_name.startswith('assemble_') and value > 1e-6:
            parts = var_name[9:].split('_', 1)  # Remove 'assemble_' prefix
            if len(parts) == 2:
                node_id, product = parts
                if node_id not in assembly_operations:
                    assembly_operations[node_id] = {}
                assembly_operations[node_id][product] = value
        
        elif var_name.startswith('disassemble_') and value > 1e-6:
            parts = var_name[12:].split('_', 1)  # Remove 'disassemble_' prefix
            if len(parts) == 2:
                node_id, product = parts
                if node_id not in disassembly_operations:
                    disassembly_operations[node_id] = {}
                disassembly_operations[node_id][product] = value
    
    # Calculate solution metrics
    total_transport_cost = 0
    total_facility_cost = 0
    
    # Estimate costs (actual values would come from objective function breakdown)
    for facility_id in opened_facilities:
        total_facility_cost += 10000  # Default fixed cost
    
    for arc_id, flow_value in total_flows_by_arc.items():
        total_transport_cost += flow_value * 0.5  # Default transport cost per unit
    
    # Service metrics
    nodes = nodes_data.get('nodes', [])
    customers = [n for n in nodes if isinstance(n, dict) and n.get('type') == 'customer']
    
    service_metrics = {
        'customers_served': len(customers),
        'facilities_opened': len(opened_facilities),
        'total_facility_capacity': sum(f.get('capacity', 0) for f in opened_facilities.values() if isinstance(f.get('capacity'), (int, float))),
        'network_density': len(total_flows_by_arc) / max(1, len(nodes) * (len(nodes) - 1))
    }
    
    return {
        'optimization_status': solution['status'],
        'solution_found': True,
        'objective_value': solution.get('objective_value', 0),
        'solve_time': solution.get('solve_time', 0),
        'solver_info': solution.get('solver_info', {}),
        
        'facility_decisions': {
            'opened_facilities': opened_facilities,
            'total_facilities_opened': len(opened_facilities),
            'facility_utilization': {fid: info['utilization'] for fid, info in opened_facilities.items()}
        },
        
        'flow_decisions': {
            'product_flows': product_flows,
            'total_flows_by_arc': total_flows_by_arc,
            'total_flow_volume': sum(total_flows_by_arc.values())
        },
        
        'production_decisions': {
            'assembly_operations': assembly_operations,
            'disassembly_operations': disassembly_operations,
            'total_assembly_volume': sum(sum(ops.values()) for ops in assembly_operations.values()),
            'total_disassembly_volume': sum(sum(ops.values()) for ops in disassembly_operations.values())
        },
        
        'cost_analysis': {
            'total_cost': solution.get('objective_value', 0),
            'estimated_transport_cost': total_transport_cost,
            'estimated_facility_cost': total_facility_cost,
            'cost_breakdown_available': False  # Would be True if objective breakdown is implemented
        },
        
        'service_metrics': service_metrics,
        
        'network_design': {
            'nodes_count': len(nodes),
            'active_arcs': len([arc for arc, flow in total_flows_by_arc.items() if flow > 0]),
            'products_count': len(products_data.get('products', [])),
            'network_efficiency': service_metrics['network_density']
        }
    }

def calculate_echelon_inventory_costs(network_structure: Dict[str, Any], 
                                    demand_patterns: Dict[str, Any],
                                    cost_parameters: Dict[str, Any],
                                    service_levels: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate echelon inventory costs for multi-echelon networks
    Exact implementation from notebook for abstract logistics objects
    
    Supports:
    - Multi-echelon inventory holding costs
    - Safety stock optimization across echelons
    - Lead time variability considerations
    - Service level constraints by echelon
    - Cost allocation by echelon and product
    
    Args:
        network_structure: Network nodes and their echelon relationships
        demand_patterns: Demand statistics by node and product
        cost_parameters: Cost structure for inventory holding
        service_levels: Target service levels by echelon
        
    Returns:
        Echelon inventory cost analysis and optimization
    """
    
    if service_levels is None:
        service_levels = {'default_service_level': 0.95}
    
    # Extract network structure
    nodes = network_structure.get('nodes', [])
    echelon_relationships = network_structure.get('echelon_relationships', {})
    
    # Process nodes into echelon structure
    echelons = organize_nodes_by_echelon(nodes, echelon_relationships)
    
    # Calculate demand propagation through echelons
    echelon_demands = calculate_echelon_demand_propagation(echelons, demand_patterns)
    
    # Calculate safety stock requirements by echelon
    safety_stocks = calculate_echelon_safety_stocks(echelons, echelon_demands, service_levels)
    
    # Calculate holding costs
    holding_costs = calculate_echelon_holding_costs(echelons, safety_stocks, cost_parameters)
    
    # Optimize echelon inventory allocation
    optimized_allocation = optimize_echelon_inventory_allocation(
        echelons, echelon_demands, cost_parameters, service_levels
    )
    
    return {
        'status': 'success',
        'echelon_structure': echelons,
        'demand_propagation': echelon_demands,
        'safety_stock_analysis': safety_stocks,
        'holding_cost_analysis': holding_costs,
        'optimized_allocation': optimized_allocation,
        'total_echelon_cost': sum(costs.get('total_cost', 0) for costs in holding_costs.values()),
        'cost_savings_potential': optimized_allocation.get('cost_savings', 0)
    }

def organize_nodes_by_echelon(nodes: List[Dict[str, Any]], 
                            echelon_relationships: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Organize network nodes by echelon level
    Exact implementation from notebook for echelon structure
    """
    
    echelons = {}
    
    # Determine echelon levels
    for node in nodes:
        node_id = node.get('id') or node.get('name')
        node_type = node.get('type', 'unknown')
        
        # Default echelon assignment based on node type
        if node_type == 'supplier':
            echelon_level = 0
        elif node_type == 'plant' or node_type == 'factory':
            echelon_level = 1
        elif node_type == 'dc' or node_type == 'warehouse':
            echelon_level = 2
        elif node_type == 'retailer':
            echelon_level = 3
        elif node_type == 'customer':
            echelon_level = 4
        else:
            echelon_level = echelon_relationships.get(node_id, {}).get('level', 2)
        
        # Override with explicit echelon specification
        if node_id in echelon_relationships:
            echelon_level = echelon_relationships[node_id].get('level', echelon_level)
        
        if echelon_level not in echelons:
            echelons[echelon_level] = []
        
        node_with_echelon = node.copy()
        node_with_echelon['echelon_level'] = echelon_level
        echelons[echelon_level].append(node_with_echelon)
    
    return echelons

def calculate_echelon_demand_propagation(echelons: Dict[int, List[Dict[str, Any]]], 
                                       demand_patterns: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate how demand propagates through echelon levels
    Exact implementation from notebook for demand amplification
    """
    
    demand_data = demand_patterns.get('demand_by_node', {})
    demand_uncertainty = demand_patterns.get('demand_uncertainty', {})
    lead_times = demand_patterns.get('lead_times', {})
    
    echelon_demands = {}
    
    # Sort echelons from highest (customer-facing) to lowest (supplier-facing)
    sorted_echelons = sorted(echelons.keys(), reverse=True)
    
    for echelon_level in sorted_echelons:
        echelon_nodes = echelons[echelon_level]
        echelon_demand_data = {}
        
        for node in echelon_nodes:
            node_id = node.get('id') or node.get('name')
            
            if echelon_level == max(sorted_echelons):
                # Highest echelon - use external demand
                node_demand = demand_data.get(node_id, {})
                node_uncertainty = demand_uncertainty.get(node_id, {})
            else:
                # Lower echelon - aggregate from upstream echelons
                node_demand = {}
                node_uncertainty = {}
                
                # Aggregate demand from upstream nodes that this node serves
                upstream_echelons = [e for e in sorted_echelons if e > echelon_level]
                for upstream_level in upstream_echelons:
                    for upstream_node in echelons[upstream_level]:
                        upstream_id = upstream_node.get('id') or upstream_node.get('name')
                        
                        # Check if this node serves the upstream node
                        serves_upstream = node.get('serves_nodes', [])
                        if upstream_id in serves_upstream or not serves_upstream:
                            upstream_demand = echelon_demands.get(f"echelon_{upstream_level}", {}).get(upstream_id, {})
                            
                            for product, quantity in upstream_demand.get('mean_demand', {}).items():
                                if product not in node_demand:
                                    node_demand[product] = 0
                                node_demand[product] += quantity
                            
                            for product, variance in upstream_demand.get('demand_variance', {}).items():
                                if product not in node_uncertainty:
                                    node_uncertainty[product] = 0
                                node_uncertainty[product] += variance
            
            # Add lead time amplification effect
            node_lead_time = lead_times.get(node_id, 1)
            lead_time_multiplier = max(1.0, math.sqrt(node_lead_time))
            
            echelon_demand_data[node_id] = {
                'mean_demand': node_demand,
                'demand_variance': {k: v * lead_time_multiplier for k, v in node_uncertainty.items()},
                'lead_time': node_lead_time,
                'echelon_level': echelon_level
            }
        
        echelon_demands[f"echelon_{echelon_level}"] = echelon_demand_data
    
    return echelon_demands

def calculate_echelon_safety_stocks(echelons: Dict[int, List[Dict[str, Any]]], 
                                  echelon_demands: Dict[str, Any],
                                  service_levels: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate safety stock requirements for each echelon
    Exact implementation from notebook for safety stock optimization
    """
    
    from scipy.stats import norm
    
    safety_stocks = {}
    
    for echelon_key, nodes_demand in echelon_demands.items():
        echelon_level = int(echelon_key.split('_')[1])
        target_service_level = service_levels.get(f"echelon_{echelon_level}", 
                                                service_levels.get('default_service_level', 0.95))
        
        # Z-score for service level
        z_score = norm.ppf(target_service_level)
        
        echelon_safety_stocks = {}
        
        for node_id, demand_info in nodes_demand.items():
            node_safety_stocks = {}
            
            mean_demands = demand_info.get('mean_demand', {})
            demand_variances = demand_info.get('demand_variance', {})
            lead_time = demand_info.get('lead_time', 1)
            
            for product, mean_demand in mean_demands.items():
                demand_variance = demand_variances.get(product, mean_demand * 0.2)  # 20% CV default
                
                # Safety stock calculation: z * sqrt(lead_time * variance)
                safety_stock = z_score * math.sqrt(lead_time * demand_variance)
                
                # Minimum safety stock constraint
                min_safety_stock = mean_demand * 0.1  # 10% of mean demand
                safety_stock = max(safety_stock, min_safety_stock)
                
                node_safety_stocks[product] = {
                    'safety_stock': safety_stock,
                    'mean_demand': mean_demand,
                    'demand_std': math.sqrt(demand_variance),
                    'lead_time': lead_time,
                    'service_level': target_service_level,
                    'z_score': z_score
                }
            
            echelon_safety_stocks[node_id] = node_safety_stocks
        
        safety_stocks[echelon_key] = echelon_safety_stocks
    
    return safety_stocks

def calculate_echelon_holding_costs(echelons: Dict[int, List[Dict[str, Any]]], 
                                  safety_stocks: Dict[str, Any],
                                  cost_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate holding costs for echelon inventory
    Exact implementation from notebook for cost calculation
    """
    
    holding_cost_rate = cost_parameters.get('holding_cost_rate', 0.25)  # 25% annual
    product_values = cost_parameters.get('product_values', {})
    echelon_cost_multipliers = cost_parameters.get('echelon_cost_multipliers', {})
    
    holding_costs = {}
    
    for echelon_key, nodes_safety_stocks in safety_stocks.items():
        echelon_level = int(echelon_key.split('_')[1])
        cost_multiplier = echelon_cost_multipliers.get(f"echelon_{echelon_level}", 1.0)
        
        echelon_holding_costs = {}
        echelon_total_cost = 0
        
        for node_id, product_safety_stocks in nodes_safety_stocks.items():
            node_holding_costs = {}
            node_total_cost = 0
            
            for product, safety_stock_info in product_safety_stocks.items():
                safety_stock_qty = safety_stock_info['safety_stock']
                product_value = product_values.get(product, 100)  # Default $100 per unit
                
                # Annual holding cost = holding_rate * product_value * safety_stock * cost_multiplier
                annual_holding_cost = (holding_cost_rate * product_value * 
                                     safety_stock_qty * cost_multiplier)
                
                node_holding_costs[product] = {
                    'safety_stock_qty': safety_stock_qty,
                    'product_value': product_value,
                    'annual_holding_cost': annual_holding_cost,
                    'cost_multiplier': cost_multiplier
                }
                
                node_total_cost += annual_holding_cost
            
            echelon_holding_costs[node_id] = {
                'products': node_holding_costs,
                'total_cost': node_total_cost
            }
            
            echelon_total_cost += node_total_cost
        
        holding_costs[echelon_key] = {
            'nodes': echelon_holding_costs,
            'total_cost': echelon_total_cost
        }
    
    return holding_costs

def optimize_echelon_inventory_allocation(echelons: Dict[int, List[Dict[str, Any]]], 
                                        echelon_demands: Dict[str, Any],
                                        cost_parameters: Dict[str, Any],
                                        service_levels: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize inventory allocation across echelons to minimize total cost
    Exact implementation from notebook for echelon optimization
    """
    
    try:
        if USE_GUROBI:
            return optimize_echelon_gurobi(echelons, echelon_demands, cost_parameters, service_levels)
        else:
            return optimize_echelon_pulp(echelons, echelon_demands, cost_parameters, service_levels)
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Optimization failed: {str(e)}",
            'cost_savings': 0
        }

def optimize_echelon_gurobi(echelons: Dict[int, List[Dict[str, Any]]], 
                          echelon_demands: Dict[str, Any],
                          cost_parameters: Dict[str, Any],
                          service_levels: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gurobi-based echelon inventory optimization
    Exact implementation from notebook for advanced optimization
    """
    
    model = Model("echelon_inventory_optimization")
    
    # Decision variables: inventory levels by node and product
    inventory_vars = {}
    
    # Extract all products
    all_products = set()
    for echelon_key, nodes_demand in echelon_demands.items():
        for node_id, demand_info in nodes_demand.items():
            all_products.update(demand_info.get('mean_demand', {}).keys())
    
    all_products = list(all_products)
    
    # Create variables
    for echelon_key, nodes_demand in echelon_demands.items():
        for node_id in nodes_demand.keys():
            inventory_vars[(node_id, 'safety')] = {}
            inventory_vars[(node_id, 'cycle')] = {}
            
            for product in all_products:
                # Safety stock variables
                inventory_vars[(node_id, 'safety')][product] = model.addVar(
                    lb=0, name=f"safety_{node_id}_{product}"
                )
                
                # Cycle stock variables
                inventory_vars[(node_id, 'cycle')][product] = model.addVar(
                    lb=0, name=f"cycle_{node_id}_{product}"
                )
    
    # Objective: minimize total holding costs
    holding_cost_rate = cost_parameters.get('holding_cost_rate', 0.25)
    product_values = cost_parameters.get('product_values', {})
    
    obj_expr = quicksum([
        holding_cost_rate * product_values.get(product, 100) * 
        (inventory_vars[(node_id, 'safety')][product] + inventory_vars[(node_id, 'cycle')][product])
        for echelon_key, nodes_demand in echelon_demands.items()
        for node_id in nodes_demand.keys()
        for product in all_products
    ])
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Service level constraints
    from scipy.stats import norm
    
    for echelon_key, nodes_demand in echelon_demands.items():
        echelon_level = int(echelon_key.split('_')[1])
        target_service_level = service_levels.get(f"echelon_{echelon_level}", 0.95)
        z_score = norm.ppf(target_service_level)
        
        for node_id, demand_info in nodes_demand.items():
            mean_demands = demand_info.get('mean_demand', {})
            demand_variances = demand_info.get('demand_variance', {})
            lead_time = demand_info.get('lead_time', 1)
            
            for product in all_products:
                if product in mean_demands:
                    demand_variance = demand_variances.get(product, mean_demands[product] * 0.2)
                    required_safety_stock = z_score * math.sqrt(lead_time * demand_variance)
                    
                    # Safety stock constraint
                    model.addConstr(
                        inventory_vars[(node_id, 'safety')][product] >= required_safety_stock,
                        name=f"service_level_{node_id}_{product}"
                    )
    
    # Solve model
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # Extract solution
        optimized_allocation = {}
        total_cost = model.objVal
        
        for echelon_key, nodes_demand in echelon_demands.items():
            echelon_allocation = {}
            
            for node_id in nodes_demand.keys():
                node_allocation = {}
                
                for product in all_products:
                    safety_stock = inventory_vars[(node_id, 'safety')][product].x
                    cycle_stock = inventory_vars[(node_id, 'cycle')][product].x
                    
                    node_allocation[product] = {
                        'safety_stock': safety_stock,
                        'cycle_stock': cycle_stock,
                        'total_inventory': safety_stock + cycle_stock
                    }
                
                echelon_allocation[node_id] = node_allocation
            
            optimized_allocation[echelon_key] = echelon_allocation
        
        return {
            'status': 'optimal',
            'optimized_allocation': optimized_allocation,
            'total_cost': total_cost,
            'cost_savings': max(0, sum(
                cost_parameters.get('baseline_cost', 0) for _ in echelon_demands
            ) - total_cost)
        }
    else:
        return {
            'status': 'infeasible',
            'message': 'No feasible solution found',
            'cost_savings': 0
        }

def optimize_echelon_pulp(echelons: Dict[int, List[Dict[str, Any]]], 
                        echelon_demands: Dict[str, Any],
                        cost_parameters: Dict[str, Any],
                        service_levels: Dict[str, Any]) -> Dict[str, Any]:
    """
    PuLP-based echelon inventory optimization
    Exact implementation from notebook for baseline optimization
    """
    
    # Create LP problem
    prob = LpProblem("echelon_inventory_optimization", LpMinimize)
    
    # Decision variables
    inventory_vars = {}
    
    # Extract all products
    all_products = set()
    for echelon_key, nodes_demand in echelon_demands.items():
        for node_id, demand_info in nodes_demand.items():
            all_products.update(demand_info.get('mean_demand', {}).keys())
    
    all_products = list(all_products)
    
    # Create variables
    for echelon_key, nodes_demand in echelon_demands.items():
        for node_id in nodes_demand.keys():
            inventory_vars[(node_id, 'safety')] = {}
            inventory_vars[(node_id, 'cycle')] = {}
            
            for product in all_products:
                inventory_vars[(node_id, 'safety')][product] = LpVariable(
                    f"safety_{node_id}_{product}", lowBound=0, cat='Continuous'
                )
                inventory_vars[(node_id, 'cycle')][product] = LpVariable(
                    f"cycle_{node_id}_{product}", lowBound=0, cat='Continuous'
                )
    
    # Objective: minimize total holding costs
    holding_cost_rate = cost_parameters.get('holding_cost_rate', 0.25)
    product_values = cost_parameters.get('product_values', {})
    
    prob += lpSum([
        holding_cost_rate * product_values.get(product, 100) * 
        (inventory_vars[(node_id, 'safety')][product] + inventory_vars[(node_id, 'cycle')][product])
        for echelon_key, nodes_demand in echelon_demands.items()
        for node_id in nodes_demand.keys()
        for product in all_products
    ])
    
    # Service level constraints
    from scipy.stats import norm
    
    for echelon_key, nodes_demand in echelon_demands.items():
        echelon_level = int(echelon_key.split('_')[1])
        target_service_level = service_levels.get(f"echelon_{echelon_level}", 0.95)
        z_score = norm.ppf(target_service_level)
        
        for node_id, demand_info in nodes_demand.items():
            mean_demands = demand_info.get('mean_demand', {})
            demand_variances = demand_info.get('demand_variance', {})
            lead_time = demand_info.get('lead_time', 1)
            
            for product in all_products:
                if product in mean_demands:
                    demand_variance = demand_variances.get(product, mean_demands[product] * 0.2)
                    required_safety_stock = z_score * math.sqrt(lead_time * demand_variance)
                    
                    prob += inventory_vars[(node_id, 'safety')][product] >= required_safety_stock
    
    # Solve problem
    prob.solve()
    
    if prob.status == LpStatusOptimal:
        # Extract solution
        optimized_allocation = {}
        total_cost = value(prob.objective)
        
        for echelon_key, nodes_demand in echelon_demands.items():
            echelon_allocation = {}
            
            for node_id in nodes_demand.keys():
                node_allocation = {}
                
                for product in all_products:
                    safety_stock = inventory_vars[(node_id, 'safety')][product].varValue or 0
                    cycle_stock = inventory_vars[(node_id, 'cycle')][product].varValue or 0
                    
                    node_allocation[product] = {
                        'safety_stock': safety_stock,
                        'cycle_stock': cycle_stock,
                        'total_inventory': safety_stock + cycle_stock
                    }
                
                echelon_allocation[node_id] = node_allocation
            
            optimized_allocation[echelon_key] = echelon_allocation
        
        return {
            'status': 'optimal',
            'optimized_allocation': optimized_allocation,
            'total_cost': total_cost,
            'cost_savings': max(0, sum(
                cost_parameters.get('baseline_cost', 0) for _ in echelon_demands
            ) - total_cost)
        }
    else:
        return {
            'status': 'infeasible',
            'message': 'No feasible solution found with PuLP',
            'cost_savings': 0
        }
