import pandas as pd
import numpy as np
import io
from typing import Dict, Any

def generate_lnd_sample_data(n_customers: int = 50, 
                            center_lat: float = 35.6762, 
                            center_lon: float = 139.6503,
                            radius_km: float = 50) -> pd.DataFrame:
    """
    Generate sample customer data for Logistics Network Design testing
    
    Args:
        n_customers: Number of customers to generate
        center_lat: Center latitude (default: Tokyo)
        center_lon: Center longitude (default: Tokyo)
        radius_km: Radius in km around center point
    
    Returns:
        DataFrame with customer data including lat, lon, demand, customer_id
    """
    np.random.seed(42)  # For reproducible results
    
    # Convert radius to degrees (approximate)
    radius_deg = radius_km / 111.0  # Roughly 1 degree = 111 km
    
    customers = []
    
    for i in range(n_customers):
        # Generate random points around the center using polar coordinates
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, radius_deg) * np.sqrt(np.random.uniform(0, 1))
        
        # Convert to lat/lon
        lat = center_lat + distance * np.cos(angle)
        lon = center_lon + distance * np.sin(angle) / np.cos(np.radians(center_lat))
        
        # Generate demand with some variation
        base_demand = np.random.lognormal(mean=3, sigma=0.8)  # Log-normal distribution
        demand = max(1, int(base_demand * 10))  # Ensure minimum demand of 1
        
        # Generate customer metadata
        customer_type = np.random.choice(['retail', 'wholesale', 'industrial'], p=[0.6, 0.3, 0.1])
        region = f"Region_{chr(65 + (i % 5))}"  # Regions A-E
        
        customers.append({
            'customer_id': f'CUST_{i+1:03d}',
            'customer_name': f'Customer_{i+1}',
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'demand': demand,
            'customer_type': customer_type,
            'region': region,
            'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.2, 0.6, 0.2])
        })
    
    return pd.DataFrame(customers)

def generate_facility_sample_data(n_facilities: int = 10,
                                 center_lat: float = 35.6762,
                                 center_lon: float = 139.6503,
                                 radius_km: float = 30) -> pd.DataFrame:
    """
    Generate sample facility candidate data
    """
    np.random.seed(123)  # Different seed for facilities
    
    radius_deg = radius_km / 111.0
    
    facilities = []
    
    for i in range(n_facilities):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, radius_deg) * np.sqrt(np.random.uniform(0, 1))
        
        lat = center_lat + distance * np.cos(angle)
        lon = center_lon + distance * np.sin(angle) / np.cos(np.radians(center_lat))
        
        # Generate facility metadata
        capacity = np.random.randint(100, 1000)
        fixed_cost = np.random.randint(10000, 50000)
        
        facilities.append({
            'facility_id': f'FAC_{i+1:03d}',
            'facility_name': f'Facility_{i+1}',
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'capacity': capacity,
            'fixed_cost': fixed_cost,
            'facility_type': np.random.choice(['Distribution Center', 'Warehouse', 'Cross-Dock'])
        })
    
    return pd.DataFrame(facilities)

def create_multiple_source_lnd_datasets() -> Dict[str, str]:
    """
    Create sample datasets for Multiple Source LND and return as CSV strings
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate customers (smaller set for MS-LND complexity)
    n_customers = 15
    customers = []
    for i in range(n_customers):
        # Tokyo area customers
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, 0.3) * np.sqrt(np.random.uniform(0, 1))
        lat = 35.6762 + distance * np.cos(angle)
        lon = 139.6503 + distance * np.sin(angle) / np.cos(np.radians(35.6762))
        
        customers.append({
            'customer_id': f'C{i+1:02d}',
            'customer_name': f'Customer_{i+1}',
            'lat': round(lat, 6),
            'lon': round(lon, 6)
        })
    customers_df = pd.DataFrame(customers)
    
    # Generate warehouses
    n_warehouses = 8
    warehouses = []
    for i in range(n_warehouses):
        # Strategic warehouse locations around Tokyo
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0.1, 0.4) * np.sqrt(np.random.uniform(0, 1))
        lat = 35.6762 + distance * np.cos(angle)
        lon = 139.6503 + distance * np.sin(angle) / np.cos(np.radians(35.6762))
        
        # Random capacity bounds
        lower_bound = np.random.randint(50, 150)
        upper_bound = lower_bound + np.random.randint(200, 500)
        
        warehouses.append({
            'warehouse_id': f'W{i+1:02d}',
            'warehouse_name': f'Warehouse_{i+1}',
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'is_available': 1,  # All warehouses available
            'fixed_cost': np.random.randint(8000, 15000),
            'variable_cost': round(np.random.uniform(0.5, 2.0), 2)
        })
    warehouses_df = pd.DataFrame(warehouses)
    
    # Generate factories
    n_factories = 4
    factories = []
    for i in range(n_factories):
        # Factories positioned somewhat further from city center
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0.2, 0.6) * np.sqrt(np.random.uniform(0, 1))
        lat = 35.6762 + distance * np.cos(angle)
        lon = 139.6503 + distance * np.sin(angle) / np.cos(np.radians(35.6762))
        
        factories.append({
            'factory_id': f'F{i+1:02d}',
            'factory_name': f'Factory_{i+1}',
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'production_cost': round(np.random.uniform(10, 30), 2)
        })
    factories_df = pd.DataFrame(factories)
    
    # Generate products
    n_products = 5
    products = []
    for i in range(n_products):
        products.append({
            'product_id': f'P{i+1:02d}',
            'product_name': f'Product_{i+1}',
            'unit_weight': round(np.random.uniform(0.5, 5.0), 2),
            'unit_volume': round(np.random.uniform(0.1, 2.0), 3),
            'category': np.random.choice(['Electronics', 'Textiles', 'Food', 'Chemicals', 'Machinery'])
        })
    products_df = pd.DataFrame(products)
    
    # Generate demand (customer-product combinations)
    demand_data = []
    for customer_id in customers_df['customer_id']:
        # Each customer demands 2-4 products randomly
        n_products_demanded = np.random.randint(2, 5)
        demanded_products = np.random.choice(products_df['product_id'], n_products_demanded, replace=False)
        
        for product_id in demanded_products:
            demand = np.random.randint(10, 100)
            demand_data.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'demand': demand,
                'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.2, 0.6, 0.2])
            })
    
    demand_df = pd.DataFrame(demand_data)
    
    # Generate factory capacity (factory-product combinations)
    factory_capacity_data = []
    for factory_id in factories_df['factory_id']:
        for product_id in products_df['product_id']:
            # Not all factories produce all products
            if np.random.random() > 0.3:  # 70% chance factory produces this product
                capacity = np.random.randint(200, 800)
                factory_capacity_data.append({
                    'factory_id': factory_id,
                    'product_id': product_id,
                    'capacity': capacity,
                    'production_rate': round(np.random.uniform(0.8, 1.2), 2)
                })
    
    factory_capacity_df = pd.DataFrame(factory_capacity_data)
    
    return {
        'ms_lnd_customers': customers_df.to_csv(index=False),
        'ms_lnd_warehouses': warehouses_df.to_csv(index=False),
        'ms_lnd_factories': factories_df.to_csv(index=False),
        'ms_lnd_products': products_df.to_csv(index=False),
        'ms_lnd_demand': demand_df.to_csv(index=False),
        'ms_lnd_factory_capacity': factory_capacity_df.to_csv(index=False)
    }


def create_elbow_method_datasets() -> Dict[str, str]:
    """
    Create sample datasets specifically for Elbow Method analysis
    """
    # Generate customer data with clear cluster structure for better elbow visualization
    np.random.seed(42)
    
    # Create customers with 3 distinct clusters for clear elbow point at 3
    cluster_centers = [
        (35.6762, 139.6503),  # Tokyo center
        (34.6937, 135.5023),  # Osaka center  
        (35.1815, 136.9066),  # Nagoya center
    ]
    
    customers_data = []
    customer_id = 1
    
    for i, (center_lat, center_lon) in enumerate(cluster_centers):
        # Generate 20 customers per cluster
        for j in range(20):
            # Add some noise to create natural clustering
            lat_offset = np.random.normal(0, 0.05)  # ~5km standard deviation
            lon_offset = np.random.normal(0, 0.05)
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
            
            # Generate realistic demand (higher for central customers)
            base_demand = np.random.uniform(50, 200)
            demand = max(10, base_demand + np.random.normal(0, 20))
            
            customers_data.append({
                'customer_id': f'C{customer_id:03d}',
                'name': f'Customer {customer_id}',
                'lat': round(lat, 6),
                'lon': round(lon, 6),
                'demand': round(demand, 1),
                'region': f'Cluster_{i+1}',
                'city': ['Tokyo', 'Osaka', 'Nagoya'][i]
            })
            customer_id += 1
    
    # Add some scattered customers to make the analysis more realistic
    for j in range(10):
        # Random locations within broader Japan area
        lat = np.random.uniform(34.0, 36.5)
        lon = np.random.uniform(135.0, 140.5)
        demand = np.random.uniform(20, 100)
        
        customers_data.append({
            'customer_id': f'C{customer_id:03d}',
            'name': f'Customer {customer_id}',
            'lat': round(lat, 6),
            'lon': round(lon, 6), 
            'demand': round(demand, 1),
            'region': 'Scattered',
            'city': 'Other'
        })
        customer_id += 1
    
    customers_df = pd.DataFrame(customers_data)
    
    # Create a second dataset with 2 clear clusters
    customers_2cluster_data = []
    cluster_centers_2 = [
        (35.6762, 139.6503),  # Tokyo area
        (34.6937, 135.5023),  # Osaka area
    ]
    
    customer_id = 1
    for i, (center_lat, center_lon) in enumerate(cluster_centers_2):
        for j in range(25):  # 25 customers per cluster
            lat_offset = np.random.normal(0, 0.08)
            lon_offset = np.random.normal(0, 0.08)
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
            demand = np.random.uniform(30, 180)
            
            customers_2cluster_data.append({
                'customer_id': f'C{customer_id:03d}',
                'name': f'Customer {customer_id}',
                'lat': round(lat, 6),
                'lon': round(lon, 6),
                'demand': round(demand, 1),
                'region': f'Zone_{i+1}',
                'city': ['Tokyo', 'Osaka'][i]
            })
            customer_id += 1
    
    customers_2cluster_df = pd.DataFrame(customers_2cluster_data)
    
    # Create a dataset with 5 clusters
    customers_5cluster_data = []
    cluster_centers_5 = [
        (35.6762, 139.6503),  # Tokyo
        (34.6937, 135.5023),  # Osaka
        (35.1815, 136.9066),  # Nagoya
        (33.5904, 130.4017),  # Fukuoka
        (43.0642, 141.3469),  # Sapporo
    ]
    
    customer_id = 1
    for i, (center_lat, center_lon) in enumerate(cluster_centers_5):
        for j in range(15):  # 15 customers per cluster
            lat_offset = np.random.normal(0, 0.06)
            lon_offset = np.random.normal(0, 0.06)
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
            demand = np.random.uniform(40, 160)
            
            customers_5cluster_data.append({
                'customer_id': f'C{customer_id:03d}',
                'name': f'Customer {customer_id}',
                'lat': round(lat, 6),
                'lon': round(lon, 6),
                'demand': round(demand, 1),
                'region': f'Area_{i+1}',
                'city': ['Tokyo', 'Osaka', 'Nagoya', 'Fukuoka', 'Sapporo'][i]
            })
            customer_id += 1
    
    customers_5cluster_df = pd.DataFrame(customers_5cluster_data)
    
    return {
        'elbow_customers_3clusters': customers_df.to_csv(index=False),
        'elbow_customers_2clusters': customers_2cluster_df.to_csv(index=False), 
        'elbow_customers_5clusters': customers_5cluster_df.to_csv(index=False)
    }

def create_lnd_sample_datasets() -> Dict[str, str]:
    """
    Create sample datasets for LND and return as CSV strings
    """
    # Generate customer data
    customers_df = generate_lnd_sample_data(n_customers=50)
    
    # Generate facility candidates
    facilities_df = generate_facility_sample_data(n_facilities=15)
    
    # Create smaller dataset for quick testing
    customers_small_df = generate_lnd_sample_data(n_customers=20, radius_km=30)
    
    # Create regional dataset with multiple clusters
    customers_regional_df = pd.concat([
        generate_lnd_sample_data(n_customers=15, center_lat=35.6762, center_lon=139.6503, radius_km=20),  # Tokyo
        generate_lnd_sample_data(n_customers=15, center_lat=34.6937, center_lon=135.5023, radius_km=20),  # Osaka
        generate_lnd_sample_data(n_customers=10, center_lat=35.1815, center_lon=136.9066, radius_km=15),  # Nagoya
    ], ignore_index=True)
    
    # Reset customer IDs for regional dataset
    customers_regional_df['customer_id'] = [f'CUST_{i+1:03d}' for i in range(len(customers_regional_df))]
    
    # Generate Multiple Source LND datasets
    ms_lnd_datasets = create_multiple_source_lnd_datasets()
    
    # Generate Elbow Method datasets
    elbow_datasets = create_elbow_method_datasets()
    
    return {
        'customers_standard': customers_df.to_csv(index=False),
        'customers_small': customers_small_df.to_csv(index=False),
        'customers_regional': customers_regional_df.to_csv(index=False),
        'facilities': facilities_df.to_csv(index=False),
        **ms_lnd_datasets,
        **elbow_datasets
    }

def get_sample_data_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available sample datasets
    """
    return {
        'customers_standard': {
            'name': 'Standard Customer Dataset',
            'description': '東京周辺の50顧客のサンプルデータ',
            'customers': 50,
            'area': 'Tokyo area (50km radius)',
            'use_case': 'General LND analysis and optimization'
        },
        'customers_small': {
            'name': 'Small Customer Dataset', 
            'description': '東京周辺の20顧客のサンプルデータ（テスト用）',
            'customers': 20,
            'area': 'Tokyo area (30km radius)',
            'use_case': 'Quick testing and algorithm validation'
        },
        'customers_regional': {
            'name': 'Multi-Regional Dataset',
            'description': '東京・大阪・名古屋地域の40顧客のサンプルデータ',
            'customers': 40,
            'area': 'Tokyo, Osaka, Nagoya regions',
            'use_case': 'Multi-cluster analysis and regional facility planning'
        },
        'facilities': {
            'name': 'Facility Candidates',
            'description': '15の施設候補地のサンプルデータ',
            'facilities': 15,
            'area': 'Tokyo area (30km radius)', 
            'use_case': 'Facility location optimization with predefined candidates'
        },
        'ms_lnd_customers': {
            'name': 'Multiple Source LND - Customers',
            'description': '複数拠点LND用の15顧客データ',
            'customers': 15,
            'area': 'Tokyo area (30km radius)',
            'use_case': 'Multiple Source Logistics Network Design optimization'
        },
        'ms_lnd_warehouses': {
            'name': 'Multiple Source LND - Warehouses',
            'description': '複数拠点LND用の8倉庫候補データ（容量制約付き）',
            'warehouses': 8,
            'area': 'Tokyo area (40km radius)',
            'use_case': 'Warehouse selection with capacity constraints'
        },
        'ms_lnd_factories': {
            'name': 'Multiple Source LND - Factories',
            'description': '複数拠点LND用の4工場データ',
            'factories': 4,
            'area': 'Tokyo area (60km radius)',
            'use_case': 'Multi-factory production planning'
        },
        'ms_lnd_products': {
            'name': 'Multiple Source LND - Products',
            'description': '複数拠点LND用の5製品データ',
            'products': 5,
            'area': 'N/A',
            'use_case': 'Multi-product optimization'
        },
        'ms_lnd_demand': {
            'name': 'Multiple Source LND - Demand',
            'description': '顧客-製品需要データ（約40レコード）',
            'records': '~40',
            'area': 'Customer-Product matrix',
            'use_case': 'Demand fulfillment optimization'
        },
        'ms_lnd_factory_capacity': {
            'name': 'Multiple Source LND - Factory Capacity',
            'description': '工場-製品生産能力データ（約14レコード）',
            'records': '~14',
            'area': 'Factory-Product matrix',
            'use_case': 'Production capacity planning'
        },
        'elbow_customers_3clusters': {
            'name': 'Elbow Method - 3 Clusters Dataset',
            'description': 'エルボー法分析用データ（3クラスター構造）',
            'customers': 70,
            'area': 'Tokyo-Osaka-Nagoya regions + scattered',
            'use_case': 'Elbow method analysis with clear optimal point at 3 facilities'
        },
        'elbow_customers_2clusters': {
            'name': 'Elbow Method - 2 Clusters Dataset',
            'description': 'エルボー法分析用データ（2クラスター構造）',
            'customers': 50,
            'area': 'Tokyo-Osaka regions',
            'use_case': 'Elbow method analysis with clear optimal point at 2 facilities'
        },
        'elbow_customers_5clusters': {
            'name': 'Elbow Method - 5 Clusters Dataset', 
            'description': 'エルボー法分析用データ（5クラスター構造）',
            'customers': 75,
            'area': 'Tokyo-Osaka-Nagoya-Fukuoka-Sapporo regions',
            'use_case': 'Elbow method analysis with clear optimal point at 5 facilities'
        }
    }