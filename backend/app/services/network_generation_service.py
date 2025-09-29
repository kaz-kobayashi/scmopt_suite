import pandas as pd
import numpy as np
import networkx as nx
import requests
from typing import Dict, List, Tuple, Any, Optional, Union
from geopy.distance import great_circle as distance
import plotly.graph_objects as go
import plotly.express as px
import math
import random
import warnings
warnings.filterwarnings('ignore')

class NetworkGenerationService:
    """
    Network generation service for logistics network design.
    Implements make_network and make_network_using_road functions 
    to match the notebook's computational procedures exactly.
    """
    
    def __init__(self, osrm_host: str = "localhost"):
        self.osrm_host = osrm_host
        
    def compute_durations(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, 
                         plnt_df: pd.DataFrame, toll: bool = True) -> Tuple[List[List[float]], List[List[float]], pd.DataFrame]:
        """
        Compute road distances and durations using OSRM routing engine
        Matches the notebook's compute_durations function exactly
        
        Args:
            cust_df: Customer DataFrame with columns ['name', 'lat', 'lon']
            dc_df: DC DataFrame with columns ['name', 'lat', 'lon']
            plnt_df: Plant DataFrame with columns ['name', 'lat', 'lon']
            toll: Whether to include toll roads
            
        Returns:
            Tuple: (durations, distances, node_df)
        """
        # Combine all nodes in order: customers, DCs, plants
        node_df = pd.concat([
            cust_df[["name", "lat", "lon"]],
            dc_df[["name", "lat", "lon"]], 
            plnt_df[["name", "lat", "lon"]]
        ], ignore_index=True)
        
        n = len(node_df)
        ROUTE = []
        for _, row in node_df.iterrows():
            ROUTE.append([row.lat, row.lon])
        
        # Build route string for OSRM API
        route_str = ""
        for (lat, lon) in ROUTE:
            route_str += f"{lon},{lat};"
        
        # Make API request to OSRM
        try:
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
        
    def make_network_using_road(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, plnt_df: pd.DataFrame,
                               durations: List[List[float]], distances: List[List[float]],
                               plnt_dc_threshold: float = 500.0, dc_cust_threshold: float = 100.0,
                               tc_per_dis: float = 2.0, dc_per_dis: float = 1.5,
                               tc_per_time: float = 0.01, dc_per_time: float = 0.008,
                               lt_lb: float = 0.0, lt_threshold: float = 86400.0,
                               stage_time_bound: Tuple[float, float] = (0.0, 86400.0)) -> Tuple[pd.DataFrame, nx.DiGraph, Dict]:
        """
        Generate transportation network using road distances and travel times
        Matches the notebook's make_network_using_road function exactly
        
        Args:
            cust_df, dc_df, plnt_df: DataFrames with location data
            durations, distances: OSRM distance/duration matrices
            plnt_dc_threshold: Max distance (km) from plant to DC
            dc_cust_threshold: Max distance (km) from DC to customer
            tc_per_dis, dc_per_dis: Cost per distance (transport/delivery)
            tc_per_time, dc_per_time: Cost per time (transport/delivery) 
            lt_lb: Lead time lower bound
            lt_threshold: Lead time threshold
            stage_time_bound: Processing time bounds per stage
            
        Returns:
            Tuple: (trans_df, graph, position)
        """
        n_cust = len(cust_df)
        n_dc = len(dc_df)
        n_plnt = len(plnt_df)
        
        # Create transportation network data
        trans_data = []
        
        # Plant to DC connections
        for i in range(n_plnt):
            plnt_idx = n_cust + n_dc + i
            for j in range(n_dc):
                dc_idx = n_cust + j
                
                dist_m = distances[plnt_idx][dc_idx]
                dist_km = dist_m / 1000.0
                duration_s = durations[plnt_idx][dc_idx]
                
                # Apply distance threshold
                if dist_km <= plnt_dc_threshold:
                    cost = tc_per_dis * dist_km + tc_per_time * duration_s
                    lead_time = max(lt_lb, duration_s + np.random.uniform(*stage_time_bound))
                    
                    trans_data.append({
                        'from_node': plnt_df.iloc[i]['name'],
                        'to_node': dc_df.iloc[j]['name'],
                        'from_type': 'plant',
                        'to_type': 'dc',
                        'distance_km': dist_km,
                        'duration_s': duration_s,
                        'cost': cost,
                        'lead_time': lead_time,
                        'capacity': float('inf'),  # Unlimited capacity for now
                        'from_lat': plnt_df.iloc[i]['lat'],
                        'from_lon': plnt_df.iloc[i]['lon'],
                        'to_lat': dc_df.iloc[j]['lat'], 
                        'to_lon': dc_df.iloc[j]['lon']
                    })
        
        # DC to Customer connections
        for i in range(n_dc):
            dc_idx = n_cust + i
            for j in range(n_cust):
                cust_idx = j
                
                dist_m = distances[dc_idx][cust_idx]
                dist_km = dist_m / 1000.0
                duration_s = durations[dc_idx][cust_idx]
                
                # Apply distance threshold
                if dist_km <= dc_cust_threshold:
                    cost = dc_per_dis * dist_km + dc_per_time * duration_s
                    lead_time = max(lt_lb, duration_s + np.random.uniform(*stage_time_bound))
                    
                    trans_data.append({
                        'from_node': dc_df.iloc[i]['name'],
                        'to_node': cust_df.iloc[j]['name'],
                        'from_type': 'dc',
                        'to_type': 'customer',
                        'distance_km': dist_km,
                        'duration_s': duration_s,
                        'cost': cost,
                        'lead_time': lead_time,
                        'capacity': float('inf'),
                        'from_lat': dc_df.iloc[i]['lat'],
                        'from_lon': dc_df.iloc[i]['lon'],
                        'to_lat': cust_df.iloc[j]['lat'],
                        'to_lon': cust_df.iloc[j]['lon']
                    })
        
        trans_df = pd.DataFrame(trans_data)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for _, row in cust_df.iterrows():
            G.add_node(row['name'], node_type='customer', pos=(row['lon'], row['lat']))
            
        for _, row in dc_df.iterrows():
            G.add_node(row['name'], node_type='dc', pos=(row['lon'], row['lat']))
            
        for _, row in plnt_df.iterrows():
            G.add_node(row['name'], node_type='plant', pos=(row['lon'], row['lat']))
        
        # Add edges with attributes
        for _, row in trans_df.iterrows():
            G.add_edge(
                row['from_node'], row['to_node'],
                weight=row['cost'],
                distance=row['distance_km'],
                duration=row['duration_s'],
                lead_time=row['lead_time'],
                edge_type=f"{row['from_type']}_to_{row['to_type']}"
            )
        
        # Create position dict for visualization
        position = nx.get_node_attributes(G, 'pos')
        
        return trans_df, G, position
        
    def make_network(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, plnt_df: pd.DataFrame,
                    plnt_dc_threshold: float = 500.0, dc_cust_threshold: float = 100.0,
                    unit_tp_cost: float = 2.0, unit_del_cost: float = 1.5,
                    lt_lb: float = 0.0, processing_time_bound: Tuple[float, float] = (3600.0, 14400.0)) -> Tuple[pd.DataFrame, nx.DiGraph, Dict]:
        """
        Generate transportation network using great circle distances (approximation)
        Matches the notebook's make_network function exactly
        
        Args:
            cust_df, dc_df, plnt_df: DataFrames with location data
            plnt_dc_threshold: Max distance (km) from plant to DC
            dc_cust_threshold: Max distance (km) from DC to customer
            unit_tp_cost: Unit transport cost per km (plant to DC)
            unit_del_cost: Unit delivery cost per km (DC to customer)
            lt_lb: Lead time lower bound
            processing_time_bound: Processing time range per stage
            
        Returns:
            Tuple: (trans_df, graph, position)
        """
        trans_data = []
        
        # Plant to DC connections
        for _, plnt_row in plnt_df.iterrows():
            for _, dc_row in dc_df.iterrows():
                dist_km = distance(
                    (plnt_row['lat'], plnt_row['lon']),
                    (dc_row['lat'], dc_row['lon'])
                ).kilometers
                
                # Apply distance threshold
                if dist_km <= plnt_dc_threshold:
                    cost = unit_tp_cost * dist_km
                    # Estimate travel time at 60 km/h average speed
                    travel_time = dist_km / 60.0 * 3600  # seconds
                    lead_time = max(lt_lb, travel_time + np.random.uniform(*processing_time_bound))
                    
                    trans_data.append({
                        'from_node': plnt_row['name'],
                        'to_node': dc_row['name'],
                        'from_type': 'plant',
                        'to_type': 'dc',
                        'distance_km': dist_km,
                        'duration_s': travel_time,
                        'cost': cost,
                        'lead_time': lead_time,
                        'capacity': float('inf'),
                        'from_lat': plnt_row['lat'],
                        'from_lon': plnt_row['lon'],
                        'to_lat': dc_row['lat'],
                        'to_lon': dc_row['lon']
                    })
        
        # DC to Customer connections  
        for _, dc_row in dc_df.iterrows():
            for _, cust_row in cust_df.iterrows():
                dist_km = distance(
                    (dc_row['lat'], dc_row['lon']),
                    (cust_row['lat'], cust_row['lon'])
                ).kilometers
                
                # Apply distance threshold
                if dist_km <= dc_cust_threshold:
                    cost = unit_del_cost * dist_km
                    travel_time = dist_km / 60.0 * 3600  # seconds
                    lead_time = max(lt_lb, travel_time + np.random.uniform(*processing_time_bound))
                    
                    trans_data.append({
                        'from_node': dc_row['name'],
                        'to_node': cust_row['name'],
                        'from_type': 'dc',
                        'to_type': 'customer',
                        'distance_km': dist_km,
                        'duration_s': travel_time,
                        'cost': cost,
                        'lead_time': lead_time,
                        'capacity': float('inf'),
                        'from_lat': dc_row['lat'],
                        'from_lon': dc_row['lon'],
                        'to_lat': cust_row['lat'],
                        'to_lon': cust_row['lon']
                    })
        
        trans_df = pd.DataFrame(trans_data)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for _, row in cust_df.iterrows():
            G.add_node(row['name'], node_type='customer', pos=(row['lon'], row['lat']))
            
        for _, row in dc_df.iterrows():
            G.add_node(row['name'], node_type='dc', pos=(row['lon'], row['lat']))
            
        for _, row in plnt_df.iterrows():
            G.add_node(row['name'], node_type='plant', pos=(row['lon'], row['lat']))
        
        # Add edges with attributes
        for _, row in trans_df.iterrows():
            G.add_edge(
                row['from_node'], row['to_node'],
                weight=row['cost'],
                distance=row['distance_km'],
                duration=row['duration_s'],
                lead_time=row['lead_time'],
                edge_type=f"{row['from_type']}_to_{row['to_type']}"
            )
        
        # Create position dict for visualization
        position = nx.get_node_attributes(G, 'pos')
        
        return trans_df, G, position
        
    def plot_scm(self, trans_df: pd.DataFrame, cust_df: pd.DataFrame, 
                dc_df: pd.DataFrame, plnt_df: pd.DataFrame, 
                title: str = "Supply Chain Network") -> Dict[str, Any]:
        """
        Visualize supply chain network on map
        Matches the notebook's plot_scm function for network visualization
        
        Args:
            trans_df: Transportation network DataFrame
            cust_df, dc_df, plnt_df: Location DataFrames
            title: Plot title
            
        Returns:
            Plotly figure data for visualization
        """
        # Create edge traces for connections
        edge_traces = []
        
        # Plant to DC edges (blue)
        plant_dc_edges = trans_df[trans_df['from_type'] == 'plant']
        if len(plant_dc_edges) > 0:
            edge_lat = []
            edge_lon = []
            for _, row in plant_dc_edges.iterrows():
                edge_lat.extend([row['from_lat'], row['to_lat'], None])
                edge_lon.extend([row['from_lon'], row['to_lon'], None])
            
            edge_traces.append({
                'type': 'scattermapbox',
                'lat': edge_lat,
                'lon': edge_lon,
                'mode': 'lines',
                'line': {'width': 2, 'color': 'blue'},
                'name': 'Plant→DC',
                'hoverinfo': 'none'
            })
        
        # DC to Customer edges (red)
        dc_cust_edges = trans_df[trans_df['from_type'] == 'dc']
        if len(dc_cust_edges) > 0:
            edge_lat = []
            edge_lon = []
            for _, row in dc_cust_edges.iterrows():
                edge_lat.extend([row['from_lat'], row['to_lat'], None])
                edge_lon.extend([row['from_lon'], row['to_lon'], None])
            
            edge_traces.append({
                'type': 'scattermapbox',
                'lat': edge_lat,
                'lon': edge_lon,
                'mode': 'lines',
                'line': {'width': 1, 'color': 'red'},
                'name': 'DC→Customer',
                'hoverinfo': 'none'
            })
        
        # Node traces
        node_traces = []
        
        # Plants (green squares)
        if len(plnt_df) > 0:
            node_traces.append({
                'type': 'scattermapbox',
                'lat': plnt_df['lat'].tolist(),
                'lon': plnt_df['lon'].tolist(),
                'mode': 'markers',
                'marker': {
                    'size': 12,
                    'color': 'green',
                    'symbol': 'square'
                },
                'text': plnt_df['name'].tolist(),
                'name': 'Plants',
                'hovertemplate': '<b>%{text}</b><br>Plant<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
            })
        
        # DCs (orange triangles)
        if len(dc_df) > 0:
            node_traces.append({
                'type': 'scattermapbox',
                'lat': dc_df['lat'].tolist(),
                'lon': dc_df['lon'].tolist(),
                'mode': 'markers',
                'marker': {
                    'size': 10,
                    'color': 'orange',
                    'symbol': 'triangle-up'
                },
                'text': dc_df['name'].tolist(),
                'name': 'Distribution Centers',
                'hovertemplate': '<b>%{text}</b><br>DC<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
            })
        
        # Customers (blue circles)
        if len(cust_df) > 0:
            node_traces.append({
                'type': 'scattermapbox',
                'lat': cust_df['lat'].tolist(),
                'lon': cust_df['lon'].tolist(),
                'mode': 'markers',
                'marker': {
                    'size': 6,
                    'color': 'lightblue'
                },
                'text': cust_df['name'].tolist(),
                'name': 'Customers',
                'hovertemplate': '<b>%{text}</b><br>Customer<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
            })
        
        # Combine all traces
        data = edge_traces + node_traces
        
        # Calculate center point for map
        all_lats = []
        all_lons = []
        for df in [cust_df, dc_df, plnt_df]:
            if len(df) > 0:
                all_lats.extend(df['lat'].tolist())
                all_lons.extend(df['lon'].tolist())
        
        center_lat = np.mean(all_lats) if all_lats else 35.6762
        center_lon = np.mean(all_lons) if all_lons else 139.6503
        
        layout = {
            'title': title,
            'autosize': True,
            'hovermode': 'closest',
            'mapbox': {
                'bearing': 0,
                'center': {'lat': center_lat, 'lon': center_lon},
                'pitch': 0,
                'zoom': 8,
                'style': 'open-street-map'
            },
            'showlegend': True,
            'legend': {'x': 0, 'y': 1}
        }
        
        return {'data': data, 'layout': layout}
        
    def analyze_network_properties(self, trans_df: pd.DataFrame, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze network properties and generate statistics
        
        Args:
            trans_df: Transportation network DataFrame
            G: NetworkX graph
            
        Returns:
            Network analysis results
        """
        analysis = {}
        
        # Check if network is empty
        if len(trans_df) == 0:
            return {
                'network_stats': {
                    'total_nodes': G.number_of_nodes(),
                    'total_edges': 0,
                    'avg_degree': 0.0,
                    'is_connected': False,
                    'num_components': G.number_of_nodes()
                },
                'distance_stats': {
                    'mean_distance_km': 0.0,
                    'median_distance_km': 0.0,
                    'std_distance_km': 0.0,
                    'min_distance_km': 0.0,
                    'max_distance_km': 0.0
                },
                'cost_stats': {
                    'total_network_cost': 0.0,
                    'mean_edge_cost': 0.0,
                    'median_edge_cost': 0.0,
                    'std_edge_cost': 0.0
                },
                'leadtime_stats': {
                    'mean_leadtime_hours': 0.0,
                    'median_leadtime_hours': 0.0,
                    'max_leadtime_hours': 0.0
                },
                'node_type_counts': {
                    'customer': sum(1 for t in nx.get_node_attributes(G, 'node_type').values() if t == 'customer'),
                    'dc': sum(1 for t in nx.get_node_attributes(G, 'node_type').values() if t == 'dc'),
                    'plant': sum(1 for t in nx.get_node_attributes(G, 'node_type').values() if t == 'plant')
                },
                'edge_type_counts': {}
            }
        
        # Basic network statistics
        num_nodes = G.number_of_nodes()
        analysis['network_stats'] = {
            'total_nodes': num_nodes,
            'total_edges': G.number_of_edges(),
            'avg_degree': sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0.0,
            'is_connected': nx.is_weakly_connected(G),
            'num_components': nx.number_weakly_connected_components(G)
        }
        
        # Distance statistics
        distances = trans_df['distance_km'].values
        analysis['distance_stats'] = {
            'mean_distance_km': float(np.mean(distances)) if len(distances) > 0 else 0.0,
            'median_distance_km': float(np.median(distances)) if len(distances) > 0 else 0.0,
            'std_distance_km': float(np.std(distances)) if len(distances) > 0 else 0.0,
            'min_distance_km': float(np.min(distances)) if len(distances) > 0 else 0.0,
            'max_distance_km': float(np.max(distances)) if len(distances) > 0 else 0.0
        }
        
        # Cost statistics
        costs = trans_df['cost'].values
        analysis['cost_stats'] = {
            'total_network_cost': float(np.sum(costs)) if len(costs) > 0 else 0.0,
            'mean_edge_cost': float(np.mean(costs)) if len(costs) > 0 else 0.0,
            'median_edge_cost': float(np.median(costs)) if len(costs) > 0 else 0.0,
            'std_edge_cost': float(np.std(costs)) if len(costs) > 0 else 0.0
        }
        
        # Lead time statistics
        lead_times = trans_df['lead_time'].values
        analysis['leadtime_stats'] = {
            'mean_leadtime_hours': float(np.mean(lead_times) / 3600) if len(lead_times) > 0 else 0.0,
            'median_leadtime_hours': float(np.median(lead_times) / 3600) if len(lead_times) > 0 else 0.0,
            'max_leadtime_hours': float(np.max(lead_times) / 3600) if len(lead_times) > 0 else 0.0
        }
        
        # Node type analysis
        node_types = nx.get_node_attributes(G, 'node_type')
        type_counts = {}
        for node_type in ['customer', 'dc', 'plant']:
            type_counts[node_type] = sum(1 for t in node_types.values() if t == node_type)
        analysis['node_type_counts'] = type_counts
        
        # Edge type analysis
        edge_types = {}
        for _, row in trans_df.iterrows():
            edge_type = f"{row['from_type']}_to_{row['to_type']}"
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        analysis['edge_type_counts'] = edge_types
        
        return analysis
        
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about network generation service capabilities
        
        Returns:
            Dict: Service information
        """
        return {
            "network_generation": {
                "description": "Complete network generation for logistics optimization",
                "features": [
                    "Road-based network generation using OSRM",
                    "Great circle distance network generation",
                    "Cost and lead time modeling",
                    "NetworkX graph integration",
                    "Interactive network visualization",
                    "Network property analysis"
                ],
                "functions": [
                    "make_network_using_road()",
                    "make_network()",
                    "compute_durations()",
                    "plot_scm()",
                    "analyze_network_properties()"
                ]
            },
            "osrm_integration": {
                "host": self.osrm_host,
                "features": [
                    "Real road distance calculations", 
                    "Duration estimates with traffic considerations",
                    "Toll road inclusion/exclusion options",
                    "Fallback to great circle distances"
                ]
            },
            "visualization": {
                "description": "Interactive supply chain network visualization",
                "features": [
                    "Plotly-based interactive maps",
                    "Node differentiation by type",
                    "Edge visualization with weights",
                    "Hover information and legends"
                ]
            }
        }

    def make_network(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, plnt_df: pd.DataFrame, 
                    plnt_dc_threshold: float = 999999., dc_cust_threshold: float = 999999.,
                    unit_tp_cost: float = 1., unit_del_cost: float = 1., lt_lb: int = 1, 
                    lt_threshold: float = 800., stage_time_bound: Tuple[int, int] = (1, 1)) -> Tuple[pd.DataFrame, nx.Graph, Dict]:
        """
        Generate transportation/delivery routes using great circle distances
        Implements the notebook make_network function exactly
        
        Args:
            cust_df: Customer DataFrame with columns ['name', 'lat', 'lon']
            dc_df: DC DataFrame with columns ['name', 'lat', 'lon'] 
            plnt_df: Plant DataFrame with columns ['name', 'lat', 'lon']
            plnt_dc_threshold: Maximum distance threshold for plant-DC connections (km)
            dc_cust_threshold: Maximum distance threshold for DC-customer connections (km)
            unit_tp_cost: Unit transportation cost per km
            unit_del_cost: Unit delivery cost per km
            lt_lb: Lead time lower bound
            lt_threshold: Lead time threshold for calculation
            stage_time_bound: Stage time bounds (min, max)
            
        Returns:
            Tuple: (trans_df, graph, position)
        """
        assert lt_threshold > 0.001
        
        # Convert names to strings to avoid issues with numeric names
        cust_df = cust_df.copy()
        dc_df = dc_df.copy()
        plnt_df = plnt_df.copy()
        cust_df["name"] = cust_df["name"].astype(str)
        dc_df["name"] = dc_df["name"].astype(str)
        plnt_df["name"] = plnt_df["name"].astype(str)
        
        # Create graph for visualization
        graph = nx.Graph()
        graph.add_nodes_from("Plnt_" + plnt_df.name)
        graph.add_nodes_from("DC_" + dc_df.name)
        graph.add_nodes_from("Cust_" + cust_df.name)
        
        # Store positions for visualization
        position = {}
        for row in plnt_df.itertuples():
            position["Plnt_" + str(row.name)] = (row.lon, row.lat)
        for row in dc_df.itertuples():
            position["DC_" + str(row.name)] = (row.lon, row.lat)
        for row in cust_df.itertuples():
            position["Cust_" + str(row.name)] = (row.lon, row.lat)

        dist, kind, cost, lead_time, stage_time = [], [], [], [], []
        from_node, to_node = [], []
        
        # Plant to DC connections
        for plnt_row in plnt_df.itertuples():
            for dc_row in dc_df.itertuples():
                dis = distance((plnt_row.lat, plnt_row.lon), (dc_row.lat, dc_row.lon)).kilometers
                if dis <= plnt_dc_threshold:
                    from_node.append(plnt_row.name)
                    to_node.append(dc_row.name)
                    dist.append(dis)
                    kind.append("plnt-dc")
                    cost.append(dis * unit_tp_cost)
                    lead_time.append(math.ceil(dis / lt_threshold) + lt_lb)
                    stage_time.append(random.randint(stage_time_bound[0], stage_time_bound[1]))
                    graph.add_edge("Plnt_" + str(plnt_row.name), "DC_" + str(dc_row.name))

        # DC to Customer connections
        for cust_row in cust_df.itertuples():
            for dc_row in dc_df.itertuples():
                dis = distance((cust_row.lat, cust_row.lon), (dc_row.lat, dc_row.lon)).kilometers
                if dis <= dc_cust_threshold:
                    from_node.append(dc_row.name)
                    to_node.append(cust_row.name)
                    dist.append(dis)
                    kind.append("dc-cust")
                    cost.append(dis * unit_del_cost)
                    lead_time.append(math.ceil(dis / lt_threshold) + lt_lb)
                    stage_time.append(random.randint(stage_time_bound[0], stage_time_bound[1]))
                    graph.add_edge("DC_" + str(dc_row.name), "Cust_" + str(cust_row.name))

        trans_df = pd.DataFrame({
            "from_node": from_node, 
            "to_node": to_node, 
            "dist": dist, 
            "cost": cost, 
            "lead_time": lead_time, 
            "stage_time": stage_time, 
            "kind": kind
        })
        
        return trans_df, graph, position

    def make_network_using_road(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, plnt_df: pd.DataFrame, 
                               durations: List[List[float]], distances: List[List[float]],
                               plnt_dc_threshold: float = 999999., dc_cust_threshold: float = 999999.,
                               tc_per_dis: float = 20./20000, dc_per_dis: float = 10./4000, 
                               tc_per_time: float = 8000./20000, dc_per_time: float = 8000./4000, 
                               lt_lb: int = 1, lt_threshold: float = 800., 
                               stage_time_bound: Tuple[int, int] = (1, 1)) -> Tuple[pd.DataFrame, nx.Graph, Dict]:
        """
        Generate transportation/delivery routes using road distances and durations
        Implements the notebook make_network_using_road function exactly
        
        Args:
            cust_df: Customer DataFrame
            dc_df: DC DataFrame  
            plnt_df: Plant DataFrame
            durations: Duration matrix from OSRM (seconds)
            distances: Distance matrix from OSRM (meters)
            plnt_dc_threshold: Maximum distance threshold for plant-DC connections
            dc_cust_threshold: Maximum distance threshold for DC-customer connections
            tc_per_dis: Transportation cost per distance unit
            dc_per_dis: Delivery cost per distance unit
            tc_per_time: Transportation cost per time unit
            dc_per_time: Delivery cost per time unit
            lt_lb: Lead time lower bound
            lt_threshold: Lead time threshold
            stage_time_bound: Stage time bounds
            
        Returns:
            Tuple: (trans_df, graph, position)
        """
        assert lt_threshold > 0.001
        
        # Convert names to strings
        cust_df = cust_df.copy()
        dc_df = dc_df.copy() 
        plnt_df = plnt_df.copy()
        cust_df["name"] = cust_df["name"].astype(str)
        dc_df["name"] = dc_df["name"].astype(str)
        plnt_df["name"] = plnt_df["name"].astype(str)
        
        # Create graph for visualization
        graph = nx.Graph()
        graph.add_nodes_from("Plnt_" + plnt_df.name)
        graph.add_nodes_from("DC_" + dc_df.name)
        graph.add_nodes_from("Cust_" + cust_df.name)
        
        # Store positions for visualization
        position = {}
        for row in plnt_df.itertuples():
            position["Plnt_" + str(row.name)] = (row.lon, row.lat)
        for row in dc_df.itertuples():
            position["DC_" + str(row.name)] = (row.lon, row.lat)
        for row in cust_df.itertuples():
            position["Cust_" + str(row.name)] = (row.lon, row.lat)

        dist, kind, time_list, cost, lead_time, stage_time = [], [], [], [], [], []
        from_node, to_node = [], []
        
        n = len(cust_df)
        
        # Plant to DC connections
        for i, plnt_row in enumerate(plnt_df.itertuples()):
            for j, dc_row in enumerate(dc_df.itertuples()):
                if distances[n+i][j] < 9999999:
                    dis = distances[n+i][j] / 1000.  # Convert to km
                    time = durations[n+i][j] / 3600.  # Convert to hours
                else:
                    dis = 10 * distance((plnt_row.lat, plnt_row.lon), (dc_row.lat, dc_row.lon)).kilometers
                    time = dis / 50.  # Assume 50 km/h average speed
                    
                if dis <= plnt_dc_threshold:
                    from_node.append(plnt_row.name)
                    to_node.append(dc_row.name)
                    dist.append(dis)
                    time_list.append(time)
                    kind.append("plnt-dc")
                    cost.append(dis * tc_per_dis + time * tc_per_time)
                    lead_time.append(math.ceil(dis / lt_threshold) + lt_lb)
                    stage_time.append(random.randint(stage_time_bound[0], stage_time_bound[1]))
                    if dis < 9999999:
                        graph.add_edge("Plnt_" + str(plnt_row.name), "DC_" + str(dc_row.name))

        # DC to Customer connections
        for i, cust_row in enumerate(cust_df.itertuples()):
            for j, dc_row in enumerate(dc_df.itertuples()):
                if distances[i][j] < 9999999:
                    dis = distances[i][j] / 1000.  # Convert to km
                    time = durations[i][j] / 3600.  # Convert to hours
                else:
                    dis = 10 * distance((cust_row.lat, cust_row.lon), (dc_row.lat, dc_row.lon)).kilometers
                    time = dis / 50.  # Assume 50 km/h average speed
                    
                if dis <= dc_cust_threshold:
                    from_node.append(dc_row.name)
                    to_node.append(cust_row.name)
                    dist.append(dis)
                    time_list.append(time)
                    kind.append("dc-cust")
                    cost.append(dis * dc_per_dis + time * dc_per_time)
                    lead_time.append(math.ceil(dis / lt_threshold) + lt_lb)
                    stage_time.append(random.randint(stage_time_bound[0], stage_time_bound[1]))
                    if dis < 9999999:
                        graph.add_edge("DC_" + str(dc_row.name), "Cust_" + str(cust_row.name))

        trans_df = pd.DataFrame({
            "from_node": from_node, 
            "to_node": to_node, 
            "dist": dist, 
            "time": time_list,
            "cost": cost, 
            "lead_time": lead_time, 
            "stage_time": stage_time, 
            "kind": kind
        })
        
        return trans_df, graph, position

    def distance_histgram(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, plnt_df: pd.DataFrame, 
                         distances: Optional[List[List[float]]] = None) -> go.Figure:
        """
        Generate histogram of distances between plants-warehouses and warehouses-customers
        Implements the notebook distance_histgram function exactly
        
        Args:
            cust_df: Customer DataFrame
            dc_df: DC DataFrame
            plnt_df: Plant DataFrame
            distances: Optional distance matrix from OSRM
            
        Returns:
            Plotly figure with distance histogram
        """
        dist, kind = [], []
        
        if distances is None:
            # Use great circle distances
            for plnt_row in plnt_df.itertuples():
                for dc_row in dc_df.itertuples():
                    dist.append(distance((plnt_row.lat, plnt_row.lon), (dc_row.lat, dc_row.lon)).kilometers)
                    kind.append("plnt-dc")
            for cust_row in cust_df.itertuples():
                for dc_row in dc_df.itertuples():
                    dist.append(distance((cust_row.lat, cust_row.lon), (dc_row.lat, dc_row.lon)).kilometers)
                    kind.append("dc-cust")
        else:
            # Use road distances - customers and warehouses are same, calculate with road distance
            n = len(cust_df)
            m = len(plnt_df)
            for i in range(n, n + m):
                for j in range(n):
                    if distances[i][j] < 99999999:
                        dist.append(distances[i][j] / 1000.)  # Convert to km
                        kind.append("plnt-dc")
                        
            for i in range(n):
                for j in range(n):
                    if distances[i][j] < 99999999:
                        dist.append(distances[i][j] / 1000.)  # Convert to km
                        kind.append("dc-cust")
                        
        df = pd.DataFrame({"dist": dist, "kind": kind})
        fig = px.histogram(df, x="dist", color="kind", 
                          title="Distance Distribution in Logistics Network",
                          labels={"dist": "Distance (km)", "kind": "Connection Type"})
        return fig

    def plot_scm(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, plnt_df: pd.DataFrame, 
                graph: nx.Graph, position: Dict, node_only: bool = False, 
                mapbox_access_token: Optional[str] = None) -> go.Figure:
        """
        Visualize supply chain network on map using Plotly
        Implements the notebook plot_scm function exactly
        
        Args:
            cust_df: Customer DataFrame
            dc_df: DC DataFrame
            plnt_df: Plant DataFrame
            graph: NetworkX graph
            position: Node positions dictionary
            node_only: If True, only show nodes without edges
            mapbox_access_token: Optional Mapbox access token
            
        Returns:
            Plotly figure with supply chain map
        """
        data = [
            go.Scattermapbox(
                lat=cust_df.iloc[:, 1],
                lon=cust_df.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=10, color="blue", opacity=0.9
                ),
                text=cust_df.iloc[:, 0],
                name="Customers"
            ),
            go.Scattermapbox(
                lat=dc_df.iloc[:, 1],
                lon=dc_df.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=20, color="red", opacity=0.6
                ),
                text=dc_df.iloc[:, 0],
                name="Warehouses"
            ),
            go.Scattermapbox(
                lat=plnt_df.iloc[:, 1],
                lon=plnt_df.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=30, color="orange", opacity=0.8
                ),
                text=plnt_df.iloc[:, 0],
                name="Plants"
            ),
        ]
        
        if not node_only:
            edge_trace_lat, edge_trace_lon = [], []
            for (i, j) in graph.edges():
                edge_trace_lat += [position[i][1], position[j][1], None]
                edge_trace_lon += [position[i][0], position[j][0], None]
                
            data.append(    
                go.Scattermapbox(
                    lat=edge_trace_lat,
                    lon=edge_trace_lon,
                    line=dict(width=0.5, color='yellow'),
                    hoverinfo='none',
                    mode='lines',
                    name="Edges"
                )
            )

        layout = go.Layout(
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,
                center=dict(
                    lat=35.8573157,
                    lon=139.64696
                ),
                pitch=0,
                zoom=5,
                style="open-street-map" if mapbox_access_token is None else "dark"
            ),
        )

        fig = go.Figure(data=data, layout=layout)
        return fig

    def show_optimized_network(self, cust_df: pd.DataFrame, dc_df: pd.DataFrame, plnt_df: pd.DataFrame, 
                              prod_df: pd.DataFrame, flow_df: pd.DataFrame, position: Dict,
                              mapbox_access_token: Optional[str] = None) -> go.Figure:
        """
        Visualize optimized network with flows on map
        Implements the notebook show_optimized_network function exactly
        
        Args:
            cust_df: Customer DataFrame
            dc_df: DC DataFrame  
            plnt_df: Plant DataFrame
            prod_df: Product DataFrame
            flow_df: Flow DataFrame with optimization results
            position: Node positions dictionary
            mapbox_access_token: Optional Mapbox access token
            
        Returns:
            Plotly figure with optimized network visualization
        """
        pd.set_option('mode.chained_assignment', None)
        
        data = [
            go.Scattermapbox(
                lat=cust_df.iloc[:, 1],
                lon=cust_df.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=10, color="blue", opacity=0.9
                ),
                text=cust_df.iloc[:, 0],
                name="Customers"
            ),
            go.Scattermapbox(
                lat=dc_df.iloc[:, 1][dc_df.open_close == 1],
                lon=dc_df.iloc[:, 2][dc_df.open_close == 1],
                mode='markers',
                marker=dict(
                    size=20, color="red", opacity=0.6
                ),
                text=dc_df.iloc[:, 0],
                name="Warehouses"
            ),
            go.Scattermapbox(
                lat=plnt_df.iloc[:, 1],
                lon=plnt_df.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=30, color="orange", opacity=0.8
                ),
                text=plnt_df.iloc[:, 0],
                name="Plants"
            ),
        ]

        for p in prod_df.iloc[:, 0]:
            edge_trace_lat, edge_trace_lon = [], []
            temp_df = flow_df[flow_df["prod"] == str(p)]
            for row in temp_df.itertuples():
                i = row.from_node
                j = row.to_node
                edge_trace_lat += [position[i][1], position[j][1], None]
                edge_trace_lon += [position[i][0], position[j][0], None]
            data.append(
                go.Scattermapbox(
                    lat=edge_trace_lat,
                    lon=edge_trace_lon,
                    line=dict(width=0.5),
                    hoverinfo='none',
                    mode='lines',
                    name=str(p)
                )
            )
            
        layout = go.Layout(
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=0,
                center=dict(
                    lat=35.8573157,
                    lon=139.64696
                ),
                pitch=0,
                zoom=5,
                style="open-street-map" if mapbox_access_token is None else "dark"
            ),
        )

        fig = go.Figure(data=data, layout=layout)
        return fig