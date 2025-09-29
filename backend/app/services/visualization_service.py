import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class VisualizationService:
    """
    Comprehensive visualization service for logistics network design.
    Integrates visualization capabilities across all LND components
    to match the notebook's visualization procedures exactly.
    """
    
    def __init__(self):
        self.default_colors = {
            'customer': 'lightblue',
            'facility': 'red', 
            'dc': 'orange',
            'plant': 'green',
            'selected_facility': 'darkred',
            'edge_plant_dc': 'blue',
            'edge_dc_customer': 'red',
            'cluster_colors': px.colors.qualitative.Set1
        }
        
    def create_facility_location_map(self, 
                                   customer_df: pd.DataFrame,
                                   facility_locations: List[Tuple[float, float]],
                                   assignments: Optional[List[int]] = None,
                                   title: str = "Facility Location Optimization Results") -> Dict[str, Any]:
        """
        Create interactive map for facility location optimization results
        Matches the notebook's facility location visualization
        
        Args:
            customer_df: Customer DataFrame with lat/lon columns
            facility_locations: List of (lat, lon) tuples for facilities
            assignments: Optional customer assignments to facilities
            title: Map title
            
        Returns:
            Plotly figure data for visualization
        """
        traces = []
        
        # Customer points
        customer_colors = 'lightblue'
        if assignments is not None:
            # Color customers by assignment
            customer_colors = [self.default_colors['cluster_colors'][a % len(self.default_colors['cluster_colors'])] 
                             for a in assignments]
        
        traces.append({
            'type': 'scattermapbox',
            'lat': customer_df['lat'].tolist(),
            'lon': customer_df['lon'].tolist(),
            'mode': 'markers',
            'marker': {
                'size': 8,
                'color': customer_colors,
                'opacity': 0.7
            },
            'text': customer_df.get('name', customer_df.index).tolist(),
            'name': 'Customers',
            'hovertemplate': '<b>%{text}</b><br>Customer<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
        })
        
        # Facility points
        if facility_locations:
            facility_lats = [loc[0] for loc in facility_locations]
            facility_lons = [loc[1] for loc in facility_locations]
            
            traces.append({
                'type': 'scattermapbox',
                'lat': facility_lats,
                'lon': facility_lons,
                'mode': 'markers',
                'marker': {
                    'size': 15,
                    'color': self.default_colors['selected_facility'],
                    'symbol': 'star'
                },
                'text': [f'Facility {i+1}' for i in range(len(facility_locations))],
                'name': 'Selected Facilities',
                'hovertemplate': '<b>%{text}</b><br>Facility<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
            })
            
            # Connection lines if assignments provided
            if assignments is not None:
                edge_lats = []
                edge_lons = []
                for i, assignment in enumerate(assignments):
                    if assignment < len(facility_locations):
                        # Customer to facility line
                        edge_lats.extend([customer_df.iloc[i]['lat'], facility_lats[assignment], None])
                        edge_lons.extend([customer_df.iloc[i]['lon'], facility_lons[assignment], None])
                
                if edge_lats:
                    traces.append({
                        'type': 'scattermapbox',
                        'lat': edge_lats,
                        'lon': edge_lons,
                        'mode': 'lines',
                        'line': {'width': 1, 'color': 'gray'},
                        'name': 'Assignments',
                        'hoverinfo': 'none'
                    })
        
        # Calculate map center
        all_lats = customer_df['lat'].tolist()
        all_lons = customer_df['lon'].tolist()
        if facility_locations:
            all_lats.extend([loc[0] for loc in facility_locations])
            all_lons.extend([loc[1] for loc in facility_locations])
        
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        layout = {
            'title': title,
            'autosize': True,
            'hovermode': 'closest',
            'mapbox': {
                'bearing': 0,
                'center': {'lat': center_lat, 'lon': center_lon},
                'pitch': 0,
                'zoom': 10,
                'style': 'open-street-map'
            },
            'showlegend': True,
            'legend': {'x': 0, 'y': 1}
        }
        
        return {'data': traces, 'layout': layout}
    
    def create_network_visualization(self, 
                                   customer_df: pd.DataFrame,
                                   dc_df: Optional[pd.DataFrame] = None,
                                   plant_df: Optional[pd.DataFrame] = None,
                                   network_edges: Optional[List[Dict]] = None,
                                   title: str = "Supply Chain Network") -> Dict[str, Any]:
        """
        Create supply chain network visualization
        Matches the notebook's network visualization procedures
        
        Args:
            customer_df: Customer locations
            dc_df: Distribution center locations (optional)
            plant_df: Plant locations (optional) 
            network_edges: List of edge dictionaries with from/to coordinates
            title: Visualization title
            
        Returns:
            Plotly figure data
        """
        traces = []
        
        # Add network edges first (so they appear behind nodes)
        if network_edges:
            # Group edges by type
            edge_types = {}
            for edge in network_edges:
                edge_type = edge.get('type', 'unknown')
                if edge_type not in edge_types:
                    edge_types[edge_type] = {'lats': [], 'lons': []}
                
                edge_types[edge_type]['lats'].extend([
                    edge['from_lat'], edge['to_lat'], None
                ])
                edge_types[edge_type]['lons'].extend([
                    edge['from_lon'], edge['to_lon'], None
                ])
            
            # Add traces for each edge type
            for edge_type, coords in edge_types.items():
                color = self.default_colors.get(f'edge_{edge_type}', 'gray')
                width = 2 if 'plant' in edge_type else 1
                
                traces.append({
                    'type': 'scattermapbox',
                    'lat': coords['lats'],
                    'lon': coords['lons'],
                    'mode': 'lines',
                    'line': {'width': width, 'color': color},
                    'name': f'{edge_type.replace("_", " ").title()} Routes',
                    'hoverinfo': 'none'
                })
        
        # Add plant nodes (largest, green squares)
        if plant_df is not None and len(plant_df) > 0:
            traces.append({
                'type': 'scattermapbox',
                'lat': plant_df['lat'].tolist(),
                'lon': plant_df['lon'].tolist(),
                'mode': 'markers',
                'marker': {
                    'size': 15,
                    'color': self.default_colors['plant'],
                    'symbol': 'square'
                },
                'text': plant_df.get('name', plant_df.index).tolist(),
                'name': 'Plants',
                'hovertemplate': '<b>%{text}</b><br>Plant<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
            })
        
        # Add DC nodes (medium, orange triangles)
        if dc_df is not None and len(dc_df) > 0:
            traces.append({
                'type': 'scattermapbox',
                'lat': dc_df['lat'].tolist(),
                'lon': dc_df['lon'].tolist(),
                'mode': 'markers',
                'marker': {
                    'size': 12,
                    'color': self.default_colors['dc'],
                    'symbol': 'triangle-up'
                },
                'text': dc_df.get('name', dc_df.index).tolist(),
                'name': 'Distribution Centers',
                'hovertemplate': '<b>%{text}</b><br>DC<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
            })
        
        # Add customer nodes (small, blue circles)
        traces.append({
            'type': 'scattermapbox',
            'lat': customer_df['lat'].tolist(),
            'lon': customer_df['lon'].tolist(),
            'mode': 'markers',
            'marker': {
                'size': 8,
                'color': self.default_colors['customer'],
                'opacity': 0.8
            },
            'text': customer_df.get('name', customer_df.index).tolist(),
            'name': 'Customers',
            'hovertemplate': '<b>%{text}</b><br>Customer<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
        })
        
        # Calculate map center from all locations
        all_lats = customer_df['lat'].tolist()
        all_lons = customer_df['lon'].tolist()
        
        if dc_df is not None and len(dc_df) > 0:
            all_lats.extend(dc_df['lat'].tolist())
            all_lons.extend(dc_df['lon'].tolist())
            
        if plant_df is not None and len(plant_df) > 0:
            all_lats.extend(plant_df['lat'].tolist())
            all_lons.extend(plant_df['lon'].tolist())
        
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        layout = {
            'title': title,
            'autosize': True,
            'hovermode': 'closest',
            'mapbox': {
                'bearing': 0,
                'center': {'lat': center_lat, 'lon': center_lon},
                'pitch': 0,
                'zoom': 9,
                'style': 'open-street-map'
            },
            'showlegend': True,
            'legend': {'x': 0, 'y': 1}
        }
        
        return {'data': traces, 'layout': layout}
    
    def create_cost_analysis_charts(self, 
                                  optimization_results: Dict[str, Any],
                                  title_prefix: str = "Optimization Analysis") -> Dict[str, Any]:
        """
        Create comprehensive cost and performance analysis charts
        
        Args:
            optimization_results: Results from optimization algorithms
            title_prefix: Prefix for chart titles
            
        Returns:
            Dictionary with multiple chart configurations
        """
        charts = {}
        
        # Convergence chart (if convergence data available)
        if 'lower_bounds' in optimization_results and 'upper_bounds' in optimization_results:
            lb = optimization_results['lower_bounds']
            ub = optimization_results['upper_bounds']
            iterations = list(range(len(lb)))
            
            charts['convergence'] = {
                'data': [
                    {
                        'type': 'scatter',
                        'x': iterations,
                        'y': ub,
                        'mode': 'lines+markers',
                        'name': 'Upper Bound',
                        'line': {'color': 'red'}
                    },
                    {
                        'type': 'scatter', 
                        'x': iterations,
                        'y': lb,
                        'mode': 'lines+markers',
                        'name': 'Lower Bound',
                        'line': {'color': 'blue'}
                    }
                ],
                'layout': {
                    'title': f'{title_prefix}: Convergence Analysis',
                    'xaxis': {'title': 'Iteration'},
                    'yaxis': {'title': 'Objective Value'},
                    'showlegend': True
                }
            }
        
        # Learning rate schedule (if available)
        if 'learning_rates' in optimization_results:
            lr = optimization_results['learning_rates']
            iterations = list(range(len(lr)))
            
            charts['learning_rate'] = {
                'data': [
                    {
                        'type': 'scatter',
                        'x': iterations,
                        'y': lr,
                        'mode': 'lines+markers',
                        'name': 'Learning Rate',
                        'line': {'color': 'green'}
                    }
                ],
                'layout': {
                    'title': f'{title_prefix}: Learning Rate Schedule',
                    'xaxis': {'title': 'Iteration'},
                    'yaxis': {'title': 'Learning Rate'},
                    'showlegend': True
                }
            }
        
        # Facility statistics (if available)
        if 'facility_stats' in optimization_results:
            stats = optimization_results['facility_stats']
            facility_names = [f"Facility {stat.get('facility_index', i)+1}" for i, stat in enumerate(stats)]
            demands = [stat.get('total_demand_served', 0) for stat in stats]
            distances = [stat.get('average_distance', 0) for stat in stats]
            
            charts['facility_demand'] = {
                'data': [
                    {
                        'type': 'bar',
                        'x': facility_names,
                        'y': demands,
                        'name': 'Demand Served',
                        'marker': {'color': 'lightblue'}
                    }
                ],
                'layout': {
                    'title': f'{title_prefix}: Demand Distribution by Facility',
                    'xaxis': {'title': 'Facility'},
                    'yaxis': {'title': 'Total Demand Served'},
                    'showlegend': True
                }
            }
            
            charts['facility_distance'] = {
                'data': [
                    {
                        'type': 'bar',
                        'x': facility_names,
                        'y': distances,
                        'name': 'Average Distance',
                        'marker': {'color': 'orange'}
                    }
                ],
                'layout': {
                    'title': f'{title_prefix}: Average Service Distance by Facility',
                    'xaxis': {'title': 'Facility'},
                    'yaxis': {'title': 'Average Distance (km)'},
                    'showlegend': True
                }
            }
        
        return charts
    
    def create_demand_heatmap(self, 
                            customer_df: pd.DataFrame,
                            demand_col: str = 'demand',
                            title: str = "Customer Demand Heatmap") -> Dict[str, Any]:
        """
        Create heatmap visualization of customer demand distribution
        
        Args:
            customer_df: DataFrame with customer locations and demand
            demand_col: Column name for demand values
            title: Chart title
            
        Returns:
            Plotly heatmap configuration
        """
        if demand_col not in customer_df.columns:
            demand_col = 'demand' if 'demand' in customer_df.columns else customer_df.columns[-1]
        
        # Create grid for heatmap
        lat_min, lat_max = customer_df['lat'].min(), customer_df['lat'].max()
        lon_min, lon_max = customer_df['lon'].min(), customer_df['lon'].max()
        
        # Create grid points
        n_points = 50
        lat_grid = np.linspace(lat_min, lat_max, n_points)
        lon_grid = np.linspace(lon_min, lon_max, n_points)
        
        # Interpolate demand onto grid using simple inverse distance weighting
        demand_grid = np.zeros((n_points, n_points))
        
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                weights = []
                demands = []
                
                for _, customer in customer_df.iterrows():
                    dist = np.sqrt((customer['lat'] - lat)**2 + (customer['lon'] - lon)**2)
                    if dist < 1e-6:  # Very close to customer
                        demand_grid[i, j] = customer[demand_col]
                        break
                    else:
                        weights.append(1.0 / (dist**2 + 1e-6))
                        demands.append(customer[demand_col])
                else:
                    # Weighted average of nearby demand
                    if weights:
                        demand_grid[i, j] = np.average(demands, weights=weights)
        
        return {
            'data': [
                {
                    'type': 'heatmap',
                    'z': demand_grid.tolist(),
                    'x': lon_grid.tolist(),
                    'y': lat_grid.tolist(),
                    'colorscale': 'YlOrRd',
                    'name': 'Demand Density'
                }
            ],
            'layout': {
                'title': title,
                'xaxis': {'title': 'Longitude'},
                'yaxis': {'title': 'Latitude'},
                'showlegend': True
            }
        }
    
    def create_distance_histogram(self, 
                                distances: List[float],
                                bins: int = 20,
                                title: str = "Distance Distribution") -> Dict[str, Any]:
        """
        Create histogram of distance distributions
        Matches the notebook's distance analysis procedures
        
        Args:
            distances: List of distance values
            bins: Number of histogram bins
            title: Chart title
            
        Returns:
            Plotly histogram configuration
        """
        return {
            'data': [
                {
                    'type': 'histogram',
                    'x': distances,
                    'nbinsx': bins,
                    'name': 'Distances',
                    'marker': {'color': 'skyblue'}
                }
            ],
            'layout': {
                'title': title,
                'xaxis': {'title': 'Distance (km)'},
                'yaxis': {'title': 'Frequency'},
                'showlegend': True,
                'bargap': 0.1
            }
        }
    
    def create_comprehensive_dashboard(self, 
                                     optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive visualization dashboard
        Integrates multiple visualization components for complete analysis
        
        Args:
            optimization_data: Complete optimization results and data
            
        Returns:
            Dashboard configuration with multiple charts
        """
        dashboard = {
            'title': 'Logistics Network Design Dashboard',
            'charts': {}
        }
        
        # Main map visualization
        if 'customer_df' in optimization_data:
            customer_df = optimization_data['customer_df']
            facility_locations = optimization_data.get('facility_locations', [])
            assignments = optimization_data.get('assignments', None)
            
            dashboard['charts']['main_map'] = self.create_facility_location_map(
                customer_df, facility_locations, assignments,
                title="Optimized Facility Locations"
            )
        
        # Network visualization (if network data available)
        if 'network_edges' in optimization_data:
            dashboard['charts']['network_map'] = self.create_network_visualization(
                optimization_data['customer_df'],
                optimization_data.get('dc_df'),
                optimization_data.get('plant_df'),
                optimization_data['network_edges'],
                title="Supply Chain Network"
            )
        
        # Analysis charts
        if 'optimization_results' in optimization_data:
            analysis_charts = self.create_cost_analysis_charts(
                optimization_data['optimization_results'],
                title_prefix="Algorithm Performance"
            )
            dashboard['charts'].update(analysis_charts)
        
        # Demand heatmap
        if 'customer_df' in optimization_data:
            customer_df = optimization_data['customer_df']
            if 'demand' in customer_df.columns:
                dashboard['charts']['demand_heatmap'] = self.create_demand_heatmap(
                    customer_df, title="Customer Demand Distribution"
                )
        
        # Distance histogram
        if 'distances' in optimization_data:
            dashboard['charts']['distance_histogram'] = self.create_distance_histogram(
                optimization_data['distances'], title="Service Distance Analysis"
            )
        
        return dashboard
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about visualization service capabilities
        
        Returns:
            Service information dictionary
        """
        return {
            "visualization_service": {
                "description": "Comprehensive visualization for logistics network design",
                "capabilities": [
                    "Interactive facility location maps",
                    "Supply chain network visualization", 
                    "Optimization convergence analysis",
                    "Cost and performance charts",
                    "Demand distribution heatmaps",
                    "Distance analysis histograms",
                    "Comprehensive dashboards"
                ],
                "chart_types": [
                    "scattermapbox",
                    "heatmap", 
                    "histogram",
                    "line_charts",
                    "bar_charts",
                    "network_graphs"
                ],
                "integration": {
                    "plotly": "Interactive web-based visualizations",
                    "mapbox": "Geographic mapping capabilities",
                    "notebook": "Matches 05lnd.ipynb visualization procedures"
                }
            },
            "supported_data": {
                "facility_location": "Customer locations and facility optimization results",
                "network_design": "Multi-echelon supply chain networks",
                "optimization_results": "Algorithm convergence and performance data", 
                "demand_analysis": "Customer demand patterns and distributions"
            },
            "output_formats": {
                "plotly_json": "Plotly figure dictionaries for web integration",
                "dashboard": "Multi-chart dashboard configurations",
                "standalone_charts": "Individual chart configurations"
            }
        }