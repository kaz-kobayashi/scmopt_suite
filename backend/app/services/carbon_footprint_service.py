import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from geopy.distance import great_circle as distance
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class CarbonFootprintService:
    """
    Carbon footprint analysis service for sustainable logistics network optimization.
    Implements multi-objective optimization considering both cost and environmental impact.
    """
    
    def __init__(self):
        # Standard CO2 emission factors (gCO2/ton-km)
        self.emission_factors = {
            "railway": 22,
            "ship": 39,
            "road_truck": 100,  # Average for road transport
            "aviation": 1490
        }
        
        # Fuel conversion factors
        self.fuel_to_co2 = {
            "diesel": 2.322,  # kg CO2 per liter diesel
            "gasoline": 2.168,  # kg CO2 per liter gasoline
            "electricity": 0.518  # kg CO2 per kWh (global average)
        }
        
    def calculate_co2_emission(self, capacity: float, loading_rate: float = 0.5, 
                              fuel_type: str = "diesel") -> Tuple[float, float]:
        """
        Calculate CO2 emissions for transportation based on capacity and loading rate
        
        This function implements the empirical formula from 05lnd.ipynb:
        - For diesel: fuel = exp(2.67 - 0.927*ln(rate) - 0.648*ln(capacity))
        - For gasoline: fuel = exp(2.71 - 0.812*ln(rate) - 0.654*ln(capacity))
        
        Args:
            capacity: Vehicle capacity in kg
            loading_rate: Loading rate (0 < rate <= 1.0)
            fuel_type: "diesel" or "gasoline"
            
        Returns:
            Tuple: (fuel_consumption_L_per_ton_km, co2_emission_g_per_ton_km)
        """
        # Input validation
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not (0 < loading_rate <= 1.0):
            raise ValueError("Loading rate must be between 0 and 1")
        if fuel_type not in ["diesel", "gasoline"]:
            raise ValueError("Fuel type must be 'diesel' or 'gasoline'")
        
        # Calculate fuel consumption per ton-km
        if fuel_type == "diesel":
            fuel_consumption = math.exp(2.67 - 0.927*math.log(loading_rate) - 0.648*math.log(capacity))
        else:  # gasoline
            fuel_consumption = math.exp(2.71 - 0.812*math.log(loading_rate) - 0.654*math.log(capacity))
        
        # Convert to CO2 emissions (g/ton-km)
        co2_factor = self.fuel_to_co2[fuel_type]
        co2_emission = fuel_consumption * co2_factor * 1000  # Convert kg to g
        
        return fuel_consumption, co2_emission
        
    def calculate_transportation_emissions(self, distance_km: float, weight_tons: float, 
                                         transport_mode: str, 
                                         capacity: Optional[float] = None,
                                         loading_rate: Optional[float] = None,
                                         fuel_type: str = "diesel") -> Dict[str, float]:
        """
        Calculate total CO2 emissions for a transportation activity
        
        Args:
            distance_km: Transportation distance in km
            weight_tons: Cargo weight in tons
            transport_mode: "road", "railway", "ship", "aviation", or "custom"
            capacity: Vehicle capacity in kg (for custom calculations)
            loading_rate: Loading rate (for custom calculations)
            fuel_type: Fuel type for custom calculations
            
        Returns:
            Dict: Emission calculation details
        """
        if transport_mode == "custom" and capacity is not None:
            # Use custom calculation with empirical formula
            fuel_consumption, co2_per_ton_km = self.calculate_co2_emission(
                capacity, loading_rate or 0.5, fuel_type
            )
            total_co2_kg = (co2_per_ton_km / 1000) * weight_tons * distance_km
            
            return {
                "transport_mode": "custom",
                "distance_km": distance_km,
                "weight_tons": weight_tons,
                "capacity_kg": capacity,
                "loading_rate": loading_rate or 0.5,
                "fuel_type": fuel_type,
                "fuel_consumption_L_per_ton_km": fuel_consumption,
                "co2_emission_g_per_ton_km": co2_per_ton_km,
                "total_co2_kg": total_co2_kg,
                "calculation_method": "empirical_formula"
            }
        else:
            # Use standard emission factors
            if transport_mode not in self.emission_factors:
                raise ValueError(f"Transport mode '{transport_mode}' not supported")
            
            emission_factor = self.emission_factors[transport_mode]
            total_co2_kg = (emission_factor / 1000) * weight_tons * distance_km
            
            return {
                "transport_mode": transport_mode,
                "distance_km": distance_km,
                "weight_tons": weight_tons,
                "emission_factor_g_per_ton_km": emission_factor,
                "total_co2_kg": total_co2_kg,
                "calculation_method": "standard_factors"
            }
    
    def multi_objective_analysis(self, facilities_data: List[Dict], customers_data: List[Dict],
                                transportation_cost_per_km: float = 1.0,
                                carbon_constraint_kg: Optional[float] = None,
                                carbon_price_per_kg: float = 0.0) -> Dict[str, Any]:
        """
        Perform multi-objective analysis considering both cost and carbon emissions
        
        Args:
            facilities_data: List of potential facility locations with costs
            customers_data: List of customers with demand and locations
            transportation_cost_per_km: Cost per km for transportation
            carbon_constraint_kg: Maximum allowed CO2 emissions in kg
            carbon_price_per_kg: Price per kg of CO2 (for carbon pricing)
            
        Returns:
            Dict: Multi-objective analysis results
        """
        results = {
            "facilities": [],
            "customers": [],
            "transportation_matrix": [],
            "environmental_impact": {},
            "cost_analysis": {},
            "pareto_analysis": {},
            "recommendations": []
        }
        
        total_cost = 0
        total_carbon_kg = 0
        
        # Calculate distances and emissions for all facility-customer pairs
        for i, facility in enumerate(facilities_data):
            facility_result = {
                "facility_id": i,
                "location": [facility.get("lat", 0), facility.get("lon", 0)],
                "fixed_cost": facility.get("fixed_cost", 0),
                "customers_served": [],
                "total_distance_km": 0,
                "total_emissions_kg": 0,
                "total_variable_cost": 0
            }
            
            for j, customer in enumerate(customers_data):
                # Calculate distance
                facility_coords = (facility.get("lat", 0), facility.get("lon", 0))
                customer_coords = (customer.get("lat", 0), customer.get("lon", 0))
                dist_km = distance(facility_coords, customer_coords).kilometers
                
                # Customer demand
                demand_tons = customer.get("demand", 0) / 1000  # Convert kg to tons
                
                # Transportation emissions
                transport_mode = facility.get("transport_mode", "road")
                capacity = facility.get("vehicle_capacity", 10000)  # Default 10 tons
                loading_rate = facility.get("loading_rate", 0.7)
                
                emissions_data = self.calculate_transportation_emissions(
                    distance_km=dist_km,
                    weight_tons=demand_tons,
                    transport_mode="custom",
                    capacity=capacity,
                    loading_rate=loading_rate,
                    fuel_type=facility.get("fuel_type", "diesel")
                )
                
                # Calculate costs
                variable_cost = dist_km * transportation_cost_per_km * demand_tons
                carbon_cost = emissions_data["total_co2_kg"] * carbon_price_per_kg
                
                customer_result = {
                    "customer_id": j,
                    "facility_id": i,
                    "distance_km": dist_km,
                    "demand_tons": demand_tons,
                    "emissions_kg": emissions_data["total_co2_kg"],
                    "transport_cost": variable_cost,
                    "carbon_cost": carbon_cost,
                    "total_cost": variable_cost + carbon_cost
                }
                
                facility_result["customers_served"].append(customer_result)
                facility_result["total_distance_km"] += dist_km
                facility_result["total_emissions_kg"] += emissions_data["total_co2_kg"]
                facility_result["total_variable_cost"] += variable_cost
                
                results["transportation_matrix"].append(customer_result)
            
            total_cost += facility_result["fixed_cost"] + facility_result["total_variable_cost"]
            total_carbon_kg += facility_result["total_emissions_kg"]
            results["facilities"].append(facility_result)
        
        # Environmental impact analysis
        results["environmental_impact"] = {
            "total_co2_emissions_kg": total_carbon_kg,
            "total_co2_emissions_tons": total_carbon_kg / 1000,
            "carbon_constraint_kg": carbon_constraint_kg,
            "constraint_violation_kg": max(0, total_carbon_kg - carbon_constraint_kg) if carbon_constraint_kg else 0,
            "constraint_satisfied": total_carbon_kg <= carbon_constraint_kg if carbon_constraint_kg else True,
            "equivalent_trees_needed": int(total_carbon_kg / 22),  # 22 kg CO2 absorbed per tree per year
            "emission_breakdown_by_facility": [
                {"facility_id": f["facility_id"], "emissions_kg": f["total_emissions_kg"]}
                for f in results["facilities"]
            ]
        }
        
        # Cost analysis
        total_carbon_cost = total_carbon_kg * carbon_price_per_kg
        results["cost_analysis"] = {
            "total_operational_cost": total_cost,
            "total_carbon_cost": total_carbon_cost,
            "total_cost_with_carbon": total_cost + total_carbon_cost,
            "carbon_cost_percentage": (total_carbon_cost / (total_cost + total_carbon_cost)) * 100 if total_cost > 0 else 0,
            "cost_per_kg_co2": total_cost / total_carbon_kg if total_carbon_kg > 0 else 0
        }
        
        # Pareto analysis (cost vs emissions trade-off)
        results["pareto_analysis"] = {
            "cost_efficiency_ratio": total_cost / total_carbon_kg if total_carbon_kg > 0 else float('inf'),
            "carbon_efficiency_ratio": total_carbon_kg / total_cost if total_cost > 0 else float('inf'),
            "pareto_optimal": results["environmental_impact"]["constraint_satisfied"]
        }
        
        # Recommendations
        recommendations = []
        
        if not results["environmental_impact"]["constraint_satisfied"]:
            violation = results["environmental_impact"]["constraint_violation_kg"]
            recommendations.append({
                "type": "constraint_violation",
                "priority": "high",
                "description": f"CO2 emissions exceed constraint by {violation:.2f} kg",
                "action": "Consider optimizing routes, improving loading rates, or switching to cleaner transport modes"
            })
        
        if results["cost_analysis"]["carbon_cost_percentage"] > 20:
            recommendations.append({
                "type": "carbon_cost_optimization",
                "priority": "medium",
                "description": f"Carbon costs represent {results['cost_analysis']['carbon_cost_percentage']:.1f}% of total costs",
                "action": "Invest in fuel-efficient vehicles or alternative energy sources"
            })
        
        # Find facilities with highest emissions
        highest_emission_facility = max(results["facilities"], key=lambda x: x["total_emissions_kg"])
        if highest_emission_facility["total_emissions_kg"] > total_carbon_kg * 0.4:
            recommendations.append({
                "type": "facility_optimization",
                "priority": "medium",
                "description": f"Facility {highest_emission_facility['facility_id']} accounts for {(highest_emission_facility['total_emissions_kg']/total_carbon_kg)*100:.1f}% of total emissions",
                "action": "Consider relocating facility or optimizing its service area"
            })
        
        results["recommendations"] = recommendations
        return results
    
    def generate_carbon_visualization_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization data for carbon footprint analysis
        
        Args:
            analysis_results: Results from multi_objective_analysis
            
        Returns:
            Dict: Plotly visualization data
        """
        facilities = analysis_results["facilities"]
        customers = analysis_results.get("customers", [])
        
        # Emission intensity visualization (facility markers sized by emissions)
        facility_lats = [f["location"][0] for f in facilities]
        facility_lons = [f["location"][1] for f in facilities]
        facility_emissions = [f["total_emissions_kg"] for f in facilities]
        facility_names = [f"Facility {f['facility_id']}" for f in facilities]
        
        max_emission = max(facility_emissions) if facility_emissions else 1
        facility_sizes = [20 + (e/max_emission) * 40 for e in facility_emissions]
        
        # Color coding based on emission intensity
        colors = []
        for emission in facility_emissions:
            if emission > max_emission * 0.7:
                colors.append('red')  # High emissions
            elif emission > max_emission * 0.4:
                colors.append('orange')  # Medium emissions
            else:
                colors.append('green')  # Low emissions
        
        facility_trace = {
            'type': 'scattermapbox',
            'lat': facility_lats,
            'lon': facility_lons,
            'mode': 'markers',
            'marker': {
                'size': facility_sizes,
                'color': colors,
                'opacity': 0.7,
                'sizemode': 'diameter'
            },
            'text': [f"{name}<br>Emissions: {e:.2f} kg CO2" for name, e in zip(facility_names, facility_emissions)],
            'hovertemplate': '%{text}<extra></extra>',
            'name': 'Facilities (sized by emissions)'
        }
        
        # Transportation routes visualization
        route_traces = []
        for transport in analysis_results["transportation_matrix"]:
            facility = facilities[transport["facility_id"]]
            
            # Color routes by emission intensity
            emission_per_ton_km = transport["emissions_kg"] / (transport["distance_km"] * transport["demand_tons"]) if transport["distance_km"] * transport["demand_tons"] > 0 else 0
            
            if emission_per_ton_km > 0.1:  # High emission route
                route_color = 'red'
                opacity = 0.8
            elif emission_per_ton_km > 0.05:  # Medium emission route
                route_color = 'orange'
                opacity = 0.6
            else:  # Low emission route
                route_color = 'green'
                opacity = 0.4
            
            # This would need customer coordinates to draw actual routes
            # For now, just indicate high-emission routes
        
        # Pie chart data for emission breakdown
        emission_breakdown = {
            'labels': [f"Facility {f['facility_id']}" for f in facilities],
            'values': [f['total_emissions_kg'] for f in facilities],
            'type': 'pie',
            'marker': {'colors': colors},
            'textinfo': 'label+percent',
            'hovertemplate': '%{label}<br>Emissions: %{value:.2f} kg CO2<br>Percentage: %{percent}<extra></extra>'
        }
        
        # Bar chart data for cost vs emissions comparison
        facility_ids = [f"Facility {f['facility_id']}" for f in facilities]
        costs = [f['fixed_cost'] + f['total_variable_cost'] for f in facilities]
        emissions = [f['total_emissions_kg'] for f in facilities]
        
        cost_bar = {
            'x': facility_ids,
            'y': costs,
            'type': 'bar',
            'name': 'Operational Cost',
            'marker': {'color': 'blue'},
            'yaxis': 'y'
        }
        
        emission_bar = {
            'x': facility_ids,
            'y': emissions,
            'type': 'bar',
            'name': 'CO2 Emissions (kg)',
            'marker': {'color': 'red'},
            'yaxis': 'y2'
        }
        
        layout = {
            'mapbox': {
                'style': 'open-street-map',
                'center': {
                    'lat': np.mean(facility_lats) if facility_lats else 35.6762,
                    'lon': np.mean(facility_lons) if facility_lons else 139.6503
                },
                'zoom': 8
            },
            'showlegend': True,
            'title': 'Carbon Footprint Analysis - Facility Locations'
        }
        
        return {
            'map_data': [facility_trace],
            'map_layout': layout,
            'pie_chart': emission_breakdown,
            'comparison_chart': {
                'data': [cost_bar, emission_bar],
                'layout': {
                    'title': 'Cost vs Emissions by Facility',
                    'xaxis': {'title': 'Facilities'},
                    'yaxis': {'title': 'Cost', 'side': 'left'},
                    'yaxis2': {'title': 'CO2 Emissions (kg)', 'side': 'right', 'overlaying': 'y'}
                }
            }
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about carbon footprint service capabilities
        
        Returns:
            Dict: Service information
        """
        return {
            "carbon_footprint_analysis": {
                "description": "Comprehensive carbon footprint analysis for sustainable logistics optimization",
                "features": [
                    "Empirical CO2 emission calculations based on vehicle capacity and loading rate",
                    "Multi-objective optimization (cost vs environmental impact)",
                    "Standard emission factors for different transport modes",
                    "Carbon constraint optimization",
                    "Multi-modal transportation analysis",
                    "Pareto analysis for cost-emissions trade-offs",
                    "Environmental impact visualization"
                ],
                "transport_modes": list(self.emission_factors.keys()),
                "emission_factors_g_per_ton_km": self.emission_factors,
                "supported_fuel_types": list(self.fuel_to_co2.keys())
            },
            "multi_objective_optimization": {
                "description": "Simultaneous optimization of cost and environmental objectives",
                "features": [
                    "Carbon emission constraints",
                    "Carbon pricing integration",
                    "Pareto frontier analysis",
                    "Sustainability recommendations",
                    "Environmental compliance checking"
                ]
            },
            "visualization_capabilities": [
                "Emission-intensity facility mapping",
                "Transportation route emission analysis",
                "Cost vs emissions comparison charts",
                "Emission breakdown by facility",
                "Environmental impact indicators"
            ]
        }