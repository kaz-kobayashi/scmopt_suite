from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import io
from app.services.core_service import (
    co2,
    compute_durations_simple,
    make_time_df
)
from app.services.routing_service import (
    optimize_vehicle_routing,
    calculate_co2_emissions,
    optimize_delivery_schedule,
    calculate_distance_matrix
)
from app.services.pyvrp_service import optimize_vrp_with_pyvrp
from app.services.hybrid_vrp_service import optimize_vrp_hybrid
from app.services.pyvrp_standard_service import (
    solve_vrplib_instance, 
    solve_vrp_from_coordinates_standard
)

router = APIRouter()

@router.post("/co2-calculation")
async def calculate_co2_emissions(
    capacity: float = Form(..., description="Vehicle capacity in tons"),
    rate: float = Form(0.5, description="Loading rate (0-1)"),
    diesel: bool = Form(False, description="True for diesel, False for gasoline")
):
    """
    Calculate CO2 emissions and fuel consumption for transportation
    """
    try:
        # Validate inputs
        if capacity <= 0:
            raise HTTPException(
                status_code=400,
                detail="Vehicle capacity must be greater than 0"
            )
        
        if not 0.1 <= rate <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Loading rate must be between 0.1 and 1.0"
            )
        
        # Calculate emissions
        fuel_consumption, co2_emissions = co2(capacity, rate, diesel)
        
        # Calculate additional metrics
        fuel_type = "Diesel" if diesel else "Gasoline"
        annual_distance = 50000  # Example: 50,000 km/year
        annual_fuel = fuel_consumption * capacity * rate * annual_distance
        annual_co2 = co2_emissions * capacity * rate * annual_distance / 1000  # Convert to kg
        
        return {
            "emissions_calculation": {
                "fuel_consumption_L_per_ton_km": float(fuel_consumption),
                "co2_emissions_g_per_ton_km": float(co2_emissions),
                "fuel_type": fuel_type,
                "vehicle_capacity_tons": capacity,
                "loading_rate": rate
            },
            "annual_estimates": {
                "estimated_annual_distance_km": annual_distance,
                "estimated_annual_fuel_consumption_L": float(annual_fuel),
                "estimated_annual_co2_emissions_kg": float(annual_co2)
            },
            "optimization_suggestions": {
                "improve_loading_rate": rate < 0.8,
                "consider_larger_vehicle": capacity < 10 and rate > 0.8,
                "efficiency_score": min(100, int((rate * 0.7 + (1 - fuel_consumption/10) * 0.3) * 100))
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating CO2 emissions: {str(e)}")

@router.post("/compute-durations")
async def compute_travel_durations(
    customer_file: UploadFile = File(...),
    plant_file: Optional[UploadFile] = File(None),
    toll: bool = Form(True, description="Include toll roads in routing")
):
    """
    Compute travel durations and distances between locations
    """
    try:
        # Read customer file
        cust_contents = await customer_file.read()
        cust_df = pd.read_csv(io.StringIO(cust_contents.decode('utf-8')))
        
        # Validate customer file columns
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in cust_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Customer file missing required columns: {missing_cols}"
            )
        
        # Read plant file if provided
        plnt_df = None
        if plant_file:
            plnt_contents = await plant_file.read()
            plnt_df = pd.read_csv(io.StringIO(plnt_contents.decode('utf-8')))
            
            # Validate plant file columns
            plnt_missing = [col for col in required_cols if col not in plnt_df.columns]
            if plnt_missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Plant file missing required columns: {plnt_missing}"
                )
        
        # Compute durations and distances
        durations, distances, node_df = compute_durations_simple(cust_df, plnt_df)
        
        # Create time dataframe
        time_df = make_time_df(node_df, durations, distances)
        
        # Calculate statistics
        n_nodes = len(node_df)
        total_pairs = n_nodes * (n_nodes - 1)
        avg_distance = float(np.mean(distances[distances > 0]))
        avg_duration = float(np.mean(durations[durations > 0]))
        max_distance = float(np.max(distances))
        max_duration = float(np.max(durations))
        
        return {
            "routing_results": {
                "durations": durations.tolist(),
                "distances": distances.tolist(),
                "time_matrix": time_df.to_dict("records"),
                "node_locations": node_df.to_dict("records")
            },
            "statistics": {
                "number_of_locations": n_nodes,
                "number_of_customer_locations": len(cust_df),
                "number_of_plant_locations": len(plnt_df) if plnt_df is not None else 0,
                "total_route_pairs": total_pairs,
                "average_distance_km": avg_distance,
                "average_duration_seconds": avg_duration,
                "maximum_distance_km": max_distance,
                "maximum_duration_seconds": max_duration
            },
            "parameters": {
                "include_toll_roads": toll,
                "calculation_method": "Haversine distance approximation"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error computing durations: {str(e)}")

@router.post("/distance-matrix")
async def create_distance_matrix(
    file: UploadFile = File(...),
    output_format: str = Form("json", description="Output format: json or csv")
):
    """
    Create distance matrix from location data
    """
    try:
        # Read location file
        contents = await file.read()
        locations_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in locations_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Compute distances
        durations, distances, node_df = compute_durations_simple(locations_df)
        
        # Create distance matrix dataframe
        location_names = node_df['name'].tolist()
        distance_matrix_df = pd.DataFrame(
            distances,
            index=location_names,
            columns=location_names
        )
        
        duration_matrix_df = pd.DataFrame(
            durations,
            index=location_names,
            columns=location_names
        )
        
        if output_format.lower() == "csv":
            # Return as CSV string
            distance_csv = distance_matrix_df.to_csv()
            duration_csv = duration_matrix_df.to_csv()
            
            return {
                "distance_matrix_csv": distance_csv,
                "duration_matrix_csv": duration_csv,
                "location_count": len(location_names)
            }
        else:
            # Return as JSON
            return {
                "distance_matrix": distance_matrix_df.to_dict(),
                "duration_matrix": duration_matrix_df.to_dict(),
                "locations": location_names,
                "matrix_size": len(location_names),
                "units": {
                    "distance": "kilometers",
                    "duration": "seconds"
                }
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating distance matrix: {str(e)}")

@router.post("/route-optimization")
async def optimize_routes(
    file: UploadFile = File(...),
    vehicle_capacity: float = Form(..., description="Vehicle capacity in kg"),
    max_routes: int = Form(5, description="Maximum number of routes"),
    depot_name: str = Form("Depot", description="Name of depot location"),
    max_runtime: int = Form(30, description="Maximum optimization runtime in seconds")
):
    """
    Advanced route optimization using PyVRP
    """
    try:
        # Read location file
        contents = await file.read()
        locations_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in locations_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Add default demand if not present
        if 'demand' not in locations_df.columns:
            locations_df['demand'] = 1.0
        
        # Use hybrid optimization (tries both PyVRP and Simple VRP, returns better result)
        result = optimize_vrp_hybrid(
            locations_df, 
            vehicle_capacity, 
            max_routes, 
            depot_name,
            max_runtime
        )
        
        # Format response for compatibility with frontend
        optimized_routes = []
        for route in result['routes']:
            optimized_routes.append({
                "route_id": route['route_id'],
                "locations": route['locations'],
                "total_distance": route['total_distance'],
                "total_load": route['total_demand']
            })
        
        return {
            "optimized_routes": optimized_routes,
            "summary": {
                "total_routes": result['summary']['total_routes'],
                "total_distance_km": result['summary']['total_distance'],
                "total_load": result['summary']['total_demand_served'],
                "depot_location": result['parameters']['depot'],
                "vehicle_capacity": vehicle_capacity,
                "locations_served": result['summary']['customers_served'],
                "locations_unserved": result['summary']['customers_unserved'],
                "optimization_cost": result['summary'].get('optimization_cost', 0),
                "avg_capacity_utilization": result['summary']['avg_capacity_utilization']
            },
            "optimization_info": {
                "algorithm": result['optimization_stats']['algorithm'],
                "capacity_constrained": True,
                "depot_based": True,
                "runtime_seconds": result['optimization_stats']['runtime_seconds'],
                "iterations": result['optimization_stats'].get('iterations', 0),
                "convergence": result['optimization_stats']['convergence']
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error optimizing routes: {str(e)}")

@router.post("/advanced-vrp")
async def advanced_vehicle_routing(
    file: UploadFile = File(...),
    vehicle_capacity: float = Form(..., description="Vehicle capacity"),
    max_routes: int = Form(5, description="Maximum number of routes"),
    depot_name: str = Form("Depot", description="Name of depot location"),
    max_runtime: int = Form(30, description="Maximum optimization runtime in seconds")
):
    """
    Advanced Vehicle Routing Problem optimization using PyVRP
    """
    try:
        # Read location file
        contents = await file.read()
        locations_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in locations_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Add default demand if not present
        if 'demand' not in locations_df.columns:
            locations_df['demand'] = 1.0
        
        # Use hybrid optimization for advanced VRP
        result = optimize_vrp_hybrid(
            locations_df, 
            vehicle_capacity, 
            max_routes, 
            depot_name,
            max_runtime
        )
        
        return {
            "vrp_result": result,
            "optimization_method": result['optimization_stats']['algorithm'],
            "performance_metrics": {
                "route_efficiency": result["summary"]["avg_capacity_utilization"],
                "distance_optimization": "Minimized total travel distance with PyVRP",
                "service_coverage": f"{result['summary']['customers_served']} customers served",
                "optimization_runtime": f"{result['optimization_stats']['runtime_seconds']:.2f}s",
                "iterations": result['optimization_stats']['iterations'],
                "convergence": result['optimization_stats']['convergence']
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in advanced VRP optimization: {str(e)}")

@router.post("/delivery-schedule")
async def create_delivery_schedule(
    file: UploadFile = File(...),
    working_start: int = Form(8, description="Working hours start (24h format)"),
    working_end: int = Form(18, description="Working hours end (24h format)"),
    service_time: int = Form(30, description="Service time per delivery (minutes)")
):
    """
    Optimize delivery schedule within working hours
    """
    try:
        # Read orders file
        contents = await file.read()
        orders_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Add default values if columns are missing
        if 'travel_time_minutes' not in orders_df.columns:
            orders_df['travel_time_minutes'] = np.random.randint(15, 45, len(orders_df))
        
        if 'order_id' not in orders_df.columns:
            orders_df['order_id'] = range(1, len(orders_df) + 1)
        
        # Optimize schedule
        result = optimize_delivery_schedule(
            orders_df, 
            working_hours=(working_start, working_end),
            service_time_minutes=service_time
        )
        
        return {
            "delivery_schedule": result,
            "optimization_summary": {
                "schedule_efficiency": f"{result['summary']['schedule_efficiency']:.1%}",
                "working_hours_utilization": f"{(result['summary']['total_service_time_hours'] + result['summary']['total_travel_time_hours']) / (working_end - working_start):.1%}",
                "recommendations": "Focus on reducing travel time between deliveries" if result['summary']['total_travel_time_hours'] > result['summary']['total_service_time_hours'] else "Schedule is well optimized"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating delivery schedule: {str(e)}")

@router.post("/emissions-analysis")
async def analyze_route_emissions(
    distance_km: float = Form(..., description="Total route distance in kilometers"),
    capacity_kg: float = Form(..., description="Vehicle capacity in kilograms"),
    loading_rate: float = Form(0.7, description="Loading rate (0-1)"),
    fuel_type: str = Form("gasoline", description="Fuel type: gasoline or diesel")
):
    """
    Analyze CO2 emissions for a specific route
    """
    try:
        # Validate inputs
        if distance_km <= 0:
            raise HTTPException(status_code=400, detail="Distance must be greater than 0")
        if capacity_kg <= 0:
            raise HTTPException(status_code=400, detail="Capacity must be greater than 0")
        if not 0.1 <= loading_rate <= 1.0:
            raise HTTPException(status_code=400, detail="Loading rate must be between 0.1 and 1.0")
        
        # Calculate emissions
        emissions_result = calculate_co2_emissions(capacity_kg, loading_rate, distance_km, fuel_type)
        
        # Add sustainability recommendations
        sustainability_score = 100 - (emissions_result['co2_emissions_kg'] / distance_km * 10)
        sustainability_score = max(0, min(100, sustainability_score))
        
        recommendations = []
        if loading_rate < 0.8:
            recommendations.append("Increase loading rate to reduce emissions per unit")
        if fuel_type.lower() == "gasoline":
            recommendations.append("Consider switching to diesel for better fuel efficiency")
        if emissions_result['co2_emissions_kg'] > distance_km * 0.5:
            recommendations.append("Route generates high emissions - consider consolidation")
        
        return {
            "emissions_analysis": emissions_result,
            "sustainability": {
                "sustainability_score": round(sustainability_score, 1),
                "emissions_per_km": round(emissions_result['co2_emissions_kg'] / distance_km, 3),
                "efficiency_rating": "High" if sustainability_score > 75 else "Medium" if sustainability_score > 50 else "Low"
            },
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing route emissions: {str(e)}")

@router.post("/vrplib-solve")
async def solve_vrplib_file(
    file: UploadFile = File(...),
    max_runtime: int = Form(60, description="Maximum optimization runtime in seconds"),
    max_iterations: int = Form(10000, description="Maximum iterations"),
    seed: int = Form(42, description="Random seed for reproducibility")
):
    """
    Solve a VRPLIB format VRP instance using standard PyVRP
    """
    try:
        # Validate file format
        if not file.filename.endswith('.vrp'):
            raise HTTPException(
                status_code=400,
                detail="File must be in VRPLIB format with .vrp extension"
            )
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.vrp') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        try:
            # Solve using standard PyVRP
            result = solve_vrplib_instance(
                temp_file_path,
                max_runtime_seconds=max_runtime,
                max_iterations=max_iterations,
                seed=seed
            )
            
            return {
                "vrplib_result": result,
                "optimization_method": result['optimization_stats']['algorithm'],
                "instance_info": result['optimization_stats'].get('instance_info', {}),
                "performance_metrics": {
                    "objective_value": result['summary']['optimization_cost'],
                    "routes_generated": result['summary']['total_routes'],
                    "customers_served": result['summary']['customers_served'],
                    "optimization_runtime": f"{result['optimization_stats']['runtime_seconds']:.2f}s",
                    "iterations": result['optimization_stats']['iterations'],
                    "convergence": result['optimization_stats']['convergence']
                }
            }
        
        finally:
            # Clean up temporary file
            import os
            os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error solving VRPLIB instance: {str(e)}")

@router.post("/standard-pyvrp")
async def solve_with_standard_pyvrp(
    file: UploadFile = File(...),
    vehicle_capacity: float = Form(..., description="Vehicle capacity"),
    max_routes: int = Form(5, description="Maximum number of routes"),
    depot_name: str = Form("Depot", description="Name of depot location"),
    max_runtime: int = Form(30, description="Maximum optimization runtime in seconds")
):
    """
    Solve VRP using standard PyVRP approach (improved algorithm)
    """
    try:
        # Read location file
        contents = await file.read()
        locations_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in locations_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Add default demand if not present
        if 'demand' not in locations_df.columns:
            locations_df['demand'] = 1.0
        
        # Solve using standard PyVRP approach
        result = solve_vrp_from_coordinates_standard(
            locations_df, 
            vehicle_capacity, 
            max_routes, 
            depot_name,
            max_runtime
        )
        
        # Format response for compatibility with frontend
        optimized_routes = []
        for route in result['routes']:
            optimized_routes.append({
                "route_id": route['route_id'],
                "locations": route['locations'],
                "total_distance": route['total_distance'],
                "total_load": route['total_demand']
            })
        
        return {
            "optimized_routes": optimized_routes,
            "summary": {
                "total_routes": result['summary']['total_routes'],
                "total_distance_km": result['summary']['total_distance'],
                "total_load": result['summary']['total_demand_served'],
                "depot_location": result['parameters']['depot'],
                "vehicle_capacity": vehicle_capacity,
                "locations_served": result['summary']['customers_served'],
                "locations_unserved": result['summary']['customers_unserved'],
                "optimization_cost": result['summary'].get('optimization_cost', 0),
                "avg_capacity_utilization": result['summary']['avg_capacity_utilization']
            },
            "optimization_info": {
                "algorithm": result['optimization_stats']['algorithm'],
                "capacity_constrained": True,
                "depot_based": True,
                "runtime_seconds": result['optimization_stats']['runtime_seconds'],
                "iterations": result['optimization_stats']['iterations'],
                "convergence": result['optimization_stats']['convergence']
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in standard PyVRP optimization: {str(e)}")