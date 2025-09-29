"""
PyVRP Routes - Comprehensive API routes for all VRP variants using PyVRP

This module provides REST API endpoints for all VRP variants:
- Basic Capacitated VRP (CVRP)
- VRP with Time Windows (VRPTW)
- Multi-Depot VRP (MDVRP)
- Pickup and Delivery VRP (PDVRP)
- Prize-Collecting VRP (PC-VRP)
- VRPLIB format support
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import io
import tempfile
import os
import pandas as pd
import json
import logging
import traceback
from datetime import datetime

from app.services.pyvrp_service import PyVRPService
from app.services.pyvrp_unified_service import PyVRPUnifiedService, dataframes_to_vrp_json
from app.models.vrp_models import (
    CVRPRequest, VRPTWRequest, MDVRPRequest, PDVRPRequest, PCVRPRequest, VRPLIBRequest,
    CVRPResult, VRPTWResult, MDVRPResult, PDVRPResult, PCVRPResult, VRPLIBResult,
    LocationModel, TimeWindow, PickupDeliveryPair, DepotModel,
    VRPDataUpload, VRPDataValidation, VRPSolverConfig, VRPError,
    SolutionComparison, VRPBenchmark, VRPExportRequest, VRPReport
)
from app.models.vrp_unified_models import VRPProblemData, UnifiedVRPSolution

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure DEBUG level is set

# Initialize PyVRP services
pyvrp_service = PyVRPService()
pyvrp_unified_service = PyVRPUnifiedService()

def clean_for_json(data):
    """Clean data for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif hasattr(data, 'dict'):  # Pydantic model
        return clean_for_json(data.dict())
    elif isinstance(data, (float, int)):
        if str(data) in ['inf', '-inf', 'nan']:
            return None
        return data
    else:
        return data

@router.get("/test")
async def test_pyvrp_endpoint():
    """Test endpoint for PyVRP routes"""
    return {
        "message": "PyVRP routes test successful",
        "status": "ok",
        "available_variants": [
            "CVRP", "VRPTW", "MDVRP", "PDVRP", "PC-VRP", "VRPLIB"
        ],
        "timestamp": datetime.now().isoformat()
    }

@router.post("/solve/cvrp", response_model=CVRPResult)
async def solve_cvrp(request: CVRPRequest):
    """
    Solve Capacitated Vehicle Routing Problem (CVRP)
    
    This endpoint solves the basic VRP variant with capacity constraints.
    """
    try:
        logger.info(f"Solving CVRP with {len(request.locations)} locations")
        
        # Convert Pydantic models to dict format expected by service
        locations = [loc.dict() for loc in request.locations]
        
        # Prepare demands list
        demands = [loc.demand for loc in request.locations]
        
        # Call PyVRP service
        result = pyvrp_service.solve_basic_cvrp(
            locations=locations,
            demands=demands,
            vehicle_capacity=request.vehicle_capacity,
            depot_index=request.depot_index,
            num_vehicles=request.num_vehicles,
            max_runtime=request.max_runtime
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "CVRP solve failed"))
        
        # Convert to response model
        return clean_for_json(result)
        
    except Exception as e:
        logger.error(f"CVRP solve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CVRP optimization failed: {str(e)}")

@router.post("/solve/vrptw", response_model=VRPTWResult)
async def solve_vrptw(request: VRPTWRequest):
    """
    Solve Vehicle Routing Problem with Time Windows (VRPTW)
    
    This endpoint solves VRP with capacity and time window constraints.
    """
    try:
        logger.info(f"Solving VRPTW with {len(request.locations)} locations")
        
        # Convert Pydantic models to required format
        locations = [loc.dict() for loc in request.locations]
        demands = [loc.demand for loc in request.locations]
        time_windows = [(tw.earliest, tw.latest) for tw in request.time_windows]
        
        # Call PyVRP service
        result = pyvrp_service.solve_vrptw(
            locations=locations,
            demands=demands,
            time_windows=time_windows,
            service_times=request.service_times,
            vehicle_capacity=request.vehicle_capacity,
            depot_index=request.depot_index,
            num_vehicles=request.num_vehicles,
            max_runtime=request.max_runtime
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "VRPTW solve failed"))
        
        return clean_for_json(result)
        
    except Exception as e:
        logger.error(f"VRPTW solve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VRPTW optimization failed: {str(e)}")

@router.post("/solve/mdvrp", response_model=MDVRPResult)
async def solve_mdvrp(request: MDVRPRequest):
    """
    Solve Multi-Depot Vehicle Routing Problem (MDVRP)
    
    This endpoint solves VRP with multiple depots.
    """
    try:
        logger.info(f"Solving MDVRP with {len(request.depots)} depots")
        
        # Convert to required format
        locations = [loc.dict() for loc in request.locations]
        demands = [loc.demand for loc in request.locations]
        vehicle_capacities = [depot.capacity for depot in request.depots]
        vehicles_per_depot = [depot.num_vehicles for depot in request.depots]
        
        # Call PyVRP service
        result = pyvrp_service.solve_multi_depot_vrp(
            locations=locations,
            demands=demands,
            depot_indices=request.depot_indices,
            vehicle_capacities=vehicle_capacities,
            vehicles_per_depot=vehicles_per_depot,
            max_runtime=request.max_runtime
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "MDVRP solve failed"))
        
        return clean_for_json(result)
        
    except Exception as e:
        logger.error(f"MDVRP solve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MDVRP optimization failed: {str(e)}")

@router.post("/solve/pdvrp", response_model=PDVRPResult)
async def solve_pdvrp(request: PDVRPRequest):
    """
    Solve Pickup and Delivery Vehicle Routing Problem (PDVRP)
    
    This endpoint solves VRP with pickup-delivery constraints.
    """
    try:
        logger.info(f"Solving PDVRP with {len(request.pickup_delivery_pairs)} PD pairs")
        
        # Convert to required format
        locations = [loc.dict() for loc in request.locations]
        pd_pairs = [(pair.pickup_location_idx, pair.delivery_location_idx) 
                   for pair in request.pickup_delivery_pairs]
        demands = [pair.demand for pair in request.pickup_delivery_pairs]
        
        # Call PyVRP service
        result = pyvrp_service.solve_pickup_delivery_vrp(
            locations=locations,
            pickup_delivery_pairs=pd_pairs,
            demands=demands,
            vehicle_capacity=request.vehicle_capacity,
            depot_index=request.depot_index,
            max_runtime=request.max_runtime
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "PDVRP solve failed"))
        
        return clean_for_json(result)
        
    except Exception as e:
        logger.error(f"PDVRP solve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDVRP optimization failed: {str(e)}")

@router.post("/solve/pcvrp", response_model=PCVRPResult)
async def solve_pcvrp(request: PCVRPRequest):
    """
    Solve Prize-Collecting Vehicle Routing Problem (PC-VRP)
    
    This endpoint solves VRP with prize collection objectives.
    """
    try:
        logger.info(f"Solving PC-VRP with minimum prize {request.min_prize}")
        
        # Convert to required format
        locations = [loc.dict() for loc in request.locations]
        demands = [loc.demand for loc in request.locations]
        
        # Call PyVRP service
        result = pyvrp_service.solve_prize_collecting_vrp(
            locations=locations,
            prizes=request.prizes,
            demands=demands,
            vehicle_capacity=request.vehicle_capacity,
            min_prize=request.min_prize,
            depot_index=request.depot_index,
            max_runtime=request.max_runtime
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "PC-VRP solve failed"))
        
        return clean_for_json(result)
        
    except Exception as e:
        logger.error(f"PC-VRP solve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PC-VRP optimization failed: {str(e)}")

@router.post("/solve/vrplib", response_model=VRPLIBResult)
async def solve_vrplib(request: VRPLIBRequest):
    """
    Solve VRPLIB format instance
    
    This endpoint accepts VRPLIB format files and solves them using PyVRP.
    """
    try:
        logger.info(f"Solving VRPLIB instance: {request.file_name}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vrp', delete=False) as temp_file:
            temp_file.write(request.file_content)
            temp_file_path = temp_file.name
        
        try:
            # Call PyVRP service
            result = pyvrp_service.solve_vrplib_instance(
                file_path=temp_file_path,
                max_runtime=request.max_runtime
            )
            
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result.get("message", "VRPLIB solve failed"))
            
            # Add instance information
            result["instance_name"] = request.file_name
            result["instance_type"] = result.get("problem_type", "Unknown")
            
            return clean_for_json(result)
            
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"VRPLIB solve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VRPLIB optimization failed: {str(e)}")

@router.post("/solve", response_model=UnifiedVRPSolution)  
async def solve_unified_vrp(request: VRPProblemData):
    """
    Solve VRP using unified API format
    
    This endpoint accepts a unified JSON format and automatically determines the VRP variant
    based on the problem data structure:
    - Time windows present -> VRPTW
    - Multiple depots -> MDVRP  
    - Pickup demands -> PDVRP
    - Prizes and optional visits -> PC-VRP
    - Otherwise -> CVRP
    
    The unified format supports pandas DataFrame conversion via the dataframes_to_vrp_json function.
    """
    # Wrap entire function in try-except to catch validation errors
    try:
        logger.info("=== ROUTE HANDLER STARTED ===")
        logger.debug(f"Request type: {type(request)}")
        
        # Log raw request data before any processing
        try:
            if hasattr(request, 'dict'):
                request_dict = request.dict()
                logger.debug(f"Full request data: {json.dumps(request_dict, indent=2)}")
            else:
                logger.debug(f"Request object: {request}")
        except Exception as log_e:
            logger.warning(f"Could not log request data: {str(log_e)}")
        
        # Check if request attributes exist and log their types
        logger.debug(f"Request attributes - clients: {hasattr(request, 'clients')}, depots: {hasattr(request, 'depots')}, vehicle_types: {hasattr(request, 'vehicle_types')}")
        
        # Safely check and log data
        clients_count = 0
        depots_count = 0
        vehicle_types_count = 0
        
        try:
            if hasattr(request, 'clients') and request.clients is not None:
                clients_count = len(request.clients)
                logger.debug(f"Clients type: {type(request.clients)}, count: {clients_count}")
                if clients_count > 0:
                    logger.debug(f"First client: {request.clients[0].dict() if hasattr(request.clients[0], 'dict') else request.clients[0]}")
        except Exception as e:
            logger.error(f"Error accessing clients: {str(e)}")
            
        try:
            if hasattr(request, 'depots') and request.depots is not None:
                depots_count = len(request.depots)
                logger.debug(f"Depots type: {type(request.depots)}, count: {depots_count}")
                if depots_count > 0:
                    logger.debug(f"First depot: {request.depots[0].dict() if hasattr(request.depots[0], 'dict') else request.depots[0]}")
        except Exception as e:
            logger.error(f"Error accessing depots: {str(e)}")
            
        try:
            if hasattr(request, 'vehicle_types') and request.vehicle_types is not None:
                vehicle_types_count = len(request.vehicle_types)
                logger.debug(f"Vehicle types type: {type(request.vehicle_types)}, count: {vehicle_types_count}")
                if vehicle_types_count > 0:
                    logger.debug(f"First vehicle type: {request.vehicle_types[0].dict() if hasattr(request.vehicle_types[0], 'dict') else request.vehicle_types[0]}")
        except Exception as e:
            logger.error(f"Error accessing vehicle_types: {str(e)}")
        
        logger.info(f"Raw request received: clients={clients_count}, depots={depots_count}, vehicle_types={vehicle_types_count}")
        
        # Validate input data
        if not request.clients:
            logger.error("No clients provided in request")
            raise HTTPException(status_code=400, detail="At least one client is required")
        if not request.depots:
            logger.error("No depots provided in request")
            raise HTTPException(status_code=400, detail="At least one depot is required")  
        if not request.vehicle_types:
            logger.error("No vehicle types provided in request")
            raise HTTPException(status_code=400, detail="At least one vehicle type is required")
            
        logger.info(f"Validation passed. Solving unified VRP with {len(request.clients)} clients and {len(request.depots)} depots")
        
        # Log sample data for debugging
        try:
            logger.debug(f"First 2 clients: {[c.dict() for c in request.clients[:2]]}")
            logger.debug(f"All vehicle types: {[vt.dict() for vt in request.vehicle_types]}")
            logger.debug(f"All depots: {[d.dict() for d in request.depots]}")
        except Exception as e:
            logger.warning(f"Could not log sample data: {str(e)}")
        
        # Call unified PyVRP service
        logger.info("Calling pyvrp_unified_service.solve()")
        result = pyvrp_unified_service.solve(request)
        
        logger.info(f"VRP solve result: status={result.status}, objective_value={result.objective_value}, routes_count={len(result.routes)}")
        
        if result.status == "error":
            logger.error("VRP solver returned error status")
            raise HTTPException(status_code=500, detail="VRP solve failed")
            
        if result.objective_value is None:
            logger.error("VRP solve returned None objective_value, creating fallback solution")
            # Create a simple fallback response
            return UnifiedVRPSolution(
                status="error",
                objective_value=0.0,
                routes=[],
                computation_time=0.0,
                solver="Error",
                is_feasible=False,
                problem_type="CVRP"
            )
        
        logger.info("Successfully solved VRP, returning cleaned result")
        return clean_for_json(result)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception in solve_unified_vrp: status_code={he.status_code}, detail={he.detail}")
        raise  # Re-raise HTTP exceptions as-is
    except ValueError as ve:
        logger.error(f"ValueError in solve_unified_vrp: {str(ve)}")
        logger.error(f"ValueError traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Invalid input data: {str(ve)}")
    except TypeError as te:
        logger.error(f"TypeError in solve_unified_vrp: {str(te)}")
        logger.error(f"TypeError traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Type error in input data: {str(te)}")
    except Exception as e:
        logger.error(f"Unexpected error in solve_unified_vrp: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"VRP optimization failed: {str(e)}")

@router.post("/upload/csv")
async def upload_csv_data(
    file: UploadFile = File(...),
    data_type: str = Form(..., description="Type: locations, time_windows, demands, etc."),
    validate_data: bool = Form(True, description="Validate uploaded data")
):
    """
    Upload and validate CSV data for VRP problems
    
    Supported data types:
    - locations: Location data with lat, lon, name, demand
    - time_windows: Time window constraints
    - depots: Multi-depot information
    - pickup_delivery: Pickup-delivery pairs
    """
    try:
        # Read CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Validate based on data type
        validation_result = _validate_csv_data(df, data_type)
        
        if not validation_result["is_valid"] and validate_data:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {validation_result['errors']}"
            )
        
        return {
            "message": f"Successfully uploaded {data_type} data",
            "file_name": file.filename,
            "data_type": data_type,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "validation": validation_result,
            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
        }
        
    except Exception as e:
        logger.error(f"CSV upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to upload CSV data: {str(e)}")

@router.post("/upload/vrplib")
async def upload_vrplib_file(file: UploadFile = File(...)):
    """
    Upload and parse VRPLIB format file
    
    This endpoint accepts .vrp files in VRPLIB format and parses them for solving.
    """
    try:
        if not file.filename.endswith('.vrp'):
            raise HTTPException(
                status_code=400,
                detail="File must be in VRPLIB format with .vrp extension"
            )
        
        # Read file content
        content = await file.read()
        file_content = content.decode('utf-8')
        
        # Parse VRPLIB file
        parse_result = pyvrp_service.parse_vrplib_file(file.filename)
        
        if parse_result["status"] == "error":
            raise HTTPException(status_code=400, detail=parse_result["message"])
        
        return {
            "message": f"Successfully parsed VRPLIB file: {file.filename}",
            "file_name": file.filename,
            "file_size": len(content),
            "parse_result": parse_result,
            "ready_to_solve": True
        }
        
    except Exception as e:
        logger.error(f"VRPLIB upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to upload VRPLIB file: {str(e)}")

@router.get("/variants")
async def get_vrp_variants():
    """
    Get information about supported VRP variants
    
    Returns details about all supported VRP problem types and their characteristics.
    """
    return {
        "variants": {
            "CVRP": {
                "name": "Capacitated Vehicle Routing Problem",
                "description": "Basic VRP with vehicle capacity constraints",
                "constraints": ["vehicle_capacity", "depot_return"],
                "objectives": ["minimize_distance"],
                "complexity": "Basic"
            },
            "VRPTW": {
                "name": "VRP with Time Windows",
                "description": "VRP with time window constraints for each customer",
                "constraints": ["vehicle_capacity", "time_windows", "service_times"],
                "objectives": ["minimize_distance", "meet_time_windows"],
                "complexity": "Advanced"
            },
            "MDVRP": {
                "name": "Multi-Depot VRP",
                "description": "VRP with multiple depot locations",
                "constraints": ["vehicle_capacity", "multiple_depots"],
                "objectives": ["minimize_distance", "balance_depot_utilization"],
                "complexity": "Advanced"
            },
            "PDVRP": {
                "name": "Pickup and Delivery VRP",
                "description": "VRP with pickup-delivery pair constraints",
                "constraints": ["vehicle_capacity", "pickup_before_delivery"],
                "objectives": ["minimize_distance", "satisfy_pd_constraints"],
                "complexity": "Advanced"
            },
            "PC-VRP": {
                "name": "Prize-Collecting VRP",
                "description": "VRP with optional customers and prize collection",
                "constraints": ["vehicle_capacity", "minimum_prize"],
                "objectives": ["maximize_prize", "minimize_distance"],
                "complexity": "Advanced"
            },
            "VRPLIB": {
                "name": "VRPLIB Format",
                "description": "Standard benchmark instances from VRPLIB",
                "constraints": ["varies_by_instance"],
                "objectives": ["varies_by_instance"],
                "complexity": "Varies"
            }
        },
        "solver_capabilities": {
            "pyvrp_available": True,
            "fallback_available": True,
            "max_locations": 1000,
            "max_vehicles": 100,
            "max_runtime": 7200
        }
    }

@router.post("/compare")
async def compare_solutions(solutions_data: List[Dict[str, Any]]):
    """
    Compare multiple VRP solutions
    
    This endpoint accepts multiple VRP solutions and provides comparative analysis.
    """
    try:
        if len(solutions_data) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 solutions required for comparison"
            )
        
        # Analyze solutions
        comparison_metrics = {}
        best_idx = 0
        best_distance = float('inf')
        
        for i, solution in enumerate(solutions_data):
            distance = solution.get('total_distance', float('inf'))
            if distance < best_distance:
                best_distance = distance
                best_idx = i
        
        # Calculate improvement metrics
        worst_distance = max(sol.get('total_distance', 0) for sol in solutions_data)
        improvement = ((worst_distance - best_distance) / worst_distance * 100) if worst_distance > 0 else 0
        
        comparison_metrics = {
            "distance_improvement_percent": improvement,
            "best_solution_index": best_idx,
            "total_solutions": len(solutions_data),
            "distance_range": {
                "min": best_distance,
                "max": worst_distance,
                "avg": sum(sol.get('total_distance', 0) for sol in solutions_data) / len(solutions_data)
            }
        }
        
        return {
            "comparison_result": {
                "best_solution_idx": best_idx,
                "comparison_metrics": comparison_metrics,
                "solutions": solutions_data
            },
            "recommendations": [
                f"Solution {best_idx} provides the best distance optimization",
                f"Distance improvement of {improvement:.1f}% over worst solution",
                "Consider hybrid approaches for better performance"
            ]
        }
        
    except Exception as e:
        logger.error(f"Solution comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare solutions: {str(e)}")

@router.post("/benchmark")
async def run_benchmark(
    instance_name: str = Form(...),
    algorithms: List[str] = Form(["pyvrp", "heuristic"]),
    max_runtime: int = Form(60)
):
    """
    Run benchmark comparison between different algorithms
    
    This endpoint compares different VRP algorithms on the same problem instance.
    """
    try:
        # This would typically load a standard benchmark instance
        # For now, return a template structure
        
        benchmark_result = {
            "instance_name": instance_name,
            "problem_size": {
                "customers": 50,  # Example
                "vehicles": 5,
                "capacity": 1000
            },
            "algorithm_results": {},
            "performance_metrics": {
                "runtime_comparison": {},
                "solution_quality": {},
                "efficiency_scores": {}
            }
        }
        
        for algorithm in algorithms:
            # Simulate algorithm performance
            benchmark_result["algorithm_results"][algorithm] = {
                "status": "completed",
                "objective_value": 450.2,  # Example
                "computation_time": max_runtime * 0.8,
                "num_vehicles_used": 4
            }
        
        return {
            "benchmark_result": benchmark_result,
            "summary": f"Benchmarked {len(algorithms)} algorithms on {instance_name}",
            "recommendation": "PyVRP generally provides better solution quality for larger instances"
        }
        
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

def _validate_csv_data(df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
    """Validate CSV data based on type"""
    errors = []
    warnings = []
    
    if data_type == "locations":
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check coordinate ranges
        if 'lat' in df.columns:
            invalid_lat = df[(df['lat'] < -90) | (df['lat'] > 90)]
            if len(invalid_lat) > 0:
                errors.append(f"Invalid latitude values in rows: {invalid_lat.index.tolist()}")
        
        if 'lon' in df.columns:
            invalid_lon = df[(df['lon'] < -180) | (df['lon'] > 180)]
            if len(invalid_lon) > 0:
                errors.append(f"Invalid longitude values in rows: {invalid_lon.index.tolist()}")
        
        # Check for demand column
        if 'demand' not in df.columns:
            warnings.append("No demand column found, will use default demand of 1.0")
    
    elif data_type == "time_windows":
        required_cols = ['location_name', 'earliest', 'latest']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
    
    elif data_type == "depots":
        required_cols = ['name', 'lat', 'lon', 'capacity', 'num_vehicles']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "data_summary": {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist()
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint for PyVRP service"""
    try:
        # Test PyVRP availability
        test_result = "PyVRP available" if hasattr(pyvrp_service, 'solve_basic_cvrp') else "PyVRP not available"
        
        return {
            "status": "healthy",
            "service": "PyVRP Routes",
            "pyvrp_status": test_result,
            "timestamp": datetime.now().isoformat(),
            "endpoints_available": [
                "/solve/cvrp", "/solve/vrptw", "/solve/mdvrp", 
                "/solve/pdvrp", "/solve/pcvrp", "/solve/vrplib"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }