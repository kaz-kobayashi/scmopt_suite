from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import json
import tempfile
import os
import uuid
from collections import defaultdict
import math

from app.services.scrm_service import SCRMService
from app.models.scrm import (
    SCRMDataGenerationRequest,
    SCRMDataGenerationResult,
    SCRMUploadRequest,
    SCRMAnalysisRequest,
    SCRMAnalysisResult,
    SCRMVisualizationRequest,
    SCRMVisualizationResult,
    VisualizationOptions,
    CriticalNode,
    ServiceInfo,
    FileDownloadInfo,
    BOMEdgeData,
    PlantTransportData,
    PlantProductData,
    PlantCapacityData,
    SCRMDataGenerationOptions
)

router = APIRouter()

# Initialize SCRM service - will be created on first use
scrm_service = None

# In-memory storage for session data (in production, use Redis or database)
session_data_store = {}

def clean_for_json(data):
    """
    Clean data for JSON serialization by replacing NaN and Inf values
    """
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return clean_for_json(data.tolist())
    elif isinstance(data, (float, np.floating)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif pd.isna(data):
        return None
    else:
        return data

@router.get("/test")
async def test_endpoint():
    """
    Simple test endpoint
    """
    return {"message": "SCRM route test successful", "status": "ok"}

@router.get("/generate-sample-data")
async def generate_sample_data():
    """
    Generate sample SCRM data for frontend testing - GET version for testing
    """
    try:
        # Return simplified mock data to test the frontend connection
        return {
            "bom_data": [
                {"child": "Component_A", "parent": "Product_A", "units": 2},
                {"child": "Component_B", "parent": "Product_A", "units": 1},
                {"child": "Component_A", "parent": "Product_B", "units": 1},
                {"child": "Raw_Material", "parent": "Component_A", "units": 3},
                {"child": "Raw_Material", "parent": "Component_B", "units": 2}
            ],
            "plant_data": [
                {"name": "Plant_1", "ub": 1000},
                {"name": "Plant_2", "ub": 800},
                {"name": "Plant_3", "ub": 1200}
            ],
            "transport_data": [
                {"from_node": "Plant_1", "to_node": "Plant_2"},
                {"from_node": "Plant_2", "to_node": "Plant_3"},
                {"from_node": "Plant_1", "to_node": "Plant_3"}
            ],
            "plant_product_data": [
                {"plnt": "Plant_1", "prod": "Product_A", "demand": 100, "ub": 150, "pipeline": 20},
                {"plnt": "Plant_2", "prod": "Product_B", "demand": 80, "ub": 120, "pipeline": 15},
                {"plnt": "Plant_3", "prod": "Component_A", "demand": 200, "ub": 250, "pipeline": 30}
            ]
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"SCRM Error: {error_details}")
        return {"error": str(e), "traceback": error_details}

@router.post("/generate-sample-data")
async def generate_sample_data_post(request: dict = None):
    """
    Generate sample SCRM data for frontend testing
    """
    # Log the request for debugging
    print(f"SCRM generate-sample-data called with request: {request}")
    
    try:
        # Initialize SCRM service
        global scrm_service
        if scrm_service is None:
            scrm_service = SCRMService()
        
        # Extract parameters from request
        benchmark_id = request.get('benchmark_id', '01') if request else '01'
        n_plants = request.get('n_plants', 3) if request else 3
        n_flex = request.get('n_flex', 2) if request else 2
        seed = request.get('seed', 1) if request else 1
        
        # Generate test data using the SCRM service
        data = scrm_service.generate_test_data(
            benchmark_id=benchmark_id,
            n_plnts=n_plants,
            n_flex=n_flex,
            seed=seed
        )
        
        # Convert DataFrames to frontend-compatible format with NaN cleaning
        result = {
            "bom_data": clean_for_json(data["bom_df"].to_dict('records')) if 'bom_df' in data else [],
            "plant_data": clean_for_json(data["plnt_df"].to_dict('records')) if 'plnt_df' in data else [],
            "transport_data": clean_for_json(data["trans_df"].to_dict('records')) if 'trans_df' in data else [],
            "plant_product_data": clean_for_json(data["plnt_prod_df"].to_dict('records')) if 'plnt_prod_df' in data else []
        }
        
        # Store the generated data in session for later use
        session_id = str(uuid.uuid4())
        session_data_store[session_id] = data
        result["session_id"] = session_id
        
        return clean_for_json(result)
        
        # Original code commented out for now to avoid NaN serialization errors
        # # Generate actual SCRM test data
        # data = scrm_service.generate_test_data(
        #     benchmark_id=benchmark_id,
        #     n_plnts=n_plants,
        #     n_flex=n_flex,
        #     seed=seed
        # )
        
        # # Convert DataFrames to frontend-compatible format
        # return {
        #     "bom_data": data["bom_df"].to_dict('records') if 'bom_df' in data else [],
        #     "plant_data": data["plnt_df"].to_dict('records') if 'plnt_df' in data else [],
        #     "transport_data": data["trans_df"].to_dict('records') if 'trans_df' in data else [],
        #     "plant_product_data": data["plnt_prod_df"].to_dict('records') if 'plnt_prod_df' in data else [],
        #     "metadata": {
        #         "benchmark_id": benchmark_id,
        #         "total_demand": data.get("total_demand", 0),
        #         "generation_parameters": {
        #             "n_plants": n_plants,
        #             "n_flex": n_flex,
        #             "seed": seed
        #         }
        #     }
        # }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"SCRM Error: {error_details}")
        
        # Return fallback mock data
        sample_data = {
            "bom_data": [
                {"child": "Product_A", "parent": "Raw_Material_1", "units": 2},
                {"child": "Product_A", "parent": "Raw_Material_2", "units": 1},
                {"child": "Product_B", "parent": "Raw_Material_1", "units": 1},
                {"child": "Product_B", "parent": "Product_A", "units": 1},
                {"child": "Product_C", "parent": "Product_B", "units": 2}
            ],
            "plant_data": [
                {"name": 0, "ub": 100},
                {"name": 1, "ub": 150},
                {"name": 2, "ub": 120}
            ],
            "transport_data": [
                {"from_node": 0, "to_node": 1, "kind": "plnt-plnt"},
                {"from_node": 1, "to_node": 2, "kind": "plnt-plnt"},
                {"from_node": 0, "to_node": 2, "kind": "plnt-plnt"}
            ],
            "plant_product_data": [
                {"plnt": 0, "prod": "Product_A", "ub": 100, "pipeline": 50, "demand": 120},
                {"plnt": 0, "prod": "Product_B", "ub": 100, "pipeline": 100, "demand": None},
                {"plnt": 1, "prod": "Component_A", "ub": 100, "pipeline": 150, "demand": None},
                {"plnt": 1, "prod": "Component_B", "ub": 100, "pipeline": 200, "demand": None},
                {"plnt": 2, "prod": "Raw_Material", "ub": 100, "pipeline": 250, "demand": None}
            ],
            "metadata": {
                "benchmark_id": benchmark_id,
                "error": str(e),
                "fallback_data": True
            }
        }
        return sample_data

@router.post("/analyze")
async def analyze_scrm(request: dict):
    """
    Run SCRM analysis with the provided data and parameters
    """
    try:
        # Initialize SCRM service if not already done
        global scrm_service
        if scrm_service is None:
            scrm_service = SCRMService()
            
        # Extract data from request
        data = request.get('data', {})
        parameters = request.get('parameters', {})
        
        # Extract model type and analysis parameters
        model_type = parameters.get('optimization_model', 'tts')
        time_horizon = parameters.get('time_horizon', 10)
        risk_level = parameters.get('risk_level', 0.95)
        solver = parameters.get('solver', 'pulp')
        
        # Convert frontend data to backend format
        backend_data = {}
        
        if 'bom_data' in data and data['bom_data']:
            backend_data['bom_df'] = pd.DataFrame(data['bom_data'])
        
        if 'plant_data' in data and data['plant_data']:
            backend_data['plnt_df'] = pd.DataFrame(data['plant_data'])
            
        if 'transport_data' in data and data['transport_data']:
            backend_data['trans_df'] = pd.DataFrame(data['transport_data'])
            
        if 'plant_product_data' in data and data['plant_product_data']:
            backend_data['plnt_prod_df'] = pd.DataFrame(data['plant_product_data'])
        
        # Set analysis parameters for advanced models
        analysis_params = {
            "max_disruptions": 2,
            "inventory_cost": 1.0,
            "backorder_cost": 1000.0,
            "time_horizon": time_horizon,
            "risk_level": risk_level
        }
        
        # Run SCRM analysis with the specified model
        if backend_data and len(backend_data) > 0:
            analysis_results = scrm_service.run_full_analysis(
                backend_data, 
                model_type=model_type, 
                analysis_params=analysis_params
            )
            
            # Format results for frontend based on model type
            if model_type == "tts":
                return {
                    "status": "success",
                    "model_type": "tts",
                    "survival_times": analysis_results.get("survival_time", []),
                    "critical_nodes": [
                        {
                            "id": f"{node[0]}_{node[1]}" if isinstance(node, tuple) else str(node),
                            "survival_time": time,
                            "risk": "CRITICAL" if time == 0 else "HIGH" if time < 5 else "MEDIUM"
                        }
                        for node, time in analysis_results.get("critical_nodes", [])
                    ],
                    "analysis_summary": {
                        "total_nodes": analysis_results.get("total_nodes", 0),
                        "average_survival_time": analysis_results.get("average_survival_time", 0),
                        "min_survival_time": analysis_results.get("min_survival_time", 0),
                        "max_survival_time": analysis_results.get("max_survival_time", 0)
                    }
                }
                
            elif model_type == "expected":
                ev_result = analysis_results.get("expected_value_result", {})
                return {
                    "status": "success",
                    "model_type": "expected_value",
                    "objective_value": analysis_results.get("objective_value", 0),
                    "total_inventory_cost": analysis_results.get("total_inventory_cost", 0),
                    "expected_backorder_cost": analysis_results.get("expected_backorder_cost", 0),
                    "scenarios_analyzed": analysis_results.get("scenarios_analyzed", 0),
                    "inventory_solution": ev_result.get("inventory_solution", {}),
                    "analysis_summary": {
                        "total_nodes": analysis_results.get("total_nodes", 0),
                        "optimization_status": ev_result.get("status", "unknown")
                    }
                }
                
            elif model_type == "cvar":
                cvar_result = analysis_results.get("cvar_result", {})
                return {
                    "status": "success", 
                    "model_type": "cvar",
                    "cvar_value": analysis_results.get("cvar_value", 0),
                    "var_value": analysis_results.get("var_value", 0),
                    "expected_cost": analysis_results.get("expected_cost", 0),
                    "beta": analysis_results.get("beta", risk_level),
                    "scenarios_analyzed": analysis_results.get("scenarios_analyzed", 0),
                    "inventory_solution": cvar_result.get("inventory_solution", {}),
                    "analysis_summary": {
                        "total_nodes": analysis_results.get("total_nodes", 0),
                        "optimization_status": cvar_result.get("status", "unknown")
                    }
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        else:
            # Return mock results if no data provided (TTS model)
            return {
                "status": "success",
                "model_type": "tts",
                "survival_times": [0, 2.5, 5.0, 12.5, 18.0, 25.0, 30.0, 40.0, 50.0],
                "critical_nodes": [
                    {"id": "Plant_1_Product_A", "survival_time": 0, "risk": "CRITICAL"},
                    {"id": "Plant_2_Product_B", "survival_time": 2.5, "risk": "HIGH"},
                    {"id": "Plant_3_Product_C", "survival_time": 5.0, "risk": "MEDIUM"}
                ],
                "analysis_summary": {
                    "total_nodes": 20,
                    "average_survival_time": 12.5,
                    "min_survival_time": 0,
                    "max_survival_time": 50
                }
            }
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"SCRM Analysis Error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-tts") 
async def analyze_tts(request: dict):
    """
    Run Time-to-Survival (TTS) analysis using PuLP solver
    """
    try:
        global scrm_service
        if scrm_service is None:
            scrm_service = SCRMService()
        
        # Get session data or use provided data
        session_id = request.get('session_id')
        if session_id and session_id in session_data_store:
            data = session_data_store[session_id]
        else:
            # If no session, try to reconstruct from request data
            if 'bom_data' in request:
                data = {
                    'bom_df': pd.DataFrame(request['bom_data']),
                    'plnt_df': pd.DataFrame(request['plant_data']),
                    'plnt_prod_df': pd.DataFrame(request['plant_product_data']),
                    'trans_df': pd.DataFrame(request['transport_data'])
                }
            else:
                # Generate default data if nothing provided
                data = scrm_service.generate_test_data()
        
        # Prepare data structures from DataFrames
        prepare_result = scrm_service.prepare_from_dataframes(
            data['bom_df'], data['plnt_df'], 
            data['plnt_prod_df'], data['trans_df']
        )
        
        Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = prepare_result
        
        # Run TTS analysis using PuLP solver
        print("Running TTS analysis with PuLP solver...")
        survival_time = scrm_service.solve_scrm(
            Demand, UB, Capacity, Pipeline, R, Product, ProdGraph, BOM
        )
        
        # Create node list with survival times
        node_list = list(ProdGraph.nodes())
        critical_nodes = []
        for i, st in enumerate(survival_time):
            if i < len(node_list):
                critical_nodes.append((str(node_list[i]), float(st)))
        
        # Sort by survival time to get critical nodes
        critical_nodes.sort(key=lambda x: x[1])
        
        # Calculate statistics
        avg_survival = np.mean(survival_time) if survival_time else 0
        min_survival = min(survival_time) if survival_time else 0
        max_survival = max(survival_time) if survival_time else 0
        
        # Create network visualization data
        nodes = []
        for i, (node, st) in enumerate(zip(node_list[:len(survival_time)], survival_time)):
            nodes.append({
                "id": f"node_{i}",
                "name": str(node),
                "survival_time": float(st)
            })
        
        edges = []
        for i, (u, v) in enumerate(list(ProdGraph.edges())[:20]):  # Limit edges for visualization
            u_idx = node_list.index(u) if u in node_list else -1
            v_idx = node_list.index(v) if v in node_list else -1
            if u_idx >= 0 and v_idx >= 0:
                edges.append({
                    "source": f"node_{u_idx}",
                    "target": f"node_{v_idx}", 
                    "weight": R.get((u, v), 1)
                })
        
        result = {
            "status": "success",
            "model_type": "tts",
            "total_nodes": len(node_list),
            "survival_time": [float(st) for st in survival_time],
            "critical_nodes": critical_nodes[:10],  # Top 10 critical nodes
            "average_survival_time": float(avg_survival),
            "min_survival_time": float(min_survival),
            "max_survival_time": float(max_survival),
            "network": {
                "nodes": nodes,
                "edges": edges
            }
        }
        
        return clean_for_json(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS analysis failed: {str(e)}")


@router.post("/analyze-expected-value")
async def analyze_expected_value(request: dict):
    """
    Run Expected Value Minimization analysis using PuLP solver
    Exact implementation from notebook
    """
    try:
        global scrm_service
        if scrm_service is None:
            scrm_service = SCRMService()
        
        # Get session data or use provided data
        session_id = request.get('session_id')
        if session_id and session_id in session_data_store:
            data = session_data_store[session_id]
        else:
            # Generate default data if nothing provided
            data = scrm_service.generate_test_data()
        
        # Prepare data structures from DataFrames
        prepare_result = scrm_service.prepare_from_dataframes(
            data['bom_df'], data['plnt_df'], 
            data['plnt_prod_df'], data['trans_df']
        )
        
        Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = prepare_result
        
        # Create disruption scenarios
        prob = {(0,): 0.7, (1,): 0.2, (2,): 0.1}  # Plant disruption probabilities
        TTR = {0: 0, 1: 10, 2: 15}  # Time-to-recovery for each plant
        h = {node: 1.0 for node in ProdGraph.nodes()}  # Inventory holding costs
        b = {node: 1000.0 for node in ProdGraph.nodes()}  # Backorder costs
        
        # Run Expected Value analysis using PuLP solver
        print("Running Expected Value analysis with PuLP solver...")
        result = scrm_service.solve_expected_value_minimization(
            Demand, UB, Capacity, Pipeline, R, Product, ProdGraph, BOM,
            prob, TTR, h, b
        )
        
        # Format response
        response = {
            "status": result["status"],
            "model_type": "expected_value",
            "total_nodes": len(ProdGraph.nodes()),
            "scenarios_analyzed": result.get("scenarios_analyzed", 0),
            "total_inventory_cost": result.get("total_inventory_cost", 0),
            "expected_backorder_cost": result.get("expected_backorder_cost", 0),
            "objective_value": result.get("objective_value", 0),
            "expected_value_result": {
                "inventory_solution": result.get("inventory_solution", {}),
                "optimization_status": result["status"]
            }
        }
        
        return clean_for_json(response)
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Expected Value Analysis Error: {tb}")
        raise HTTPException(status_code=500, detail=f"Expected value analysis failed: {str(e)}\nTraceback: {tb}")


@router.post("/analyze-cvar")
async def analyze_cvar(request: dict):
    """
    Run CVaR (Conditional Value at Risk) analysis using PuLP solver
    Exact implementation from notebook
    """
    try:
        global scrm_service
        if scrm_service is None:
            scrm_service = SCRMService()
        
        # Get session data or use provided data
        session_id = request.get('session_id')
        if session_id and session_id in session_data_store:
            data = session_data_store[session_id]
        else:
            # Generate default data if nothing provided
            data = scrm_service.generate_test_data()
        
        # Prepare data structures from DataFrames
        prepare_result = scrm_service.prepare_from_dataframes(
            data['bom_df'], data['plnt_df'], 
            data['plnt_prod_df'], data['trans_df']
        )
        
        Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = prepare_result
        
        # Create disruption scenarios
        prob = {(0,): 0.7, (1,): 0.2, (2,): 0.1}  # Plant disruption probabilities
        TTR = {0: 0, 1: 10, 2: 15}  # Time-to-recovery for each plant
        h = {node: 1.0 for node in ProdGraph.nodes()}  # Inventory holding costs
        b = {node: 1000.0 for node in ProdGraph.nodes()}  # Backorder costs
        beta = request.get('beta', 0.95)  # Risk level for CVaR
        
        # Run CVaR analysis using PuLP solver
        print("Running CVaR analysis with PuLP solver...")
        result = scrm_service.solve_cvar_model(
            Demand, UB, Capacity, Pipeline, R, Product, ProdGraph, BOM,
            prob, TTR, h, b, beta
        )
        
        # Format response
        response = {
            "status": result["status"],
            "model_type": "cvar",
            "total_nodes": len(ProdGraph.nodes()),
            "scenarios_analyzed": result.get("scenarios_analyzed", 0),
            "beta": result.get("beta", beta),
            "cvar_value": result.get("cvar_value", 0),
            "var_value": result.get("var_value", 0),
            "expected_cost": result.get("expected_cost", 0),
            "cvar_result": {
                "inventory_solution": result.get("inventory_solution", {}),
                "scenario_costs": result.get("scenario_costs", {}),
                "optimization_status": result["status"]
            }
        }
        
        return clean_for_json(response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CVaR analysis failed: {str(e)}")

@router.post("/export-results")
async def export_scrm_results(request: dict):
    """
    Export SCRM analysis results to Excel format
    """
    try:
        # This would normally create an Excel file with the results
        # For now, return a simple response
        return {"message": "Export functionality not yet implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/generate", response_model=SCRMDataGenerationResult)
async def generate_scrm_data(request: SCRMDataGenerationRequest):
    """
    Generate SCRM test data from Willems benchmark problems
    Exact implementation from 09scrm.ipynb notebook
    """
    try:
        # Generate test data using SCRM service
        data = scrm_service.generate_test_data(
            benchmark_id=request.options.benchmark_id,
            n_plnts=request.options.n_plnts,
            n_flex=request.options.n_flex,
            seed=request.options.seed
        )
        
        # Store data in session for later use
        session_id = f"scrm_session_{uuid.uuid4().hex[:8]}"
        session_data_store[session_id] = data
        
        # Create result summary
        result = SCRMDataGenerationResult(
            benchmark_id=data["benchmark_id"],
            total_demand=data["total_demand"],
            total_plants=len(data["plnt_df"]),
            total_products=len(data["bom_df"]["child"].unique()),
            total_nodes=len(data["plnt_prod_df"]),
            bom_edges=len(data["bom_df"]),
            plant_edges=len(data["trans_df"]),
            generation_options=request.options
        )
        
        # Add session ID to response headers
        response = JSONResponse(content=result.dict())
        response.headers["X-Session-ID"] = session_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"データ生成エラー: {str(e)}")


@router.post("/upload")
async def upload_scrm_data(
    bom_file: UploadFile = File(..., description="BOM CSV file"),
    transport_file: UploadFile = File(..., description="Plant transport CSV file"),
    plant_product_file: UploadFile = File(..., description="Plant-product CSV file"),
    plant_capacity_file: UploadFile = File(..., description="Plant capacity CSV file")
):
    """
    Upload CSV files for SCRM analysis
    """
    try:
        # Read CSV files
        bom_content = await bom_file.read()
        bom_df = pd.read_csv(io.StringIO(bom_content.decode('utf-8')))
        
        transport_content = await transport_file.read()
        trans_df = pd.read_csv(io.StringIO(transport_content.decode('utf-8')))
        
        plant_product_content = await plant_product_file.read()
        plnt_prod_df = pd.read_csv(io.StringIO(plant_product_content.decode('utf-8')))
        
        plant_capacity_content = await plant_capacity_file.read()
        plnt_df = pd.read_csv(io.StringIO(plant_capacity_content.decode('utf-8')))
        
        # Validate required columns
        required_bom_cols = ['child', 'parent', 'units']
        required_trans_cols = ['from_node', 'to_node', 'kind']
        required_plnt_prod_cols = ['plnt', 'prod', 'ub', 'pipeline', 'demand']
        required_plnt_cols = ['name', 'ub']
        
        # Check columns
        if not all(col in bom_df.columns for col in required_bom_cols):
            raise ValueError(f"BOM file missing required columns: {required_bom_cols}")
        if not all(col in trans_df.columns for col in required_trans_cols):
            raise ValueError(f"Transport file missing required columns: {required_trans_cols}")
        if not all(col in plnt_prod_df.columns for col in required_plnt_prod_cols):
            raise ValueError(f"Plant-product file missing required columns: {required_plnt_prod_cols}")
        if not all(col in plnt_df.columns for col in required_plnt_cols):
            raise ValueError(f"Plant capacity file missing required columns: {required_plnt_cols}")
        
        # Store data in session
        session_id = f"scrm_session_{uuid.uuid4().hex[:8]}"
        data = {
            "bom_df": bom_df,
            "trans_df": trans_df,
            "plnt_prod_df": plnt_prod_df,
            "plnt_df": plnt_df,
            "data_source": "uploaded"
        }
        session_data_store[session_id] = data
        
        return {
            "status": "success",
            "session_id": session_id,
            "summary": {
                "bom_records": len(bom_df),
                "transport_records": len(trans_df),
                "plant_product_records": len(plnt_prod_df),
                "plant_records": len(plnt_df)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファイルアップロードエラー: {str(e)}")


@router.post("/analyze", response_model=SCRMAnalysisResult)
async def analyze_scrm(request: SCRMAnalysisRequest):
    """
    Run SCRM analysis on generated or uploaded data
    """
    try:
        if request.data_source == "generated":
            if not request.generation_options:
                raise ValueError("Generation options required for generated data source")
            
            # Generate data
            data = scrm_service.generate_test_data(
                benchmark_id=request.generation_options.benchmark_id,
                n_plnts=request.generation_options.n_plnts,
                n_flex=request.generation_options.n_flex,
                seed=request.generation_options.seed
            )
        else:
            if not request.upload_data:
                raise ValueError("Upload data required for uploaded data source")
            
            # Convert Pydantic models to DataFrames
            bom_data = [item.dict() for item in request.upload_data.bom_data]
            bom_df = pd.DataFrame(bom_data)
            
            transport_data = [item.dict() for item in request.upload_data.transport_data]
            trans_df = pd.DataFrame(transport_data)
            
            plant_product_data = [item.dict() for item in request.upload_data.plant_product_data]
            plnt_prod_df = pd.DataFrame(plant_product_data)
            
            plant_capacity_data = [item.dict() for item in request.upload_data.plant_capacity_data]
            plnt_df = pd.DataFrame(plant_capacity_data)
            
            data = {
                "bom_df": bom_df,
                "trans_df": trans_df,
                "plnt_prod_df": plnt_prod_df,
                "plnt_df": plnt_df,
                "data_source": "uploaded"
            }
        
        # Run full SCRM analysis
        analysis_results = scrm_service.run_full_analysis(data)
        
        # Convert critical nodes to proper format
        critical_nodes = []
        for node, survival_time in analysis_results["critical_nodes"]:
            critical_nodes.append(CriticalNode(
                plant=node[0],
                product=node[1],
                survival_time=survival_time
            ))
        
        # Create analysis summary
        survival_times = analysis_results["survival_time"]
        zero_survival = sum(1 for t in survival_times if t == 0.0)
        high_risk = sum(1 for t in survival_times if 0.0 < t <= 1.0)
        resilient = sum(1 for t in survival_times if t > 1.0)
        
        analysis_summary = {
            "zero_survival_nodes": zero_survival,
            "high_risk_nodes": high_risk,
            "resilient_nodes": resilient,
            "risk_distribution": {
                "critical": f"{zero_survival / len(survival_times) * 100:.1f}%",
                "high_risk": f"{high_risk / len(survival_times) * 100:.1f}%",
                "resilient": f"{resilient / len(survival_times) * 100:.1f}%"
            }
        }
        
        # Store results for visualization
        session_id = f"scrm_analysis_{uuid.uuid4().hex[:8]}"
        session_data_store[session_id] = {
            **data,
            "analysis_results": analysis_results
        }
        
        result = SCRMAnalysisResult(
            status="success",
            total_nodes=analysis_results["total_nodes"],
            survival_time=survival_times,
            critical_nodes=critical_nodes,
            average_survival_time=analysis_results["average_survival_time"],
            min_survival_time=analysis_results["min_survival_time"],
            max_survival_time=analysis_results["max_survival_time"],
            analysis_summary=analysis_summary
        )
        
        response = JSONResponse(content=result.dict())
        response.headers["X-Analysis-Session-ID"] = session_id
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"分析エラー: {str(e)}")


@router.post("/visualize/{graph_type}", response_model=SCRMVisualizationResult)
async def visualize_scrm(
    graph_type: str,
    session_id: str = Form(...),
    title: str = Form(""),
    node_size: int = Form(30),
    node_color: str = Form("Yellow"),
    width: int = Form(800),
    height: int = Form(600)
):
    """
    Generate visualization for SCRM analysis
    """
    try:
        # Validate graph type
        allowed_types = ["plant_graph", "bom_graph", "production_graph", "risk_analysis"]
        if graph_type not in allowed_types:
            raise ValueError(f"Graph type must be one of: {allowed_types}")
        
        # Get session data
        if session_id not in session_data_store:
            raise ValueError(f"Session ID not found: {session_id}")
        
        data = session_data_store[session_id]
        
        # Generate appropriate visualization
        if graph_type == "risk_analysis":
            if "analysis_results" not in data:
                raise ValueError("Analysis results not found. Run analysis first.")
            
            # Reconstruct data structures for risk visualization
            result = scrm_service.prepare_from_dataframes(
                data["bom_df"], data["plnt_df"], 
                data["plnt_prod_df"], data["trans_df"]
            )
            Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = result
            
            # Generate risk visualization
            analysis_results = data["analysis_results"]
            fig = scrm_service.draw_scrm(
                ProdGraph, analysis_results["survival_time"], Pipeline, UB, pos3
            )
            
        elif graph_type == "plant_graph":
            result = scrm_service.prepare_from_dataframes(
                data["bom_df"], data["plnt_df"], 
                data["plnt_prod_df"], data["trans_df"]
            )
            Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = result
            
            fig = scrm_service.draw_graph(G, pos2, title or "Plant Graph", node_size, "Red")
            
        elif graph_type == "bom_graph":
            result = scrm_service.prepare_from_dataframes(
                data["bom_df"], data["plnt_df"], 
                data["plnt_prod_df"], data["trans_df"]
            )
            Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = result
            
            fig = scrm_service.draw_graph(BOM, pos, title or "BOM Graph", 20, "Green")
            
        elif graph_type == "production_graph":
            result = scrm_service.prepare_from_dataframes(
                data["bom_df"], data["plnt_df"], 
                data["plnt_prod_df"], data["trans_df"]
            )
            Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = result
            
            fig = scrm_service.draw_graph(ProdGraph, pos3, title or "Production Graph", node_size, node_color)
        
        # Update figure layout
        fig.update_layout(
            width=width,
            height=height,
            title=title or f"SCRM {graph_type.replace('_', ' ').title()}"
        )
        
        # Create summary statistics
        summary = {
            "graph_type": graph_type,
            "total_nodes": len(fig.data[0]['x']) if fig.data else 0,
            "total_edges": len(fig.data[1]['x']) // 3 if len(fig.data) > 1 and fig.data[1]['x'] else 0
        }
        
        if graph_type == "risk_analysis" and "analysis_results" in data:
            summary["critical_nodes_count"] = len(data["analysis_results"]["critical_nodes"])
            summary["average_survival_time"] = data["analysis_results"]["average_survival_time"]
        
        return SCRMVisualizationResult(
            graph_type=graph_type,
            plotly_json=fig.to_dict(),
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"可視化エラー: {str(e)}")


@router.get("/download/{session_id}/{file_name}")
async def download_scrm_file(session_id: str, file_name: str):
    """
    Download SCRM analysis results or data files
    """
    try:
        if session_id not in session_data_store:
            raise ValueError(f"Session ID not found: {session_id}")
        
        data = session_data_store[session_id]
        
        # Determine file type and data type from filename
        if file_name.endswith('.csv'):
            file_type = 'csv'
            media_type = 'text/csv'
        elif file_name.endswith('.xlsx'):
            file_type = 'xlsx'
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif file_name.endswith('.json'):
            file_type = 'json'
            media_type = 'application/json'
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            if 'bom' in file_name.lower():
                if file_type == 'csv':
                    data["bom_df"].to_csv(temp_file.name, index=False)
                elif file_type == 'xlsx':
                    data["bom_df"].to_excel(temp_file.name, index=False, sheet_name="BOM")
                    
            elif 'transport' in file_name.lower():
                if file_type == 'csv':
                    data["trans_df"].to_csv(temp_file.name, index=False)
                elif file_type == 'xlsx':
                    data["trans_df"].to_excel(temp_file.name, index=False, sheet_name="Transport")
                    
            elif 'plant_product' in file_name.lower():
                if file_type == 'csv':
                    data["plnt_prod_df"].to_csv(temp_file.name, index=False)
                elif file_type == 'xlsx':
                    data["plnt_prod_df"].to_excel(temp_file.name, index=False, sheet_name="PlantProducts")
                    
            elif 'plant_capacity' in file_name.lower() or 'plants' in file_name.lower():
                if file_type == 'csv':
                    data["plnt_df"].to_csv(temp_file.name, index=False)
                elif file_type == 'xlsx':
                    data["plnt_df"].to_excel(temp_file.name, index=False, sheet_name="Plants")
                    
            elif 'analysis_results' in file_name.lower() and "analysis_results" in data:
                analysis_results = data["analysis_results"]
                
                # Create results DataFrame
                node_list = list(range(len(analysis_results["survival_time"])))
                results_df = pd.DataFrame({
                    'node_id': node_list,
                    'survival_time': analysis_results["survival_time"],
                    'is_critical': [t == 0.0 for t in analysis_results["survival_time"]],
                    'risk_level': ['Critical' if t == 0.0 else 'High Risk' if t <= 1.0 else 'Resilient' 
                                  for t in analysis_results["survival_time"]]
                })
                
                if file_type == 'csv':
                    results_df.to_csv(temp_file.name, index=False)
                elif file_type == 'xlsx':
                    results_df.to_excel(temp_file.name, index=False, sheet_name="AnalysisResults")
                elif file_type == 'json':
                    with open(temp_file.name, 'w') as f:
                        json.dump(analysis_results, f, indent=2)
            
            else:
                raise ValueError(f"Unknown file type: {file_name}")
            
            return FileResponse(
                path=temp_file.name,
                filename=file_name,
                media_type=media_type
            )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファイルダウンロードエラー: {str(e)}")


@router.get("/service-info", response_model=ServiceInfo)
async def get_scrm_service_info():
    """
    Get SCRM service information
    """
    return ServiceInfo(
        service_name="Supply Chain Risk Management (SCRM) Service",
        version="1.0.0",
        description="Complete SCRM analysis system from 09scrm.ipynb implementing MERIODAS framework (MEta RIsk Oriented Disruption Analysis System)",
        features=[
            "Time-to-Survival (TTS) analysis",
            "Critical node identification", 
            "Risk visualization with Plotly",
            "Willems benchmark data generation",
            "CSV data import/export",
            "Multi-scenario disruption analysis",
            "Interactive graph visualization",
            "Supply chain resilience assessment",
            "MIT Simchi-Levi optimization model",
            "Multi-echelon risk propagation"
        ],
        supported_benchmarks=[f"{i:02d}" for i in range(1, 39)]  # 01-38
    )


@router.get("/benchmark-list")
async def get_benchmark_list():
    """
    Get list of available Willems benchmark problems
    """
    return {
        "available_benchmarks": [f"{i:02d}" for i in range(1, 39)],
        "total_count": 38,
        "description": "Willems benchmark problems for supply chain analysis",
        "source": "Willems, S.P. (2000). Data set: Multi-echelon inventory systems"
    }


@router.post("/generate-template")
async def generate_csv_template(
    template_type: str = Form(...),
    benchmark_id: str = Form("01")
):
    """
    Generate CSV template files for SCRM data upload
    """
    try:
        if template_type not in ["bom", "transport", "plant_product", "plant_capacity", "all"]:
            raise ValueError(f"Template type must be one of: bom, transport, plant_product, plant_capacity, all")
        
        # Generate sample data
        data = scrm_service.generate_test_data(benchmark_id=benchmark_id, n_plnts=2, n_flex=1, seed=1)
        
        if template_type == "bom":
            return Response(
                content=data["bom_df"].to_csv(index=False),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=bom_template.csv"}
            )
        elif template_type == "transport":
            return Response(
                content=data["trans_df"].to_csv(index=False),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=transport_template.csv"}
            )
        elif template_type == "plant_product":
            return Response(
                content=data["plnt_prod_df"].to_csv(index=False),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=plant_product_template.csv"}
            )
        elif template_type == "plant_capacity":
            return Response(
                content=data["plnt_df"].to_csv(index=False),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=plant_capacity_template.csv"}
            )
        elif template_type == "all":
            # Create ZIP file with all templates
            import zipfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                with zipfile.ZipFile(temp_file.name, 'w') as zip_file:
                    zip_file.writestr("bom_template.csv", data["bom_df"].to_csv(index=False))
                    zip_file.writestr("transport_template.csv", data["trans_df"].to_csv(index=False))
                    zip_file.writestr("plant_product_template.csv", data["plnt_prod_df"].to_csv(index=False))
                    zip_file.writestr("plant_capacity_template.csv", data["plnt_df"].to_csv(index=False))
                
                return FileResponse(
                    path=temp_file.name,
                    filename="scrm_templates.zip",
                    media_type="application/zip"
                )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"テンプレート生成エラー: {str(e)}")


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session data from memory
    """
    if session_id in session_data_store:
        del session_data_store[session_id]
        return {"status": "success", "message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@router.get("/sessions")
async def list_sessions():
    """
    List active sessions (for debugging/monitoring)
    """
    sessions = []
    for session_id, data in session_data_store.items():
        session_info = {
            "session_id": session_id,
            "data_source": data.get("data_source", "generated"),
            "has_analysis": "analysis_results" in data,
            "total_nodes": len(data.get("plnt_prod_df", [])) if "plnt_prod_df" in data else 0
        }
        sessions.append(session_info)
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }