from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
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

from app.services.snd_service import SNDService
from app.models.snd import (
    SNDOptimizationRequest,
    SNDOptimizationResult,
    SNDVisualizationRequest,
    SNDVisualizationResult,
    SNDDataUploadResult,
    SNDExportRequest,
    PathResult,
    VehicleResult,
    CostBreakdown,
    NetworkNode,
    NetworkEdge,
    SolverError
)

router = APIRouter()

# Initialize SND service
snd_service = None

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
    Simple test endpoint for SND routes
    """
    return {"message": "SND route test successful", "status": "ok"}

@router.post("/upload-data")
async def upload_snd_data(
    dc_file: UploadFile = File(..., description="Distribution Centers CSV file"),
    od_file: UploadFile = File(..., description="Origin-Destination demand CSV file")
):
    """
    Upload DC and OD data files for SND analysis
    """
    try:
        # Read DC file
        dc_content = await dc_file.read()
        dc_df = pd.read_csv(io.StringIO(dc_content.decode('utf-8')))
        
        # Read OD file
        od_content = await od_file.read()
        od_df = pd.read_csv(io.StringIO(od_content.decode('utf-8')), index_col=0)
        
        # Validate DC data structure
        required_dc_columns = ['name', 'lat', 'lon', 'ub', 'vc']
        missing_dc_cols = [col for col in required_dc_columns if col not in dc_df.columns]
        if missing_dc_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"DC file missing required columns: {missing_dc_cols}"
            )
        
        # Store data in session
        session_id = str(uuid.uuid4())
        session_data_store[session_id] = {
            "dc_df": dc_df,
            "od_df": od_df,
            "upload_time": pd.Timestamp.now().isoformat()
        }
        
        return SNDDataUploadResult(
            session_id=session_id,
            dc_count=len(dc_df),
            od_pairs=len(od_df) * len(od_df.columns),
            message="Data uploaded successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")

@router.get("/sample-data")
async def generate_sample_data():
    """
    Generate sample SND data for testing
    """
    try:
        # Load sample data from nbs/data directory
        dc_df = pd.read_csv("/Users/kazuhiro/Documents/2509/scmopt_suite/nbs/data/DC.csv", index_col=0)
        od_df = pd.read_csv("/Users/kazuhiro/Documents/2509/scmopt_suite/nbs/data/od.csv", index_col=0)
        
        # Take first 10 DCs for performance
        n = 10
        dc_df = dc_df.iloc[:n, :]
        od_df = od_df.iloc[:n, :n]
        
        # Store in session
        session_id = str(uuid.uuid4())
        session_data_store[session_id] = {
            "dc_df": dc_df,
            "od_df": od_df,
            "upload_time": pd.Timestamp.now().isoformat()
        }
        
        return clean_for_json({
            "session_id": session_id,
            "dc_data": dc_df.to_dict('records'),
            "od_data": od_df.to_dict(),
            "message": f"Sample data generated with {len(dc_df)} DCs"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample data generation failed: {str(e)}")

@router.post("/optimize")
async def optimize_snd(request: SNDOptimizationRequest):
    """
    Run SND optimization with given parameters
    """
    try:
        global snd_service
        if snd_service is None:
            snd_service = SNDService()
        
        # Get data from request or session
        if request.dc_data and request.od_data:
            # Use data from request
            dc_df = pd.DataFrame(request.dc_data)
            # Handle OD data format conversion
            if isinstance(request.od_data, dict):
                od_df = pd.DataFrame(request.od_data)
            else:
                od_df = pd.DataFrame(request.od_data)
        else:
            raise HTTPException(status_code=400, detail="No data provided for optimization")
        
        # Validate required columns
        if 'name' not in dc_df.columns or 'lat' not in dc_df.columns or 'lon' not in dc_df.columns:
            raise HTTPException(status_code=400, detail="DC data missing required columns (name, lat, lon)")
        
        # Run optimization
        print(f"Running SND optimization with {len(dc_df)} DCs...")
        result = snd_service.solve_sndp(
            dc_df=dc_df,
            od_df=od_df,
            cost_per_dis=request.cost_per_distance,
            cost_per_time=request.cost_per_time,
            capacity=request.capacity,
            max_cpu=request.max_cpu_time,
            scaling=request.use_scaling,
            k=request.k_paths,
            alpha=request.alpha,
            max_iter=request.max_iterations,
            use_osrm=request.use_osrm
        )
        
        # Store result in session
        session_id = str(uuid.uuid4())
        session_data_store[session_id] = {
            "dc_df": dc_df,
            "od_df": od_df,
            "result": result,
            "optimization_time": pd.Timestamp.now().isoformat()
        }
        
        # Convert DataFrames to lists for JSON response
        paths = []
        for _, row in result["path_df"].iterrows():
            paths.append(PathResult(
                origin=row["origin"],
                destination=row["destination"],
                path=row["path"],
                cost=0.0  # Could calculate path cost if needed
            ))
        
        vehicles = []
        for _, row in result["vehicle_df"].iterrows():
            vehicles.append(VehicleResult(
                from_id=int(row["from_id"]),
                to_id=int(row["to_id"]),
                from_name=row["from"],
                to_name=row["to"],
                flow=float(row["flow"]),
                number=float(row["number"])
            ))
        
        cost_breakdown = CostBreakdown(
            transfer_cost=float(result["cost_breakdown"]["transfer_cost"]),
            vehicle_cost=float(result["cost_breakdown"]["vehicle_cost"]),
            total_cost=float(result["cost_breakdown"]["total_cost"])
        )
        
        response = SNDOptimizationResult(
            status=result["status"],
            total_cost=float(result["cost_breakdown"]["total_cost"]),
            paths=paths,
            vehicles=vehicles,
            cost_breakdown=cost_breakdown,
            computation_time=float(result["computation_time"]),
            iterations=result.get("iterations"),
            paths_generated=int(result["paths_generated"])
        )
        
        # Add session ID to response
        response_dict = response.dict()
        response_dict["session_id"] = session_id
        
        return clean_for_json(response_dict)
        
    except SolverError as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@router.post("/visualize")
async def visualize_snd(request: SNDVisualizationRequest):
    """
    Generate visualization data for SND solution
    """
    try:
        global snd_service
        if snd_service is None:
            snd_service = SNDService()
        
        # Get session data
        if request.session_id not in session_data_store:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = session_data_store[request.session_id]
        
        if "result" not in session_data:
            raise HTTPException(status_code=400, detail="No optimization result found in session")
        
        dc_df = session_data["dc_df"]
        result = session_data["result"]
        
        # Generate visualization data
        viz_data = snd_service.generate_visualization_data(
            dc_df=dc_df,
            path_df=result["path_df"],
            vehicle_df=result["vehicle_df"],
            pos=result["pos"],
            destination_filter=request.destination_filter
        )
        
        # Convert to response model
        nodes = [NetworkNode(**node) for node in viz_data["nodes"]]
        edges = [NetworkEdge(**edge) for edge in viz_data["edges"]]
        
        response = SNDVisualizationResult(
            nodes=nodes,
            edges=edges,
            center_lat=viz_data["center_lat"],
            center_lon=viz_data["center_lon"],
            zoom_level=viz_data["zoom_level"]
        )
        
        return clean_for_json(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")

@router.post("/export-results")
async def export_snd_results(request: SNDExportRequest):
    """
    Export SND optimization results to CSV or Excel
    """
    try:
        # Get session data
        if request.session_id not in session_data_store:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = session_data_store[request.session_id]
        
        if "result" not in session_data:
            raise HTTPException(status_code=400, detail="No optimization result found in session")
        
        result = session_data["result"]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{request.format}') as tmp_file:
            if request.format == "csv":
                # Export paths and vehicles as separate CSV files in a zip
                import zipfile
                zip_path = tmp_file.name.replace('.csv', '.zip')
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    # Save paths
                    paths_csv = result["path_df"].to_csv(index=False)
                    zipf.writestr("snd_paths.csv", paths_csv)
                    
                    # Save vehicles
                    vehicles_csv = result["vehicle_df"].to_csv(index=False)
                    zipf.writestr("snd_vehicles.csv", vehicles_csv)
                    
                    # Save cost breakdown
                    cost_df = pd.DataFrame([result["cost_breakdown"]])
                    cost_csv = cost_df.to_csv(index=False)
                    zipf.writestr("snd_costs.csv", cost_csv)
                
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename="snd_results.zip"
                )
            
            elif request.format == "excel":
                # Create Excel file with multiple sheets
                excel_path = tmp_file.name.replace('.csv', '.xlsx')
                
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    result["path_df"].to_excel(writer, sheet_name='Paths', index=False)
                    result["vehicle_df"].to_excel(writer, sheet_name='Vehicles', index=False)
                    
                    cost_df = pd.DataFrame([result["cost_breakdown"]])
                    cost_df.to_excel(writer, sheet_name='Costs', index=False)
                
                return FileResponse(
                    excel_path,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    filename="snd_results.xlsx"
                )
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """
    Get information about a specific session
    """
    try:
        if session_id not in session_data_store:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = session_data_store[session_id]
        
        info = {
            "session_id": session_id,
            "dc_count": len(session_data["dc_df"]),
            "od_pairs": len(session_data["od_df"]) * len(session_data["od_df"].columns),
            "has_result": "result" in session_data,
            "upload_time": session_data.get("upload_time"),
            "optimization_time": session_data.get("optimization_time")
        }
        
        if "result" in session_data:
            result = session_data["result"]
            info["optimization_status"] = result["status"]
            info["total_cost"] = result["cost_breakdown"]["total_cost"]
            info["computation_time"] = result["computation_time"]
            info["paths_generated"] = result["paths_generated"]
        
        return clean_for_json(info)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session info retrieval failed: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its data
    """
    try:
        if session_id not in session_data_store:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del session_data_store[session_id]
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")