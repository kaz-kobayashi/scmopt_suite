from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np


class BOMEdgeData(BaseModel):
    """BOM edge data model"""
    child: str = Field(..., description="Child product/component name")
    parent: str = Field(..., description="Parent product name") 
    units: float = Field(1.0, ge=0, description="Units of child required per parent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "child": "Component_A",
                "parent": "Product_B",
                "units": 2.0
            }
        }


class PlantTransportData(BaseModel):
    """Plant transportation data model"""
    from_node: int = Field(..., description="Source plant ID")
    to_node: int = Field(..., description="Destination plant ID")
    kind: str = Field("plnt-plnt", description="Transportation type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "from_node": 1,
                "to_node": 2,
                "kind": "plnt-plnt"
            }
        }


class PlantProductData(BaseModel):
    """Plant-Product relationship data model"""
    plnt: int = Field(..., description="Plant ID")
    prod: str = Field(..., description="Product name")
    ub: float = Field(..., ge=0, description="Production upper bound")
    pipeline: float = Field(..., ge=0, description="Pipeline inventory")
    demand: Optional[float] = Field(None, ge=0, description="External demand (if any)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "plnt": 1,
                "prod": "Product_A",
                "ub": 1000.0,
                "pipeline": 50.0,
                "demand": 100.0
            }
        }


class PlantCapacityData(BaseModel):
    """Plant capacity data model"""
    name: int = Field(..., description="Plant ID")
    ub: float = Field(..., ge=0, description="Plant capacity upper bound")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": 1,
                "ub": 2000.0
            }
        }


class SCRMDataGenerationOptions(BaseModel):
    """SCRM data generation options"""
    benchmark_id: str = Field("01", description="Willems benchmark problem ID")
    n_plnts: int = Field(3, ge=1, le=10, description="Number of plants per stage")
    n_flex: int = Field(2, ge=1, le=5, description="Production flexibility (products per plant)")
    prob: float = Field(0.5, ge=0, le=1, description="Edge probability in plant graph")
    capacity_factor: float = Field(1.0, gt=0, le=10, description="Plant capacity multiplier")
    production_factor: float = Field(1.0, gt=0, le=10, description="Production limit multiplier")
    pipeline_factor: float = Field(1.0, gt=0, le=10, description="Pipeline inventory multiplier")
    seed: int = Field(1, ge=1, description="Random seed for reproducibility")
    
    @validator('benchmark_id')
    def validate_benchmark_id(cls, v):
        # Ensure benchmark ID is zero-padded 2-digit string
        try:
            id_int = int(v)
            if not (1 <= id_int <= 38):
                raise ValueError(f"Benchmark ID must be between 01 and 38")
            return f"{id_int:02d}"
        except ValueError:
            raise ValueError(f"Invalid benchmark ID: {v}")
    
    class Config:
        json_schema_extra = {
            "example": {
                "benchmark_id": "01",
                "n_plnts": 3,
                "n_flex": 2,
                "prob": 0.5,
                "capacity_factor": 1.0,
                "production_factor": 1.0,
                "pipeline_factor": 1.0,
                "seed": 1
            }
        }


class SCRMDataGenerationRequest(BaseModel):
    """SCRM data generation request"""
    options: SCRMDataGenerationOptions = Field(default_factory=SCRMDataGenerationOptions)
    
    class Config:
        json_schema_extra = {
            "example": {
                "options": {
                    "benchmark_id": "01",
                    "n_plnts": 3,
                    "n_flex": 2,
                    "seed": 42
                }
            }
        }


class SCRMUploadRequest(BaseModel):
    """SCRM CSV upload request"""
    bom_data: List[BOMEdgeData] = Field(..., description="BOM structure data")
    transport_data: List[PlantTransportData] = Field(..., description="Plant transportation data")
    plant_product_data: List[PlantProductData] = Field(..., description="Plant-product relationship data")
    plant_capacity_data: List[PlantCapacityData] = Field(..., description="Plant capacity data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bom_data": [
                    {"child": "Component_A", "parent": "Product_B", "units": 1.0}
                ],
                "transport_data": [
                    {"from_node": 1, "to_node": 2, "kind": "plnt-plnt"}
                ],
                "plant_product_data": [
                    {"plnt": 1, "prod": "Product_A", "ub": 1000.0, "pipeline": 50.0, "demand": 100.0}
                ],
                "plant_capacity_data": [
                    {"name": 1, "ub": 2000.0}
                ]
            }
        }


class SCRMAnalysisRequest(BaseModel):
    """SCRM analysis request"""
    data_source: str = Field(..., description="Data source type: 'generated' or 'uploaded'")
    generation_options: Optional[SCRMDataGenerationOptions] = Field(None, description="Options for data generation")
    upload_data: Optional[SCRMUploadRequest] = Field(None, description="Uploaded data")
    
    @validator('data_source')
    def validate_data_source(cls, v):
        allowed_sources = ["generated", "uploaded"]
        if v not in allowed_sources:
            raise ValueError(f"Data source must be one of: {allowed_sources}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_source": "generated",
                "generation_options": {
                    "benchmark_id": "01",
                    "n_plnts": 3,
                    "n_flex": 2
                }
            }
        }


class CriticalNode(BaseModel):
    """Critical node information"""
    plant: int = Field(..., description="Plant ID")
    product: str = Field(..., description="Product name")
    survival_time: float = Field(..., ge=0, description="Time-to-Survival when this node is disrupted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "plant": 1,
                "product": "Product_A",
                "survival_time": 2.5
            }
        }


class SCRMDataGenerationResult(BaseModel):
    """SCRM data generation result"""
    benchmark_id: str = Field(..., description="Benchmark problem ID used")
    total_demand: float = Field(..., ge=0, description="Total system demand")
    total_plants: int = Field(..., ge=1, description="Total number of plants")
    total_products: int = Field(..., ge=1, description="Total number of products")
    total_nodes: int = Field(..., ge=1, description="Total nodes in production graph")
    bom_edges: int = Field(..., ge=0, description="Number of BOM edges")
    plant_edges: int = Field(..., ge=0, description="Number of plant transportation edges")
    generation_options: SCRMDataGenerationOptions = Field(..., description="Options used for generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "benchmark_id": "01",
                "total_demand": 500.0,
                "total_plants": 9,
                "total_products": 8,
                "total_nodes": 24,
                "bom_edges": 7,
                "plant_edges": 12,
                "generation_options": {
                    "benchmark_id": "01",
                    "n_plnts": 3,
                    "n_flex": 2
                }
            }
        }


class SCRMAnalysisResult(BaseModel):
    """SCRM analysis result"""
    status: str = Field(..., description="Analysis status")
    total_nodes: int = Field(..., ge=0, description="Total nodes analyzed")
    survival_time: List[float] = Field(..., description="Time-to-Survival for each disruption scenario")
    critical_nodes: List[CriticalNode] = Field(..., description="Most critical nodes (lowest survival times)")
    average_survival_time: float = Field(..., ge=0, description="Average survival time across all scenarios")
    min_survival_time: float = Field(..., ge=0, description="Minimum survival time")
    max_survival_time: float = Field(..., ge=0, description="Maximum survival time")
    analysis_summary: Dict[str, Any] = Field(..., description="Additional analysis metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_nodes": 24,
                "survival_time": [0.0, 2.5, 1.8, 0.0, 3.2],
                "critical_nodes": [
                    {"plant": 1, "product": "Product_A", "survival_time": 0.0},
                    {"plant": 2, "product": "Component_B", "survival_time": 1.2}
                ],
                "average_survival_time": 1.75,
                "min_survival_time": 0.0,
                "max_survival_time": 3.2,
                "analysis_summary": {
                    "zero_survival_nodes": 2,
                    "high_risk_nodes": 5,
                    "resilient_nodes": 17
                }
            }
        }


class VisualizationOptions(BaseModel):
    """Visualization options for SCRM graphs"""
    graph_type: str = Field(..., description="Type of graph to visualize")
    title: str = Field("", description="Graph title")
    node_size: int = Field(30, ge=5, le=100, description="Node size")
    node_color: str = Field("Yellow", description="Node color")
    width: int = Field(800, ge=400, le=2000, description="Graph width in pixels")
    height: int = Field(600, ge=300, le=1500, description="Graph height in pixels")
    
    @validator('graph_type')
    def validate_graph_type(cls, v):
        allowed_types = ["plant_graph", "bom_graph", "production_graph", "risk_analysis"]
        if v not in allowed_types:
            raise ValueError(f"Graph type must be one of: {allowed_types}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "graph_type": "risk_analysis",
                "title": "Supply Chain Risk Analysis",
                "node_size": 40,
                "node_color": "Red",
                "width": 1000,
                "height": 700
            }
        }


class SCRMVisualizationRequest(BaseModel):
    """SCRM visualization request"""
    data_id: str = Field(..., description="Data session ID from previous generation/upload")
    options: VisualizationOptions = Field(..., description="Visualization options")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_id": "scrm_session_123",
                "options": {
                    "graph_type": "risk_analysis",
                    "title": "Risk Analysis Network"
                }
            }
        }


class SCRMVisualizationResult(BaseModel):
    """SCRM visualization result"""
    graph_type: str = Field(..., description="Type of graph visualized")
    plotly_json: Dict[str, Any] = Field(..., description="Plotly figure as JSON")
    summary: Dict[str, Any] = Field(..., description="Graph summary statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "graph_type": "risk_analysis",
                "plotly_json": {"data": [], "layout": {}},
                "summary": {
                    "total_nodes": 24,
                    "total_edges": 18,
                    "critical_nodes_count": 3
                }
            }
        }


class FileDownloadInfo(BaseModel):
    """File download information"""
    file_name: str = Field(..., description="Name of the file to download")
    file_type: str = Field(..., description="File type (csv, xlsx, json)")
    data_type: str = Field(..., description="Type of data (bom, plants, analysis_results)")
    
    @validator('file_type')
    def validate_file_type(cls, v):
        allowed_types = ["csv", "xlsx", "json"]
        if v not in allowed_types:
            raise ValueError(f"File type must be one of: {allowed_types}")
        return v
    
    @validator('data_type')
    def validate_data_type(cls, v):
        allowed_types = ["bom", "plants", "transport", "plant_products", "analysis_results"]
        if v not in allowed_types:
            raise ValueError(f"Data type must be one of: {allowed_types}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "scrm_analysis_results_01.csv",
                "file_type": "csv",
                "data_type": "analysis_results"
            }
        }


class ServiceInfo(BaseModel):
    """SCRM service information"""
    service_name: str
    version: str = "1.0.0"
    description: str
    features: List[str]
    supported_benchmarks: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "service_name": "Supply Chain Risk Management (SCRM) Service",
                "version": "1.0.0",
                "description": "Complete SCRM analysis system from 09scrm.ipynb implementing MERIODAS framework",
                "features": [
                    "Time-to-Survival (TTS) analysis",
                    "Critical node identification",
                    "Risk visualization with Plotly",
                    "Willems benchmark data generation",
                    "CSV data import/export",
                    "Multi-scenario disruption analysis",
                    "Interactive graph visualization"
                ],
                "supported_benchmarks": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
            }
        }