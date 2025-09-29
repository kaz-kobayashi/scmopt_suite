from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class ItemMasterData(BaseModel):
    """品目マスタデータモデル"""
    name: str = Field(..., description="品目名")
    holding_cost: float = Field(1.0, ge=0, description="在庫保管コスト単価")
    setup_cost: float = Field(100.0, ge=0, description="段取りコスト")
    target_inventory: float = Field(1000.0, ge=0, description="目標在庫レベル")
    initial_inventory: Optional[float] = Field(0.0, ge=0, description="初期在庫")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Product_A",
                "holding_cost": 2.5,
                "setup_cost": 150.0,
                "target_inventory": 500.0,
                "initial_inventory": 100.0
            }
        }


class ProcessMasterData(BaseModel):
    """プロセスマスタデータモデル"""
    name: str = Field(..., description="品目名")
    process: str = Field(..., description="プロセス名")
    processing_time: float = Field(1.0, ge=0, description="処理時間（分/単位）")
    setup_time: float = Field(60.0, ge=0, description="段取り時間（分）")
    resource: Optional[str] = Field(None, description="使用資源名")
    mode: Optional[str] = Field(None, description="生産モード")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Product_A",
                "process": "Assembly",
                "processing_time": 2.0,
                "setup_time": 90.0,
                "resource": "Machine_1",
                "mode": "Normal"
            }
        }


class BOMData(BaseModel):
    """部品表データモデル"""
    parent: str = Field(..., description="親品目")
    child: str = Field(..., description="子品目")
    units: float = Field(1.0, gt=0, description="必要数量")
    
    class Config:
        schema_extra = {
            "example": {
                "parent": "Product_A",
                "child": "Component_B",
                "units": 2.0
            }
        }


class ResourceData(BaseModel):
    """資源データモデル"""
    name: str = Field(..., description="資源名")
    period: int = Field(..., ge=1, description="期間")
    capacity: float = Field(..., ge=0, description="資源容量（分）")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Machine_1",
                "period": 1,
                "capacity": 2400.0
            }
        }


class OrderData(BaseModel):
    """注文データモデル"""
    item: str = Field(..., description="品目名")
    period: int = Field(..., ge=1, description="期間")
    demand: float = Field(..., ge=0, description="需要量")
    due_date: int = Field(..., ge=1, description="納期")
    
    class Config:
        schema_extra = {
            "example": {
                "item": "Product_A",
                "period": 1,
                "demand": 100.0,
                "due_date": 1
            }
        }


class DemandMatrix(BaseModel):
    """需要マトリクスデータモデル"""
    items: List[str] = Field(..., description="品目リスト")
    periods: int = Field(..., ge=1, description="計画期間数")
    demand_data: List[List[float]] = Field(..., description="需要データ（品目×期間）")
    
    @validator('demand_data')
    def validate_demand_data(cls, v, values):
        if 'items' in values and 'periods' in values:
            expected_rows = len(values['items'])
            expected_cols = values['periods']
            
            if len(v) != expected_rows:
                raise ValueError(f"需要データの行数が品目数と一致しません: {len(v)} != {expected_rows}")
            
            for i, row in enumerate(v):
                if len(row) != expected_cols:
                    raise ValueError(f"需要データの列数が期間数と一致しません（行{i}）: {len(row)} != {expected_cols}")
                
                if any(demand < 0 for demand in row):
                    raise ValueError(f"需要は非負でなければなりません（行{i}）")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "items": ["Product_A", "Product_B"],
                "periods": 3,
                "demand_data": [
                    [100.0, 150.0, 120.0],
                    [80.0, 90.0, 110.0]
                ]
            }
        }


class OptimizationOptions(BaseModel):
    """最適化オプション"""
    max_cpu: int = Field(10, ge=1, le=3600, description="最大計算時間（秒）")
    solver: str = Field("CBC", description="ソルバータイプ")
    gap_tolerance: Optional[float] = Field(0.01, ge=0, le=1, description="最適性ギャップ許容値")
    use_gurobi: bool = Field(False, description="Gurobiソルバー使用フラグ")
    
    @validator('solver')
    def validate_solver(cls, v):
        allowed_solvers = ["GRB", "CBC", "SCIP", "GLPK"]
        if v not in allowed_solvers:
            raise ValueError(f"サポートされていないソルバー: {v}. 使用可能: {allowed_solvers}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "max_cpu": 300,
                "solver": "GRB",
                "gap_tolerance": 0.05,
                "use_gurobi": True
            }
        }


class LotsizingRequest(BaseModel):
    """ロットサイズ決定問題リクエスト"""
    prod_data: List[ItemMasterData] = Field(..., description="品目データ")
    production_data: List[ProcessMasterData] = Field(..., description="生産データ")
    bom_data: Optional[List[BOMData]] = Field(None, description="部品表データ")
    demand: DemandMatrix = Field(..., description="需要データ")
    resource_data: Optional[List[ResourceData]] = Field(None, description="資源データ")
    optimization_options: OptimizationOptions = Field(default_factory=OptimizationOptions)
    
    class Config:
        schema_extra = {
            "example": {
                "prod_data": [
                    {
                        "name": "Product_A",
                        "holding_cost": 2.0,
                        "setup_cost": 100.0,
                        "target_inventory": 500.0
                    }
                ],
                "production_data": [
                    {
                        "name": "Product_A",
                        "process": "Assembly",
                        "processing_time": 1.5,
                        "setup_time": 60.0,
                        "resource": "Machine_1"
                    }
                ],
                "demand": {
                    "items": ["Product_A"],
                    "periods": 5,
                    "demand_data": [[100.0, 120.0, 80.0, 150.0, 110.0]]
                },
                "optimization_options": {
                    "max_cpu": 300,
                    "solver": "CBC"
                }
            }
        }


class MultiModeLotsizingRequest(BaseModel):
    """マルチモードロットサイズ決定問題リクエスト"""
    prod_data: List[ItemMasterData] = Field(..., description="品目データ")
    production_data: List[ProcessMasterData] = Field(..., description="生産データ（複数モード含む）")
    bom_data: Optional[List[BOMData]] = Field(None, description="部品表データ")
    demand: DemandMatrix = Field(..., description="需要データ")
    resource_data: List[ResourceData] = Field(..., description="資源データ（必須）")
    modes: List[str] = Field(..., description="生産モードリスト")
    optimization_options: OptimizationOptions = Field(default_factory=OptimizationOptions)
    
    @validator('modes')
    def validate_modes(cls, v):
        if len(v) == 0:
            raise ValueError("少なくとも1つのモードが必要です")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prod_data": [
                    {
                        "name": "Product_A",
                        "holding_cost": 2.0,
                        "setup_cost": 100.0,
                        "target_inventory": 500.0
                    }
                ],
                "production_data": [
                    {
                        "name": "Product_A",
                        "process": "Assembly_Normal",
                        "processing_time": 2.0,
                        "setup_time": 90.0,
                        "resource": "Machine_1",
                        "mode": "Normal"
                    },
                    {
                        "name": "Product_A",
                        "process": "Assembly_Fast",
                        "processing_time": 1.5,
                        "setup_time": 120.0,
                        "resource": "Machine_1",
                        "mode": "Fast"
                    }
                ],
                "demand": {
                    "items": ["Product_A"],
                    "periods": 5,
                    "demand_data": [[100.0, 120.0, 80.0, 150.0, 110.0]]
                },
                "resource_data": [
                    {"name": "Machine_1", "period": 1, "capacity": 2400.0},
                    {"name": "Machine_1", "period": 2, "capacity": 2400.0}
                ],
                "modes": ["Normal", "Fast"]
            }
        }


class ProductionScheduleResult(BaseModel):
    """生産スケジュール結果"""
    item: str
    period: int
    quantity: float
    mode: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "item": "Product_A",
                "period": 1,
                "quantity": 150.0,
                "mode": "Normal"
            }
        }


class InventoryResult(BaseModel):
    """在庫結果"""
    item: str
    period: int
    level: float
    
    class Config:
        schema_extra = {
            "example": {
                "item": "Product_A",
                "period": 1,
                "level": 50.0
            }
        }


class SetupResult(BaseModel):
    """段取り結果"""
    item: str
    period: int
    setup: bool
    mode: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "item": "Product_A",
                "period": 1,
                "setup": True,
                "mode": "Normal"
            }
        }


class CostSummary(BaseModel):
    """コスト要約"""
    total_cost: float
    holding_cost: float
    setup_cost: float
    details: Optional[Dict[str, float]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "total_cost": 1250.5,
                "holding_cost": 350.5,
                "setup_cost": 900.0,
                "details": {
                    "Product_A_holding": 200.0,
                    "Product_A_setup": 500.0,
                    "Product_B_holding": 150.5,
                    "Product_B_setup": 400.0
                }
            }
        }


class ResourceUtilization(BaseModel):
    """資源利用状況"""
    resource: str
    period: int
    capacity: float
    usage: float
    utilization_rate: float
    mode: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "resource": "Machine_1",
                "period": 1,
                "capacity": 2400.0,
                "usage": 1800.0,
                "utilization_rate": 0.75,
                "mode": "Normal"
            }
        }


class LotsizingResult(BaseModel):
    """ロットサイズ決定結果"""
    status: str = Field(..., description="最適化ステータス")
    objective_value: Optional[float] = Field(None, description="目的関数値")
    production_schedule: List[ProductionScheduleResult] = Field(..., description="生産スケジュール")
    inventory_levels: List[InventoryResult] = Field(..., description="在庫レベル")
    setup_schedule: List[SetupResult] = Field(..., description="段取りスケジュール")
    cost_summary: CostSummary = Field(..., description="コスト要約")
    resource_utilization: Optional[List[ResourceUtilization]] = Field(None, description="資源利用状況")
    solution_time: Optional[float] = Field(None, description="求解時間（秒）")
    gap: Optional[float] = Field(None, description="最適性ギャップ")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "Optimal",
                "objective_value": 1250.5,
                "production_schedule": [
                    {
                        "item": "Product_A",
                        "period": 1,
                        "quantity": 150.0
                    }
                ],
                "inventory_levels": [
                    {
                        "item": "Product_A",
                        "period": 1,
                        "level": 50.0
                    }
                ],
                "setup_schedule": [
                    {
                        "item": "Product_A",
                        "period": 1,
                        "setup": True
                    }
                ],
                "cost_summary": {
                    "total_cost": 1250.5,
                    "holding_cost": 350.5,
                    "setup_cost": 900.0
                },
                "solution_time": 2.5,
                "gap": 0.01
            }
        }


class ExcelLotsizeRequest(BaseModel):
    """Excelファイルベースのロットサイズ最適化リクエスト"""
    filename: str = Field(..., description="Excelファイル名")
    optimization_options: OptimizationOptions = Field(default_factory=OptimizationOptions)
    output_filename: Optional[str] = Field(None, description="結果出力ファイル名")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "lotsize_input.xlsx",
                "optimization_options": {
                    "max_cpu": 300,
                    "solver": "CBC"
                },
                "output_filename": "lotsize_result.xlsx"
            }
        }


class CostAnalysisRequest(BaseModel):
    """コスト分析リクエスト"""
    lotsize_result: LotsizingResult = Field(..., description="ロットサイズ決定結果")
    analysis_options: Optional[Dict[str, Any]] = Field(None, description="分析オプション")
    
    class Config:
        schema_extra = {
            "example": {
                "lotsize_result": {
                    "status": "Optimal",
                    "objective_value": 1250.5,
                    "production_schedule": [],
                    "inventory_levels": [],
                    "setup_schedule": [],
                    "cost_summary": {
                        "total_cost": 1250.5,
                        "holding_cost": 350.5,
                        "setup_cost": 900.0
                    }
                },
                "analysis_options": {
                    "breakdown_by_item": True,
                    "breakdown_by_period": True
                }
            }
        }


class ServiceInfo(BaseModel):
    """サービス情報"""
    service_name: str
    version: str = "1.0.0"
    description: str
    features: List[str]
    supported_solvers: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "service_name": "Lot Size Optimization Service",
                "version": "1.0.0",
                "description": "Dynamic lot sizing optimization with multi-stage BOM support",
                "features": [
                    "Basic lot sizing optimization",
                    "Multi-mode production planning",
                    "BOM hierarchy processing",
                    "Resource capacity constraints",
                    "Excel integration"
                ],
                "supported_solvers": ["GRB", "CBC", "SCIP", "GLPK"]
            }
        }