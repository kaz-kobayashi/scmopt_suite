from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np


class NetworkItemData(BaseModel):
    """ネットワーク品目データモデル"""
    stage: int = Field(..., ge=1, description="ステージ番号")
    item: str = Field(..., description="品目名")
    parent: Optional[str] = Field(None, description="親品目")
    lead_time: float = Field(..., ge=0, description="リードタイム")
    
    class Config:
        json_schema_extra = {
            "example": {
                "stage": 1,
                "item": "Product_A",
                "parent": None,
                "lead_time": 7.0
            }
        }


class DemandItemData(BaseModel):
    """需要品目データモデル"""
    item: str = Field(..., description="品目名")
    mean_demand: float = Field(..., ge=0, description="平均需要")
    demand_std: float = Field(..., ge=0, description="需要標準偏差")
    cv_demand: Optional[float] = Field(None, ge=0, description="需要変動係数")
    
    @validator('cv_demand', always=True)
    def compute_cv_demand(cls, v, values):
        if v is None and 'mean_demand' in values and 'demand_std' in values:
            mean_demand = values['mean_demand']
            demand_std = values['demand_std']
            return demand_std / mean_demand if mean_demand > 0 else 0
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "item": "Product_A",
                "mean_demand": 100.0,
                "demand_std": 15.0,
                "cv_demand": 0.15
            }
        }


class CostItemData(BaseModel):
    """コスト品目データモデル"""
    item: str = Field(..., description="品目名")
    holding_cost: float = Field(..., ge=0, description="在庫保管コスト単価")
    shortage_cost: float = Field(..., ge=0, description="欠品コスト単価")
    service_level: float = Field(0.95, ge=0, le=1, description="目標サービスレベル")
    
    class Config:
        json_schema_extra = {
            "example": {
                "item": "Product_A",
                "holding_cost": 2.5,
                "shortage_cost": 50.0,
                "service_level": 0.95
            }
        }


class MESSAOptimizationOptions(BaseModel):
    """MESSA最適化オプション"""
    solver: str = Field("CBC", description="ソルバータイプ")
    max_time: int = Field(300, ge=1, le=3600, description="最大計算時間（秒）")
    gap_tolerance: float = Field(0.01, ge=0, le=1, description="最適性ギャップ許容値")
    service_level_constraint: float = Field(0.95, ge=0, le=1, description="サービスレベル制約")
    include_correlation: bool = Field(False, description="需要相関を考慮するか")
    
    @validator('solver')
    def validate_solver(cls, v):
        allowed_solvers = ["GRB", "CBC", "SCIP", "GLPK"]
        if v not in allowed_solvers:
            raise ValueError(f"サポートされていないソルバー: {v}. 使用可能: {allowed_solvers}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "solver": "CBC",
                "max_time": 300,
                "gap_tolerance": 0.01,
                "service_level_constraint": 0.95,
                "include_correlation": False
            }
        }


class MESSARequest(BaseModel):
    """MESSAリクエストモデル"""
    network_data: List[NetworkItemData] = Field(..., description="ネットワーク構造データ")
    demand_data: List[DemandItemData] = Field(..., description="需要データ")
    cost_data: List[CostItemData] = Field(..., description="コストデータ")
    optimization_options: MESSAOptimizationOptions = Field(default_factory=MESSAOptimizationOptions)
    
    class Config:
        json_schema_extra = {
            "example": {
                "network_data": [
                    {
                        "stage": 1,
                        "item": "Product_A",
                        "parent": None,
                        "lead_time": 7.0
                    }
                ],
                "demand_data": [
                    {
                        "item": "Product_A",
                        "mean_demand": 100.0,
                        "demand_std": 15.0
                    }
                ],
                "cost_data": [
                    {
                        "item": "Product_A",
                        "holding_cost": 2.5,
                        "shortage_cost": 50.0,
                        "service_level": 0.95
                    }
                ]
            }
        }


class MESSAExcelRequest(BaseModel):
    """MESSAExcelリクエストモデル"""
    network_sheet: str = Field("network", description="ネットワーク構造シート名")
    demand_sheet: str = Field("demand", description="需要データシート名")
    cost_sheet: str = Field("cost", description="コストデータシート名")
    optimization_options: MESSAOptimizationOptions = Field(default_factory=MESSAOptimizationOptions)
    output_filename: Optional[str] = Field(None, description="結果出力ファイル名")
    
    class Config:
        json_schema_extra = {
            "example": {
                "network_sheet": "network",
                "demand_sheet": "demand", 
                "cost_sheet": "cost",
                "output_filename": "messa_results.xlsx"
            }
        }


class SafetyStockResult(BaseModel):
    """安全在庫結果"""
    item: str
    safety_stock_level: float
    holding_cost: float
    target_service_level: float
    achieved_service_level: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "item": "Product_A",
                "safety_stock_level": 50.0,
                "holding_cost": 125.0,
                "target_service_level": 0.95,
                "achieved_service_level": 0.952
            }
        }


class MESSACostBreakdown(BaseModel):
    """MESSAコスト内訳"""
    total_holding_cost: float
    holding_costs_by_item: Dict[str, float]
    holding_costs_by_stage: Optional[Dict[int, float]] = None
    cost_percentages: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_holding_cost": 250.0,
                "holding_costs_by_item": {
                    "Product_A": 125.0,
                    "Product_B": 125.0
                },
                "cost_percentages": {
                    "Product_A": 50.0,
                    "Product_B": 50.0
                }
            }
        }


class MESSAResult(BaseModel):
    """MESSA最適化結果"""
    status: str = Field(..., description="最適化ステータス")
    objective_value: Optional[float] = Field(None, description="目的関数値")
    safety_stock_results: List[SafetyStockResult] = Field(..., description="安全在庫結果")
    total_safety_stock: float = Field(..., description="総安全在庫")
    cost_breakdown: MESSACostBreakdown = Field(..., description="コスト内訳")
    solution_time: Optional[float] = Field(None, description="求解時間（秒）")
    gap: Optional[float] = Field(None, description="最適性ギャップ")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "Optimal",
                "objective_value": 250.0,
                "safety_stock_results": [
                    {
                        "item": "Product_A",
                        "safety_stock_level": 50.0,
                        "holding_cost": 125.0,
                        "target_service_level": 0.95,
                        "achieved_service_level": 0.952
                    }
                ],
                "total_safety_stock": 100.0,
                "cost_breakdown": {
                    "total_holding_cost": 250.0,
                    "holding_costs_by_item": {"Product_A": 125.0},
                    "cost_percentages": {"Product_A": 100.0}
                },
                "solution_time": 2.5,
                "gap": 0.01
            }
        }


class EOQRequest(BaseModel):
    """EOQリクエストモデル"""
    fixed_cost: float = Field(..., ge=0, description="固定発注コスト")
    demand_rate: float = Field(..., gt=0, description="需要レート（期間あたり）")
    holding_cost: float = Field(..., gt=0, description="在庫保管コスト単価")
    backorder_cost: float = Field(0, ge=0, description="バックオーダーコスト単価")
    interest_rate: float = Field(0, ge=0, description="金利")
    unit_cost: float = Field(1, gt=0, description="単位コスト")
    service_level: float = Field(1, ge=0, le=1, description="サービスレベル")
    discount_schedule: Optional[List[Tuple[float, float]]] = Field(None, description="数量割引スケジュール")
    discount_type: str = Field("incremental", description="割引タイプ")
    allow_backorder: bool = Field(False, description="バックオーダー許可フラグ")
    
    @validator('discount_type')
    def validate_discount_type(cls, v):
        allowed_types = ["incremental", "all_units"]
        if v not in allowed_types:
            raise ValueError(f"サポートされていない割引タイプ: {v}. 使用可能: {allowed_types}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "fixed_cost": 100.0,
                "demand_rate": 1000.0,
                "holding_cost": 2.0,
                "backorder_cost": 10.0,
                "unit_cost": 5.0,
                "service_level": 0.95,
                "allow_backorder": True
            }
        }


class EOQResult(BaseModel):
    """EOQ結果モデル"""
    optimization_model_type: str
    optimal_order_quantity: float
    optimal_reorder_point: Optional[float] = None
    total_cost: float
    holding_cost: float
    ordering_cost: float
    shortage_cost: Optional[float] = None
    order_frequency: float
    cycle_time: float
    parameters: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "optimization_model_type": "basic_eoq",
                "optimal_order_quantity": 316.23,
                "total_cost": 632.46,
                "holding_cost": 316.23,
                "ordering_cost": 316.23,
                "order_frequency": 3.16,
                "cycle_time": 0.316,
                "parameters": {
                    "fixed_cost": 100.0,
                    "demand_rate": 1000.0,
                    "holding_cost": 2.0
                }
            }
        }


class InventorySimulationRequest(BaseModel):
    """在庫シミュレーションリクエスト"""
    n_samples: int = Field(1000, ge=1, le=10000, description="シミュレーション回数")
    n_periods: int = Field(365, ge=1, le=3650, description="シミュレーション期間")
    mean_demand: float = Field(..., gt=0, description="平均需要")
    demand_std: float = Field(..., ge=0, description="需要標準偏差")
    lead_time: int = Field(..., ge=0, description="リードタイム")
    order_quantity: float = Field(..., gt=0, description="発注量")
    reorder_point: float = Field(..., ge=0, description="再発注点")
    backorder_cost: float = Field(..., ge=0, description="欠品コスト単価")
    holding_cost: float = Field(..., ge=0, description="在庫保管コスト単価")
    fixed_cost: float = Field(..., ge=0, description="固定発注コスト")
    order_up_to_level: Optional[float] = Field(None, description="(s,S)政策のSレベル")
    
    class Config:
        json_schema_extra = {
            "example": {
                "n_samples": 1000,
                "n_periods": 365,
                "mean_demand": 10.0,
                "demand_std": 3.0,
                "lead_time": 7,
                "order_quantity": 100.0,
                "reorder_point": 70.0,
                "backorder_cost": 5.0,
                "holding_cost": 1.0,
                "fixed_cost": 50.0
            }
        }


class InventorySimulationResult(BaseModel):
    """在庫シミュレーション結果"""
    average_cost: float
    cost_std: float
    inventory_history: List[float]
    service_level_achieved: float
    average_inventory: float
    stockout_frequency: float
    total_orders: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "average_cost": 1250.5,
                "cost_std": 150.3,
                "inventory_history": [50.0, 45.0, 40.0],
                "service_level_achieved": 0.95,
                "average_inventory": 42.5,
                "stockout_frequency": 0.05,
                "total_orders": 36
            }
        }


class BaseStockSimulationRequest(BaseModel):
    """ベースストック政策シミュレーションリクエスト"""
    n_samples: int = Field(1000, ge=1, le=10000, description="シミュレーション回数")
    n_periods: int = Field(365, ge=1, le=3650, description="シミュレーション期間")
    demand_data: List[float] = Field(..., description="需要データ系列")
    capacity: float = Field(..., gt=0, description="生産能力")
    lead_time: int = Field(..., ge=0, description="リードタイム")
    backorder_cost: float = Field(..., ge=0, description="欠品コスト単価")
    holding_cost: float = Field(..., ge=0, description="在庫保管コスト単価")
    base_stock_level: float = Field(..., ge=0, description="ベースストックレベル")
    
    class Config:
        json_schema_extra = {
            "example": {
                "n_samples": 1000,
                "n_periods": 365,
                "demand_data": [10, 12, 8, 15, 9],
                "capacity": 20.0,
                "lead_time": 7,
                "backorder_cost": 5.0,
                "holding_cost": 1.0,
                "base_stock_level": 100.0
            }
        }


class NetworkOptimizationRequest(BaseModel):
    """ネットワーク最適化リクエスト"""
    network_structure: Dict[str, Any] = Field(..., description="ネットワーク構造")
    base_stock_levels: Dict[str, float] = Field(..., description="ベースストックレベル")
    demand_data: List[List[float]] = Field(..., description="需要データ配列")
    n_periods: int = Field(365, ge=1, le=3650, description="シミュレーション期間")
    n_simulations: int = Field(100, ge=1, le=1000, description="シミュレーション回数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "network_structure": {
                    "nodes": ["A", "B", "C"],
                    "edges": [["A", "B"], ["B", "C"]],
                    "lead_times": {"A": 5, "B": 3, "C": 2}
                },
                "base_stock_levels": {"A": 100, "B": 80, "C": 60},
                "demand_data": [[10, 12, 8], [15, 18, 12]],
                "n_periods": 365,
                "n_simulations": 100
            }
        }


class NetworkOptimizationResult(BaseModel):
    """ネットワーク最適化結果"""
    total_cost: float
    cost_by_node: Dict[str, float]
    inventory_levels: Dict[str, List[float]]
    service_levels: Dict[str, float]
    fill_rates: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_cost": 5000.0,
                "cost_by_node": {"A": 2000, "B": 1800, "C": 1200},
                "inventory_levels": {"A": [100, 95, 90], "B": [80, 75, 70]},
                "service_levels": {"A": 0.95, "B": 0.92, "C": 0.98},
                "fill_rates": {"A": 0.96, "B": 0.94, "C": 0.99}
            }
        }


class InventoryOptimizationServiceInfo(BaseModel):
    """在庫最適化サービス情報"""
    service_name: str
    version: str = "1.0.0"
    description: str
    features: List[str]
    supported_models: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "service_name": "Advanced Inventory Optimization Service",
                "version": "1.0.0", 
                "description": "Complete inventory optimization suite from 03inventory.ipynb",
                "features": [
                    "MESSA safety stock allocation",
                    "Advanced EOQ models",
                    "Inventory policy simulation",
                    "Multi-echelon optimization",
                    "Network-based optimization",
                    "Excel integration"
                ],
                "supported_models": [
                    "EOQ", "EOQ with backorders", "EOQ with quantity discounts",
                    "(Q,R) policy", "(s,S) policy", "Base stock policy",
                    "Multi-stage base stock", "Network base stock"
                ]
            }
        }