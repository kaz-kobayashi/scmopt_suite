from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class DynamicPricingRequest(BaseModel):
    """動的価格付けリクエストモデル"""
    actions: List[float] = Field(default=[15.0, 20.0, 25.0, 30.0, 35.0], description="価格候補")
    epochs: int = Field(default=10, description="エポック数")
    beta_params: Tuple[float, float] = Field(default=(50.0, -1.0), description="需要関数パラメータ (beta0, beta1)")
    sigma: float = Field(default=1.0, description="需要ノイズの標準偏差")
    t_periods: int = Field(default=512, description="計画期間")
    delta: float = Field(default=0.1, description="確率的保証パラメータ")
    scaling: float = Field(default=1.0, description="報酬スケーリング")


class ValueFunctionRequest(BaseModel):
    """価値関数計算リクエストモデル"""
    capacity: int = Field(default=1000, description="在庫容量")
    periods: int = Field(default=50, description="期間数")
    actions: List[float] = Field(default=[15.0, 20.0, 25.0, 30.0, 35.0], description="価格候補")
    beta_params: Tuple[float, float] = Field(default=(50.0, -1.0), description="需要関数パラメータ")
    sigma: float = Field(default=0.0, description="需要の不確実性")
    n_samples: int = Field(default=1, description="サンプル数")


class RevenueManagementRequest(BaseModel):
    """収益管理最適化リクエストモデル"""
    demand: Dict[int, float] = Field(description="旅程別需要量")
    capacity: Dict[int, float] = Field(description="行程別容量")
    revenue: Dict[int, float] = Field(description="旅程別収益")
    usage_matrix: Dict[str, int] = Field(description="旅程-行程使用関係 '(i,j)': 1")
    method: int = Field(default=0, description="最適化手法 0:deterministic, 1:sampling, 2:recourse")
    n_samples: int = Field(default=100, description="サンプリング数")


class BidPriceControlRequest(BaseModel):
    """入札価格コントロール方策リクエストモデル"""
    demand: Dict[int, float] = Field(description="旅程別需要量")
    capacity: Dict[int, float] = Field(description="行程別容量")
    revenue: Dict[int, float] = Field(description="旅程別収益")
    usage_matrix: Dict[str, int] = Field(description="旅程-行程使用関係")
    method: int = Field(default=0, description="双対変数計算手法")
    n_samples: int = Field(default=100, description="サンプリング数")
    random_seed: int = Field(default=123, description="乱数シード")


class NestedBookingLimitRequest(BaseModel):
    """入れ子上限コントロール方策リクエストモデル"""
    demand: Dict[int, float] = Field(description="旅程別需要量")
    capacity: Dict[int, float] = Field(description="行程別容量")
    revenue: Dict[int, float] = Field(description="旅程別収益")
    usage_matrix: Dict[str, int] = Field(description="旅程-行程使用関係")
    method: int = Field(default=0, description="双対変数計算手法")
    n_samples: int = Field(default=100, description="サンプリング数")
    random_seed: int = Field(default=123, description="乱数シード")


class ProspectPricingRequest(BaseModel):
    """プロスペクト理論価格戦略リクエストモデル"""
    periods: int = Field(default=200, description="計画期間")
    alpha: float = Field(default=0.5, description="参照価格更新パラメータ")
    beta: float = Field(default=0.8, description="利益域リスク回避パラメータ")
    gamma: float = Field(default=0.8, description="損失域リスク回避パラメータ")
    zeta: float = Field(default=8.0, description="利益域需要パラメータ")
    eta: float = Field(default=12.0, description="損失域需要パラメータ")
    initial_reference_price: float = Field(default=25.0, description="初期参照価格")
    base_demand_params: Tuple[float, float, float] = Field(default=(100.0, 2.0, 25.0), description="基本需要パラメータ (d0, a, p0)")


class BookingForecastRequest(BaseModel):
    """予約予測リクエストモデル"""
    booking_data: List[Dict[str, Any]] = Field(description="予約履歴データ")
    horizon_days: int = Field(default=100, description="予測期間")
    leadtime_days: int = Field(default=15, description="最大リードタイム")
    current_period: int = Field(default=90, description="現在期間")
    method: str = Field(default="multiplicative", description="予測手法 multiplicative/additive")


class SampleRMDataRequest(BaseModel):
    """サンプル収益管理データ生成リクエストモデル"""
    num_periods: int = Field(default=5, description="期間数")
    demand_types: Dict[int, float] = Field(default={1: 20.0, 2: 6.0, 3: 3.0}, description="需要タイプ別需要量")
    revenue_types: Dict[int, float] = Field(default={1: 1000.0, 2: 3000.0, 3: 5000.0}, description="需要タイプ別収益")
    capacity_per_period: float = Field(default=10.0, description="期間あたり容量")


# 結果モデル
class DynamicPricingResult(BaseModel):
    """動的価格付け結果モデル"""
    total_reward: float
    price_history: List[float]
    demand_history: List[float]
    reward_history: List[float]
    estimated_beta: Tuple[float, float]
    epochs_data: List[Dict[str, Any]]


class ValueFunctionResult(BaseModel):
    """価値関数計算結果モデル"""
    value_function: List[List[float]]
    action_function: List[List[float]]
    simulation_results: Dict[str, Any]
    total_reward: float
    price_history: List[float]
    inventory_history: List[int]


class RevenueManagementResult(BaseModel):
    """収益管理最適化結果モデル"""
    objective_value: float
    dual_variables: Dict[int, float]
    optimal_solution: Dict[int, float]
    method_used: str
    computation_time: float


class ControlPolicyResult(BaseModel):
    """制御方策結果モデル"""
    total_revenue: float
    acceptance_history: List[Dict[str, Any]]
    capacity_history: List[Dict[int, float]]
    dual_variables_history: List[Dict[int, float]]
    method_used: str


class ProspectPricingResult(BaseModel):
    """プロスペクト理論価格戦略結果モデル"""
    optimal_prices: List[float]
    reference_prices: List[float]
    demands: List[float]
    revenues: List[float]
    total_revenue: float
    prospect_effects: Dict[str, Any]


class BookingForecastResult(BaseModel):
    """予約予測結果モデル"""
    forecast_matrix: List[List[float]]
    cumulative_bookings: List[List[float]]
    multiplicative_ratios: List[float]
    forecast_accuracy: Dict[str, float]
    method_used: str


class SampleRMDataResult(BaseModel):
    """サンプル収益管理データ結果モデル"""
    demand: Dict[int, float]
    revenue: Dict[int, float]
    usage_matrix: Dict[str, int]
    capacity: Dict[int, float]
    description: str


class CSVUploadRequest(BaseModel):
    """CSVファイルアップロードリクエストモデル"""
    file_content: str = Field(description="CSVファイルの内容")
    file_name: str = Field(description="ファイル名")
    data_type: str = Field(description="データタイプ: demand, revenue, capacity, usage_matrix")


class CSVUploadResult(BaseModel):
    """CSVアップロード結果モデル"""
    parsed_data: Dict[str, Any]
    row_count: int
    column_count: int
    data_type: str
    validation_status: str
    error_messages: List[str] = []


class RMError(Exception):
    """収益管理カスタム例外"""
    pass