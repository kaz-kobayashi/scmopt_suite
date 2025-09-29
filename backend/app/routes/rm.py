from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any
import numpy as np
import time as time_module
import traceback
import os

from app.services.rm_service import RMService
from app.models.rm import (
    DynamicPricingRequest,
    ValueFunctionRequest,
    RevenueManagementRequest,
    BidPriceControlRequest,
    NestedBookingLimitRequest,
    ProspectPricingRequest,
    BookingForecastRequest,
    SampleRMDataRequest,
    CSVUploadRequest,
    DynamicPricingResult,
    ValueFunctionResult,
    RevenueManagementResult,
    ControlPolicyResult,
    ProspectPricingResult,
    BookingForecastResult,
    SampleRMDataResult,
    CSVUploadResult,
    RMError
)

router = APIRouter()

# Initialize RM service
rm_service = None

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
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif hasattr(data, 'item'):  # numpy scalar
        return clean_for_json(data.item())
    else:
        return data

@router.get("/test")
async def test_endpoint():
    """
    Simple test endpoint for RM routes
    """
    return {"message": "RM route test successful", "status": "ok"}

@router.post("/dynamic-pricing")
async def dynamic_pricing_optimization(request: DynamicPricingRequest):
    """
    動的価格最適化（強化学習）
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Running dynamic pricing with {len(request.actions)} actions...")
        
        actions = np.array(request.actions)
        result = rm_service.dynamic_pricing_learning(
            actions=actions,
            beta_params=request.beta_params,
            epochs=request.epochs,
            sigma=request.sigma,
            delta=request.delta,
            scaling=request.scaling
        )
        
        response = DynamicPricingResult(
            total_reward=result["total_reward"],
            price_history=result["price_history"],
            demand_history=result["demand_history"],
            reward_history=result["reward_history"],
            estimated_beta=result["estimated_beta"],
            epochs_data=result["epochs_data"]
        )
        
        return clean_for_json(response.dict())
        
    except RMError as e:
        raise HTTPException(status_code=500, detail=f"RM optimization failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dynamic pricing error: {str(e)}")

@router.post("/value-function")
async def value_function_calculation(request: ValueFunctionRequest):
    """
    価値関数計算（動的計画法）
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Computing value function for {request.capacity} capacity, {request.periods} periods...")
        
        start_time = time_module.time()
        
        # Compute value function
        V, A = rm_service.value_function_dp(
            capacity=request.capacity,
            periods=request.periods,
            actions=request.actions,
            beta_params=request.beta_params,
            sigma=request.sigma,
            n_samples=request.n_samples
        )
        
        # Run simulation
        simulation = rm_service.simulate_inventory_pricing(
            V=V, A=A,
            initial_capacity=min(request.capacity, 800),
            periods=request.periods,
            beta_params=request.beta_params,
            sigma=0.1
        )
        
        computation_time = time_module.time() - start_time
        
        response = ValueFunctionResult(
            value_function=V.tolist(),
            action_function=A.tolist(),
            simulation_results=simulation,
            total_reward=simulation["total_reward"],
            price_history=simulation["price_history"],
            inventory_history=simulation["inventory_history"]
        )
        
        return clean_for_json(response.dict())
        
    except RMError as e:
        raise HTTPException(status_code=500, detail=f"Value function calculation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Value function error: {str(e)}")

@router.post("/revenue-optimization")
async def revenue_management_optimization(request: RevenueManagementRequest):
    """
    収益管理最適化（確定的・サンプリング・リコース）
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Running revenue optimization with method {request.method}...")
        
        start_time = time_module.time()
        
        # Convert usage matrix from string keys to tuple keys
        a = {}
        for key_str, value in request.usage_matrix.items():
            # Parse "(i,j)" format
            key_str = key_str.strip("()")
            i, j = map(int, key_str.split(","))
            a[(i, j)] = value
        
        # Select optimization method
        if request.method == 0:
            obj, dual, ystar = rm_service.rm_deterministic(
                request.demand, request.capacity, request.revenue, a
            )
            method_name = "deterministic"
        elif request.method == 1:
            obj, dual, ystar = rm_service.rm_sampling(
                request.demand, request.capacity, request.revenue, a, request.n_samples
            )
            method_name = "sampling"
        elif request.method == 2:
            obj, dual, ystar = rm_service.rm_recourse(
                request.demand, request.capacity, request.revenue, a
            )
            method_name = "recourse"
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Must be 0, 1, or 2")
        
        computation_time = time_module.time() - start_time
        
        response = RevenueManagementResult(
            objective_value=obj,
            dual_variables=dual,
            optimal_solution=ystar,
            method_used=method_name,
            computation_time=computation_time
        )
        
        return clean_for_json(response.dict())
        
    except RMError as e:
        raise HTTPException(status_code=500, detail=f"Revenue optimization failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Revenue optimization error: {str(e)}")

@router.post("/bid-price-control")
async def bid_price_control_simulation(request: BidPriceControlRequest):
    """
    入札価格コントロール方策シミュレーション
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Running bid price control simulation with method {request.method}...")
        
        # Convert usage matrix
        a = {}
        for key_str, value in request.usage_matrix.items():
            key_str = key_str.strip("()")
            i, j = map(int, key_str.split(","))
            a[(i, j)] = value
        
        total_revenue, acceptance_history = rm_service.bid_price_control(
            demand=request.demand,
            revenue=request.revenue,
            a=a,
            capacity=request.capacity,
            n_samples=request.n_samples,
            method=request.method,
            random_seed=request.random_seed
        )
        
        response = ControlPolicyResult(
            total_revenue=total_revenue,
            acceptance_history=acceptance_history,
            capacity_history=[],  # Will be filled from acceptance_history
            dual_variables_history=[],  # Will be filled from acceptance_history
            method_used=f"bid_price_control_method_{request.method}"
        )
        
        return clean_for_json(response.dict())
        
    except RMError as e:
        raise HTTPException(status_code=500, detail=f"Bid price control simulation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bid price control error: {str(e)}")

@router.get("/sample-data")
async def generate_sample_rm_data():
    """
    サンプル収益管理データ生成
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print("Generating sample RM data...")
        
        demand, revenue, a, capacity = rm_service.make_sample_data_for_rm(5)
        
        # Convert usage matrix to string keys for JSON
        usage_matrix = {}
        for (i, j), value in a.items():
            usage_matrix[f"({i},{j})"] = value
        
        response = SampleRMDataResult(
            demand=demand,
            revenue=revenue,
            usage_matrix=usage_matrix,
            capacity=capacity,
            description="Sample data with 5 periods, 3 demand types (duration 1,2,3)"
        )
        
        return clean_for_json(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample data generation failed: {str(e)}")

@router.post("/nested-booking-limit")
async def nested_booking_limit_simulation(request: NestedBookingLimitRequest):
    """
    入れ子上限コントロール方策シミュレーション
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Running nested booking limit simulation with method {request.method}...")
        
        # Convert usage matrix
        a = {}
        for key_str, value in request.usage_matrix.items():
            key_str = key_str.strip("()")
            i, j = map(int, key_str.split(","))
            a[(i, j)] = value
        
        total_revenue, acceptance_history = rm_service.nested_booking_limit_control(
            demand=request.demand,
            revenue=request.revenue,
            a=a,
            capacity=request.capacity,
            n_samples=request.n_samples,
            method=request.method,
            random_seed=request.random_seed
        )
        
        response = ControlPolicyResult(
            total_revenue=total_revenue,
            acceptance_history=acceptance_history,
            capacity_history=[],  # Will be filled from acceptance_history
            dual_variables_history=[],  # Will be filled from acceptance_history
            method_used=f"nested_booking_limit_method_{request.method}"
        )
        
        return clean_for_json(response.dict())
        
    except RMError as e:
        raise HTTPException(status_code=500, detail=f"Nested booking limit simulation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nested booking limit error: {str(e)}")

@router.post("/prospect-pricing")
async def prospect_theory_pricing(request: ProspectPricingRequest):
    """
    プロスペクト理論価格戦略
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Running prospect theory pricing with parameters: alpha={request.alpha}, periods={request.periods}...")
        
        # Extract base demand parameters
        d0, a, p0 = request.base_demand_params
        
        result = rm_service.prospect_theory_pricing(
            base_demand=d0,
            base_price=p0,
            reference_price=request.initial_reference_price,
            alpha=request.alpha,
            zeta=request.zeta,
            eta=request.eta,
            beta=request.beta,
            gamma=request.gamma,
            periods=request.periods
        )
        
        response = ProspectPricingResult(
            optimal_prices=result["optimal_prices"],
            reference_prices=result["reference_prices"],
            demands=result["demands"],
            revenues=result["revenues"],
            total_revenue=result["total_revenue"],
            prospect_effects=result["prospect_effects"]
        )
        
        return clean_for_json(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prospect pricing error: {str(e)}")

@router.post("/booking-forecast")
async def booking_demand_forecast(request: BookingForecastRequest):
    """
    予約需要予測
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Running booking forecast with method {request.method}...")
        
        # Convert booking matrix from list to numpy array
        booking_matrix = np.array(request.booking_matrix)
        
        if request.method == "multiplicative":
            result = rm_service.multiplicative_booking_forecast(
                booking_matrix=booking_matrix,
                current_period=request.current_period,
                max_leadtime=request.max_leadtime
            )
        else:
            # For now, only multiplicative method is implemented
            raise HTTPException(status_code=400, detail=f"Method '{request.method}' not implemented yet")
        
        response = BookingForecastResult(
            forecast_matrix=result["forecast_matrix"],
            cumulative_bookings=result["cumulative_bookings"],
            multiplicative_ratios=result["multiplicative_ratios"],
            forecast_accuracy=result["forecast_accuracy"],
            method_used=result["method_used"]
        )
        
        return clean_for_json(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Booking forecast error: {str(e)}")

@router.get("/methods")
async def get_available_methods():
    """
    利用可能な手法の一覧
    """
    return {
        "revenue_optimization_methods": {
            0: "deterministic",
            1: "sampling",
            2: "recourse"
        },
        "control_policies": [
            "bid_price_control",
            "nested_booking_limit"
        ],
        "pricing_strategies": [
            "dynamic_pricing",
            "prospect_theory"
        ],
        "forecasting_methods": [
            "multiplicative",
            "additive"
        ]
    }

@router.post("/upload-csv")
async def upload_csv_data(file: UploadFile = File(...), data_type: str = Form(...)):
    """
    CSVファイルアップロード機能
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print(f"Uploading CSV file: {file.filename}, type: {data_type}")
        
        # ファイル内容を読み込み
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # CSVデータを解析
        parse_result = rm_service.parse_csv_data(csv_content, data_type)
        
        if parse_result["validation"] == "error":
            raise HTTPException(status_code=400, detail=f"CSV解析エラー: {', '.join(parse_result['errors'])}")
        
        # データの行数・列数を計算
        lines = csv_content.strip().split('\n')
        row_count = len(lines) - 1  # ヘッダーを除く
        column_count = len(lines[0].split(',')) if lines else 0
        
        response = CSVUploadResult(
            parsed_data=parse_result["data"],
            row_count=row_count,
            column_count=column_count,
            data_type=data_type,
            validation_status=parse_result["validation"],
            error_messages=parse_result.get("errors", [])
        )
        
        return clean_for_json(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV upload error: {str(e)}")

@router.post("/validate-data")
async def validate_rm_data(
    demand: Dict[int, float],
    revenue: Dict[int, float], 
    capacity: Dict[int, float],
    usage_matrix: Dict[str, int]
):
    """
    収益管理データの整合性検証
    """
    try:
        global rm_service
        if rm_service is None:
            rm_service = RMService()
        
        print("Validating RM data consistency...")
        
        validation_result = rm_service.validate_rm_data(demand, revenue, capacity, usage_matrix)
        
        return clean_for_json(validation_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data validation error: {str(e)}")

@router.get("/download-sample/{data_type}")
async def download_sample_csv(data_type: str):
    """
    サンプルCSVファイルダウンロード
    """
    try:
        print(f"Downloading sample CSV for data type: {data_type}")
        
        # ファイルパスの設定（バックエンドディレクトリ内のsample_data）
        import os
        from pathlib import Path
        
        # バックエンドディレクトリのsample_dataを使用
        current_dir = Path(__file__).parent.parent.parent
        base_path = current_dir / "sample_data"
        file_mapping = {
            "demand": "demand_sample.csv",
            "revenue": "revenue_sample.csv", 
            "capacity": "capacity_sample.csv",
            "usage_matrix": "usage_matrix_sample.csv"
        }
        
        if data_type not in file_mapping:
            raise HTTPException(status_code=400, detail=f"Invalid data type: {data_type}")
        
        file_path = base_path / file_mapping[data_type]
        
        print(f"Looking for file at: {file_path}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample file not found: {file_mapping[data_type]} at {file_path}")
        
        return FileResponse(
            path=str(file_path),
            filename=file_mapping[data_type],
            media_type='text/csv'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")