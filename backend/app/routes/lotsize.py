from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import json
import tempfile
import os
from app.services.lotsize_service import LotsizeOptimizationService
from app.services.excel_lotsize_service import ExcelLotsizeService
from app.models.lotsize import (
    LotsizingRequest,
    MultiModeLotsizingRequest,
    LotsizingResult,
    ExcelLotsizeRequest,
    CostAnalysisRequest,
    ServiceInfo,
    ItemMasterData,
    ProcessMasterData,
    BOMData,
    ResourceData,
    DemandMatrix,
    OptimizationOptions,
    ProductionScheduleResult,
    InventoryResult,
    SetupResult,
    CostSummary,
    ResourceUtilization
)

router = APIRouter()

# Initialize services
lotsize_service = LotsizeOptimizationService()
excel_service = ExcelLotsizeService()


@router.post("/optimize", response_model=LotsizingResult)
async def optimize_lotsize(request: LotsizingRequest):
    """
    ロットサイズ決定問題を最適化
    Exact implementation from 11lotsize.ipynb notebook
    """
    try:
        # Convert Pydantic models to DataFrames
        prod_df = pd.DataFrame([item.dict() for item in request.prod_data])
        production_df = pd.DataFrame([prod.dict() for prod in request.production_data])
        
        bom_df = None
        if request.bom_data:
            bom_df = pd.DataFrame([bom.dict() for bom in request.bom_data])
        
        resource_df = None
        if request.resource_data:
            resource_df = pd.DataFrame([res.dict() for res in request.resource_data])
        
        # Convert demand matrix
        demand = np.array(request.demand.demand_data)
        
        # Run optimization
        model, T = lotsize_service.lotsizing(
            prod_df=prod_df,
            production_df=production_df,
            bom_df=bom_df,
            demand=demand,
            resource_df=resource_df,
            max_cpu=request.optimization_options.max_cpu,
            solver=request.optimization_options.solver
        )
        
        # Extract results
        result = _extract_optimization_results(model, prod_df, production_df, T)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"最適化エラー: {str(e)}")


@router.post("/optimize-multimode", response_model=LotsizingResult)
async def optimize_multimode_lotsize(request: MultiModeLotsizingRequest):
    """
    マルチモードロットサイズ決定問題を最適化
    Exact implementation from 11lotsize.ipynb notebook
    """
    try:
        # Convert Pydantic models to DataFrames
        prod_df = pd.DataFrame([item.dict() for item in request.prod_data])
        production_df = pd.DataFrame([prod.dict() for prod in request.production_data])
        
        bom_df = None
        if request.bom_data:
            bom_df = pd.DataFrame([bom.dict() for bom in request.bom_data])
        
        resource_df = pd.DataFrame([res.dict() for res in request.resource_data])
        
        # Convert demand matrix
        demand = np.array(request.demand.demand_data)
        
        # Run multi-mode optimization
        model, T = lotsize_service.multi_mode_lotsizing(
            prod_df=prod_df,
            production_df=production_df,
            bom_df=bom_df,
            demand=demand,
            resource_df=resource_df,
            modes=request.modes,
            max_cpu=request.optimization_options.max_cpu,
            solver=request.optimization_options.solver
        )
        
        # Extract results
        result = _extract_multimode_optimization_results(
            model, prod_df, production_df, T, request.modes
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"マルチモード最適化エラー: {str(e)}")


@router.post("/excel-optimize")
async def optimize_from_excel(
    file: UploadFile = File(...),
    max_cpu: int = Form(10),
    solver: str = Form("CBC"),
    output_filename: Optional[str] = Form(None)
):
    """
    Excelファイルからロットサイズ決定問題を最適化
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_filename = temp_file.name
        
        try:
            # Read data from Excel
            prod_df, production_df, bom_df, resource_df, order_df = excel_service.read_dfs_from_excel_lot(temp_filename)
            
            # Convert order data to demand matrix
            demand = _convert_orders_to_demand(order_df, prod_df)
            
            # Run optimization
            model, T = lotsize_service.lotsizing(
                prod_df=prod_df,
                production_df=production_df,
                bom_df=bom_df,
                demand=demand,
                resource_df=resource_df,
                max_cpu=max_cpu,
                solver=solver
            )
            
            # Generate output Excel file
            if output_filename is None:
                output_filename = f"lotsize_result_{file.filename}"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as output_file:
                output_path = excel_service.lot_output_excel(
                    model, prod_df, production_df, T, output_file.name
                )
                
                return FileResponse(
                    path=output_path,
                    filename=output_filename,
                    media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        
        finally:
            # Clean up temporary files
            os.unlink(temp_filename)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Excel最適化エラー: {str(e)}")


@router.post("/generate-excel-template")
async def generate_excel_template(
    prod_data: List[ItemMasterData],
    production_data: List[ProcessMasterData],
    bom_data: Optional[List[BOMData]] = None,
    resource_data: Optional[List[ResourceData]] = None,
    demand: Optional[DemandMatrix] = None
):
    """
    ロットサイズ最適化用Excelテンプレートを生成
    """
    try:
        # Convert to DataFrames
        prod_df = pd.DataFrame([item.dict() for item in prod_data])
        production_df = pd.DataFrame([prod.dict() for prod in production_data])
        
        bom_df = None
        if bom_data:
            bom_df = pd.DataFrame([bom.dict() for bom in bom_data])
        
        resource_df = None
        if resource_data:
            resource_df = pd.DataFrame([res.dict() for res in resource_data])
        
        # Convert demand if provided
        demand_array = None
        if demand:
            demand_array = np.array(demand.demand_data)
        else:
            # Generate sample demand
            demand_array = np.random.randint(50, 200, size=(len(prod_data), 5))
        
        # Generate Excel template
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            filename = excel_service.generate_lotsize_master(
                prod_df, production_df, bom_df, demand_array, resource_df, temp_file.name
            )
            
            return FileResponse(
                path=filename,
                filename="lotsize_template.xlsx",
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"テンプレート生成エラー: {str(e)}")


@router.post("/show-results")
async def show_optimization_results(request: LotsizingRequest):
    """
    ロットサイズ最適化結果を表示用に整理
    """
    try:
        # Run optimization first
        optimization_result = await optimize_lotsize(request)
        
        # Generate detailed results using notebook function
        prod_df = pd.DataFrame([item.dict() for item in request.prod_data])
        production_df = pd.DataFrame([prod.dict() for prod in request.production_data])
        
        # Convert to numpy array for result processing
        demand = np.array(request.demand.demand_data)
        T = request.demand.periods
        
        # Use service function to show results
        results = lotsize_service.show_result_for_lotsizing(
            None, prod_df, production_df, demand, T
        )
        
        return {
            "optimization_result": optimization_result,
            "detailed_analysis": results,
            "cost_breakdown": _generate_cost_breakdown(optimization_result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"結果表示エラー: {str(e)}")


@router.post("/cost-analysis")
async def analyze_costs(request: CostAnalysisRequest):
    """
    コスト分析を実行
    """
    try:
        result = request.lotsize_result
        
        # Generate cost analysis
        analysis = {
            "total_cost_breakdown": {
                "holding_cost": result.cost_summary.holding_cost,
                "setup_cost": result.cost_summary.setup_cost,
                "total": result.cost_summary.total_cost
            },
            "cost_by_period": _analyze_costs_by_period(result),
            "cost_by_item": _analyze_costs_by_item(result),
            "utilization_analysis": _analyze_resource_utilization(result)
        }
        
        # Add optional detailed analysis
        if request.analysis_options:
            if request.analysis_options.get("breakdown_by_item", False):
                analysis["detailed_item_costs"] = _detailed_item_cost_analysis(result)
            
            if request.analysis_options.get("breakdown_by_period", False):
                analysis["detailed_period_costs"] = _detailed_period_cost_analysis(result)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"コスト分析エラー: {str(e)}")


@router.get("/service-info", response_model=ServiceInfo)
async def get_service_info():
    """
    サービス情報を取得
    """
    return ServiceInfo(
        service_name="Lot Size Optimization Service",
        version="1.0.0",
        description="Dynamic lot sizing optimization with multi-stage BOM support from 11lotsize.ipynb",
        features=[
            "Basic lot sizing optimization",
            "Multi-mode production planning", 
            "BOM hierarchy processing",
            "Resource capacity constraints",
            "Excel integration",
            "Cost analysis and visualization",
            "Rolling horizon optimization support"
        ],
        supported_solvers=["GRB", "CBC", "SCIP", "GLPK"]
    )


@router.post("/generate-demand-from-order")
async def generate_demand_from_order(
    order_data: List[Dict[str, Any]], 
    items: List[str],
    start: str = "2024-01-01",
    finish: str = "2024-01-31"
):
    """
    注文データから需要配列を生成
    """
    try:
        # Use helper function to convert orders directly
        order_df = pd.DataFrame(order_data)
        demand = _convert_orders_to_demand(order_df, pd.DataFrame({'name': items}))
        
        return {
            "items": items,
            "periods": demand.shape[1],
            "demand_matrix": demand.tolist(),
            "total_demand_by_item": demand.sum(axis=1).tolist(),
            "total_demand_by_period": demand.sum(axis=0).tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"需要生成エラー: {str(e)}")


# Helper functions

def _extract_optimization_results(model: Any, prod_df: pd.DataFrame, 
                                production_df: pd.DataFrame, T: int) -> LotsizingResult:
    """最適化結果を抽出"""
    
    # Production schedule
    production_schedule = []
    for item_idx, item_name in enumerate(prod_df['name']):
        for t in range(T):
            try:
                if hasattr(model, 'x') and (t, item_name) in model.x:
                    value = model.x[t, item_name].x if hasattr(model.x[t, item_name], 'x') else model.x[t, item_name]
                    if value > 0.01:
                        production_schedule.append(ProductionScheduleResult(
                            item=item_name,
                            period=t + 1,
                            quantity=round(value, 2)
                        ))
            except:
                continue
    
    # Inventory levels
    inventory_levels = []
    for item_idx, item_name in enumerate(prod_df['name']):
        for t in range(T):
            try:
                if hasattr(model, 'I') and (t, item_name) in model.I:
                    value = model.I[t, item_name].x if hasattr(model.I[t, item_name], 'x') else model.I[t, item_name]
                    inventory_levels.append(InventoryResult(
                        item=item_name,
                        period=t + 1,
                        level=round(value, 2)
                    ))
            except:
                continue
    
    # Setup schedule
    setup_schedule = []
    for item_idx, item_name in enumerate(prod_df['name']):
        for t in range(T):
            try:
                if hasattr(model, 'y') and (t, item_name) in model.y:
                    value = model.y[t, item_name].x if hasattr(model.y[t, item_name], 'x') else model.y[t, item_name]
                    if value > 0.5:
                        setup_schedule.append(SetupResult(
                            item=item_name,
                            period=t + 1,
                            setup=True
                        ))
            except:
                continue
    
    # Calculate costs
    total_holding_cost = 0
    total_setup_cost = 0
    
    for item_idx, item_name in enumerate(prod_df['name']):
        holding_cost = prod_df.iloc[item_idx].get('holding_cost', 1.0)
        setup_cost = prod_df.iloc[item_idx].get('setup_cost', 100.0)
        
        for t in range(T):
            try:
                # Inventory holding cost
                if hasattr(model, 'I') and (t, item_name) in model.I:
                    inventory = model.I[t, item_name].x if hasattr(model.I[t, item_name], 'x') else model.I[t, item_name]
                    total_holding_cost += holding_cost * inventory
                
                # Setup cost
                if hasattr(model, 'y') and (t, item_name) in model.y:
                    setup = model.y[t, item_name].x if hasattr(model.y[t, item_name], 'x') else model.y[t, item_name]
                    total_setup_cost += setup_cost * setup
            except:
                continue
    
    cost_summary = CostSummary(
        total_cost=total_holding_cost + total_setup_cost,
        holding_cost=total_holding_cost,
        setup_cost=total_setup_cost
    )
    
    # Get status
    status = "Optimal"
    objective_value = total_holding_cost + total_setup_cost
    
    try:
        if hasattr(model, 'status'):
            status = str(model.status)
        if hasattr(model, 'objVal'):
            objective_value = model.objVal
    except:
        pass
    
    return LotsizingResult(
        status=status,
        objective_value=objective_value,
        production_schedule=production_schedule,
        inventory_levels=inventory_levels,
        setup_schedule=setup_schedule,
        cost_summary=cost_summary
    )


def _extract_multimode_optimization_results(model: Any, prod_df: pd.DataFrame, 
                                          production_df: pd.DataFrame, T: int,
                                          modes: List[str]) -> LotsizingResult:
    """マルチモード最適化結果を抽出"""
    
    # Production schedule with modes
    production_schedule = []
    for item_idx, item_name in enumerate(prod_df['name']):
        for t in range(T):
            for mode in modes:
                try:
                    if hasattr(model, 'x') and (t, item_name, mode) in model.x:
                        value = model.x[t, item_name, mode].x if hasattr(model.x[t, item_name, mode], 'x') else model.x[t, item_name, mode]
                        if value > 0.01:
                            production_schedule.append(ProductionScheduleResult(
                                item=item_name,
                                period=t + 1,
                                quantity=round(value, 2),
                                mode=mode
                            ))
                except:
                    continue
    
    # Setup schedule with modes
    setup_schedule = []
    for item_idx, item_name in enumerate(prod_df['name']):
        for t in range(T):
            for mode in modes:
                try:
                    if hasattr(model, 'y') and (t, item_name, mode) in model.y:
                        value = model.y[t, item_name, mode].x if hasattr(model.y[t, item_name, mode], 'x') else model.y[t, item_name, mode]
                        if value > 0.5:
                            setup_schedule.append(SetupResult(
                                item=item_name,
                                period=t + 1,
                                setup=True,
                                mode=mode
                            ))
                except:
                    continue
    
    # Inventory levels (same as single mode)
    inventory_levels = []
    for item_idx, item_name in enumerate(prod_df['name']):
        for t in range(T):
            try:
                if hasattr(model, 'I') and (t, item_name) in model.I:
                    value = model.I[t, item_name].x if hasattr(model.I[t, item_name], 'x') else model.I[t, item_name]
                    inventory_levels.append(InventoryResult(
                        item=item_name,
                        period=t + 1,
                        level=round(value, 2)
                    ))
            except:
                continue
    
    # Resource utilization
    resource_utilization = []
    resources = production_df['resource'].unique() if 'resource' in production_df.columns else []
    
    for resource in resources:
        for t in range(T):
            for mode in modes:
                try:
                    # Calculate usage based on production and processing times
                    usage = 0
                    for item_name in prod_df['name']:
                        if hasattr(model, 'x') and (t, item_name, mode) in model.x:
                            prod_qty = model.x[t, item_name, mode].x if hasattr(model.x[t, item_name, mode], 'x') else model.x[t, item_name, mode]
                            
                            # Get processing time for this item-resource-mode combination
                            proc_data = production_df[
                                (production_df['name'] == item_name) & 
                                (production_df['resource'] == resource) &
                                (production_df.get('mode', mode) == mode)
                            ]
                            
                            if len(proc_data) > 0:
                                proc_time = proc_data.iloc[0].get('processing_time', 1.0)
                                usage += prod_qty * proc_time
                    
                    if usage > 0.01:
                        resource_utilization.append(ResourceUtilization(
                            resource=resource,
                            period=t + 1,
                            capacity=2400.0,  # Default capacity
                            usage=usage,
                            utilization_rate=usage / 2400.0,
                            mode=mode
                        ))
                except:
                    continue
    
    # Calculate costs (similar to single mode)
    total_holding_cost = 0
    total_setup_cost = 0
    
    for item_idx, item_name in enumerate(prod_df['name']):
        holding_cost = prod_df.iloc[item_idx].get('holding_cost', 1.0)
        setup_cost = prod_df.iloc[item_idx].get('setup_cost', 100.0)
        
        for t in range(T):
            try:
                # Inventory holding cost
                if hasattr(model, 'I') and (t, item_name) in model.I:
                    inventory = model.I[t, item_name].x if hasattr(model.I[t, item_name], 'x') else model.I[t, item_name]
                    total_holding_cost += holding_cost * inventory
                
                # Setup cost (sum over all modes)
                for mode in modes:
                    if hasattr(model, 'y') and (t, item_name, mode) in model.y:
                        setup = model.y[t, item_name, mode].x if hasattr(model.y[t, item_name, mode], 'x') else model.y[t, item_name, mode]
                        total_setup_cost += setup_cost * setup
            except:
                continue
    
    cost_summary = CostSummary(
        total_cost=total_holding_cost + total_setup_cost,
        holding_cost=total_holding_cost,
        setup_cost=total_setup_cost
    )
    
    status = "Optimal"
    objective_value = total_holding_cost + total_setup_cost
    
    try:
        if hasattr(model, 'status'):
            status = str(model.status)
        if hasattr(model, 'objVal'):
            objective_value = model.objVal
    except:
        pass
    
    return LotsizingResult(
        status=status,
        objective_value=objective_value,
        production_schedule=production_schedule,
        inventory_levels=inventory_levels,
        setup_schedule=setup_schedule,
        cost_summary=cost_summary,
        resource_utilization=resource_utilization
    )


def _convert_orders_to_demand(order_df: pd.DataFrame, prod_df: pd.DataFrame) -> np.ndarray:
    """注文データから需要マトリクスを生成"""
    if order_df is None or len(order_df) == 0:
        # Generate sample demand
        n_items = len(prod_df)
        T = 5  # Default periods
        return np.random.randint(50, 200, size=(n_items, T)).astype(float)
    
    # Get unique items and periods
    items = list(prod_df['name'])
    max_period = int(order_df['period'].max()) if 'period' in order_df.columns else 5
    
    # Initialize demand matrix
    demand = np.zeros((len(items), max_period))
    
    # Fill demand matrix
    for _, row in order_df.iterrows():
        item_name = row['item']
        period = int(row['period']) - 1  # Convert to 0-based index
        demand_qty = row['demand']
        
        if item_name in items and 0 <= period < max_period:
            item_idx = items.index(item_name)
            demand[item_idx, period] += demand_qty
    
    return demand


def _generate_cost_breakdown(result: LotsizingResult) -> Dict[str, Any]:
    """コスト内訳を生成"""
    return {
        "cost_per_period": _calculate_cost_per_period(result),
        "cost_per_item": _calculate_cost_per_item(result),
        "setup_vs_holding_ratio": result.cost_summary.setup_cost / max(result.cost_summary.holding_cost, 0.1)
    }


def _calculate_cost_per_period(result: LotsizingResult) -> Dict[int, float]:
    """期間別コストを計算"""
    period_costs = {}
    
    # Group costs by period
    for inv in result.inventory_levels:
        period = inv.period
        if period not in period_costs:
            period_costs[period] = 0
        # Assume holding cost of 1.0 per unit for this calculation
        period_costs[period] += inv.level * 1.0
    
    for setup in result.setup_schedule:
        period = setup.period
        if period not in period_costs:
            period_costs[period] = 0
        # Assume setup cost of 100.0 for this calculation
        period_costs[period] += 100.0
    
    return period_costs


def _calculate_cost_per_item(result: LotsizingResult) -> Dict[str, float]:
    """品目別コストを計算"""
    item_costs = {}
    
    # Group costs by item
    for inv in result.inventory_levels:
        item = inv.item
        if item not in item_costs:
            item_costs[item] = 0
        item_costs[item] += inv.level * 1.0  # Assume holding cost of 1.0
    
    for setup in result.setup_schedule:
        item = setup.item
        if item not in item_costs:
            item_costs[item] = 0
        item_costs[item] += 100.0  # Assume setup cost of 100.0
    
    return item_costs


def _analyze_costs_by_period(result: LotsizingResult) -> Dict[int, Dict[str, float]]:
    """期間別詳細コスト分析"""
    period_analysis = {}
    
    for inv in result.inventory_levels:
        period = inv.period
        if period not in period_analysis:
            period_analysis[period] = {"holding_cost": 0, "setup_cost": 0}
        period_analysis[period]["holding_cost"] += inv.level * 1.0
    
    for setup in result.setup_schedule:
        period = setup.period
        if period not in period_analysis:
            period_analysis[period] = {"holding_cost": 0, "setup_cost": 0}
        period_analysis[period]["setup_cost"] += 100.0
    
    return period_analysis


def _analyze_costs_by_item(result: LotsizingResult) -> Dict[str, Dict[str, float]]:
    """品目別詳細コスト分析"""
    item_analysis = {}
    
    for inv in result.inventory_levels:
        item = inv.item
        if item not in item_analysis:
            item_analysis[item] = {"holding_cost": 0, "setup_cost": 0}
        item_analysis[item]["holding_cost"] += inv.level * 1.0
    
    for setup in result.setup_schedule:
        item = setup.item
        if item not in item_analysis:
            item_analysis[item] = {"holding_cost": 0, "setup_cost": 0}
        item_analysis[item]["setup_cost"] += 100.0
    
    return item_analysis


def _analyze_resource_utilization(result: LotsizingResult) -> Dict[str, Any]:
    """資源利用状況分析"""
    if not result.resource_utilization:
        return {"message": "No resource utilization data available"}
    
    utilization_summary = {}
    
    for util in result.resource_utilization:
        resource = util.resource
        if resource not in utilization_summary:
            utilization_summary[resource] = {
                "average_utilization": 0,
                "max_utilization": 0,
                "periods_over_80_percent": 0,
                "total_periods": 0
            }
        
        utilization_summary[resource]["total_periods"] += 1
        utilization_summary[resource]["average_utilization"] += util.utilization_rate
        utilization_summary[resource]["max_utilization"] = max(
            utilization_summary[resource]["max_utilization"], 
            util.utilization_rate
        )
        
        if util.utilization_rate > 0.8:
            utilization_summary[resource]["periods_over_80_percent"] += 1
    
    # Calculate averages
    for resource, data in utilization_summary.items():
        if data["total_periods"] > 0:
            data["average_utilization"] /= data["total_periods"]
    
    return utilization_summary


def _detailed_item_cost_analysis(result: LotsizingResult) -> Dict[str, Any]:
    """詳細品目別コスト分析"""
    # Implementation for detailed item cost analysis
    return {"message": "Detailed item cost analysis not yet implemented"}


def _detailed_period_cost_analysis(result: LotsizingResult) -> Dict[str, Any]:
    """詳細期間別コスト分析"""
    # Implementation for detailed period cost analysis
    return {"message": "Detailed period cost analysis not yet implemented"}