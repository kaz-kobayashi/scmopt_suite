from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
from app.services.inventory_service import (
    eoq,
    simulate_inventory,
    optimize_qr,
    approximate_ss,
    base_stock_simulation,
    multi_echelon_optimization,
    inventory_abc_classification,
    ww,
    best_distribution,
    best_histogram,
    messa_optimization,
    newsvendor_model,
    periodic_review_optimization,
    seasonal_inventory_management,
    inventory_cost_sensitivity_analysis,
    safety_stock_allocation_dp_tabu,
    inventory_allocation_optimization,
    multi_stage_base_stock_simulation,
    network_base_stock_simulation,
    periodic_inv_opt,
    make_excel_messa,
    read_willems,
    draw_graph_for_SSA
)
from app.services.inventory_optimization_service import InventoryOptimizationService
from app.models.inventory_optimization import (
    MESSARequest,
    MESSAExcelRequest,
    MESSAResult,
    EOQRequest,
    EOQResult,
    InventorySimulationRequest,
    InventorySimulationResult,
    BaseStockSimulationRequest,
    NetworkOptimizationRequest,
    NetworkOptimizationResult,
    InventoryOptimizationServiceInfo
)

router = APIRouter()

# Initialize inventory optimization service
inventory_optimization_service = InventoryOptimizationService()

class MultiEchelonRequest(BaseModel):
    network_structure: Dict[str, Any]
    demand_data: List[List[float]]
    cost_parameters: Dict[str, float]

@router.post("/eoq")
async def calculate_eoq(
    K: float = Form(..., description="Fixed ordering cost"),
    d: float = Form(..., description="Demand rate per period"),
    h: float = Form(..., description="Holding cost per unit per period"),
    b: float = Form(0, description="Backorder cost per unit per period"),
    r: float = Form(0.1, description="Interest rate"),
    c: float = Form(1, description="Unit cost"),
    theta: float = Form(0.95, description="Service level")
):
    """
    Calculate Economic Order Quantity (EOQ)
    """
    try:
        result = eoq(K, d, h, b, r, c, theta)
        
        # Extract values from the result dictionary
        Q_optimal = result['optimal_order_quantity']
        total_cost = result['total_relevant_cost']
        
        # Calculate additional metrics
        annual_ordering_cost = K * d / Q_optimal
        annual_holding_cost = h * Q_optimal / 2
        cycle_time = Q_optimal / d
        
        return {
            "optimal_order_quantity": float(Q_optimal),
            "total_annual_cost": float(total_cost),
            "annual_ordering_cost": float(annual_ordering_cost),
            "annual_holding_cost": float(annual_holding_cost),
            "cycle_time_periods": float(cycle_time),
            "model_type": result['model_type'],
            "service_level_achieved": result['service_level_achieved'],
            "parameters": {
                "fixed_cost": K,
                "demand_rate": d,
                "holding_cost": h,
                "backorder_cost": b,
                "interest_rate": r,
                "unit_cost": c,
                "service_level": theta
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating EOQ: {str(e)}")

@router.post("/simulate")
async def simulate_inventory_policy(
    n_samples: int = Form(1000, description="Number of simulation runs"),
    n_periods: int = Form(365, description="Number of periods per run"),
    mu: float = Form(..., description="Mean demand per period"),
    sigma: float = Form(..., description="Standard deviation of demand"),
    LT: int = Form(1, description="Lead time in periods"),
    Q: float = Form(..., description="Order quantity"),
    R: float = Form(..., description="Reorder point"),
    b: float = Form(10, description="Backorder cost per unit"),
    h: float = Form(1, description="Holding cost per unit per period"),
    fc: float = Form(100, description="Fixed ordering cost"),
    S: Optional[float] = Form(None, description="Order-up-to level (for s,S policy)")
):
    """
    Simulate inventory policies (Q,R) or (s,S)
    """
    try:
        # Run simulation
        costs, inventory_levels = simulate_inventory(
            n_samples, n_periods, mu, sigma, LT, Q, R, b, h, fc, S
        )
        
        # Calculate statistics
        avg_cost = float(np.mean(costs))
        std_cost = float(np.std(costs))
        min_cost = float(np.min(costs))
        max_cost = float(np.max(costs))
        
        # Calculate confidence intervals
        confidence_95 = {
            "lower": float(np.percentile(costs, 2.5)),
            "upper": float(np.percentile(costs, 97.5))
        }
        
        # Calculate average inventory levels
        avg_inventory = float(np.mean(inventory_levels))
        
        policy_type = "(s,S)" if S is not None else "(Q,R)"
        
        return {
            "simulation_results": {
                "policy_type": policy_type,
                "average_cost_per_period": avg_cost,
                "cost_standard_deviation": std_cost,
                "cost_range": {"min": min_cost, "max": max_cost},
                "confidence_interval_95": confidence_95,
                "average_inventory_level": avg_inventory,
                "number_of_simulations": n_samples,
                "periods_per_simulation": n_periods
            },
            "parameters": {
                "mean_demand": mu,
                "demand_std": sigma,
                "lead_time": LT,
                "order_quantity": Q,
                "reorder_point": R,
                "order_up_to_level": S,
                "backorder_cost": b,
                "holding_cost": h,
                "fixed_cost": fc
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in inventory simulation: {str(e)}")

@router.post("/optimize-qr")
async def optimize_qr_policy(
    mu: float = Form(..., description="Mean demand per period"),
    sigma: float = Form(..., description="Standard deviation of demand"),
    LT: int = Form(1, description="Lead time in periods"),
    b: float = Form(10, description="Backorder cost per unit"),
    h: float = Form(1, description="Holding cost per unit per period"),
    fc: float = Form(100, description="Fixed ordering cost"),
    alpha: float = Form(0.95, description="Service level"),
    n_samples: int = Form(1000, description="Number of simulation samples"),
    n_periods: int = Form(100, description="Number of periods for optimization")
):
    """
    Optimize (Q,R) policy parameters
    """
    try:
        # Get initial Q and R values
        Q_initial = np.sqrt(2 * fc * mu / h)  # EOQ
        R_initial = mu * LT + 1.65 * sigma * np.sqrt(LT)  # Safety stock
        
        # Optimize parameters
        R_optimal, Q_optimal = optimize_qr(
            n_samples, n_periods, mu, sigma, LT, 
            Q_initial, R_initial, 1.65, b, h, fc, alpha
        )
        
        # Simulate optimized policy
        costs, _ = simulate_inventory(
            n_samples, n_periods, mu, sigma, LT, 
            Q_optimal, R_optimal, b, h, fc
        )
        
        avg_cost = float(np.mean(costs))
        
        return {
            "optimized_parameters": {
                "optimal_order_quantity": float(Q_optimal),
                "optimal_reorder_point": float(R_optimal),
                "expected_cost_per_period": avg_cost
            },
            "initial_parameters": {
                "initial_order_quantity": float(Q_initial),
                "initial_reorder_point": float(R_initial)
            },
            "input_parameters": {
                "mean_demand": mu,
                "demand_std": sigma,
                "lead_time": LT,
                "service_level": alpha,
                "backorder_cost": b,
                "holding_cost": h,
                "fixed_cost": fc
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error optimizing (Q,R) policy: {str(e)}")

@router.post("/optimize-ss") 
async def optimize_ss_policy(
    mu: float = Form(..., description="Mean demand per period"),
    sigma: float = Form(..., description="Standard deviation of demand"),
    LT: int = Form(1, description="Lead time in periods"),
    b: float = Form(10, description="Backorder cost per unit"),
    h: float = Form(1, description="Holding cost per unit per period"),
    fc: float = Form(100, description="Fixed ordering cost")
):
    """
    Optimize (s,S) policy parameters using approximation methods
    """
    try:
        # Calculate optimal (s,S) parameters
        s_optimal, S_optimal = approximate_ss(mu, sigma, LT, b, h, fc)
        
        # Calculate additional metrics
        EOQ = np.sqrt(2 * fc * mu / h)
        safety_stock = s_optimal - mu * LT
        
        return {
            "optimized_parameters": {
                "optimal_reorder_point_s": float(s_optimal),
                "optimal_order_up_to_S": float(S_optimal),
                "order_quantity": float(S_optimal - s_optimal),
                "safety_stock": float(safety_stock),
                "eoq_reference": float(EOQ)
            },
            "input_parameters": {
                "mean_demand": mu,
                "demand_std": sigma,
                "lead_time": LT,
                "backorder_cost": b,
                "holding_cost": h,
                "fixed_cost": fc
            },
            "policy_explanation": {
                "description": "Order to level S when inventory reaches s",
                "reorder_trigger": f"When inventory <= {s_optimal:.2f}",
                "order_up_to": f"Order to bring inventory to {S_optimal:.2f}"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error optimizing (s,S) policy: {str(e)}")

@router.post("/base-stock-simulation")
async def simulate_base_stock(
    file: UploadFile = File(...),
    capacity: float = Form(..., description="Production capacity per period"),
    LT: int = Form(1, description="Lead time in periods"),
    b: float = Form(10, description="Backorder cost per unit"),
    h: float = Form(1, description="Holding cost per unit per period"),
    S: float = Form(..., description="Base stock level"),
    n_samples: int = Form(100, description="Number of simulation runs")
):
    """
    Simulate periodic review base-stock policy with demand data
    """
    try:
        # Read demand file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate demand file
        if 'demand' not in demand_df.columns:
            raise HTTPException(
                status_code=400,
                detail="Demand file must contain 'demand' column"
            )
        
        demand_data = demand_df['demand'].values
        n_periods = len(demand_data)
        
        # Run simulation
        derivative, avg_cost, inventory_levels = base_stock_simulation(
            n_samples, n_periods, demand_data, capacity, LT, b, h, S
        )
        
        # Calculate statistics
        avg_inventory = float(np.mean(inventory_levels))
        service_level = float(np.mean(inventory_levels >= 0))  # Approximation
        
        return {
            "simulation_results": {
                "base_stock_level": S,
                "average_cost_per_period": float(avg_cost),
                "average_inventory_level": avg_inventory,
                "approximate_service_level": service_level,
                "number_of_periods": n_periods,
                "number_of_simulations": n_samples
            },
            "parameters": {
                "production_capacity": capacity,
                "lead_time": LT,
                "backorder_cost": b,
                "holding_cost": h,
                "base_stock_level": S
            },
            "demand_statistics": {
                "mean_demand": float(np.mean(demand_data)),
                "std_demand": float(np.std(demand_data)),
                "total_periods": len(demand_data)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in base stock simulation: {str(e)}")

@router.post("/multi-echelon")
async def optimize_multi_echelon(
    file: UploadFile = File(...),
    echelons: str = Form("plant,dc,retail"),
    service_level: float = Form(0.95),
    plant_holding_cost: float = Form(0.5),
    dc_holding_cost: float = Form(1.0),
    retail_holding_cost: float = Form(2.0),
    plant_ordering_cost: float = Form(500.0),
    dc_ordering_cost: float = Form(200.0),
    retail_ordering_cost: float = Form(50.0)
):
    """
    Multi-echelon inventory optimization
    """
    try:
        # Read demand file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate demand file
        if 'demand' not in demand_df.columns:
            raise HTTPException(
                status_code=400,
                detail="Demand file must contain 'demand' column"
            )
        
        demand_data = demand_df['demand'].values
        
        # Parse echelons
        echelon_list = [e.strip() for e in echelons.split(',')]
        
        # Network structure
        network_structure = {
            'echelons': echelon_list,
            'plant_lead_time': 7,
            'dc_lead_time': 3,
            'retail_lead_time': 1
        }
        
        # Cost parameters
        cost_parameters = {
            'service_level': service_level,
            'plant_holding_cost': plant_holding_cost,
            'dc_holding_cost': dc_holding_cost,
            'retail_holding_cost': retail_holding_cost,
            'plant_ordering_cost': plant_ordering_cost,
            'dc_ordering_cost': dc_ordering_cost,
            'retail_ordering_cost': retail_ordering_cost
        }
        
        # Perform optimization
        result = multi_echelon_optimization(demand_data, network_structure, cost_parameters)
        
        return {
            "optimization_result": result,
            "summary": {
                "total_echelons": len(echelon_list),
                "total_cost_estimate": result["total_cost_estimate"],
                "service_level": service_level
            },
            "recommendations": {
                "high_cost_echelon": max(result["echelon_policies"].items(), 
                                       key=lambda x: x[1]["cost_parameters"]["holding_cost"])[0],
                "optimization_focus": "Focus on reducing inventory at high-cost echelons"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in multi-echelon optimization: {str(e)}")

@router.post("/multi-echelon-json")
async def optimize_multi_echelon_json(request: MultiEchelonRequest):
    """
    Multi-echelon inventory optimization (JSON input)
    """
    try:
        # Convert demand data to numpy array
        demand_data = np.array(request.demand_data).flatten()
        
        # Create network structure with default values if missing
        network_structure = {
            'echelons': ['plant', 'dc', 'retail'],
            'plant_lead_time': 7,
            'dc_lead_time': 3,
            'retail_lead_time': 1,
            **request.network_structure
        }
        
        # Extract cost parameters
        cost_parameters = {
            'service_level': 0.95,
            'plant_holding_cost': 0.5,
            'dc_holding_cost': 1.0,
            'retail_holding_cost': 2.0,
            'plant_ordering_cost': 500.0,
            'dc_ordering_cost': 200.0,
            'retail_ordering_cost': 50.0,
            **request.cost_parameters
        }
        
        # Perform optimization
        result = multi_echelon_optimization(demand_data, network_structure, cost_parameters)
        
        return {
            "optimization_result": result,
            "summary": {
                "total_echelons": len(network_structure.get('echelons', ['plant', 'dc', 'retail'])),
                "total_cost_estimate": result.get("total_cost_estimate", 0),
                "service_level": cost_parameters.get('service_level', 0.95),
                "network_configuration": f"Plants: {len(request.network_structure.get('plants', []))}, DCs: {len(request.network_structure.get('distribution_centers', []))}, Retailers: {len(request.network_structure.get('retailers', []))}"
            },
            "performance_metrics": {
                "average_service_level": result.get("service_level", cost_parameters.get('service_level', 0.95)) * 100,
                "inventory_turnover": result.get("inventory_turnover", 12.0),
                "total_cost": result.get("total_cost_estimate", 0)
            },
            "recommendations": [
                "Consider consolidating inventory at distribution centers to reduce holding costs",
                "Monitor service levels at retail locations to ensure customer satisfaction",
                "Review ordering policies at each echelon to minimize total system cost"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in multi-echelon optimization: {str(e)}")

@router.post("/inventory-abc")
async def inventory_abc_analysis(
    file: UploadFile = File(...),
    cost_data: str = Form('{"A": 10.0, "B": 5.0, "C": 2.0}')
):
    """
    ABC classification for inventory management
    """
    try:
        # Read demand file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate demand file
        required_cols = ['prod', 'demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in demand file: {missing_cols}"
            )
        
        # Parse cost data
        try:
            import json
            inventory_costs = json.loads(cost_data)
        except:
            # Default costs if parsing fails
            inventory_costs = {prod: 1.0 for prod in demand_df['prod'].unique()}
        
        # Perform ABC classification
        result = inventory_abc_classification(demand_df, inventory_costs)
        
        return {
            "classification_result": result,
            "inventory_strategy": {
                "A_items": "High-value items requiring tight inventory control",
                "B_items": "Moderate-value items with standard inventory policies", 
                "C_items": "Low-value items suitable for bulk ordering"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in inventory ABC analysis: {str(e)}")

@router.post("/wagner-whitin")
async def wagner_whitin_optimization(
    demand: List[float] = Body(..., description="Multi-period demand list"),
    fc: float = Body(100.0, description="Fixed ordering cost"),
    vc: float = Body(0.0, description="Variable ordering cost"),
    h: float = Body(5.0, description="Holding cost per unit per period")
):
    """
    Wagner-Whitin dynamic lot sizing algorithm
    """
    try:
        optimal_cost, order_quantities = ww(demand, fc, vc, h)
        
        # Calculate performance metrics
        total_demand = sum(demand)
        total_orders = sum(1 for q in order_quantities if q > 0)
        avg_inventory = sum(order_quantities) / len(order_quantities) / 2
        
        return {
            "optimization_result": {
                "optimal_total_cost": float(optimal_cost),
                "order_quantities": [float(q) for q in order_quantities],
                "total_demand": total_demand,
                "total_ordering_periods": total_orders,
                "average_inventory_level": avg_inventory
            },
            "performance_metrics": {
                "cost_per_unit": optimal_cost / total_demand if total_demand > 0 else 0,
                "order_frequency": total_orders / len(demand) if len(demand) > 0 else 0,
                "inventory_turnover": total_demand / avg_inventory if avg_inventory > 0 else 0
            },
            "input_parameters": {
                "demand_periods": len(demand),
                "fixed_cost": fc,
                "variable_cost": vc,
                "holding_cost": h
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in Wagner-Whitin optimization: {str(e)}")

@router.post("/demand-distribution-fitting")
async def demand_distribution_analysis(
    file: UploadFile = File(...),
    demand_col: str = Form("demand", description="Demand column name"),
    analysis_type: str = Form("continuous", description="continuous or histogram")
):
    """
    Demand distribution fitting and analysis
    """
    try:
        # Read demand file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate demand file
        if demand_col not in demand_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{demand_col}' not found in file"
            )
        
        demand_data = demand_df[demand_col].values
        
        if analysis_type == "continuous":
            # Continuous distribution fitting
            fig, frozen_dist, best_fit_name, best_fit_params = best_distribution(demand_data)
            
            return {
                "analysis_type": "continuous_distribution",
                "best_fit_distribution": {
                    "distribution_name": best_fit_name,
                    "parameters": [float(p) for p in best_fit_params],
                    "mean": float(frozen_dist.mean()),
                    "std": float(frozen_dist.std()),
                    "variance": float(frozen_dist.var())
                },
                "data_statistics": {
                    "sample_mean": float(np.mean(demand_data)),
                    "sample_std": float(np.std(demand_data)),
                    "sample_size": len(demand_data),
                    "min_value": float(np.min(demand_data)),
                    "max_value": float(np.max(demand_data))
                },
                "histogram_data": {
                    "bins": fig['data_histogram'][1].tolist(),
                    "frequencies": fig['data_histogram'][0].tolist()
                }
            }
        else:
            # Histogram-based distribution
            fig, hist_dist = best_histogram(demand_data)
            
            return {
                "analysis_type": "histogram_distribution", 
                "histogram_distribution": {
                    "mean": float(hist_dist.mean()),
                    "std": float(hist_dist.std()),
                    "variance": float(hist_dist.var())
                },
                "data_statistics": {
                    "sample_mean": float(np.mean(demand_data)),
                    "sample_std": float(np.std(demand_data)),
                    "sample_size": len(demand_data),
                    "min_value": float(np.min(demand_data)),
                    "max_value": float(np.max(demand_data))
                },
                "histogram_data": {
                    "bins": fig['data_histogram'][1].tolist(),
                    "frequencies": fig['data_histogram'][0].tolist()
                }
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in demand distribution analysis: {str(e)}")

class MESSARequest(BaseModel):
    demand_data: List[float]
    network_structure: Dict[str, Any]
    cost_parameters: Dict[str, float]

@router.post("/messa-optimization")
async def messa_system_optimization(request: MESSARequest):
    """
    MESSA (Multi-Echelon Serial Stock Allocation) system optimization
    """
    try:
        demand_array = np.array(request.demand_data)
        
        # Perform MESSA optimization
        result = messa_optimization(
            demand_array, 
            request.network_structure,
            request.cost_parameters
        )
        
        return {
            "messa_optimization_result": result,
            "summary": {
                "total_system_cost": result["total_system_cost"],
                "optimization_method": result["optimization_method"],
                "num_echelons": len(result["echelon_policies"]),
                "total_safety_stock": result["performance_metrics"]["total_safety_stock"]
            },
            "key_insights": {
                "critical_echelon": max(result["echelon_policies"].items(), 
                                      key=lambda x: x[1]["echelon_cost"])[0],
                "service_level_achieved": result["echelon_policies"][list(result["echelon_policies"].keys())[0]]["service_level"],
                "inventory_investment": result["performance_metrics"]["total_safety_stock"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in MESSA optimization: {str(e)}")

@router.post("/newsvendor-model")
async def newsvendor_optimization(
    demand_mean: float = Form(..., description="Mean demand"),
    demand_std: float = Form(..., description="Standard deviation of demand"),
    selling_price: float = Form(..., description="Selling price per unit"),
    purchase_cost: float = Form(..., description="Purchase cost per unit"),
    salvage_value: float = Form(0.0, description="Salvage value for unsold units"),
    discrete: bool = Form(False, description="Use discrete version")
):
    """
    Newsvendor model for optimal ordering under demand uncertainty
    """
    try:
        result = newsvendor_model(
            demand_mean, demand_std, selling_price, 
            purchase_cost, salvage_value, discrete
        )
        
        return {
            "newsvendor_result": result,
            "decision_analysis": {
                "optimal_order_quantity": result["optimal_order_quantity"],
                "expected_profit": result["expected_profit"],
                "critical_ratio": result["critical_ratio"],
                "model_type": result["model_type"]
            },
            "risk_analysis": {
                "expected_sales": result["expected_sales"],
                "expected_leftover": result["expected_leftover"],
                "expected_shortage": result["expected_shortage"],
                "overage_cost": result["overage_cost"],
                "underage_cost": result["underage_cost"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in newsvendor optimization: {str(e)}")

@router.post("/periodic-review-optimization")
async def optimize_periodic_review(
    demand_mean: float = Form(..., description="Mean demand per period"),
    demand_std: float = Form(..., description="Standard deviation of demand"),
    lead_time: int = Form(1, description="Lead time in periods"),
    review_period: int = Form(1, description="Review period length"),
    holding_cost: float = Form(1.0, description="Holding cost per unit per period"),
    shortage_cost: float = Form(10.0, description="Shortage cost per unit per period"),
    ordering_cost: float = Form(0.0, description="Fixed ordering cost")
):
    """
    Periodic review inventory policy optimization
    """
    try:
        result = periodic_review_optimization(
            demand_mean, demand_std, lead_time, review_period,
            holding_cost, shortage_cost, ordering_cost
        )
        
        return {
            "periodic_review_result": result,
            "policy_summary": {
                "policy_type": result["policy_type"],
                "reorder_point_s": result["s_parameter"],
                "order_up_to_S": result["S_parameter"],
                "safety_stock": result["safety_stock"],
                "service_level": result["service_level"]
            },
            "cost_analysis": {
                "total_expected_cost": result["total_expected_cost"],
                "expected_holding_cost": result["expected_holding_cost"],
                "expected_ordering_cost": result["expected_ordering_cost"],
                "shortage_probability": result["shortage_probability"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in periodic review optimization: {str(e)}")

@router.post("/seasonal-inventory-management")
async def seasonal_inventory_analysis(
    file: UploadFile = File(...),
    demand_col: str = Form("demand", description="Demand column name"),
    seasonality_periods: int = Form(12, description="Number of periods in seasonal cycle"),
    forecast_periods: int = Form(12, description="Number of periods to forecast")
):
    """
    Seasonal inventory management with time series decomposition
    """
    try:
        # Read demand file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate demand file
        if demand_col not in demand_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{demand_col}' not found in file"
            )
        
        historical_demand = demand_df[demand_col].values
        
        result = seasonal_inventory_management(
            historical_demand, seasonality_periods, forecast_periods
        )
        
        return {
            "seasonal_analysis": result,
            "planning_insights": {
                "trend_direction": result["seasonality_metrics"]["trend_direction"],
                "seasonal_strength": result["historical_analysis"]["seasonal_strength"],
                "forecast_reliability": "high" if result["seasonality_metrics"]["seasonal_variance_ratio"] > 0.1 else "moderate"
            },
            "inventory_strategy": {
                "base_safety_stock": result["inventory_recommendations"]["base_safety_stock"],
                "seasonal_adjustments": len(result["inventory_recommendations"]["seasonal_safety_stock"]),
                "review_frequency": result["inventory_recommendations"]["recommended_review_frequency"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in seasonal inventory analysis: {str(e)}")

class SensitivityRequest(BaseModel):
    base_parameters: Dict[str, float]
    parameter_ranges: Dict[str, List[float]]  # [min, max] for each parameter
    n_points: int = 10

@router.post("/inventory-sensitivity-analysis")
async def inventory_sensitivity_analysis(request: SensitivityRequest):
    """
    Inventory cost and sensitivity analysis
    """
    try:
        # Convert parameter ranges to tuples
        parameter_ranges_tuples = {
            param: tuple(range_vals) for param, range_vals in request.parameter_ranges.items()
        }
        
        result = inventory_cost_sensitivity_analysis(
            request.base_parameters,
            parameter_ranges_tuples,
            request.n_points
        )
        
        return {
            "sensitivity_analysis": result,
            "base_case": {
                "parameters": request.base_parameters,
                "base_cost": result.get("base_cost", 0)
            },
            "analysis_insights": {
                "most_sensitive_parameter": max(result.get("sensitivity_results", {}).items(), 
                                             key=lambda x: max(x[1]["costs"]) - min(x[1]["costs"]))[0] if result.get("sensitivity_results") else "none",
                "analysis_points": request.n_points,
                "parameters_analyzed": len(request.parameter_ranges)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in sensitivity analysis: {str(e)}")

class SafetyStockRequest(BaseModel):
    echelons: List[str]
    demand_data: List[float]
    holding_costs: List[float]
    service_level_target: float = 0.95
    total_safety_stock_budget: Optional[float] = None
    dp_iterations: int = 100
    tabu_iterations: int = 50
    tabu_list_size: int = 10

@router.post("/safety-stock-allocation")
async def safety_stock_allocation_optimization(request: SafetyStockRequest):
    """
    Safety stock allocation optimization using Dynamic Programming + Tabu Search
    """
    try:
        demand_array = np.array(request.demand_data)
        
        result = safety_stock_allocation_dp_tabu(
            request.echelons,
            demand_array,
            request.holding_costs,
            request.service_level_target,
            request.total_safety_stock_budget,
            request.dp_iterations,
            request.tabu_iterations,
            request.tabu_list_size
        )
        
        return {
            "safety_stock_allocation": result,
            "optimization_summary": {
                "method": result["optimization_details"]["optimization_method"],
                "total_echelons": len(request.echelons),
                "service_level_target": request.service_level_target,
                "improvement_achieved": result["optimization_details"]["improvement_from_dp"]
            },
            "key_metrics": {
                "total_safety_stock": result["system_performance"]["total_safety_stock_allocated"],
                "total_cost": result["system_performance"]["total_holding_cost"],
                "system_service_level": result["system_performance"]["system_service_level"],
                "budget_utilization": result["system_performance"]["budget_utilization"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in safety stock allocation: {str(e)}")

class InventoryAllocationRequest(BaseModel):
    demand_matrix: List[List[float]]  # periods x locations
    cost_matrix: List[List[float]]    # locations x [holding_cost, shortage_cost]
    capacity_constraints: List[float]
    service_level_requirements: List[float]

@router.post("/inventory-allocation-optimization")
async def inventory_allocation_multi_stage(request: InventoryAllocationRequest):
    """
    Multi-stage inventory allocation optimization
    """
    try:
        demand_matrix = np.array(request.demand_matrix)
        cost_matrix = np.array(request.cost_matrix)
        
        result = inventory_allocation_optimization(
            demand_matrix,
            cost_matrix,
            request.capacity_constraints,
            request.service_level_requirements
        )
        
        return {
            "allocation_optimization": result,
            "summary": {
                "total_locations": result["optimization_summary"]["total_locations"],
                "periods_analyzed": result["optimization_summary"]["periods_analyzed"],
                "allocation_method": result["optimization_summary"]["allocation_method"]
            },
            "performance": {
                "total_allocated": result["system_performance"]["total_allocated_inventory"],
                "total_cost": result["system_performance"]["total_system_cost"],
                "average_service_level": result["system_performance"]["average_service_level"],
                "capacity_utilization": result["system_performance"]["total_capacity_utilization"]
            },
            "insights": {
                "bottleneck_location": result["recommendations"]["bottleneck_location"],
                "highest_cost_location": result["recommendations"]["highest_cost_location"],
                "service_level_issues": result["recommendations"]["service_level_issues"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in inventory allocation optimization: {str(e)}")

class MultiStageSimulationRequest(BaseModel):
    network_structure: Dict[str, Any]
    base_stock_levels: Dict[str, float]
    demand_data: List[float]
    n_periods: int = 365
    n_simulations: int = 100

@router.post("/multi-stage-simulation")
async def multi_stage_simulation_endpoint(request: MultiStageSimulationRequest):
    """
    Multi-stage base stock policy simulation
    """
    try:
        demand_array = np.array(request.demand_data)
        
        result = multi_stage_base_stock_simulation(
            request.network_structure,
            request.base_stock_levels,
            demand_array,
            request.n_periods,
            request.n_simulations
        )
        
        return {
            "multi_stage_simulation": result,
            "simulation_summary": {
                "simulation_method": result["simulation_method"],
                "total_stages": result["system_metrics"]["total_stages"],
                "average_cost": result["performance_metrics"]["average_cost_per_period"],
                "simulations_completed": result["system_metrics"]["total_simulations_completed"]
            },
            "stage_performance": {
                stage: {
                    "average_inventory": metrics["average_inventory"],
                    "service_level": metrics["average_service_level"],
                    "base_stock_setting": metrics["base_stock_level"]
                }
                for stage, metrics in result["stage_metrics"].items()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in multi-stage simulation: {str(e)}")

class NetworkSimulationRequest(BaseModel):
    network_data: Dict[str, Any]
    base_stock_policies: Dict[str, float] 
    demand_data: List[float]
    n_periods: int = 365
    n_simulations: int = 100

@router.post("/network-simulation")
async def network_simulation_endpoint(request: NetworkSimulationRequest):
    """
    Network-wide base stock policy simulation
    """
    try:
        demand_array = np.array(request.demand_data)
        
        result = network_base_stock_simulation(
            request.network_data,
            demand_array,
            request.base_stock_policies,
            request.n_periods,
            request.n_simulations
        )
        
        return {
            "network_simulation": result,
            "network_summary": {
                "simulation_method": result["simulation_method"],
                "network_nodes": result["network_topology"]["total_nodes"],
                "network_arcs": result["network_topology"]["total_arcs"],
                "average_system_cost": result["performance_metrics"]["average_system_cost"],
                "simulation_horizon": result["network_flow_summary"]["simulation_horizon"]
            },
            "node_performance": {
                node: {
                    "average_inventory": metrics["average_inventory"], 
                    "service_level": metrics["average_service_level"],
                    "inventory_turnover": metrics["inventory_turnover"]
                }
                for node, metrics in result["node_metrics"].items()
            },
            "optimization_insights": {
                "total_nodes_analyzed": len(result["node_metrics"]),
                "network_complexity_score": result["network_topology"]["total_arcs"] / max(1, result["network_topology"]["total_nodes"]),
                "cost_range": result["performance_metrics"]["cost_per_period_range"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in network simulation: {str(e)}")

@router.post("/advanced-eoq")
async def advanced_eoq_optimization(
    K: float = Form(..., description="Fixed ordering cost"),
    d: float = Form(..., description="Demand rate per period"),
    h: float = Form(..., description="Holding cost per unit per period"),
    b: float = Form(0, description="Backorder cost per unit per period"),
    r: float = Form(0.1, description="Interest rate"),
    c: float = Form(1, description="Unit cost"),
    theta: float = Form(0.95, description="Service level"),
    discount_schedule: str = Form(None, description="Quantity discount schedule as JSON"),
    discount_type: str = Form("incremental", description="Discount type: incremental or all_units"),
    allow_backorder: bool = Form(False, description="Allow backorders")
):
    """
    Advanced EOQ model with quantity discounts and backorder support
    """
    try:
        # Parse discount schedule if provided
        discount = None
        if discount_schedule:
            import json
            try:
                discount_data = json.loads(discount_schedule)
                discount = [(float(bp), float(price)) for bp, price in discount_data]
            except:
                discount = None
        
        # Use the enhanced EOQ function
        result = eoq(K, d, h, b, r, c, theta, discount, discount_type, allow_backorder)
        
        return {
            "advanced_eoq_result": result,
            "optimization_summary": {
                "model_type": result["model_type"],
                "optimal_quantity": result["optimal_order_quantity"],
                "total_cost": result["total_relevant_cost"],
                "has_discounts": discount is not None,
                "allows_backorders": allow_backorder
            },
            "cost_breakdown": {
                "ordering_cost": result["annual_ordering_cost"],
                "holding_cost": result["annual_holding_cost"],
                "purchase_cost": result.get("annual_purchase_cost", 0),
                "service_level_achieved": result["service_level_achieved"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in advanced EOQ optimization: {str(e)}")

class PeriodicOptimizationRequest(BaseModel):
    demand_data: List[float]
    cost_parameters: Dict[str, float]
    optimization_params: Optional[Dict[str, Any]] = None
    use_adam_optimizer: bool = True

@router.post("/periodic-optimization")
async def periodic_optimization_endpoint(request: PeriodicOptimizationRequest):
    """
    Periodic review inventory optimization with Adam optimizer support
    """
    try:
        demand_array = np.array(request.demand_data)
        
        result = periodic_inv_opt(
            demand_array,
            request.cost_parameters,
            request.optimization_params,
            request.use_adam_optimizer
        )
        
        return {
            "periodic_optimization": result,
            "optimization_summary": {
                "method": result["optimization_method"],
                "reorder_point": result["final_policy"]["reorder_point_s"],
                "order_up_to_level": result["final_policy"]["order_up_to_level_S"],
                "total_cost": result["cost_analysis"]["total_expected_cost"],
                "service_level": result["performance_metrics"]["cycle_service_level"]
            },
            "demand_insights": {
                "mean_demand": result["demand_statistics"]["mean"],
                "demand_variability": result["demand_statistics"]["cv"],
                "sample_periods": result["demand_statistics"]["sample_size"]
            },
            "policy_performance": {
                "safety_stock": result["final_policy"]["safety_stock"],
                "inventory_turnover": result["performance_metrics"]["inventory_turnover"],
                "cost_per_unit": result["cost_analysis"]["cost_per_unit_demand"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in periodic optimization: {str(e)}")

class ExcelGenerationRequest(BaseModel):
    messa_results: Dict[str, Any]
    output_filename: str = "messa_optimization_results.xlsx"
    include_charts: bool = True

@router.post("/generate-excel-messa")
async def generate_excel_messa_endpoint(request: ExcelGenerationRequest):
    """
    Generate Excel template with MESSA optimization results
    """
    try:
        result = make_excel_messa(
            request.messa_results,
            request.output_filename,
            request.include_charts
        )
        
        return {
            "excel_generation": result,
            "file_info": {
                "filename": result["filename"],
                "file_size": result.get("file_size_bytes", 0),
                "sheets_count": len(result.get("sheets_created", [])),
                "generation_status": result["excel_generation_status"]
            },
            "download_ready": result["excel_generation_status"] == "success",
            "excel_features": {
                "charts_included": request.include_charts,
                "sheets_created": result.get("sheets_created", []),
                "optimization_summary": result.get("summary_stats", {})
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating Excel report: {str(e)}")

@router.post("/load-willems-benchmark")
async def load_willems_benchmark_endpoint(
    problem_instance: str = Form("default", description="Problem instance to load"),
    benchmark_file: Optional[UploadFile] = File(None, description="Custom benchmark file")
):
    """
    Load Willems benchmark problems for multi-echelon inventory optimization
    """
    try:
        benchmark_data = None
        benchmark_file_path = None
        
        # If benchmark file is uploaded, process it
        if benchmark_file and benchmark_file.filename:
            import json
            contents = await benchmark_file.read()
            try:
                benchmark_data = json.loads(contents.decode('utf-8'))
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON format in benchmark file"
                )
        
        result = read_willems(
            benchmark_file_path=benchmark_file_path,
            benchmark_data=benchmark_data,
            problem_instance=problem_instance
        )
        
        # Check if there was an error
        if result.get('status') == 'error' or result.get('benchmark_status') != 'loaded_successfully':
            raise HTTPException(
                status_code=400,
                detail=result.get('error_message', 'Unknown error loading benchmark')
            )
        
        return {
            "benchmark_loading": result,
            "problem_summary": {
                "instance": result["problem_instance"],
                "source": result["benchmark_source"],
                "network_size": f"{result['network_analysis']['total_nodes']} nodes, {result['network_analysis']['total_arcs']} arcs",
                "complexity": result["benchmark_metadata"]["problem_complexity"],
                "optimization_ready": all(result["validation_checks"].values())
            },
            "optimization_data": result["optimization_data"],
            "recommended_next_steps": [
                "Run MESSA optimization on the loaded benchmark",
                "Compare different algorithm performance",
                "Visualize the network structure",
                "Generate Excel reports for results"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading Willems benchmark: {str(e)}")

class NetworkVisualizationRequest(BaseModel):
    network_data: Dict[str, Any]
    safety_stock_allocation: Dict[str, float]
    optimization_results: Optional[Dict[str, Any]] = None
    output_filename: str = "ssa_network_visualization.html"
    layout_algorithm: str = "spring"

@router.post("/visualize-network-ssa")
async def visualize_network_ssa_endpoint(request: NetworkVisualizationRequest):
    """
    Generate interactive network visualization for Safety Stock Allocation
    """
    try:
        result = draw_graph_for_SSA(
            request.network_data,
            request.safety_stock_allocation,
            request.optimization_results,
            request.output_filename,
            request.layout_algorithm
        )
        
        # Check if visualization was successful
        if result.get('visualization_status') != 'success':
            raise HTTPException(
                status_code=400,
                detail=result.get('error_message', 'Visualization generation failed')
            )
        
        return {
            "network_visualization": result,
            "visualization_summary": {
                "status": result["visualization_status"],
                "filename": result["filename"],
                "nodes_count": result["network_statistics"]["total_nodes"],
                "arcs_count": result["network_statistics"]["total_arcs"],
                "total_allocation": result["allocation_statistics"]["total_allocation"],
                "layout_used": request.layout_algorithm
            },
            "optimization_insights": result["optimization_insights"],
            "interactive_features": result["visualization_properties"]["interactive_features"],
            "download_info": result["download_info"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating network visualization: {str(e)}")

@router.get("/available-benchmark-instances")
async def get_available_benchmark_instances():
    """
    Get list of available Willems benchmark problem instances
    """
    try:
        # Load default benchmark data to get available instances
        result = read_willems(problem_instance="default")  # This will load defaults
        
        # Extract available instances (fallback to known instances)
        available_instances = [
            {
                "name": "default",
                "description": "標準的な5ノードサプライチェーンネットワーク",
                "complexity": "medium",
                "nodes": 5,
                "recommended_for": "基本的なアルゴリズムテストと学習"
            },
            {
                "name": "small_network",
                "description": "シンプルな2ノードネットワーク",
                "complexity": "low",
                "nodes": 2,
                "recommended_for": "アルゴリズム検証と概念理解"
            },
            {
                "name": "complex_network",
                "description": "複雑な7ノードマルチ階層ネットワーク",
                "complexity": "high", 
                "nodes": 7,
                "recommended_for": "高度な最適化アルゴリズムの評価"
            }
        ]
        
        return {
            "available_instances": available_instances,
            "total_instances": len(available_instances),
            "usage_guide": {
                "beginner": "small_network",
                "intermediate": "default",
                "advanced": "complex_network"
            },
            "supported_features": [
                "Multi-echelon inventory optimization",
                "Stochastic demand modeling",
                "Service level constraints",
                "Cost minimization objectives"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving benchmark instances: {str(e)}")


# New MESSA and Advanced Inventory Optimization Endpoints

@router.post("/messa-advanced", response_model=MESSAResult)
async def messa_advanced_optimization(request: MESSARequest):
    """
    Advanced MESSA (MEta Safety Stock Allocation) optimization
    Exact implementation from 03inventory.ipynb notebook
    """
    try:
        # Prepare data for MESSA optimization
        network_df = pd.DataFrame([item.dict() for item in request.network_data])
        demand_df = pd.DataFrame([item.dict() for item in request.demand_data])
        cost_df = pd.DataFrame([item.dict() for item in request.cost_data])
        
        # Create temporary data structure
        messa_data = {
            'network': network_df,
            'demand': demand_df,
            'cost': cost_df,
            'messa_master': network_df.merge(demand_df, on='item').merge(cost_df, on='item')
        }
        
        # Prepare optimization parameters
        optimization_params = inventory_optimization_service.prepare_opt_for_messa(
            messa_data, request.optimization_options.dict()
        )
        
        # Run MESSA optimization
        optimization_results = inventory_optimization_service.run_messa_optimization(optimization_params)
        
        # Convert results to response model
        safety_stock_results = []
        for item in optimization_results['safety_stock_levels']:
            safety_stock_results.append({
                'item': item,
                'safety_stock_level': optimization_results['safety_stock_levels'][item],
                'holding_cost': optimization_results['cost_breakdown']['holding_costs_by_item'][item],
                'target_service_level': optimization_results['target_service_levels'][item],
                'achieved_service_level': optimization_results['achieved_service_levels'][item]
            })
        
        cost_breakdown = {
            'total_holding_cost': optimization_results['total_holding_cost'],
            'holding_costs_by_item': optimization_results['cost_breakdown']['holding_costs_by_item'],
            'cost_percentages': {
                item: (cost / optimization_results['total_holding_cost'] * 100) if optimization_results['total_holding_cost'] > 0 else 0
                for item, cost in optimization_results['cost_breakdown']['holding_costs_by_item'].items()
            }
        }
        
        return MESSAResult(
            status=optimization_results['status'],
            objective_value=optimization_results['objective_value'],
            safety_stock_results=safety_stock_results,
            total_safety_stock=optimization_results['total_safety_stock'],
            cost_breakdown=cost_breakdown
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MESSA optimization error: {str(e)}")


@router.post("/messa-excel-advanced")
async def messa_excel_advanced_optimization(
    file: UploadFile = File(...),
    network_sheet: str = Form("network"),
    demand_sheet: str = Form("demand"),
    cost_sheet: str = Form("cost"),
    solver: str = Form("CBC"),
    max_time: int = Form(300),
    output_filename: Optional[str] = Form(None)
):
    """
    Advanced MESSA optimization from Excel file
    Exact implementation from 03inventory.ipynb notebook
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_filename = temp_file.name
        
        try:
            # Prepare data from Excel
            optimization_options = {
                'solver': solver,
                'max_time': max_time,
                'gap_tolerance': 0.01,
                'service_level_constraint': 0.95
            }
            
            messa_data = inventory_optimization_service.prepare_df_for_messa(
                temp_filename, network_sheet, demand_sheet, cost_sheet
            )
            
            optimization_params = inventory_optimization_service.prepare_opt_for_messa(
                messa_data, optimization_options
            )
            
            # Run MESSA optimization
            optimization_results = inventory_optimization_service.run_messa_optimization(optimization_params)
            
            # Generate Excel output
            if output_filename is None:
                output_filename = f"messa_advanced_results_{file.filename}"
            
            output_path = inventory_optimization_service.messa_for_excel(
                optimization_results, output_filename
            )
            
            from fastapi.responses import FileResponse
            return FileResponse(
                path=output_path,
                filename=output_filename,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        finally:
            # Clean up temporary file
            os.unlink(temp_filename)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Excel MESSA optimization error: {str(e)}")


@router.post("/eoq-advanced", response_model=EOQResult)
async def eoq_advanced_optimization(request: EOQRequest):
    """
    Advanced EOQ optimization with all variants
    Exact implementation from 03inventory.ipynb notebook
    """
    try:
        # Import the existing eoq function from inventory_service
        from app.services.inventory_service import eoq
        
        # Convert discount schedule if provided
        discount = None
        if request.discount_schedule:
            discount = [(bp, price) for bp, price in request.discount_schedule]
        
        # Run EOQ optimization
        result = eoq(
            K=request.fixed_cost,
            d=request.demand_rate,
            h=request.holding_cost,
            b=request.backorder_cost,
            r=request.interest_rate,
            c=request.unit_cost,
            theta=request.service_level,
            discount=discount,
            discount_type=request.discount_type,
            allow_backorder=request.allow_backorder
        )
        
        return EOQResult(
            optimization_model_type=result['model_type'],
            optimal_order_quantity=result['optimal_order_quantity'],
            optimal_reorder_point=result.get('optimal_reorder_point'),
            total_cost=result['total_cost'],
            holding_cost=result['holding_cost'],
            ordering_cost=result['ordering_cost'],
            shortage_cost=result.get('shortage_cost'),
            order_frequency=result['order_frequency'],
            cycle_time=result['cycle_time'],
            parameters=result['parameters']
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Advanced EOQ optimization error: {str(e)}")


@router.post("/inventory-simulation-advanced", response_model=InventorySimulationResult)
async def inventory_simulation_advanced(request: InventorySimulationRequest):
    """
    Advanced inventory policy simulation
    Exact implementation from 03inventory.ipynb notebook
    """
    try:
        # Import simulation function from inventory_service
        from app.services.inventory_service import simulate_inventory
        
        # Run simulation
        costs, inventory_levels = simulate_inventory(
            n_samples=request.n_samples,
            n_periods=request.n_periods,
            mu=request.mean_demand,
            sigma=request.demand_std,
            LT=request.lead_time,
            Q=request.order_quantity,
            R=request.reorder_point,
            b=request.backorder_cost,
            h=request.holding_cost,
            fc=request.fixed_cost,
            S=request.order_up_to_level
        )
        
        # Calculate performance metrics
        average_cost = float(np.mean(costs))
        cost_std = float(np.std(costs))
        average_inventory = float(np.mean(inventory_levels))
        
        # Calculate service level (approximation)
        stockouts = np.sum(inventory_levels <= 0)
        total_periods = len(inventory_levels) * request.n_samples
        stockout_frequency = stockouts / total_periods if total_periods > 0 else 0
        service_level_achieved = 1.0 - stockout_frequency
        
        # Estimate total orders
        total_orders = int(request.n_periods * request.demand_rate / request.order_quantity)
        
        return InventorySimulationResult(
            average_cost=average_cost,
            cost_std=cost_std,
            inventory_history=inventory_levels.tolist()[:100],  # Limit to first 100 for API response
            service_level_achieved=service_level_achieved,
            average_inventory=average_inventory,
            stockout_frequency=stockout_frequency,
            total_orders=total_orders
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Advanced inventory simulation error: {str(e)}")


@router.post("/base-stock-simulation-advanced")
async def base_stock_simulation_advanced(request: BaseStockSimulationRequest):
    """
    Advanced base stock policy simulation
    Exact implementation from 03inventory.ipynb notebook
    """
    try:
        # Import simulation function from inventory_service
        from app.services.inventory_service import base_stock_simulation
        
        # Convert demand data to numpy array
        demand_array = np.array(request.demand_data)
        
        # Run base stock simulation
        avg_cost, cost_std, inventory_hist = base_stock_simulation(
            n_samples=request.n_samples,
            n_periods=request.n_periods,
            demand=demand_array,
            capacity=request.capacity,
            LT=request.lead_time,
            b=request.backorder_cost,
            h=request.holding_cost,
            S=request.base_stock_level
        )
        
        return {
            "average_cost_per_period": float(avg_cost),
            "cost_standard_deviation": float(cost_std),
            "inventory_trajectory": inventory_hist.tolist()[:100],  # Limit for API response
            "base_stock_level": request.base_stock_level,
            "capacity_utilization": request.capacity,
            "simulation_parameters": {
                "samples": request.n_samples,
                "periods": request.n_periods,
                "lead_time": request.lead_time
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base stock simulation error: {str(e)}")


@router.post("/network-optimization-advanced", response_model=NetworkOptimizationResult)
async def network_optimization_advanced(request: NetworkOptimizationRequest):
    """
    Advanced network-based inventory optimization
    Exact implementation from 03inventory.ipynb notebook
    """
    try:
        # Import network simulation function from inventory_service
        from app.services.inventory_service import network_base_stock_simulation
        
        # Convert demand data to numpy array
        demand_array = np.array(request.demand_data)
        
        # Run network optimization
        results = network_base_stock_simulation(
            network_data=request.network_structure,
            demand_data=demand_array,
            base_stock_policies=request.base_stock_levels,
            n_periods=request.n_periods,
            n_simulations=request.n_simulations
        )
        
        return NetworkOptimizationResult(
            total_cost=results['total_cost'],
            cost_by_node=results['cost_by_node'],
            inventory_levels=results['inventory_levels'],
            service_levels=results['service_levels'],
            fill_rates=results['fill_rates']
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Network optimization error: {str(e)}")


@router.get("/inventory-optimization-service-info", response_model=InventoryOptimizationServiceInfo)
async def get_inventory_optimization_service_info():
    """
    Get information about the advanced inventory optimization service
    """
    return InventoryOptimizationServiceInfo(
        service_name="Advanced Inventory Optimization Service",
        version="1.0.0",
        description="Complete inventory optimization suite from 03inventory.ipynb with MESSA, advanced EOQ, and multi-echelon optimization",
        features=[
            "MESSA (MEta Safety Stock Allocation) system",
            "Advanced EOQ models with backorders and quantity discounts",
            "Comprehensive inventory policy simulation",
            "Multi-echelon inventory optimization", 
            "Network-based inventory optimization",
            "Excel integration for MESSA",
            "Base stock policy optimization",
            "Service level constraint optimization"
        ],
        supported_models=[
            "Basic EOQ", "EOQ with backorders", "EOQ with quantity discounts",
            "(Q,R) inventory policy", "(s,S) inventory policy", 
            "Base stock policy", "Multi-stage base stock",
            "Network base stock", "MESSA safety stock allocation"
        ]
    )


@router.post("/generate-messa-template")
async def generate_messa_excel_template():
    """
    Generate MESSA Excel template for input data
    """
    try:
        import tempfile
        
        # Generate template
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            template_path = inventory_optimization_service.generate_messa_excel_template(temp_file.name)
            
            from fastapi.responses import FileResponse
            return FileResponse(
                path=template_path,
                filename="messa_template.xlsx",
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Template generation error: {str(e)}")