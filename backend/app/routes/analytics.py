from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
import io
import json
from app.utils import convert_to_json_safe
from app.services.abc_service import (
    demand_tree_map,
    abc_analysis, 
    risk_pooling_analysis,
    pareto_analysis,
    inventory_analysis,
    rank_analysis,
    rank_analysis_all_periods,
    show_mean_cv,
    demand_tree_map_with_abc,
    generate_figures_for_abc_analysis,
    show_rank_analysis,
    show_inventory_reduction,
    add_abc,
    abc_analysis_all,
    inventory_simulation,
    show_prod_inv_demand,
    plot_demands,
    Scbas,
    promotion_effect_analysis
)

router = APIRouter()

@router.post("/abc-analysis")
async def perform_abc_analysis(
    file: UploadFile = File(...),
    threshold: str = Form("0.7,0.2,0.1"),
    agg_col: str = Form("prod"),
    value: str = Form("demand"),
    abc_name: str = Form("abc"),
    rank_name: str = Form("rank")
):
    """
    Perform ABC analysis on uploaded demand data
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = [agg_col, value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Parse threshold values
        threshold_values = [float(x.strip()) for x in threshold.split(",")]
        
        # Validate threshold values
        if abs(sum(threshold_values) - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="Threshold values must sum to approximately 1.0"
            )
        
        # Perform ABC analysis
        agg_df, new_df, category = abc_analysis(
            demand_df=demand_df,
            threshold=threshold_values,
            agg_col=agg_col,
            value=value,
            abc_name=abc_name,
            rank_name=rank_name
        )
        
        return {
            "aggregated_data": agg_df.to_dict("records"),
            "classified_data": new_df.head(1000).to_dict("records"),  # Limit output size
            "categories": category,
            "summary": {
                "total_items": len(agg_df),
                "total_value": float(agg_df[value].sum()),
                "thresholds": threshold_values
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing ABC analysis: {str(e)}")

@router.post("/treemap")
async def generate_treemap(
    file: UploadFile = File(...),
    parent: str = Form("cust"),
    value: str = Form("demand")
):
    """
    Generate demand treemap visualization
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = [parent, value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Generate treemap
        treemap_data = demand_tree_map(
            demand_df=demand_df,
            parent=parent,
            value=value
        )
        
        return {"treemap": treemap_data}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating treemap: {str(e)}")

@router.post("/risk-pooling")
async def analyze_risk_pooling(
    file: UploadFile = File(...),
    agg_period: str = Form("1w")
):
    """
    Perform risk pooling analysis
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['date', 'cust', 'prod', 'demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Perform risk pooling analysis
        result_df = risk_pooling_analysis(demand_df, agg_period)
        
        # Check if results are available
        if result_df.empty:
            return {
                "risk_pooling_analysis": [],
                "summary": {
                    "total_products": 0,
                    "total_reduction_abs": 0.0,
                    "avg_reduction_pct": 0.0,
                    "aggregation_period": agg_period,
                    "message": "No risk pooling opportunities found (need at least 2 customers per product)"
                }
            }
        
        return {
            "risk_pooling_analysis": result_df.to_dict("records"),
            "summary": {
                "total_products": len(result_df),
                "total_reduction_abs": float(result_df['reduction_absolute'].sum()) if 'reduction_absolute' in result_df.columns else 0.0,
                "avg_reduction_pct": float(result_df['reduction_percentage'].mean()) if 'reduction_percentage' in result_df.columns else 0.0,
                "aggregation_period": agg_period
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in risk pooling analysis: {str(e)}")

@router.post("/inventory-analysis")
async def perform_inventory_analysis(
    demand_file: UploadFile = File(...),
    product_file: UploadFile = File(...),
    z: float = Form(1.65),
    LT: int = Form(1),
    r: float = Form(0.3),
    num_days: int = Form(7)
):
    """
    Perform comprehensive inventory analysis
    """
    try:
        # Read demand file
        demand_contents = await demand_file.read()
        demand_df = pd.read_csv(io.StringIO(demand_contents.decode('utf-8')))
        
        # Read product file  
        product_contents = await product_file.read()
        prod_df = pd.read_csv(io.StringIO(product_contents.decode('utf-8')))
        
        # Validate demand file columns
        demand_required = ['date', 'prod', 'demand']
        demand_missing = [col for col in demand_required if col not in demand_df.columns]
        if demand_missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in demand file: {demand_missing}"
            )
        
        # Validate product file columns
        prod_required = ['name']
        prod_missing = [col for col in prod_required if col not in prod_df.columns]
        if prod_missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in product file: {prod_missing}"
            )
        
        # Perform risk pooling analysis first
        inv_reduction_df = risk_pooling_analysis(demand_df)
        
        # Perform inventory analysis
        result_df = inventory_analysis(
            prod_df=prod_df,
            demand_df=demand_df,
            inv_reduction_df=inv_reduction_df,
            z=z,
            LT=LT,
            r=r,
            num_days=num_days
        )
        
        return {
            "inventory_analysis": result_df.to_dict("records"),
            "parameters": {
                "safety_factor_z": z,
                "lead_time": LT,
                "holding_cost_rate": r,
                "num_days": num_days
            },
            "summary": {
                "total_products": len(result_df),
                "total_safety_stock": float(result_df['safety_inventory'].sum()),
                "total_target_inventory": float(result_df['target_inventory'].sum())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in inventory analysis: {str(e)}")

@router.post("/pareto-analysis")
async def perform_pareto_analysis(
    file: UploadFile = File(...),
    agg_col: str = Form("prod"),
    value: str = Form("demand")
):
    """
    Perform Pareto analysis (80-20 rule analysis)
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = [agg_col, value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Perform Pareto analysis
        pareto_df, pareto_stats = pareto_analysis(
            demand_df=demand_df,
            agg_col=agg_col,
            value=value
        )
        
        return convert_to_json_safe({
            "pareto_data": pareto_df.to_dict("records"),
            "pareto_statistics": pareto_stats,
            "analysis_summary": {
                "follows_80_20_rule": pareto_stats["pareto_compliance"],
                "top_20_percent_contribute": f"{pareto_stats['top_20_pct_value_ratio']:.1%}",
                "analyzed_column": agg_col,
                "value_column": value
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error performing Pareto analysis: {str(e)}")

@router.post("/rank-analysis")
async def perform_rank_analysis(
    file: UploadFile = File(...),
    agg_col: str = Form("prod"),
    value: str = Form("demand")
):
    """
    Perform rank analysis on demand data
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = [agg_col, value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Perform rank analysis
        rank_dict = rank_analysis(
            df=demand_df,
            agg_col=agg_col,
            value=value
        )
        
        return {
            "rank_analysis": rank_dict,
            "analysis_summary": {
                "total_items": len(rank_dict),
                "analyzed_column": agg_col,
                "value_column": value
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error performing rank analysis: {str(e)}")

@router.post("/rank-analysis-periods")
async def perform_rank_analysis_periods(
    file: UploadFile = File(...),
    agg_col: str = Form("prod"),
    value: str = Form("demand"),
    agg_period: str = Form("1w")
):
    """
    Perform rank analysis over multiple periods
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['date', agg_col, value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Perform periodic rank analysis
        rank_dict = rank_analysis_all_periods(
            df=demand_df,
            agg_col=agg_col,
            value=value,
            agg_period=agg_period
        )
        
        return {
            "rank_analysis_periods": rank_dict,
            "analysis_summary": {
                "total_items": len(rank_dict),
                "analyzed_column": agg_col,
                "value_column": value,
                "aggregation_period": agg_period
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error performing periodic rank analysis: {str(e)}")

@router.post("/mean-cv-analysis")
async def perform_mean_cv_analysis(
    demand_file: UploadFile = File(...),
    product_file: UploadFile = File(None),
    show_name: bool = Form(True)
):
    """
    Perform mean vs coefficient of variation analysis
    """
    try:
        # Read demand file
        demand_contents = await demand_file.read()
        demand_df = pd.read_csv(io.StringIO(demand_contents.decode('utf-8')))
        
        # Validate demand file columns
        required_cols = ['prod', 'demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns in demand file: {missing_cols}"
            )
        
        # Read product file if provided
        prod_df = None
        if product_file:
            try:
                product_contents = await product_file.read()
                prod_df = pd.read_csv(io.StringIO(product_contents.decode('utf-8')))
            except Exception as e:
                # If product file reading fails, continue without it
                prod_df = None
        
        # Perform mean-CV analysis
        result = show_mean_cv(
            demand_df=demand_df,
            prod_df=prod_df,
            show_name=show_name
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error performing mean-CV analysis: {str(e)}")

@router.post("/treemap-with-abc")
async def generate_treemap_with_abc(
    file: UploadFile = File(...),
    abc_col: str = Form("abc")
):
    """
    Generate ABC-classified treemap visualization
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['prod', 'cust', 'demand', abc_col]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}. Please ensure the data has ABC analysis results."
            )
        
        # Generate ABC treemap
        treemap_data = demand_tree_map_with_abc(
            demand_df=demand_df,
            abc_col=abc_col
        )
        
        if "error" in treemap_data:
            raise HTTPException(status_code=400, detail=treemap_data["error"])
        
        return {"treemap": treemap_data}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating ABC treemap: {str(e)}")

@router.post("/comprehensive-abc-analysis")
async def perform_comprehensive_abc_analysis(
    file: UploadFile = File(...),
    value: str = Form("demand"),
    cumsum: bool = Form(True),
    cust_thres: str = Form("0.7, 0.2, 0.1"),
    prod_thres: str = Form("0.7, 0.2, 0.1")
):
    """
    Generate comprehensive ABC analysis with both customer and product figures
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['prod', 'cust', value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Validate threshold values
        try:
            cust_threshold_values = [float(x.strip()) for x in cust_thres.split(",")]
            prod_threshold_values = [float(x.strip()) for x in prod_thres.split(",")]
            
            if abs(sum(cust_threshold_values) - 1.0) > 0.01:
                raise HTTPException(status_code=400, detail="Customer threshold values must sum to approximately 1.0")
            if abs(sum(prod_threshold_values) - 1.0) > 0.01:
                raise HTTPException(status_code=400, detail="Product threshold values must sum to approximately 1.0")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid threshold format")
        
        # Perform comprehensive ABC analysis
        result = generate_figures_for_abc_analysis(
            demand_df=demand_df,
            value=value,
            cumsum=cumsum,
            cust_thres=cust_thres,
            prod_thres=prod_thres
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error performing comprehensive ABC analysis: {str(e)}")

@router.post("/advanced-rank-analysis")
async def perform_advanced_rank_analysis(
    file: UploadFile = File(...),
    value: str = Form("demand"),
    agg_period: str = Form("1m"),
    top_rank: int = Form(10)
):
    """
    Perform advanced rank analysis with time series visualization
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['date', 'prod', value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Perform advanced rank analysis
        result = show_rank_analysis(
            demand_df=demand_df,
            agg_df_prod=None,
            value=value,
            agg_period=agg_period,
            top_rank=top_rank
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error performing advanced rank analysis: {str(e)}")

@router.post("/inventory-reduction-visualization")
async def visualize_inventory_reduction(
    file: UploadFile = File(...),
    agg_period: str = Form("1w")
):
    """
    Visualize inventory reduction potential through risk pooling
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['date', 'cust', 'prod', 'demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # First perform risk pooling analysis to get reduction data
        inv_reduction_df = risk_pooling_analysis(demand_df, agg_period)
        
        if inv_reduction_df.empty:
            return {
                "chart_data": {"x": [], "y": [], "colors": [], "type": "bar"},
                "title": "Inventory Reduction Through Risk Pooling",
                "x_label": "Products",
                "y_label": "Reduction Amount",
                "statistics": {
                    "total_reduction": 0,
                    "average_reduction": 0,
                    "max_reduction": 0,
                    "min_reduction": 0,
                    "num_products": 0
                },
                "products": [],
                "summary": {
                    "message": "No inventory reduction opportunities found (need at least 2 customers per product)"
                }
            }
        
        # Add rank column for color coding
        inv_reduction_df = inv_reduction_df.sort_values('reduction_absolute', ascending=False).reset_index(drop=True)
        inv_reduction_df['rank'] = inv_reduction_df.index + 1
        
        # Rename columns to match expected format
        if 'reduction_absolute' in inv_reduction_df.columns and 'reduction' not in inv_reduction_df.columns:
            inv_reduction_df['reduction'] = inv_reduction_df['reduction_absolute']
        
        # Generate visualization data
        result = show_inventory_reduction(inv_reduction_df)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error visualizing inventory reduction: {str(e)}")

# New Pydantic models for request bodies
class ScbasRequest(BaseModel):
    abc_thresholds: List[float] = Field([0.7, 0.2, 0.1], description="ABC classification thresholds")
    safety_factor: float = Field(1.65, description="Safety stock factor")
    lead_time: int = Field(7, description="Lead time in days")
    holding_cost_rate: float = Field(0.2, description="Holding cost rate")
    value_col: str = Field("demand", description="Value column for analysis")
    simulation_periods: int = Field(365, description="Number of periods to simulate")

@router.post("/add-abc-to-dataframe")
async def add_abc_to_existing_dataframe(
    demand_file: UploadFile = File(...),
    abc_file: UploadFile = File(...),
    merge_col: str = Form("prod", description="Column to merge on")
):
    """
    Add ABC analysis results to existing DataFrame
    """
    try:
        # Read demand file
        demand_contents = await demand_file.read()
        demand_df = pd.read_csv(io.StringIO(demand_contents.decode('utf-8')))
        
        # Read ABC analysis results file
        abc_contents = await abc_file.read()
        abc_df = pd.read_csv(io.StringIO(abc_contents.decode('utf-8')))
        
        # Validate merge column exists
        if merge_col not in demand_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Merge column '{merge_col}' not found in demand file"
            )
        
        # Add ABC results to demand dataframe
        enhanced_df = add_abc(demand_df, abc_df, merge_col)
        
        return {
            "enhanced_data": enhanced_df.head(500).to_dict("records"),  # Limit output size
            "summary": {
                "total_records": len(enhanced_df),
                "abc_categories_added": enhanced_df['abc'].value_counts().to_dict() if 'abc' in enhanced_df.columns else {},
                "merge_column": merge_col,
                "enhancement_successful": True
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding ABC to DataFrame: {str(e)}")

@router.post("/abc-analysis-all")
async def customer_product_abc_analysis(
    file: UploadFile = File(...),
    threshold: str = Form("0.7,0.2,0.1", description="ABC thresholds"),
    cust_col: str = Form("cust", description="Customer column name"),
    prod_col: str = Form("prod", description="Product column name"),
    value: str = Form("demand", description="Value column for analysis")
):
    """
    Customer-Product combination ABC analysis
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = [cust_col, prod_col, value]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Parse threshold values
        threshold_values = [float(x.strip()) for x in threshold.split(",")]
        
        # Perform ABC analysis on customer-product combinations
        agg_df, summary_stats = abc_analysis_all(
            demand_df, threshold_values, cust_col, prod_col, value
        )
        
        return {
            "aggregated_data": agg_df.head(1000).to_dict("records"),  # Limit output size
            "summary_statistics": summary_stats,
            "analysis_parameters": {
                "customer_column": cust_col,
                "product_column": prod_col,
                "value_column": value,
                "thresholds": threshold_values
            },
            "total_combinations": len(agg_df),
            "abc_distribution": agg_df['abc'].value_counts().to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in customer-product ABC analysis: {str(e)}")

@router.post("/inventory-simulation")
async def run_inventory_simulation(
    demand_file: UploadFile = File(...),
    product_file: UploadFile = File(None),
    safety_factor: float = Form(1.65, description="Safety stock factor"),
    lead_time: int = Form(7, description="Lead time in days"),
    holding_cost_rate: float = Form(0.2, description="Holding cost rate"),
    simulation_periods: int = Form(365, description="Number of periods to simulate")
):
    """
    Complete (Q,R) policy inventory simulation
    """
    try:
        # Read demand file
        demand_contents = await demand_file.read()
        demand_df = pd.read_csv(io.StringIO(demand_contents.decode('utf-8')))
        
        # Validate required columns in demand file
        required_demand_cols = ['date', 'prod', 'demand']
        missing_cols = [col for col in required_demand_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns in demand file: {missing_cols}"
            )
        
        # Read product file if provided
        prod_df = pd.DataFrame()
        if product_file and product_file.filename:
            prod_contents = await product_file.read()
            prod_df = pd.read_csv(io.StringIO(prod_contents.decode('utf-8')))
        
        # Run simulation
        simulation_results = inventory_simulation(
            demand_df, prod_df, safety_factor, lead_time, holding_cost_rate, simulation_periods
        )
        
        return simulation_results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in inventory simulation: {str(e)}")

@router.post("/production-inventory-demand-viz")
async def production_inventory_demand_visualization(
    file: UploadFile = File(...),
    product: str = Form(..., description="Product to visualize"),
    max_periods: int = Form(100, description="Maximum periods to display"),
    safety_factor: float = Form(1.65),
    lead_time: int = Form(7),
    holding_cost_rate: float = Form(0.2),
    simulation_periods: int = Form(365)
):
    """
    Production/inventory/demand time series visualization
    """
    try:
        # Read demand file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['date', 'prod', 'demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Check if product exists
        if product not in demand_df['prod'].unique():
            raise HTTPException(
                status_code=400,
                detail=f"Product '{product}' not found in data"
            )
        
        # Run simulation first
        prod_df = pd.DataFrame()
        simulation_results = inventory_simulation(
            demand_df, prod_df, safety_factor, lead_time, holding_cost_rate, simulation_periods
        )
        
        # Generate visualization data
        viz_data = show_prod_inv_demand(simulation_results, product, max_periods)
        
        return viz_data
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in production/inventory/demand visualization: {str(e)}")

@router.post("/multi-product-demand-plot")
async def multi_product_demand_visualization(
    file: UploadFile = File(...),
    products: str = Form(None, description="Comma-separated list of products (optional)"),
    max_periods: int = Form(365, description="Maximum periods to display")
):
    """
    Multi-product demand visualization
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['date', 'prod', 'demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Parse products list if provided
        product_list = None
        if products:
            product_list = [p.strip() for p in products.split(",")]
            # Validate products exist
            available_products = demand_df['prod'].unique()
            invalid_products = [p for p in product_list if p not in available_products]
            if invalid_products:
                raise HTTPException(
                    status_code=400,
                    detail=f"Products not found in data: {invalid_products}"
                )
        
        # Generate visualization data
        chart_data = plot_demands(demand_df, product_list, max_periods)
        
        return convert_to_json_safe(chart_data)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in multi-product demand visualization: {str(e)}")

@router.post("/scbas-comprehensive-analysis")
async def scbas_comprehensive_analysis(
    demand_file: UploadFile = File(...),
    product_file: UploadFile = File(None),
    customer_file: UploadFile = File(None),
    request: ScbasRequest = Body(...)
):
    """
    SCBAS Comprehensive ABC Analysis Framework
    """
    try:
        # Read demand file
        demand_contents = await demand_file.read()
        demand_df = pd.read_csv(io.StringIO(demand_contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['date', 'cust', 'prod', 'demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Read optional product file
        prod_df = None
        if product_file and product_file.filename:
            prod_contents = await product_file.read()
            prod_df = pd.read_csv(io.StringIO(prod_contents.decode('utf-8')))
        
        # Read optional customer file
        cust_df = None
        if customer_file and customer_file.filename:
            cust_contents = await customer_file.read()
            cust_df = pd.read_csv(io.StringIO(cust_contents.decode('utf-8')))
        
        # Initialize SCBAS framework
        scbas = Scbas(
            abc_thresholds=request.abc_thresholds,
            safety_factor=request.safety_factor,
            lead_time=request.lead_time,
            holding_cost_rate=request.holding_cost_rate
        )
        
        # Load data
        scbas.load_data(demand_df, prod_df, cust_df)
        
        # Run comprehensive analysis
        scbas.run_abc_analysis(value_col=request.value_col)
        scbas.run_simulation(simulation_periods=request.simulation_periods)
        
        # Get all analysis results
        abc_results = scbas.get_abc_results()
        simulation_results = scbas.get_simulation_results()
        treemap_data = scbas.get_treemap_data()
        mean_cv_analysis = scbas.get_mean_cv_analysis()
        risk_pooling_results = scbas.get_risk_pooling_analysis()
        rank_analysis_results = scbas.get_rank_analysis()
        inventory_analysis_results = scbas.get_inventory_analysis()
        ui_node_structure = scbas.generate_ui_node_structure()
        analysis_summary = scbas.get_analysis_summary()
        
        return {
            "scbas_analysis": {
                "abc_results": abc_results,
                "simulation_results": simulation_results,
                "treemap_data": treemap_data,
                "mean_cv_analysis": mean_cv_analysis,
                "risk_pooling_analysis": risk_pooling_results.head(100).to_dict("records") if not risk_pooling_results.empty else [],
                "rank_analysis": rank_analysis_results,
                "inventory_analysis": inventory_analysis_results,
                "ui_node_structure": ui_node_structure,
                "analysis_summary": analysis_summary
            },
            "configuration": {
                "abc_thresholds": request.abc_thresholds,
                "safety_factor": request.safety_factor,
                "lead_time": request.lead_time,
                "holding_cost_rate": request.holding_cost_rate,
                "value_column": request.value_col,
                "simulation_periods": request.simulation_periods
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in SCBAS comprehensive analysis: {str(e)}")

@router.post("/promotion-effect-analysis")
async def promotion_effect_analysis_endpoint(
    file: UploadFile = File(...),
    promo_col: str = Form("promo_0", description="Promotion flag column"),
    value_col: str = Form("demand", description="Value column for analysis"),
    product_col: str = Form("prod", description="Product column")
):
    """
    Analyze promotion effects on demand patterns
    """
    try:
        # Read CSV file
        contents = await file.read()
        demand_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = [promo_col, value_col, product_col]
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Run promotion analysis
        results = promotion_effect_analysis(demand_df, promo_col, value_col, product_col)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in promotion effect analysis: {str(e)}")