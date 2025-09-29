import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def demand_tree_map(demand_df: pd.DataFrame, parent: str = "cust", value: str = "demand") -> Dict[str, Any]:
    """
    Generate treemap visualization of demand/sales data
    """
    # Aggregate demand by parent column
    agg_df = demand_df.groupby([parent])[value].sum().reset_index()
    
    # Create simple data structure for treemap
    treemap_data = []
    for _, row in agg_df.iterrows():
        treemap_data.append({
            "label": row[parent],
            "value": float(row[value]),
            "parent": "",
        })
    
    # Calculate total for percentage
    total_value = agg_df[value].sum()
    for item in treemap_data:
        item["percentage"] = (item["value"] / total_value * 100) if total_value > 0 else 0
    
    return {
        "data": treemap_data,
        "title": f"Demand Treemap by {parent.title()}",
        "total_value": float(total_value),
        "parent_column": parent,
        "value_column": value,
        "num_categories": len(treemap_data)
    }

def abc_analysis(demand_df: pd.DataFrame,
                 threshold: List[float],
                 agg_col: str = "prod",
                 value: str = "demand",
                 abc_name: str = "abc",
                 rank_name: str = "rank") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, List[str]]]:
    """
    Perform ABC classification on products/customers
    """
    # Aggregate by column
    agg_df = demand_df.groupby(agg_col)[value].sum().reset_index()
    agg_df = agg_df.sort_values(value, ascending=False).reset_index(drop=True)
    
    # Calculate cumulative percentage
    total_value = agg_df[value].sum()
    agg_df['cumsum'] = agg_df[value].cumsum()
    agg_df['cum_pct'] = agg_df['cumsum'] / total_value
    
    # Assign ABC categories
    agg_df[abc_name] = 'C'
    agg_df[rank_name] = 2
    
    cum_threshold = 0
    rank = 0
    category = {}
    
    for i, thresh in enumerate(threshold):
        cum_threshold += thresh
        mask = agg_df['cum_pct'] <= cum_threshold
        abc_label = chr(65 + i)  # A, B, C, ...
        agg_df.loc[mask & (agg_df[abc_name] == 'C'), abc_name] = abc_label
        agg_df.loc[mask & (agg_df[rank_name] == 2), rank_name] = rank
        
        # Store category mapping
        category[rank] = agg_df[agg_df[rank_name] == rank][agg_col].tolist()
        rank += 1
    
    # Create mapping dictionaries
    abc_dict = dict(zip(agg_df[agg_col], agg_df[abc_name]))
    rank_dict = dict(zip(agg_df[agg_col], agg_df[rank_name]))
    
    # Apply to original dataframe
    new_df = demand_df.copy()
    new_df[abc_name] = new_df[agg_col].map(abc_dict)
    new_df[rank_name] = new_df[agg_col].map(rank_dict)
    
    return agg_df, new_df, category

def risk_pooling_analysis(demand_df: pd.DataFrame, agg_period: str = "1w") -> pd.DataFrame:
    """
    Analyze inventory reduction potential through risk pooling
    """
    # Convert date column to datetime
    demand_df = demand_df.copy()
    
    try:
        demand_df['date'] = pd.to_datetime(demand_df['date'])
    except:
        # If date conversion fails, create simple aggregation
        return _simple_risk_pooling_analysis(demand_df)
    
    # Always use simple analysis for now
    return _simple_risk_pooling_analysis(demand_df)
    
    # Set date as index for resampling
    demand_df_indexed = demand_df.set_index('date')
    
    # Group by product and resample
    results = []
    
    for prod in demand_df_indexed['prod'].unique():
        prod_data = demand_df_indexed[demand_df_indexed['prod'] == prod]
        
        # Calculate statistics by customer
        cust_stats = []
        for cust in prod_data['cust'].unique():
            cust_data = prod_data[prod_data['cust'] == cust]
            try:
                # Resample demand by period
                cust_demand = cust_data.resample(agg_period)['demand'].sum()
                mean_demand = cust_demand.mean()
                std_demand = cust_demand.std()
                
                # Skip if NaN values
                if pd.isna(mean_demand) or pd.isna(std_demand) or std_demand == 0:
                    continue
                    
                cust_stats.append({
                    'cust': cust,
                    'prod': prod,
                    'mean_demand': mean_demand,
                    'std_demand': std_demand,
                    'cv': std_demand / mean_demand if mean_demand > 0 else 0
                })
            except Exception as e:
                # Skip problematic resampling
                continue
        
        if cust_stats and len(cust_stats) > 1:  # Need at least 2 customers for pooling
            # Calculate pooled statistics
            total_mean = sum(stat['mean_demand'] for stat in cust_stats)
            total_var = sum(stat['std_demand']**2 for stat in cust_stats)
            total_std = np.sqrt(total_var)
            
            # Calculate individual safety stock (sum of individual)
            individual_ss = sum(1.65 * stat['std_demand'] for stat in cust_stats)
            
            # Calculate pooled safety stock
            pooled_ss = 1.65 * total_std
            
            # Calculate reduction
            reduction_abs = individual_ss - pooled_ss
            reduction_pct = (reduction_abs / individual_ss * 100) if individual_ss > 0 else 0
            
            results.append({
                'prod': prod,
                'num_customers': len(cust_stats),
                'total_mean_demand': total_mean,
                'total_std_demand': total_std,
                'individual_safety_stock': individual_ss,
                'pooled_safety_stock': pooled_ss,
                'reduction_absolute': reduction_abs,
                'reduction_percentage': reduction_pct
            })
    
    return pd.DataFrame(results)

def _simple_risk_pooling_analysis(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple risk pooling analysis without time-based resampling
    """
    results = []
    
    for prod in demand_df['prod'].unique():
        prod_data = demand_df[demand_df['prod'] == prod]
        
        # Calculate statistics by customer
        cust_stats = []
        customers = prod_data['cust'].unique()
        
        if len(customers) < 2:
            continue  # Need at least 2 customers for pooling
            
        for cust in customers:
            cust_data = prod_data[prod_data['cust'] == cust]
            mean_demand = cust_data['demand'].mean()
            std_demand = cust_data['demand'].std()
            
            # Use 0 std if NaN (single observation)
            if pd.isna(std_demand):
                std_demand = 0
            
            if pd.notna(mean_demand):
                cust_stats.append({
                    'cust': cust,
                    'prod': prod,
                    'mean_demand': mean_demand,
                    'std_demand': std_demand,
                    'cv': std_demand / mean_demand if mean_demand > 0 else 0
                })
        
        if len(cust_stats) >= 2:
            # Calculate pooled statistics
            total_mean = sum(stat['mean_demand'] for stat in cust_stats)
            total_var = sum(stat['std_demand']**2 for stat in cust_stats)
            total_std = np.sqrt(total_var) if total_var > 0 else 0
            
            # Calculate individual safety stock (sum of individual)
            individual_ss = sum(1.65 * stat['std_demand'] for stat in cust_stats)
            
            # Calculate pooled safety stock
            pooled_ss = 1.65 * total_std
            
            # Calculate reduction
            reduction_abs = individual_ss - pooled_ss
            reduction_pct = (reduction_abs / individual_ss * 100) if individual_ss > 0 else 0
            
            results.append({
                'prod': prod,
                'num_customers': len(cust_stats),
                'total_mean_demand': total_mean,
                'total_std_demand': total_std,
                'individual_safety_stock': individual_ss,
                'pooled_safety_stock': pooled_ss,
                'reduction_absolute': reduction_abs,
                'reduction_percentage': reduction_pct
            })
    
    return pd.DataFrame(results)

def pareto_analysis(demand_df: pd.DataFrame, 
                   agg_col: str = "prod", 
                   value: str = "demand") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform Pareto analysis (80-20 rule analysis)
    """
    # Aggregate by column
    agg_df = demand_df.groupby(agg_col)[value].sum().reset_index()
    agg_df = agg_df.sort_values(value, ascending=False).reset_index(drop=True)
    
    # Calculate cumulative percentage
    total_value = agg_df[value].sum()
    agg_df['cumsum'] = agg_df[value].cumsum()
    agg_df['cum_pct'] = agg_df['cumsum'] / total_value
    agg_df['pct'] = agg_df[value] / total_value
    
    # Find 80% point
    pareto_80_index = agg_df[agg_df['cum_pct'] <= 0.8].index[-1] if len(agg_df[agg_df['cum_pct'] <= 0.8]) > 0 else 0
    pareto_20_index = int(len(agg_df) * 0.2)
    
    # Calculate Pareto statistics
    top_20_pct_items = len(agg_df) * 0.2
    top_20_pct_value = agg_df.head(int(top_20_pct_items))[value].sum() / total_value
    
    pareto_stats = {
        "total_items": len(agg_df),
        "total_value": float(total_value),
        "pareto_80_index": pareto_80_index + 1,  # Human readable (1-indexed)
        "pareto_20_index": pareto_20_index,
        "top_20_pct_items": int(top_20_pct_items),
        "top_20_pct_value_ratio": float(top_20_pct_value),
        "pareto_compliance": top_20_pct_value >= 0.8,  # True if follows 80-20 rule
    }
    
    return agg_df, pareto_stats

def inventory_analysis(prod_df: pd.DataFrame, 
                      demand_df: pd.DataFrame,
                      inv_reduction_df: pd.DataFrame,
                      z: float = 1.65,
                      LT: int = 1,
                      r: float = 0.3,
                      num_days: int = 7) -> pd.DataFrame:
    """
    Calculate safety stock, lot sizes, and target inventory levels
    """
    result_df = prod_df.copy()
    
    # Calculate demand statistics
    demand_stats = demand_df.groupby('prod')['demand'].agg(['mean', 'std']).reset_index()
    demand_stats.columns = ['name', 'average_demand', 'standard_deviation']
    
    # Merge with product data
    result_df = result_df.merge(demand_stats, on='name', how='left')
    
    # Calculate safety stock
    result_df['safety_inventory'] = z * result_df['standard_deviation'] * np.sqrt(LT)
    
    # Calculate EOQ (Economic Order Quantity)
    # Assuming fixed ordering cost K = fixed_cost
    result_df['lot_size'] = np.sqrt(
        2 * result_df['fixed_cost'] * result_df['average_demand'] / 
        (r * result_df.get('cust_value', result_df.get('dc_value', 1)))
    )
    
    # Calculate target inventory (safety stock + average demand during lead time)
    result_df['target_inventory'] = (
        result_df['safety_inventory'] + 
        result_df['average_demand'] * LT
    )
    
    # Calculate initial inventory (example: 50% of target)
    result_df['initial_inventory'] = result_df['target_inventory'] * 0.5
    
    # Fill NaN values
    result_df = result_df.fillna(0)
    
    return result_df

def rank_analysis(df: pd.DataFrame, 
                  agg_col: str, 
                  value: str) -> Dict[str, int]:
    """
    全期間分のランク分析のための関数
    """
    temp_series = df.groupby([agg_col])[value].sum()
    sorted_series = temp_series.sort_values(ascending=False)
    rank = {}  # ランクを格納する辞書
    count = 0
    for i in sorted_series.index:
        count += 1
        rank[i] = count
    return rank

def rank_analysis_all_periods(df: pd.DataFrame, 
                              agg_col: str, 
                              value: str,
                              agg_period: str) -> Dict[str, List[int]]:
    """
    期別のランク分析のための関数
    """
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Reset index if needed
    if df.index.name == 'date' or 'date' in df.index.names:
        df = df.reset_index()
    
    agg_set = set(df[agg_col].unique())  # 集約する対象の集合
    rank = {}  # 対象の期ごとのランクのリストを保持する辞書
    for i in agg_set:
        rank[i] = []
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    start_date = pd.to_datetime(min(df.index))
    end_date = pd.to_datetime(max(df.index))
    
    for t in pd.date_range(start_date, end_date, freq=agg_period):
        selected_df = df[(df.index >= start_date) & (df.index <= t)]
        rank_in_period = rank_analysis(selected_df.reset_index(), agg_col, value)
        for i in rank:
            if i in rank_in_period:
                rank[i].append(rank_in_period[i])
            else:
                rank[i].append(None)  # Use None instead of np.nan for JSON serialization
        start_date = t
    
    return rank

def show_mean_cv(demand_df: pd.DataFrame, 
                 prod_df: Optional[pd.DataFrame] = None, 
                 show_name: bool = True) -> Dict[str, Any]:
    """
    需要の製品ごとの平均と変動係数の散布図データを生成する関数
    """
    # Copy to avoid modifying original
    df = demand_df.copy()
    
    # Handle date column
    try:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    except:
        pass
        
    # Calculate mean and std for each product
    gdf = pd.pivot_table(df, values="demand", index="prod", aggfunc=["sum", "std"])
    gdf = gdf.sort_values(by=('sum', 'demand'), ascending=False)
    gdf = gdf.reset_index()
    
    # Flatten column names
    gdf.columns = ["prod", "sum", "std"]
    
    # Calculate coefficient of variation
    gdf["cv"] = gdf["std"] / (gdf["sum"] + 0.0001)
    
    # Add product color information if available
    if prod_df is not None and "cust_value" in prod_df.columns:
        prod_color = {}
        for _, row in prod_df.iterrows():
            prod_color[row["name"]] = row["cust_value"]
        
        color = []
        for _, row in gdf.iterrows():
            color.append(prod_color.get(row["prod"], 1))  # Default color value
        gdf["price"] = color
    
    # Prepare data for frontend
    scatter_data = []
    for _, row in gdf.iterrows():
        point = {
            "x": float(row["sum"]),
            "y": float(row["cv"]),
            "prod": row["prod"],
            "size": float(row["std"]) if not pd.isna(row["std"]) else 1.0
        }
        
        if "price" in gdf.columns:
            point["color"] = float(row["price"])
            
        scatter_data.append(point)
    
    return {
        "scatter_data": scatter_data,
        "x_label": "Total Demand",
        "y_label": "Coefficient of Variation (CV)", 
        "title": "Mean vs Coefficient of Variation Analysis",
        "show_names": show_name,
        "has_price_info": prod_df is not None and "cust_value" in prod_df.columns
    }

def demand_tree_map_with_abc(demand_df: pd.DataFrame, abc_col: str) -> Dict[str, Any]:
    """
    ABC別に色分けした需要のtreemapを生成する関数
    """
    # Copy to avoid modifying original
    df = demand_df.copy()
    
    # Handle date column if present
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        except:
            pass
    
    # Aggregate demand by product, customer, and ABC category
    try:
        agg_df = pd.pivot_table(
            df, 
            index=["prod", "cust", abc_col], 
            values="demand", 
            aggfunc="sum"
        ).reset_index()
    except KeyError as e:
        # If ABC column doesn't exist, return error
        return {
            "error": f"Missing column: {str(e)}",
            "data": [],
            "title": "ABC Treemap - Error"
        }
    
    # Create treemap data with ABC color coding
    treemap_data = []
    abc_color_map = {
        'A': '#2196f3',  # Blue
        'B': '#ff9800',  # Orange  
        'C': '#4caf50'   # Green
    }
    
    for _, row in agg_df.iterrows():
        abc_category = row[abc_col]
        treemap_data.append({
            "label": f"{row['cust']}/{row['prod']}",
            "value": float(row["demand"]),
            "abc_category": abc_category,
            "color": abc_color_map.get(abc_category, '#757575'),
            "parent": row['cust'],
            "prod": row['prod'],
            "cust": row['cust']
        })
    
    # Calculate totals for percentage
    total_value = sum(item["value"] for item in treemap_data)
    for item in treemap_data:
        item["percentage"] = (item["value"] / total_value * 100) if total_value > 0 else 0
    
    return {
        "data": treemap_data,
        "title": f"ABC-Classified Demand Treemap",
        "total_value": float(total_value),
        "abc_column": abc_col,
        "value_column": "demand",
        "num_items": len(treemap_data),
        "color_mapping": abc_color_map
    }

def generate_figures_for_abc_analysis(demand_df: pd.DataFrame,
                                     value: str = "demand",
                                     cumsum: bool = True,
                                     cust_thres: str = "0.7, 0.2, 0.1",
                                     prod_thres: str = "0.7, 0.2, 0.1") -> Dict[str, Any]:
    """
    顧客・製品の両方に対するABC分析を同時に行い、結果の図とデータフレームを同時に得る関数
    """
    # Parse thresholds
    cust_threshold = [float(i.strip()) for i in cust_thres.split(",")]
    prod_threshold = [float(i.strip()) for i in prod_thres.split(",")]

    # Perform ABC analysis for products and customers
    agg_df_prod, new_df, category_prod = abc_analysis(
        demand_df, prod_threshold, 'prod', value, "prod_ABC", "prod_rank")
    agg_df_cust, new_df, category_cust = abc_analysis(
        new_df, cust_threshold, 'cust', value, "customer_ABC", "customer_rank")

    # Calculate cumulative sums if requested
    if cumsum:
        agg_df_prod["cumsum_prod"] = agg_df_prod[value].cumsum()
        agg_df_prod["cum_pct_prod"] = agg_df_prod["cumsum_prod"] / agg_df_prod["cumsum_prod"].iloc[-1]
        
        agg_df_cust["cumsum_cust"] = agg_df_cust[value].cumsum()
        agg_df_cust["cum_pct_cust"] = agg_df_cust["cumsum_cust"] / agg_df_cust["cumsum_cust"].iloc[-1]

    # Create visualization data for products
    alphabet = 'ABCDEFGHIJ'
    prod_colors = []
    for i in range(len(category_prod)):
        prod_colors.extend([f'rgba({41 + i*50}, {150 + i*30}, {243 - i*50}, 0.8)'] * len(category_prod[i]))

    prod_chart_data = {
        "x": list(agg_df_prod.index if hasattr(agg_df_prod, 'index') else range(len(agg_df_prod))),
        "y": agg_df_prod["cum_pct_prod"].tolist() if cumsum else agg_df_prod[value].tolist(),
        "labels": agg_df_prod.index.tolist() if hasattr(agg_df_prod, 'index') else list(agg_df_prod['prod']),
        "colors": prod_colors,
        "type": "bar",
        "cumulative": cumsum,
        "title": f"Product ABC Analysis ({'Cumulative' if cumsum else 'Values'})"
    }

    # Create visualization data for customers
    cust_colors = []
    for i in range(len(category_cust)):
        cust_colors.extend([f'rgba({255 - i*40}, {154 + i*20}, {26 + i*60}, 0.8)'] * len(category_cust[i]))

    cust_chart_data = {
        "x": list(agg_df_cust.index if hasattr(agg_df_cust, 'index') else range(len(agg_df_cust))),
        "y": agg_df_cust["cum_pct_cust"].tolist() if cumsum else agg_df_cust[value].tolist(),
        "labels": agg_df_cust.index.tolist() if hasattr(agg_df_cust, 'index') else list(agg_df_cust['cust']),
        "colors": cust_colors,
        "type": "bar",
        "cumulative": cumsum,
        "title": f"Customer ABC Analysis ({'Cumulative' if cumsum else 'Values'})"
    }

    return {
        "prod_chart": prod_chart_data,
        "cust_chart": cust_chart_data,
        "agg_df_prod": agg_df_prod.to_dict("records"),
        "agg_df_cust": agg_df_cust.to_dict("records"),
        "classified_data": new_df.head(1000).to_dict("records"),  # Limit output size
        "category_prod": category_prod,
        "category_cust": category_cust,
        "summary": {
            "total_products": len(agg_df_prod),
            "total_customers": len(agg_df_cust),
            "total_value": float(new_df[value].sum()),
            "prod_thresholds": prod_threshold,
            "cust_thresholds": cust_threshold,
            "value_column": value
        }
    }

def show_rank_analysis(demand_df: pd.DataFrame,
                       agg_df_prod: Optional[pd.DataFrame] = None,
                       value: str = "demand",
                       agg_period: str = "1m",
                       top_rank: int = 10) -> Dict[str, Any]:
    """
    ランク分析の可視化関数 - 時系列でのランクの変化を表示
    """
    df = demand_df.copy()
    agg_col = "prod"
    
    # Reset index if needed
    if df.index.name == 'date' or 'date' in df.index.names:
        df = df.reset_index()
    
    # Perform periodic rank analysis
    rank_dict = rank_analysis_all_periods(df, agg_col, value, agg_period)

    # Set date index for date range generation
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    start_date = pd.to_datetime(min(df.index))
    end_date = pd.to_datetime(max(df.index))
    x_range = [t.strftime('%Y-%m-%d') for t in pd.date_range(start_date, end_date, freq=agg_period)]

    # Get overall rank to determine top items
    if agg_df_prod is None:
        agg_df_prod, _, _ = abc_analysis(df.reset_index(), [0.7, 0.2, 0.1], agg_col, value, "ABC", "Rank")

    # Create time series data for top products
    time_series_data = []
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#16a085']
    
    top_products = list(agg_df_prod.index)[:top_rank] if hasattr(agg_df_prod, 'index') else list(agg_df_prod['prod'])[:top_rank]
    
    for i, product in enumerate(top_products):
        product_ranks = rank_dict.get(product, [])
        
        # Handle None values for missing periods
        y_values = []
        x_values = []
        for j, rank_val in enumerate(product_ranks):
            if rank_val is not None and j < len(x_range):
                y_values.append(rank_val)
                x_values.append(x_range[j])
        
        if y_values:  # Only add if we have data
            time_series_data.append({
                "x": x_values,
                "y": y_values,
                "name": str(product),
                "color": colors[i % len(colors)],
                "mode": "lines+markers"
            })

    return {
        "time_series_data": time_series_data,
        "x_range": x_range,
        "title": f"Product Rank Evolution Over Time (Top {top_rank})",
        "x_label": "Time Period",
        "y_label": "Rank (Lower is Better)",
        "agg_period": agg_period,
        "top_products": top_products,
        "summary": {
            "total_periods": len(x_range),
            "total_products_analyzed": len(rank_dict),
            "top_rank_displayed": top_rank,
            "aggregation_period": agg_period,
            "value_column": value
        }
    }

def show_inventory_reduction(inv_reduction_df: pd.DataFrame) -> Dict[str, Any]:
    """
    在庫削減量の可視化関数
    """
    try:
        df = inv_reduction_df.copy()
        df = df.reset_index(drop=True)
    except:
        df = inv_reduction_df.copy()
    
    # Sort by reduction amount (descending)
    if 'reduction' in df.columns:
        df = df.sort_values('reduction', ascending=False)
    elif 'reduction_absolute' in df.columns:
        df = df.sort_values('reduction_absolute', ascending=False)
    
    # Prepare data for bar chart
    products = df['prod'].tolist()
    reductions = df['reduction'].tolist() if 'reduction' in df.columns else df['reduction_absolute'].tolist()
    
    # Color coding based on rank (if available)
    if 'rank' in df.columns:
        colors = []
        for rank in df['rank']:
            if rank <= 3:
                colors.append('#e74c3c')  # Red for top 3
            elif rank <= 7:
                colors.append('#f39c12')  # Orange for 4-7
            else:
                colors.append('#2ecc71')  # Green for others
    else:
        colors = ['#3498db'] * len(products)
    
    # Create chart data
    chart_data = {
        "x": products,
        "y": reductions,
        "colors": colors,
        "type": "bar"
    }
    
    # Calculate statistics
    total_reduction = sum(reductions)
    avg_reduction = total_reduction / len(reductions) if reductions else 0
    max_reduction = max(reductions) if reductions else 0
    min_reduction = min(reductions) if reductions else 0
    
    return {
        "chart_data": chart_data,
        "title": "Inventory Reduction Through Risk Pooling",
        "x_label": "Products",
        "y_label": "Reduction Amount",
        "statistics": {
            "total_reduction": total_reduction,
            "average_reduction": avg_reduction,
            "max_reduction": max_reduction,
            "min_reduction": min_reduction,
            "num_products": len(products)
        },
        "products": products,
        "summary": {
            "message": f"Total inventory reduction potential: {total_reduction:.2f} units through risk pooling across {len(products)} products"
        }
    }

def add_abc(original_df: pd.DataFrame, 
           abc_df: pd.DataFrame,
           merge_col: str = "prod") -> pd.DataFrame:
    """
    Add ABC analysis results to existing DataFrame
    Exact implementation from notebook
    
    Args:
        original_df: Original DataFrame to enhance
        abc_df: ABC analysis results DataFrame
        merge_col: Column to merge on
        
    Returns:
        Enhanced DataFrame with ABC classifications
    """
    result_df = original_df.copy()
    
    # Create mapping dictionaries from ABC results
    if merge_col in abc_df.columns:
        # Handle case where abc_df has merge_col as regular column
        abc_mapping = dict(zip(abc_df[merge_col], abc_df.get('abc', abc_df.get('ABC', 'C'))))
        rank_mapping = dict(zip(abc_df[merge_col], abc_df.get('rank', abc_df.get('Rank', 0))))
    else:
        # Handle case where merge_col is index
        abc_mapping = dict(zip(abc_df.index, abc_df.get('abc', abc_df.get('ABC', 'C'))))
        rank_mapping = dict(zip(abc_df.index, abc_df.get('rank', abc_df.get('Rank', 0))))
    
    # Add ABC and rank columns
    result_df['abc'] = result_df[merge_col].map(abc_mapping).fillna('C')
    result_df['rank'] = result_df[merge_col].map(rank_mapping).fillna(2)
    
    return result_df

def abc_analysis_all(demand_df: pd.DataFrame,
                    threshold: List[float] = [0.7, 0.2, 0.1],
                    cust_col: str = "cust",
                    prod_col: str = "prod",
                    value: str = "demand") -> Tuple[pd.DataFrame, Dict]:
    """
    Customer-Product combination ABC analysis
    Exact implementation from notebook
    
    Args:
        demand_df: Demand DataFrame
        threshold: ABC thresholds
        cust_col: Customer column name
        prod_col: Product column name
        value: Value column for analysis
        
    Returns:
        Aggregated DataFrame with ABC classifications, summary dictionary
    """
    # Aggregate by customer-product combination
    agg_df = demand_df.groupby([cust_col, prod_col]).agg({
        value: ['sum', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [cust_col, prod_col, f'{value}_sum', f'{value}_std', f'{value}_count']
    
    # Handle NaN in std (for cases with single observation)
    agg_df[f'{value}_std'] = agg_df[f'{value}_std'].fillna(0)
    
    # Sort by total demand
    agg_df = agg_df.sort_values(f'{value}_sum', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative percentage
    total_value = agg_df[f'{value}_sum'].sum()
    agg_df['cumsum'] = agg_df[f'{value}_sum'].cumsum()
    agg_df['cum_pct'] = agg_df['cumsum'] / total_value
    
    # Assign ABC categories
    agg_df['abc'] = 'C'
    agg_df['rank'] = len(threshold) - 1  # Last category
    
    cum_threshold = 0
    for i, thresh in enumerate(threshold):
        cum_threshold += thresh
        mask = agg_df['cum_pct'] <= cum_threshold
        abc_label = chr(65 + i)  # A, B, C, ...
        agg_df.loc[mask & (agg_df['abc'] == 'C'), 'abc'] = abc_label
        agg_df.loc[mask & (agg_df['rank'] == len(threshold) - 1), 'rank'] = i
    
    # Calculate summary statistics
    summary_stats = {}
    for category in ['A', 'B', 'C']:
        cat_data = agg_df[agg_df['abc'] == category]
        if len(cat_data) > 0:
            summary_stats[category] = {
                'count': len(cat_data),
                'total_value': float(cat_data[f'{value}_sum'].sum()),
                'percentage_of_total': float(cat_data[f'{value}_sum'].sum() / total_value * 100),
                'avg_value': float(cat_data[f'{value}_sum'].mean()),
                'avg_std': float(cat_data[f'{value}_std'].mean())
            }
    
    return agg_df, summary_stats

def inventory_simulation(demand_df: pd.DataFrame,
                        prod_df: pd.DataFrame,
                        safety_factor: float = 1.65,
                        lead_time: int = 7,
                        holding_cost_rate: float = 0.2,
                        simulation_periods: int = 365) -> Dict[str, Any]:
    """
    Complete (Q,R) policy inventory simulation
    Exact implementation from notebook
    
    Args:
        demand_df: Demand data with date, prod, demand columns
        prod_df: Product data with fixed_cost, plnt_value columns
        safety_factor: Safety stock factor (z-score)
        lead_time: Lead time in days
        holding_cost_rate: Holding cost rate
        simulation_periods: Number of periods to simulate
        
    Returns:
        Simulation results with inventory levels, costs, and policies
    """
    # Prepare data
    demand_df = demand_df.copy()
    prod_df = prod_df.copy()
    
    # Ensure date column is datetime
    demand_df['date'] = pd.to_datetime(demand_df['date'])
    
    # Get unique products
    products = demand_df['prod'].unique()
    
    simulation_results = {}
    
    for product in products:
        # Filter demand for this product
        prod_demand = demand_df[demand_df['prod'] == product].copy()
        prod_demand = prod_demand.sort_values('date')
        
        # Calculate demand statistics
        daily_demand_mean = prod_demand['demand'].mean()
        daily_demand_std = prod_demand['demand'].std()
        
        if daily_demand_std == 0:
            daily_demand_std = daily_demand_mean * 0.1  # Assume 10% CV if std is 0
        
        # Get product parameters
        if product in prod_df.index:
            fixed_cost = prod_df.loc[product, 'fixed_cost'] if 'fixed_cost' in prod_df.columns else 100.0
            unit_value = prod_df.loc[product, 'plnt_value'] if 'plnt_value' in prod_df.columns else 1.0
        else:
            fixed_cost = 100.0
            unit_value = 1.0
        
        holding_cost = holding_cost_rate * unit_value
        
        # Calculate inventory policies
        # Safety stock: z * σ * sqrt(LT)
        safety_stock = safety_factor * daily_demand_std * math.sqrt(lead_time)
        
        # EOQ: sqrt(2 * FC * d / h)
        eoq = math.sqrt(2 * fixed_cost * daily_demand_mean / holding_cost)
        
        # Reorder point: mean demand during lead time + safety stock
        reorder_point = daily_demand_mean * lead_time + safety_stock
        
        # Target inventory: reorder point + EOQ
        target_inventory = reorder_point + eoq
        
        # Initial inventory: target inventory + half lot size
        initial_inventory = target_inventory + eoq / 2
        
        # Simulation
        inventory_levels = []
        order_quantities = []
        costs = []
        stockouts = []
        
        current_inventory = initial_inventory
        pending_orders = []  # List of (arrival_day, quantity) tuples
        
        # Use historical demand for simulation or generate if not enough data
        if len(prod_demand) >= simulation_periods:
            simulation_demand = prod_demand.head(simulation_periods)['demand'].values
        else:
            # Generate demand using normal distribution
            np.random.seed(42)  # For reproducibility
            simulation_demand = np.maximum(0, np.random.normal(
                daily_demand_mean, daily_demand_std, simulation_periods
            ))
        
        for day in range(simulation_periods):
            # Receive pending orders
            arriving_orders = [qty for arrival_day, qty in pending_orders if arrival_day == day]
            current_inventory += sum(arriving_orders)
            pending_orders = [(arr_day, qty) for arr_day, qty in pending_orders if arr_day != day]
            
            # Meet demand
            daily_demand = simulation_demand[day]
            if current_inventory >= daily_demand:
                current_inventory -= daily_demand
                stockout = 0
            else:
                stockout = daily_demand - current_inventory
                current_inventory = 0
            
            # Check if we need to reorder
            total_pipeline = current_inventory + sum(qty for _, qty in pending_orders)
            if total_pipeline <= reorder_point:
                # Place order
                order_qty = eoq
                arrival_day = day + lead_time
                pending_orders.append((arrival_day, order_qty))
                order_quantities.append(order_qty)
            else:
                order_quantities.append(0)
            
            # Calculate costs
            holding_cost_today = holding_cost * current_inventory
            ordering_cost_today = fixed_cost if order_quantities[-1] > 0 else 0
            stockout_cost_today = stockout * unit_value * 10  # Assume stockout cost is 10x unit value
            
            total_cost_today = holding_cost_today + ordering_cost_today + stockout_cost_today
            
            inventory_levels.append(current_inventory)
            costs.append(total_cost_today)
            stockouts.append(stockout)
        
        # Calculate performance metrics
        avg_inventory = np.mean(inventory_levels)
        total_cost = sum(costs)
        service_level = (simulation_periods - sum(1 for s in stockouts if s > 0)) / simulation_periods
        total_orders = sum(1 for q in order_quantities if q > 0)
        
        simulation_results[product] = {
            'demand_statistics': {
                'mean_daily_demand': daily_demand_mean,
                'std_daily_demand': daily_demand_std,
                'cv': daily_demand_std / daily_demand_mean if daily_demand_mean > 0 else 0
            },
            'inventory_policy': {
                'safety_stock': safety_stock,
                'eoq': eoq,
                'reorder_point': reorder_point,
                'target_inventory': target_inventory,
                'initial_inventory': initial_inventory
            },
            'simulation_results': {
                'avg_inventory_level': avg_inventory,
                'total_cost': total_cost,
                'service_level': service_level,
                'total_orders': total_orders,
                'avg_cost_per_period': total_cost / simulation_periods
            },
            'time_series': {
                'inventory_levels': inventory_levels[:100],  # Limit output size
                'order_quantities': order_quantities[:100],
                'daily_costs': costs[:100],
                'stockouts': stockouts[:100]
            },
            'parameters': {
                'fixed_cost': fixed_cost,
                'unit_value': unit_value,
                'holding_cost_rate': holding_cost_rate,
                'safety_factor': safety_factor,
                'lead_time': lead_time
            }
        }
    
    # Overall system metrics
    total_system_cost = sum(result['simulation_results']['total_cost'] for result in simulation_results.values())
    avg_system_service_level = np.mean([result['simulation_results']['service_level'] for result in simulation_results.values()])
    
    return {
        'individual_products': simulation_results,
        'system_metrics': {
            'total_system_cost': total_system_cost,
            'avg_system_service_level': avg_system_service_level,
            'total_products_simulated': len(simulation_results),
            'simulation_periods': simulation_periods
        },
        'configuration': {
            'safety_factor': safety_factor,
            'lead_time': lead_time,
            'holding_cost_rate': holding_cost_rate,
            'simulation_periods': simulation_periods
        }
    }

def show_prod_inv_demand(simulation_results: Dict[str, Any],
                        product: str,
                        max_periods: int = 100) -> Dict[str, Any]:
    """
    Production/inventory/demand time series visualization
    
    Args:
        simulation_results: Results from inventory_simulation
        product: Product to visualize
        max_periods: Maximum periods to display
        
    Returns:
        Chart data for frontend visualization
    """
    if product not in simulation_results['individual_products']:
        raise ValueError(f"Product {product} not found in simulation results")
    
    prod_data = simulation_results['individual_products'][product]
    time_series = prod_data['time_series']
    
    # Limit data size
    periods = list(range(min(max_periods, len(time_series['inventory_levels']))))
    inventory_levels = time_series['inventory_levels'][:max_periods]
    order_quantities = time_series['order_quantities'][:max_periods]
    daily_costs = time_series['daily_costs'][:max_periods]
    
    return {
        'chart_data': {
            'periods': periods,
            'inventory_levels': inventory_levels,
            'order_quantities': order_quantities,
            'daily_costs': daily_costs
        },
        'policy_info': {
            'safety_stock': prod_data['inventory_policy']['safety_stock'],
            'reorder_point': prod_data['inventory_policy']['reorder_point'],
            'eoq': prod_data['inventory_policy']['eoq']
        },
        'title': f'Production/Inventory/Demand Analysis - {product}',
        'product': product,
        'performance': prod_data['simulation_results']
    }

def plot_demands(demand_df: pd.DataFrame,
                products: List[str] = None,
                max_periods: int = 365) -> Dict[str, Any]:
    """
    Multi-product demand visualization
    
    Args:
        demand_df: Demand DataFrame
        products: List of products to visualize (if None, use top products)
        max_periods: Maximum periods to display
        
    Returns:
        Chart data for multi-product demand visualization
    """
    df = demand_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if products is None:
        # Select top products by total demand
        top_products = df.groupby('prod')['demand'].sum().nlargest(5).index.tolist()
        products = top_products
    
    # Limit products to avoid overwhelming visualization
    products = products[:10]  # Maximum 10 products
    
    chart_data = {
        'dates': [],
        'products': {},
        'title': 'Multi-Product Demand Patterns'
    }
    
    # Get date range
    date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')[:max_periods]
    chart_data['dates'] = [d.strftime('%Y-%m-%d') for d in date_range]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
              '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#16a085']
    
    for i, product in enumerate(products):
        prod_data = df[df['prod'] == product].copy()
        prod_data = prod_data.set_index('date')
        
        # Reindex to daily frequency
        daily_demand = prod_data.reindex(date_range, fill_value=0)['demand']
        
        chart_data['products'][product] = {
            'demands': daily_demand.tolist()[:max_periods],
            'color': colors[i % len(colors)],
            'total_demand': float(daily_demand.sum()),
            'avg_daily_demand': float(daily_demand.mean()),
            'max_daily_demand': float(daily_demand.max())
        }
    
    return chart_data

# SCBAS Class - Comprehensive ABC Analysis Framework
class Scbas(BaseModel):
    """
    Supply Chain Business Analysis System (SCBAS)
    Comprehensive ABC analysis framework matching notebook implementation
    """
    demand_df: Optional[pd.DataFrame] = Field(None, description="Demand DataFrame")
    prod_df: Optional[pd.DataFrame] = Field(None, description="Product DataFrame")  
    cust_df: Optional[pd.DataFrame] = Field(None, description="Customer DataFrame")
    
    # Analysis parameters
    abc_thresholds: List[float] = Field([0.7, 0.2, 0.1], description="ABC classification thresholds")
    safety_factor: float = Field(1.65, description="Safety stock factor")
    lead_time: int = Field(7, description="Lead time in days")
    holding_cost_rate: float = Field(0.2, description="Holding cost rate")
    
    # State management
    _abc_results: Optional[Dict[str, Any]] = None
    _simulation_results: Optional[Dict[str, Any]] = None
    _analysis_state: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **data):
        super().__init__(**data)
        self._analysis_state = {}
    
    def load_data(self, demand_df: pd.DataFrame, 
                  prod_df: pd.DataFrame = None,
                  cust_df: pd.DataFrame = None) -> 'Scbas':
        """Load data into SCBAS framework"""
        self.demand_df = demand_df.copy()
        if prod_df is not None:
            self.prod_df = prod_df.copy()
        if cust_df is not None:
            self.cust_df = cust_df.copy()
        return self
    
    def run_abc_analysis(self, 
                        value_col: str = "demand",
                        product_thresholds: List[float] = None,
                        customer_thresholds: List[float] = None) -> 'Scbas':
        """Run comprehensive ABC analysis"""
        if self.demand_df is None:
            raise ValueError("Demand data must be loaded first")
        
        thresholds = product_thresholds or self.abc_thresholds
        cust_thresholds = customer_thresholds or self.abc_thresholds
        
        # Run ABC analysis for products and customers
        self._abc_results = generate_figures_for_abc_analysis(
            self.demand_df, thresholds, cust_thresholds, value_col
        )
        
        # Store in state
        self._analysis_state['abc_completed'] = True
        self._analysis_state['abc_value_col'] = value_col
        
        return self
    
    def run_simulation(self, simulation_periods: int = 365) -> 'Scbas':
        """Run inventory simulation"""
        if self.demand_df is None:
            raise ValueError("Demand data must be loaded first")
        
        prod_df = self.prod_df if self.prod_df is not None else pd.DataFrame()
        
        self._simulation_results = inventory_simulation(
            self.demand_df, prod_df, self.safety_factor, 
            self.lead_time, self.holding_cost_rate, simulation_periods
        )
        
        self._analysis_state['simulation_completed'] = True
        self._analysis_state['simulation_periods'] = simulation_periods
        
        return self
    
    def get_abc_results(self) -> Dict[str, Any]:
        """Get ABC analysis results"""
        if self._abc_results is None:
            raise ValueError("ABC analysis must be run first")
        return self._abc_results
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """Get simulation results"""
        if self._simulation_results is None:
            raise ValueError("Simulation must be run first")
        return self._simulation_results
    
    def get_treemap_data(self, parent_col: str = "cust", value_col: str = "demand") -> Dict[str, Any]:
        """Generate treemap data"""
        if self.demand_df is None:
            raise ValueError("Demand data must be loaded first")
        return demand_tree_map(self.demand_df, parent_col, value_col)
    
    def get_abc_treemap_data(self, parent_col: str = "cust", value_col: str = "demand") -> Dict[str, Any]:
        """Generate ABC-classified treemap data"""
        if self.demand_df is None or self._abc_results is None:
            raise ValueError("Data and ABC analysis must be completed first")
        return demand_tree_map_with_abc(self.demand_df, parent_col, value_col)
    
    def get_mean_cv_analysis(self, value_col: str = "demand") -> Dict[str, Any]:
        """Get mean vs coefficient of variation analysis"""
        if self.demand_df is None:
            raise ValueError("Demand data must be loaded first")
        return show_mean_cv(self.demand_df, self.prod_df, value_col)
    
    def get_risk_pooling_analysis(self, agg_period: str = "1W") -> pd.DataFrame:
        """Get risk pooling analysis"""
        if self.demand_df is None:
            raise ValueError("Demand data must be loaded first")
        return risk_pooling_analysis(self.demand_df, agg_period)
    
    def get_rank_analysis(self, agg_period: str = "1M", top_rank: int = 10) -> Dict[str, Any]:
        """Get ranking analysis"""
        if self.demand_df is None:
            raise ValueError("Demand data must be loaded first")
        
        agg_df_prod = None
        if self._abc_results is not None:
            agg_df_prod = pd.DataFrame(self._abc_results.get('agg_df_prod', []))
        
        return show_rank_analysis(self.demand_df, agg_df_prod, 
                                "demand", agg_period, top_rank)
    
    def get_inventory_analysis(self) -> Dict[str, Any]:
        """Get comprehensive inventory analysis"""
        if self.demand_df is None:
            raise ValueError("Demand data must be loaded first")
        
        prod_df = self.prod_df if self.prod_df is not None else pd.DataFrame()
        return inventory_analysis(self.demand_df, prod_df, 
                                self.safety_factor, self.lead_time, self.holding_cost_rate)
    
    def generate_ui_node_structure(self) -> Dict[str, Any]:
        """Generate hierarchical node structure for UI components"""
        if self.demand_df is None:
            raise ValueError("Data must be loaded first")
        
        # Create hierarchical structure for UI
        nodes = []
        
        # Product nodes
        if 'prod' in self.demand_df.columns:
            product_summary = self.demand_df.groupby('prod')['demand'].agg(['sum', 'count', 'mean']).reset_index()
            for _, row in product_summary.iterrows():
                nodes.append({
                    'id': f"prod_{row['prod']}",
                    'label': str(row['prod']),
                    'type': 'product',
                    'value': float(row['sum']),
                    'count': int(row['count']),
                    'avg': float(row['mean']),
                    'parent': 'products'
                })
        
        # Customer nodes
        if 'cust' in self.demand_df.columns:
            customer_summary = self.demand_df.groupby('cust')['demand'].agg(['sum', 'count', 'mean']).reset_index()
            for _, row in customer_summary.iterrows():
                nodes.append({
                    'id': f"cust_{row['cust']}",
                    'label': str(row['cust']),
                    'type': 'customer',
                    'value': float(row['sum']),
                    'count': int(row['count']),
                    'avg': float(row['mean']),
                    'parent': 'customers'
                })
        
        # Root nodes
        root_nodes = [
            {'id': 'products', 'label': 'Products', 'type': 'category', 'parent': None},
            {'id': 'customers', 'label': 'Customers', 'type': 'category', 'parent': None}
        ]
        
        return {
            'nodes': root_nodes + nodes,
            'total_nodes': len(nodes) + len(root_nodes),
            'categories': ['products', 'customers'],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'data_summary': {
                    'total_demand': float(self.demand_df['demand'].sum()),
                    'unique_products': int(self.demand_df['prod'].nunique()) if 'prod' in self.demand_df.columns else 0,
                    'unique_customers': int(self.demand_df['cust'].nunique()) if 'cust' in self.demand_df.columns else 0
                }
            }
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        summary = {
            'scbas_version': '1.0',
            'analysis_state': self._analysis_state.copy(),
            'configuration': {
                'abc_thresholds': self.abc_thresholds,
                'safety_factor': self.safety_factor,
                'lead_time': self.lead_time,
                'holding_cost_rate': self.holding_cost_rate
            },
            'data_loaded': {
                'demand_df': self.demand_df is not None,
                'prod_df': self.prod_df is not None,
                'cust_df': self.cust_df is not None
            },
            'analyses_completed': {
                'abc_analysis': self._abc_results is not None,
                'simulation': self._simulation_results is not None
            }
        }
        
        if self.demand_df is not None:
            summary['data_summary'] = {
                'total_records': len(self.demand_df),
                'date_range': {
                    'start': str(self.demand_df['date'].min()) if 'date' in self.demand_df.columns else None,
                    'end': str(self.demand_df['date'].max()) if 'date' in self.demand_df.columns else None
                },
                'total_demand': float(self.demand_df['demand'].sum()),
                'unique_products': int(self.demand_df['prod'].nunique()) if 'prod' in self.demand_df.columns else 0,
                'unique_customers': int(self.demand_df['cust'].nunique()) if 'cust' in self.demand_df.columns else 0
            }
        
        return summary

def promotion_effect_analysis(demand_df: pd.DataFrame,
                             promo_col: str = "promo_0",
                             value_col: str = "demand",
                             product_col: str = "prod") -> Dict[str, Any]:
    """
    Analyze promotion effects on demand patterns
    Exact implementation from notebook for promo_0, promo_1 flags
    
    Args:
        demand_df: Demand DataFrame with promotion flags
        promo_col: Promotion flag column (promo_0 or promo_1)
        value_col: Value column for analysis
        product_col: Product column
        
    Returns:
        Promotion effect analysis results
    """
    if promo_col not in demand_df.columns:
        raise ValueError(f"Promotion column '{promo_col}' not found in data")
    
    results = {}
    
    # Overall promotion effect
    promo_data = demand_df[demand_df[promo_col] == 1]
    non_promo_data = demand_df[demand_df[promo_col] == 0]
    
    if len(promo_data) == 0:
        return {
            "error": "No promotion periods found",
            "promo_column": promo_col,
            "total_records": len(demand_df)
        }
    
    # Calculate overall metrics
    promo_avg = promo_data[value_col].mean()
    non_promo_avg = non_promo_data[value_col].mean()
    lift_percentage = ((promo_avg - non_promo_avg) / non_promo_avg * 100) if non_promo_avg > 0 else 0
    
    results['overall_effect'] = {
        'promotion_periods': len(promo_data),
        'non_promotion_periods': len(non_promo_data),
        'average_demand_with_promo': float(promo_avg),
        'average_demand_without_promo': float(non_promo_avg),
        'demand_lift_percentage': float(lift_percentage),
        'total_promo_demand': float(promo_data[value_col].sum()),
        'total_non_promo_demand': float(non_promo_data[value_col].sum())
    }
    
    # Product-level promotion effect
    product_effects = {}
    for product in demand_df[product_col].unique():
        prod_data = demand_df[demand_df[product_col] == product]
        prod_promo = prod_data[prod_data[promo_col] == 1]
        prod_non_promo = prod_data[prod_data[promo_col] == 0]
        
        if len(prod_promo) > 0 and len(prod_non_promo) > 0:
            prod_promo_avg = prod_promo[value_col].mean()
            prod_non_promo_avg = prod_non_promo[value_col].mean()
            prod_lift = ((prod_promo_avg - prod_non_promo_avg) / prod_non_promo_avg * 100) if prod_non_promo_avg > 0 else 0
            
            product_effects[product] = {
                'promotion_periods': len(prod_promo),
                'non_promotion_periods': len(prod_non_promo),
                'avg_demand_with_promo': float(prod_promo_avg),
                'avg_demand_without_promo': float(prod_non_promo_avg),
                'demand_lift_percentage': float(prod_lift),
                'promo_effectiveness': 'High' if prod_lift > 50 else 'Medium' if prod_lift > 20 else 'Low'
            }
    
    results['product_level_effects'] = product_effects
    
    # ABC analysis with promotion consideration
    # Separate ABC analysis for promotion and non-promotion periods
    if len(promo_data) > 0:
        promo_agg_df, promo_enhanced_df, promo_categories = abc_analysis(
            promo_data, [0.7, 0.2, 0.1], product_col, value_col, "promo_abc", "promo_rank"
        )
        results['promotion_abc'] = {
            'categories': promo_categories,
            'top_products': promo_agg_df.head(10)[product_col].tolist() if len(promo_agg_df) > 0 else []
        }
    
    if len(non_promo_data) > 0:
        non_promo_agg_df, non_promo_enhanced_df, non_promo_categories = abc_analysis(
            non_promo_data, [0.7, 0.2, 0.1], product_col, value_col, "non_promo_abc", "non_promo_rank"
        )
        results['non_promotion_abc'] = {
            'categories': non_promo_categories,
            'top_products': non_promo_agg_df.head(10)[product_col].tolist() if len(non_promo_agg_df) > 0 else []
        }
    
    # Statistical significance (simple t-test approximation)
    import scipy.stats as stats
    if len(promo_data) > 1 and len(non_promo_data) > 1:
        t_stat, p_value = stats.ttest_ind(
            promo_data[value_col].dropna(), 
            non_promo_data[value_col].dropna()
        )
        results['statistical_significance'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'confidence_level': '95%'
        }
    
    # Recommendations
    recommendations = []
    if lift_percentage > 50:
        recommendations.append("High promotion effectiveness - consider expanding promotion frequency")
    elif lift_percentage > 20:
        recommendations.append("Moderate promotion effectiveness - optimize promotion timing")
    elif lift_percentage > 0:
        recommendations.append("Low promotion effectiveness - review promotion strategy")
    else:
        recommendations.append("Negative promotion effect - investigate promotion cannibalization")
    
    # Identify most/least responsive products
    if product_effects:
        most_responsive = max(product_effects.items(), key=lambda x: x[1]['demand_lift_percentage'])
        least_responsive = min(product_effects.items(), key=lambda x: x[1]['demand_lift_percentage'])
        
        recommendations.append(f"Most promotion-responsive product: {most_responsive[0]} ({most_responsive[1]['demand_lift_percentage']:.1f}% lift)")
        recommendations.append(f"Least promotion-responsive product: {least_responsive[0]} ({least_responsive[1]['demand_lift_percentage']:.1f}% lift)")
    
    results['recommendations'] = recommendations
    results['analysis_metadata'] = {
        'promotion_column': promo_col,
        'value_column': value_col,
        'product_column': product_col,
        'total_products_analyzed': len(product_effects),
        'analysis_period': {
            'start': str(demand_df['date'].min()) if 'date' in demand_df.columns else None,
            'end': str(demand_df['date'].max()) if 'date' in demand_df.columns else None
        }
    }
    
    return results