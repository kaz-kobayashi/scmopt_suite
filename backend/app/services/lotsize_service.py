import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import optimization libraries
try:
    from gurobipy import Model, GRB, quicksum
    USE_GUROBI = True
except ImportError:
    USE_GUROBI = False

# Always import PuLP as fallback
try:
    from pulp import *
    USE_PULP = True
except ImportError:
    USE_PULP = False

class LotsizeOptimizationService:
    """
    Lot Size Optimization System (OptLot)
    Exact implementation from 11lotsize.ipynb notebook
    
    Supports:
    - Dynamic lot sizing optimization
    - Multi-stage lot sizing with BOM
    - Multi-mode production optimization
    - Resource capacity constraints
    - Both Gurobi and PuLP solvers
    """
    
    def __init__(self):
        self.use_gurobi = USE_GUROBI
        self.use_pulp = USE_PULP
        if not self.use_gurobi and not self.use_pulp:
            raise ImportError("Neither Gurobi nor PuLP is available. Please install at least one optimization solver.")
        
    def lotsizing(self, prod_df: pd.DataFrame, production_df: pd.DataFrame, 
                  bom_df: pd.DataFrame, demand: np.ndarray, resource_df: pd.DataFrame,
                  max_cpu: int = 10, solver: str = "CBC") -> Tuple[Any, int]:
        """
        ロットサイズ決定問題を解く関数
        Exact implementation from notebook
        
        Args:
            prod_df: 品目データフレーム
            production_df: 生産情報データフレーム
            bom_df: 部品展開表データフレーム
            demand: （期別・品目別の）需要を入れた配列（行が品目，列が期）
            resource_df: 資源データフレーム
            max_cpu: 最大計算時間（秒）
            solver: ソルバータイプ ("GRB", "CBC", "SCIP")
            
        Returns:
            model: ロットサイズ決定モデルのオブジェクト
            T: 計画期間数
        """
        
        # 原材料と完成品の抽出
        if bom_df is not None and len(bom_df) > 0:
            raw_materials = set(bom_df["child"])
            products = set(bom_df["parent"])
            items = raw_materials | products
        else:
            if 'name' not in prod_df.columns:
                prod_df.reset_index(inplace=True)
            products = set(prod_df['name'])
            items = products
            raw_materials = set()
            
        # 計画期間の数の抽出
        _, T = demand.shape
        
        # 資源の容量の抽出
        M = {}
        if resource_df is not None and len(resource_df) > 0:
            for row in resource_df.itertuples():
                M[row.name, row.period] = row.capacity

        # 親子関係の辞書を作成
        parent = defaultdict(set)
        phi = defaultdict(float)  # qを１単位生産するために必要なpのunit数
        if bom_df is not None and len(bom_df) > 0:
            for row in bom_df.itertuples():
                parent[row.child].add(row.parent)
                phi[row.child, row.parent] = row.units

        # モデル作成
        if self.use_gurobi and solver == "GRB":
            model = Model()
        else:
            model = LpProblem("LotsizeOptimization", LpMinimize)
            
        x, I, y = {}, {}, {}
        slack, surplus = {}, {}
        Ts = range(0, T)
        
        # 原材料の変数作成
        if bom_df is not None and len(bom_df) > 0:
            for i, p in enumerate(raw_materials):
                for t in Ts:
                    if self.use_gurobi and solver == "GRB":
                        x[t, p] = model.addVar(name=f"x[{p},{t}]")
                        I[t, p] = model.addVar(name=f"I[{p},{t}]", 
                                             ub=float(prod_df.loc[prod_df['name']==p, 'target_inventory'].iloc[0]))
                        y[t, p] = model.addVar(name=f"y[{p},{t}]", vtype=GRB.BINARY)
                        slack[t, p] = model.addVar(name=f"slack[{p},{t}]")
                        surplus[t, p] = model.addVar(name=f"surplus[{p},{t}]")
                    else:
                        x[t, p] = LpVariable(f"x_{p}_{t}", lowBound=0)
                        I[t, p] = LpVariable(f"I_{p}_{t}", lowBound=0, 
                                           upBound=float(prod_df.loc[prod_df['name']==p, 'target_inventory'].iloc[0]))
                        y[t, p] = LpVariable(f"y_{p}_{t}", cat='Binary')
                        slack[t, p] = LpVariable(f"slack_{p}_{t}", lowBound=0)
                        surplus[t, p] = LpVariable(f"surplus_{p}_{t}", lowBound=0)
                        
                # 初期在庫（デフォルト値を使用）
                initial_inventory = prod_df.loc[prod_df['name']==p, 'initial_inventory']
                I[-1, p] = float(initial_inventory.iloc[0]) if len(initial_inventory) > 0 and not initial_inventory.isna().iloc[0] else 0.0
                
                # 最終期の在庫量（安全在庫、なければ0）
                safety_inventory = prod_df.loc[prod_df['name']==p].get('safety_inventory', pd.Series([0.0]))
                I[T-1, p] = float(safety_inventory.iloc[0]) if len(safety_inventory) > 0 and not safety_inventory.isna().iloc[0] else 0.0

        # 完成品の変数作成
        for i, p in enumerate(products):
            for t in Ts:
                if self.use_gurobi and solver == "GRB":
                    x[t, p] = model.addVar(name=f"x[{p},{t}]")
                    # 安全在庫の下限を設定（なければ0）
                    safety_inventory = prod_df.loc[prod_df['name']==p].get('safety_inventory', pd.Series([0.0]))
                    safety_lb = float(safety_inventory.iloc[0]) if len(safety_inventory) > 0 and not safety_inventory.isna().iloc[0] else 0.0
                    target_ub = float(prod_df.loc[prod_df['name']==p, 'target_inventory'].iloc[0])
                    
                    I[t, p] = model.addVar(name=f"I[{p},{t}]", 
                                         lb=safety_lb, 
                                         ub=target_ub)
                    y[t, p] = model.addVar(name=f"y[{p},{t}]", vtype=GRB.BINARY)
                    slack[t, p] = model.addVar(name=f"slack[{p},{t}]")
                    surplus[t, p] = model.addVar(name=f"surplus[{p},{t}]")
                else:
                    x[t, p] = LpVariable(f"x_{p}_{t}", lowBound=0)
                    # 安全在庫の下限を設定（なければ0）
                    safety_inventory = prod_df.loc[prod_df['name']==p].get('safety_inventory', pd.Series([0.0]))
                    safety_lb = float(safety_inventory.iloc[0]) if len(safety_inventory) > 0 and not safety_inventory.isna().iloc[0] else 0.0
                    target_ub = float(prod_df.loc[prod_df['name']==p, 'target_inventory'].iloc[0])
                    
                    I[t, p] = LpVariable(f"I_{p}_{t}", lowBound=safety_lb, upBound=target_ub)
                    y[t, p] = LpVariable(f"y_{p}_{t}", cat='Binary')
                    slack[t, p] = LpVariable(f"slack_{p}_{t}", lowBound=0)
                    surplus[t, p] = LpVariable(f"surplus_{p}_{t}", lowBound=0)
                    
            # 初期在庫（デフォルト値を使用）
            initial_inventory = prod_df.loc[prod_df['name']==p, 'initial_inventory']
            I[-1, p] = float(initial_inventory.iloc[0]) if len(initial_inventory) > 0 and not initial_inventory.isna().iloc[0] else 0.0
            
            # 最終期の在庫量（安全在庫、なければ0）
            safety_inventory = prod_df.loc[prod_df['name']==p].get('safety_inventory', pd.Series([0.0]))
            I[T-1, p] = float(safety_inventory.iloc[0]) if len(safety_inventory) > 0 and not safety_inventory.isna().iloc[0] else 0.0

        if self.use_gurobi and solver == "GRB":
            model.update()

        # 制約条件の追加
        for t in Ts:
            # 原材料の時間容量制約
            if bom_df is not None and len(bom_df) > 0:
                if self.use_gurobi and solver == "GRB":
                    model.addConstr(
                        quicksum(production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] +
                               production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0]*y[t, p] 
                               for p in raw_materials) <= M["Res0", t],
                        f"TimeConstraint0({t})")
                else:
                    model += (lpSum([production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] +
                                   production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0]*y[t, p] 
                                   for p in raw_materials]) <= M["Res0", t])

                for p in raw_materials:
                    # フロー保存制約
                    if self.use_gurobi and solver == "GRB":
                        model.addConstr(
                            I[t-1, p] + x[t, p] + slack[t, p] - surplus[t, p] == 
                            I[t, p] + quicksum(phi[p, q]*x[t, q] for q in parent[p]), 
                            f"FlowCons({t},{p})")
                        
                        # 容量接続制約
                        model.addConstr(
                            production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] <= 
                            (M["Res0", t] - production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0])*y[t, p], 
                            f"ConstrUB({t},{p})")
                        
                        # 最小ロット制約
                        model.addConstr(x[t, p] >= 0.*y[t, p], f"ConstrLB({t},{p})")
                    else:
                        model += (I[t-1, p] + x[t, p] + slack[t, p] - surplus[t, p] == 
                                I[t, p] + lpSum([phi[p, q]*x[t, q] for q in parent[p]]))
                        
                        model += (production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] <= 
                                (M["Res0", t] - production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0])*y[t, p])
                        
                        model += (x[t, p] >= 0.*y[t, p])

            # 完成品の時間容量制約
            if self.use_gurobi and solver == "GRB":
                model.addConstr(
                    quicksum(production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] +
                           production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0]*y[t, p]
                           for p in products) <= M["Res1", t], 
                    f"TimeConstraint1({t})")
            else:
                model += (lpSum([production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] +
                               production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0]*y[t, p]
                               for p in products]) <= M["Res1", t])

            for i, p in enumerate(products):
                # フロー保存制約
                if self.use_gurobi and solver == "GRB":
                    model.addConstr(
                        I[t-1, p] + x[t, p] + slack[t, p] - surplus[t, p] ==
                        I[t, p] + demand[i, t], f"FlowCons({t},{p})")

                    # 容量接続制約
                    model.addConstr(
                        production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] <=
                        (M["Res1", t] - production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0])*y[t, p],
                        f"ConstrUB({t},{p})")

                    # 最小ロット制約
                    model.addConstr(x[t, p] >= 0.*y[t, p], f"ConstrLB({t},{p})")

                    # 強化制約
                    model.addConstr(x[t, p] <= demand[i, t]*y[t, p] + I[t, p], f"Tighten({t},{p})")
                else:
                    model += (I[t-1, p] + x[t, p] + slack[t, p] - surplus[t, p] ==
                            I[t, p] + demand[i, t])
                    
                    model += (production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]*x[t, p] <=
                            (M["Res1", t] - production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0])*y[t, p])
                    
                    model += (x[t, p] >= 0.*y[t, p])
                    model += (x[t, p] <= demand[i, t]*y[t, p] + I[t, p])

        # 目的関数の設定
        if bom_df is not None and len(bom_df) > 0:
            if self.use_gurobi and solver == "GRB":
                model.setObjective(
                    quicksum(99999999.*slack[t, p] + 9999999.*surplus[t, p] +
                           production_df.loc[production_df['name']==p, 'SetupCost'].iloc[0]*y[t, p] + 
                           production_df.loc[production_df['name']==p, 'ProdCost'].iloc[0]*x[t, p] 
                           for t in Ts for p in items)
                    + quicksum(prod_df.loc[prod_df['name']==p, 'inv_cost'].iloc[0]*I[t, p]
                             for t in Ts for p in products)
                    + quicksum(99999.*I[t, p] for t in Ts for p in raw_materials),
                    GRB.MINIMIZE)
            else:
                model += (lpSum([99999999.*slack[t, p] + 9999999.*surplus[t, p] +
                               production_df.loc[production_df['name']==p, 'SetupCost'].iloc[0]*y[t, p] + 
                               production_df.loc[production_df['name']==p, 'ProdCost'].iloc[0]*x[t, p] 
                               for t in Ts for p in items])
                        + lpSum([prod_df.loc[prod_df['name']==p, 'inv_cost'].iloc[0]*I[t, p]
                               for t in Ts for p in products])
                        + lpSum([99999.*I[t, p] for t in Ts for p in raw_materials]))
        else:
            if self.use_gurobi and solver == "GRB":
                model.setObjective(
                    quicksum(99999999.*slack[t, p] + 9999999.*surplus[t, p] +
                           production_df.loc[production_df['name']==p, 'SetupCost'].iloc[0]*y[t, p] + 
                           production_df.loc[production_df['name']==p, 'ProdCost'].iloc[0]*x[t, p] 
                           for t in Ts for p in items)
                    + quicksum(prod_df.loc[prod_df['name']==p, 'inv_cost'].iloc[0]*I[t, p]
                             for t in Ts for p in products),
                    GRB.MINIMIZE)
            else:
                model += (lpSum([99999999.*slack[t, p] + 9999999.*surplus[t, p] +
                               production_df.loc[production_df['name']==p, 'SetupCost'].iloc[0]*y[t, p] + 
                               production_df.loc[production_df['name']==p, 'ProdCost'].iloc[0]*x[t, p] 
                               for t in Ts for p in items])
                        + lpSum([prod_df.loc[prod_df['name']==p, 'inv_cost'].iloc[0]*I[t, p]
                               for t in Ts for p in products]))

        # データ保存
        model.__data = x, I, y, slack, surplus

        # 最適化実行
        if solver == "GRB" and self.use_gurobi:
            model.Params.TimeLimit = max_cpu
            model.optimize()
        elif solver == "CBC":
            solver_obj = PULP_CBC_CMD(timeLimit=max_cpu, presolve=True, msg=0)
            model.solve(solver_obj)
        elif solver == "SCIP":
            solver_obj = SCIP_CMD(timeLimit=max_cpu, msg=0)
            model.solve(solver_obj)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        return model, T

    def show_result_for_lotsizing(self, model: Any, T: int, prod_df: pd.DataFrame, 
                                 production_df: pd.DataFrame, bom_df: pd.DataFrame, 
                                 resource_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        最適化結果から図とデータフレームを生成する関数
        Exact implementation from notebook
        
        Args:
            model: ロットサイズ決定モデル
            T: 計画期間
            production_df: 生産情報データフレーム 
            bom_df: 部品展開表データフレーム
            resource_df: 資源データフレーム
            
        Returns:
            violated: 需要満足条件を逸脱した品目と期と逸脱量を保存したデータフレーム
            production: 生産量を保管したデータフレーム
            inventory: 在庫量を保管したデータフレーム
            fig_inv: 在庫量の推移を表した図オブジェクト
            fig_capacity: 容量制約を表した図オブジェクト
        """
        
        if bom_df is not None and len(bom_df) > 0:
            raw_materials = set(bom_df["child"])
            products = set(bom_df["parent"])
            items = raw_materials | products
        else:
            if 'name' not in prod_df.columns:
                prod_df.reset_index(inplace=True)
            products = set(prod_df['name'])
            items = products
            raw_materials = set()
            
        num_item = len(items)
        prod_array = np.zeros(shape=(num_item, T))
        inv_array = np.zeros(shape=(num_item, T))
        x, I, y, slack, surplus = model.__data

        slack_, surplus_, period_, prod_ = [], [], [], []
        for i, p in enumerate(items):
            for t in range(T):
                slack_val = slack[t, p].X if hasattr(slack[t, p], 'X') else slack[t, p].varValue
                surplus_val = surplus[t, p].X if hasattr(surplus[t, p], 'X') else surplus[t, p].varValue
                
                if slack_val and slack_val > 0.001:
                    period_.append(t)
                    prod_.append(p)
                    slack_.append(slack_val)
                    surplus_.append(0.)
                if surplus_val and surplus_val > 0.001:
                    period_.append(t)
                    prod_.append(p)
                    slack_.append(0.)
                    surplus_.append(surplus_val)

        violated = pd.DataFrame({"prod": prod_, "period": period_, "slack": slack_, "surplus": surplus_})

        items_list = list(items)
        for i, p in enumerate(items_list):
            for t in range(T):
                prod_time = production_df.loc[production_df['name']==p, 'ProdTime'].iloc[0]
                setup_time = production_df.loc[production_df['name']==p, 'SetupTime'].iloc[0]
                
                x_val = x[t, p].X if hasattr(x[t, p], 'X') else x[t, p].varValue
                y_val = y[t, p].X if hasattr(y[t, p], 'X') else y[t, p].varValue
                
                prod_array[i, t] = prod_time * (x_val or 0) + setup_time * (y_val or 0)
                
                try:
                    inv_val = I[t, p].X if hasattr(I[t, p], 'X') else I[t, p].varValue
                    inv_array[i, t] = inv_val or 0
                except (AttributeError, TypeError):
                    inv_array[i, t] = I[t, p] if isinstance(I[t, p], (int, float)) else 0

        production = pd.DataFrame(prod_array, index=items_list)
        inventory = pd.DataFrame(inv_array, index=items_list)

        # 在庫の推移の図データ
        fig_inv_data = {
            'data': [],
            'layout': {
                'title': 'Inventory Levels',
                'xaxis': {'title': '期'},
                'yaxis': {'title': '在庫量'},
                'autosize': True
            }
        }
        
        for i in inventory.index:
            trace_data = {
                'x': list(range(T)),
                'y': inventory.loc[i].tolist(),
                'name': str(i),
                'type': 'scatter',
                'mode': 'lines+markers'
            }
            fig_inv_data['data'].append(trace_data)

        # 容量制約の図データ
        M = {}
        for row in resource_df.itertuples():
            M[row.name, row.period] = row.capacity

        if bom_df is not None and len(bom_df) > 0:
            fig_capacity_data = {
                'data': [],
                'layout': {
                    'title': 'Production Capacity',
                    'barmode': 'stack',
                    'yaxis': {'title': '生産量'},
                    'autosize': True
                }
            }
            
            # Raw materials
            for i in raw_materials:
                trace_data = {
                    'x': list(range(T)),
                    'y': production.loc[i].tolist(),
                    'name': f'Raw Material {i}',
                    'type': 'bar'
                }
                fig_capacity_data['data'].append(trace_data)
            
            # Raw material capacity line
            capacity_trace = {
                'x': list(range(T)),
                'y': [M.get(("Res0", t), 0) for t in range(T)],
                'name': 'Raw Material Capacity',
                'type': 'scatter',
                'mode': 'lines'
            }
            fig_capacity_data['data'].append(capacity_trace)
            
            # Products
            for i in products:
                trace_data = {
                    'x': list(range(T)),
                    'y': production.loc[i].tolist(),
                    'name': f'Product {i}',
                    'type': 'bar'
                }
                fig_capacity_data['data'].append(trace_data)
            
            # Product capacity line
            product_capacity_trace = {
                'x': list(range(T)),
                'y': [M.get(("Res1", t), 0) for t in range(T)],
                'name': 'Product Capacity',
                'type': 'scatter',
                'mode': 'lines'
            }
            fig_capacity_data['data'].append(product_capacity_trace)
        else:
            fig_capacity_data = {
                'data': [],
                'layout': {
                    'title': 'Production Capacity',
                    'barmode': 'stack',
                    'yaxis': {'title': '生産量'},
                    'autosize': True
                }
            }
            
            # Capacity line
            capacity_trace = {
                'x': list(range(T)),
                'y': [M.get(("Res1", t), 0) for t in range(T)],
                'name': 'Product Capacity',
                'type': 'scatter',
                'mode': 'lines'
            }
            fig_capacity_data['data'].append(capacity_trace)
            
            # Products
            for i in products:
                trace_data = {
                    'x': list(range(T)),
                    'y': production.loc[i].tolist(),
                    'name': f'Product {i}',
                    'type': 'bar'
                }
                fig_capacity_data['data'].append(trace_data)

        return violated, production, inventory, fig_inv_data, fig_capacity_data

    def multi_mode_lotsizing(self, item_df: pd.DataFrame, resource_df: pd.DataFrame, 
                            process_df: pd.DataFrame, bom_df: pd.DataFrame, usage_df: pd.DataFrame,
                            demand: Dict[Tuple[int, str], float], capacity: Dict[Tuple[int, str], float], 
                            T: int = 1, fix_x: Optional[Dict] = None) -> Any:
        """
        多モードロットサイズ決定問題を解く関数
        Exact implementation from notebook
        
        Args:
            item_df: 品目データフレーム
            resource_df: 資源データフレーム
            process_df: 工程データフレーム
            bom_df: 部品展開表データフレーム
            usage_df: 資源使用量データフレーム
            demand: 需要量を入れた辞書
            capacity: 資源量上限を入れた辞書
            T: 計画期間数
            fix_x: 変数固定情報を入れた辞書
            
        Returns:
            model: モデルオブジェクト
        """
        
        INF = 9999999999.
        h = {}  # 在庫費用
        IUB, ILB = {}, {}  # 在庫量の下限と上限
        
        for row in item_df.itertuples():
            item_name = row[1]
            h[item_name] = row[2]
            ILB[item_name] = row[3] if not pd.isnull(row[3]) else 0.
            IUB[item_name] = row[4] if not pd.isnull(row[4]) else INF

        # 親子関係や資源必要量の辞書を作成
        parent = defaultdict(set)  # 子品目pを必要とする親品目とモードの組の集合
        phi = defaultdict(float)  # 親品目qをモードmで１単位生産するために必要な子品目pのunit数
        modes = defaultdict(set)  # 親品目qのモード集合
        resources = defaultdict(set)  # 親品目qをモードmで生産するときに必要な資源の集合
        setup_time = defaultdict(float)
        setup_cost = defaultdict(float)
        prod_time = defaultdict(float)
        prod_cost = defaultdict(float)

        items = item_df.iloc[:, 0]  # 品目のリストを準備
        item_set = set(items)
        resource_set = set(resource_df.iloc[:, 0])

        # プロセス情報の処理
        current_item = None
        for row in process_df.itertuples():
            if row[1] is not None:
                current_item = row[1]
            if row[2] is not None:  # モード
                m = row[2]
                modes[current_item].add(m)
            
            # 費用
            setup_cost[current_item, m] = row[3] if row[3] is not None and not np.isnan(row[3]) else 0.
            prod_cost[current_item, m] = row[4] if row[4] is not None and not np.isnan(row[4]) else 0.

        # 全品目にモードがあることを確認
        for q in items:
            if len(modes[q]) == 0:
                raise ValueError(f"品目{q}にモードがありません．")

        # BOM情報の処理
        current_item = None
        current_mode = None
        for row in bom_df.itertuples():
            if row[1] is not None:
                current_item = row[1]
            if row[2] is not None:  # モード
                current_mode = row[2]
            if row[3] is not None:
                p = row[3]
                phi[p, current_item, current_mode] = row[4]
                parent[p].add((current_item, current_mode))
            
            if current_item not in item_set:
                raise ValueError(f"品目{current_item}が品目シートにありません．")
            if p not in item_set:
                raise ValueError(f"品目{p}が品目シートにありません．")

        # 資源使用量情報の処理
        current_item = None
        current_mode = None
        for row in usage_df.itertuples():
            if row[1] is not None:
                current_item = row[1]
            if row[2] is not None:  # モード
                current_mode = row[2]
            # 資源と時間
            if row[3] is not None:
                r = row[3]
                if r in resource_set and current_item in item_set:
                    resources[current_item, current_mode].add(r)
                    setup_time[current_item, current_mode, r] = row[4] if row[4] is not None and not np.isnan(row[4]) else 0.
                    prod_time[current_item, current_mode, r] = row[5] if row[5] is not None and not np.isnan(row[5]) else 0.

        item_modes = defaultdict(set)  # 資源rを使用する品目とモードの組の集合（resourcesの逆写像）
        for key in resources:
            for r in resources[key]:
                item_modes[r].add(key)

        # モデル作成
        if self.use_gurobi:
            model = Model()
        else:
            model = LpProblem("MultiModeLotsizing", LpMinimize)
            
        x, I, y = {}, {}, {}
        slack, surplus = {}, {}
        inv_slack, inv_surplus = {}, {}
        Ts = range(0, T)

        # 変数作成
        for row in item_df.itertuples():
            p = row[1]
            for m in modes[p]:
                for t in Ts:
                    if self.use_gurobi:
                        x[t, m, p] = model.addVar(name=f"x({p},{m},{t})")
                        I[t, p] = model.addVar(name=f"I({p},{t})")
                        y[t, m, p] = model.addVar(name=f"y({p},{m},{t})", vtype=GRB.BINARY)
                        slack[t, p] = model.addVar(name=f"slack({p},{t})")
                        surplus[t, p] = model.addVar(name=f"surplus({p},{t})")
                        inv_slack[t, p] = model.addVar(name=f"inv_slack({p},{t})")
                        inv_surplus[t, p] = model.addVar(name=f"inv_surplus({p},{t})")
                    else:
                        x[t, m, p] = LpVariable(f"x_{p}_{m}_{t}", lowBound=0)
                        I[t, p] = LpVariable(f"I_{p}_{t}", lowBound=0)
                        y[t, m, p] = LpVariable(f"y_{p}_{m}_{t}", cat='Binary')
                        slack[t, p] = LpVariable(f"slack_{p}_{t}", lowBound=0)
                        surplus[t, p] = LpVariable(f"surplus_{p}_{t}", lowBound=0)
                        inv_slack[t, p] = LpVariable(f"inv_slack_{p}_{t}", lowBound=0)
                        inv_surplus[t, p] = LpVariable(f"inv_surplus_{p}_{t}", lowBound=0)
                        
            # 初期在庫と最終在庫
            I[-1, p] = row[5] if not pd.isnull(row[5]) else 0.  # 初期在庫
            I[T-1, p] = row[6] if not pd.isnull(row[6]) else 0.  # 最終期の在庫量

        # 各費用項目を別途合計する
        cost = {}
        for i in range(5):
            if self.use_gurobi:
                cost[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"cost[{i}]")
            else:
                cost[i] = LpVariable(f"cost_{i}", lowBound=0)

        if self.use_gurobi:
            model.update()

        # 変数の固定
        if fix_x is not None:
            for (t, m, p) in fix_x:
                if self.use_gurobi:
                    model.addConstr(x[t, m, p] == fix_x[t, m, p])
                else:
                    model += (x[t, m, p] == fix_x[t, m, p])

        # 在庫量の上下限の逸脱の計算
        for t in Ts:
            for p in items:
                if self.use_gurobi:
                    model.addConstr(I[t, p] <= IUB[p] + inv_surplus[t, p], f"IUB({t},{p})")
                    model.addConstr(ILB[p] <= I[t, p] + inv_slack[t, p], f"ILB({t},{p})")
                else:
                    model += (I[t, p] <= IUB[p] + inv_surplus[t, p])
                    model += (ILB[p] <= I[t, p] + inv_slack[t, p])

        # 資源容量制約
        for row in resource_df.itertuples():
            r = row[1]
            for t in Ts:
                if self.use_gurobi:
                    model.addConstr(
                        quicksum(prod_time[p, m, r]*x[t, m, p] + setup_time[p, m, r]*y[t, m, p] 
                               for (p, m) in item_modes[r]) <= capacity[t, r],
                        f"TimeConstraint1({r},{t})")
                else:
                    model += (lpSum([prod_time[p, m, r]*x[t, m, p] + setup_time[p, m, r]*y[t, m, p] 
                                   for (p, m) in item_modes[r]]) <= capacity[t, r])

        # フロー保存制約
        for t in Ts:
            for p in items:
                if self.use_gurobi:
                    model.addConstr(
                        I[t-1, p] + quicksum(x[t, m, p] for m in modes[p]) + slack[t, p] - surplus[t, p] == 
                        I[t, p] + demand.get((t, p), 0) + 
                        quicksum(phi[p, q, m]*x[t, m, q] for (q, m) in parent[p]), 
                        f"FlowCons({t},{p})")
                else:
                    model += (I[t-1, p] + lpSum([x[t, m, p] for m in modes[p]]) + slack[t, p] - surplus[t, p] == 
                            I[t, p] + demand.get((t, p), 0) + 
                            lpSum([phi[p, q, m]*x[t, m, q] for (q, m) in parent[p]]))

        # 容量接続制約
        for t in Ts:
            for p in items:
                for m in modes[p]:
                    for r in resources[p, m]:
                        if self.use_gurobi:
                            model.addConstr(
                                prod_time[p, m, r]*x[t, m, p] <= 
                                (capacity[t, r] - setup_time[p, m, r])*y[t, m, p], 
                                f"ConstrUB({t},{m},{r},{p})")
                        else:
                            model += (prod_time[p, m, r]*x[t, m, p] <= 
                                    (capacity[t, r] - setup_time[p, m, r])*y[t, m, p])

        # 費用制約
        if self.use_gurobi:
            model.addConstr(
                quicksum(slack[t, p] + surplus[t, p] for t in Ts for p in items) == cost[0])
            model.addConstr(
                quicksum(inv_slack[t, p] + inv_surplus[t, p] for t in Ts for p in items) == cost[1])
            model.addConstr(
                quicksum(setup_cost[p, m]*y[t, m, p] 
                       for t in Ts for p in items for m in modes[p] 
                       for r in resources[p, m]) == cost[2])
            model.addConstr(
                quicksum(prod_cost[p, m]*x[t, m, p] 
                       for t in Ts for p in items for m in modes[p] 
                       for r in resources[p, m]) == cost[3])
            model.addConstr(
                quicksum(h[p]*I[t, p] for t in Ts for p in items) == cost[4])

            model.setObjective(
                9999999.*cost[0] + 999999.*cost[1] + quicksum(cost[i] for i in range(2, 5)), 
                GRB.MINIMIZE)
        else:
            model += (lpSum([slack[t, p] + surplus[t, p] for t in Ts for p in items]) == cost[0])
            model += (lpSum([inv_slack[t, p] + inv_surplus[t, p] for t in Ts for p in items]) == cost[1])
            model += (lpSum([setup_cost[p, m]*y[t, m, p] 
                           for t in Ts for p in items for m in modes[p] 
                           for r in resources[p, m]]) == cost[2])
            model += (lpSum([prod_cost[p, m]*x[t, m, p] 
                           for t in Ts for p in items for m in modes[p] 
                           for r in resources[p, m]]) == cost[3])
            model += (lpSum([h[p]*I[t, p] for t in Ts for p in items]) == cost[4])

            model += (9999999.*cost[0] + 999999.*cost[1] + lpSum([cost[i] for i in range(2, 5)]))

        # データ保存
        model.__data = (x, I, y, slack, surplus, inv_slack, inv_surplus, cost, items, modes, 
                       item_modes, setup_time, prod_time, parent, resources)

        return model

    def make_cost_df(self, cost: Dict[int, Any]) -> pd.DataFrame:
        """
        費用内訳のデータフレームを生成する関数
        Exact implementation from notebook
        
        Args:
            cost: 費用の変数を入れた辞書
            
        Returns:
            費用内訳のデータフレーム
        """
        
        cost_values = {}
        for i, cost_var in cost.items():
            if hasattr(cost_var, 'X'):
                cost_values[i] = cost_var.X
            elif hasattr(cost_var, 'varValue'):
                cost_values[i] = cost_var.varValue
            else:
                cost_values[i] = 0.0

        df = pd.DataFrame({
            "需要逸脱ペナルティ": cost_values.get(0, 0),
            "在庫上下限逸脱ペナルティ": cost_values.get(1, 0),
            "段取り費用": cost_values.get(2, 0),
            "生産変動費用": cost_values.get(3, 0),
            "在庫費用": cost_values.get(4, 0)
        }, index=["費用"])
        
        return df.T

    def generate_demand_from_order(self, order_data: List[Dict[str, Any]], start: str, finish: str,
                                  period: int = 1, period_unit: str = "日") -> Tuple[Dict[Tuple[int, str], float], int]:
        """
        注文Workbookから需要を生成する関数
        Exact implementation from notebook
        
        Args:
            order_data: 注文情報のリスト
            start: 開始日
            finish: 終了日
            period: 期を構成する単位期間の数
            period_unit: 期の単位
            
        Returns:
            demand: 需要量を入れた辞書
            T: 計画期間数
        """
        
        trans = {"時": "h", "日": "d", "週": "w", "月": "M"}
        
        if period_unit == "週":
            weekday = pd.to_datetime(start).strftime('%a')
            freq = f"{period}{trans[period_unit]}-{weekday}"
        else:
            freq = f"{period}{trans[period_unit]}"

        # データフレーム作成
        order_df = pd.DataFrame(order_data)
        order_df.columns = ["品目(ID)", "納期", "数量"]
        order_df["納期"] = pd.to_datetime(order_df["納期"])
        order_df.set_index("納期", inplace=True)
        
        order_grouped = order_df.groupby(["品目(ID)"]).resample(freq)["数量"].sum()
        
        # 期間インデックス生成
        time_index = pd.date_range(start, finish, freq=freq)
        T = len(time_index)
        
        to_idx = {}  # 日付を期IDに変換するための辞書
        for t, t_idx in enumerate(time_index):
            date_ = t_idx.strftime('%Y-%m-%d')
            to_idx[date_] = t
        
        # 需要量 demand[t,p]
        demand = defaultdict(float)
        for idx in order_grouped.keys():
            date_ = idx[1].strftime('%Y-%m-%d')
            val = order_grouped[idx]
            if date_ in to_idx:
                demand[to_idx[date_], idx[0]] = float(val)
        
        return dict(demand), T

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about lot size optimization service capabilities
        
        Returns:
            Dict: Service information
        """
        return {
            "lotsize_optimization": {
                "description": "Complete lot size optimization system (OptLot)",
                "version": "1.0.0",
                "features": [
                    "Dynamic lot sizing optimization",
                    "Multi-stage lot sizing with BOM",
                    "Multi-mode production optimization", 
                    "Resource capacity constraints",
                    "Excel template generation and processing",
                    "Advanced visualization and analysis"
                ]
            },
            "optimization_models": {
                "single_stage": "Basic lot sizing with capacity constraints",
                "multi_stage": "Multi-level BOM with assembly/disassembly",
                "multi_mode": "Multiple production modes per item",
                "facility_location": "Facility location formulation support"
            },
            "solver_support": {
                "primary": "Gurobi (commercial solver)",
                "fallback": "PuLP with CBC/SCIP",
                "automatic_detection": True
            },
            "excel_integration": {
                "template_generation": "Master data and order templates",
                "result_export": "Formatted optimization results",
                "rolling_horizon": "Variable fixing support"
            }
        }