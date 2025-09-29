# PyVRP Implementation for Metro VI Delivery Planning System
# Compatible with existing metro system architecture

from typing import List, Optional, Union, Tuple, Dict, Set, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel
import ast
import datetime as dt

# PyVRP imports (from the existing code we can see it's already imported)
from pyvrp import Model as PyVRPModel
from pyvrp import (
    GeneticAlgorithm,
    Population, 
    PenaltyManager,
    read,
    Solution as PyVRPSolution,
    Route, 
    CostEvaluator,
    Client, 
    ProblemData, 
    VehicleType, 
    RandomNumberGenerator,
)

# Import existing classes from metro system
from scmopt2.core import *

class PyVRPModelBuilder:
    """PyVRPモデル構築用のクラス"""
    
    def __init__(self):
        self.model = PyVRPModel()
        self.location_mapping = {}  # 既存location_indexとPyVRP内部IDのマッピング
        self.client_mapping = {}    # Job IDとPyVRP Client IDのマッピング
        self.depot_mapping = {}     # デポのマッピング
        self.vehicle_mapping = {}   # 運搬車のマッピング
        
    def add_depot(self, x: float, y: float, name: str = "", location_index: int = None) -> int:
        """デポを追加"""
        # 座標をスケーリング（PyVRPは整数座標を期待）
        scaled_x = int(x * 1000000)  # 経度を6桁精度で整数化
        scaled_y = int(y * 1000000)  # 緯度を6桁精度で整数化
        
        depot = self.model.add_depot(
            x=scaled_x,
            y=scaled_y,
            name=name
        )
        
        if location_index is not None:
            self.location_mapping[location_index] = depot
        
        return depot
        
    def add_client(self, x: float, y: float, demand: int = 0, 
                  service_duration: int = 0, tw_early: int = 0, tw_late: int = 86400,
                  required: bool = True, prize: int = 0, name: str = "", 
                  location_index: int = None, job_id: int = None) -> int:
        """顧客を追加"""
        # 座標をスケーリング
        scaled_x = int(x * 1000000)
        scaled_y = int(y * 1000000)
        
        client = self.model.add_client(
            x=scaled_x,
            y=scaled_y,
            demand=demand,
            service_duration=service_duration,
            tw_early=tw_early,
            tw_late=tw_late,
            required=required,
            prize=prize,
            name=name
        )
        
        if location_index is not None:
            self.location_mapping[location_index] = client
        if job_id is not None:
            self.client_mapping[job_id] = client
            
        return client
        
    def add_vehicle_type(self, num_available: int = 1, capacity: int = 1000,
                        depot=None, fixed_cost: int = 0, tw_early: int = 0, 
                        tw_late: int = 86400, max_duration: int = None,
                        max_distance: int = None, name: str = "") -> int:
        """車両タイプを追加"""
        vehicle_type = self.model.add_vehicle_type(
            num_available=num_available,
            capacity=capacity,
            depot=depot,
            fixed_cost=fixed_cost,
            tw_early=tw_early,
            tw_late=tw_late,
            max_duration=max_duration,
            max_distance=max_distance,
            name=name
        )
        
        return vehicle_type
        
    def add_edges_from_matrix(self, duration_matrix: List[List[int]], 
                             distance_matrix: List[List[int]] = None):
        """距離・時間行列からエッジを追加"""
        n = len(duration_matrix)
        locations = list(self.location_mapping.values())
        
        if distance_matrix is None:
            distance_matrix = duration_matrix  # 時間を距離として使用
            
        for i in range(n):
            for j in range(n):
                if i != j and i < len(locations) and j < len(locations):
                    self.model.add_edge(
                        locations[i], 
                        locations[j], 
                        distance=distance_matrix[i][j],
                        duration=duration_matrix[i][j]
                    )

def convert_metro_to_pyvrp(model: 'Model') -> PyVRPModelBuilder:
    """既存のMetroモデルをPyVRPモデルに変換"""
    builder = PyVRPModelBuilder()
    
    # デポの追加（運搬車の出発地から抽出）
    depot_locations = set()
    for vehicle in model.vehicles:
        if vehicle.start_index is not None:
            depot_locations.add(vehicle.start_index)
        if vehicle.end_index is not None:
            depot_locations.add(vehicle.end_index)
    
    # ジョブから位置情報を取得してデポを追加
    if hasattr(model, 'matrices') and 'car' in model.matrices:
        # 移動時間行列がある場合、全ての位置を確認
        matrix_size = len(model.matrices['car'].durations)
        
        # デポの追加
        for depot_idx in depot_locations:
            if depot_idx < matrix_size:
                # デポの座標情報を取得（存在する場合）
                depot_name = f"depot_{depot_idx}"
                # 仮の座標（実際の実装では適切な座標を設定）
                builder.add_depot(x=139.0 + depot_idx * 0.01, y=35.0 + depot_idx * 0.01, 
                                name=depot_name, location_index=depot_idx)
    
    # 顧客（ジョブ）の追加
    if model.jobs:
        for job in model.jobs:
            # 需要量（配達量を使用）
            demand = job.delivery[0] if job.delivery else 0
            
            # 時間枠の処理
            tw_early, tw_late = 0, 86400  # デフォルト24時間
            if job.time_windows and len(job.time_windows) > 0:
                tw_early, tw_late = job.time_windows[0]  # 最初の時間枠を使用
            
            # 座標の取得
            if job.location:
                x, y = job.location[0], job.location[1]
            else:
                # location_indexからダミー座標を生成
                x = 139.0 + job.location_index * 0.001 if job.location_index else 139.0
                y = 35.0 + job.location_index * 0.001 if job.location_index else 35.0
            
            # 優先度をprizeに変換（優先度が高いほど大きなprizeにする）
            prize = job.priority * 100 if job.priority else 0
            required = job.priority > 0 if job.priority else True
            
            builder.add_client(
                x=x, y=y,
                demand=demand,
                service_duration=job.service if job.service else 0,
                tw_early=tw_early,
                tw_late=tw_late,
                required=required,
                prize=prize,
                name=job.description if job.description else f"job_{job.id}",
                location_index=job.location_index,
                job_id=job.id
            )
    
    # 運搬車の追加
    if model.vehicles:
        for vehicle in model.vehicles:
            # 容量の取得
            capacity = vehicle.capacity[0] if vehicle.capacity else 1000
            
            # 時間枠の処理
            tw_early, tw_late = 0, 86400
            if vehicle.time_window:
                tw_early, tw_late = vehicle.time_window
            
            # デポの取得
            depot = None
            if vehicle.start_index is not None and vehicle.start_index in builder.location_mapping:
                depot = builder.location_mapping[vehicle.start_index]
            
            # 固定費用
            fixed_cost = vehicle.costs.fixed if vehicle.costs else 0
            
            # 最大稼働時間
            max_duration = vehicle.max_travel_time
            
            builder.add_vehicle_type(
                num_available=1,
                capacity=capacity,
                depot=depot,
                fixed_cost=fixed_cost,
                tw_early=tw_early,
                tw_late=tw_late,
                max_duration=max_duration,
                name=vehicle.description if vehicle.description else f"vehicle_{vehicle.id}"
            )
    
    # 距離・時間行列の追加
    if hasattr(model, 'matrices') and 'car' in model.matrices:
        duration_matrix = model.matrices['car'].durations
        distance_matrix = model.matrices['car'].distances if model.matrices['car'].distances else None
        builder.add_edges_from_matrix(duration_matrix, distance_matrix)
    
    return builder

def optimize_with_pyvrp(model: 'Model', max_iterations: int = 1000, 
                       max_runtime: int = 60, seed: int = 1234) -> PyVRPSolution:
    """PyVRPを使用した最適化"""
    # MetroモデルをPyVRPモデルに変換
    builder = convert_metro_to_pyvrp(model)
    
    # 最適化の実行
    from pyvrp.stop import MaxIterations, MaxRuntime
    
    result = builder.model.solve(
        stop=MaxRuntime(max_runtime),
        seed=seed
    )
    
    return result, builder

def convert_pyvrp_solution_to_metro(solution: PyVRPSolution, builder: PyVRPModelBuilder) -> dict:
    """PyVRPソリューションを既存形式に変換"""
    output_dic = {
        "code": 0,
        "summary": {
            "cost": solution.cost(),
            "routes": len(solution.best.get_routes()),
            "unassigned": len(solution.best.get_unassigned()),
            "duration": solution.cost(),
        },
        "routes": [],
        "unassigned": []
    }
    
    # ルート情報の変換
    for route_idx, route in enumerate(solution.best.get_routes()):
        route_data = {
            "vehicle": route_idx,
            "cost": route.cost(),
            "duration": route.duration(),
            "steps": []
        }
        
        # ルートのステップを変換
        for client_idx in route:
            if client_idx == 0:  # デポ
                step = {
                    "type": "start" if not route_data["steps"] else "end",
                    "location_index": 0,
                    "arrival": 0,
                    "service": 0,
                    "waiting_time": 0
                }
            else:
                step = {
                    "type": "job",
                    "location_index": client_idx,
                    "id": client_idx,  # 実際にはマッピングが必要
                    "arrival": 0,  # 実際の到着時間計算が必要
                    "service": 0,
                    "waiting_time": 0
                }
            route_data["steps"].append(step)
        
        output_dic["routes"].append(route_data)
    
    return output_dic

# メイン関数
def optimize_vrp_with_pyvrp(model: 'Model', max_runtime: int = 60, 
                           seed: int = 1234) -> Tuple[dict, dict, str]:
    """PyVRPを使用してVRP問題を最適化"""
    try:
        # PyVRPで最適化
        solution, builder = optimize_with_pyvrp(model, max_runtime=max_runtime, seed=seed)
        
        # 結果を既存形式に変換
        output_dic = convert_pyvrp_solution_to_metro(solution, builder)
        
        # 入力データの準備（既存形式との互換性のため）
        input_dic = {
            "jobs": [],
            "vehicles": [],
            "matrices": {}
        }
        
        # ジョブ情報の追加
        if model.jobs:
            for job in model.jobs:
                job_data = {
                    "id": job.id,
                    "location": job.location if job.location else [139.0, 35.0],
                    "description": job.description,
                    "pickup": job.pickup,
                    "delivery": job.delivery,
                    "service": job.service,
                    "time_windows": job.time_windows,
                    "priority": job.priority
                }
                input_dic["jobs"].append(job_data)
        
        # 車両情報の追加
        if model.vehicles:
            for vehicle in model.vehicles:
                vehicle_data = {
                    "id": vehicle.id,
                    "start": vehicle.start,
                    "end": vehicle.end,
                    "capacity": vehicle.capacity,
                    "time_window": vehicle.time_window,
                    "description": vehicle.description
                }
                input_dic["vehicles"].append(vehicle_data)
        
        error = ""
        
    except Exception as e:
        output_dic = {"code": -1, "error": str(e)}
        input_dic = {}
        error = str(e)
    
    return input_dic, output_dic, error

print("PyVRP implementation module loaded successfully")