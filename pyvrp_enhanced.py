# Enhanced PyVRP Implementation with Advanced Features
# Supports Pickup & Delivery, Multiple Depots, Time Windows, and more

from typing import List, Optional, Union, Tuple, Dict, Set, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel
import ast
import datetime as dt
from copy import deepcopy

# PyVRP imports
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

from pyvrp.stop import MaxIterations, MaxRuntime, NoImprovement

class EnhancedPyVRPBuilder:
    """拡張PyVRPモデル構築クラス"""
    
    def __init__(self):
        self.model = PyVRPModel()
        self.location_mapping = {}      # 既存location_index -> PyVRP location
        self.client_mapping = {}        # Job ID -> PyVRP Client
        self.depot_mapping = {}         # Depot index -> PyVRP Depot
        self.vehicle_mapping = {}       # Vehicle ID -> PyVRP VehicleType
        self.pickup_delivery_pairs = [] # Pickup-Delivery ペア
        self.shipments = []             # Shipment情報
        
    def add_depot(self, x: float, y: float, name: str = "", location_index: int = None) -> int:
        """デポを追加"""
        scaled_x = int(x * 1000000)
        scaled_y = int(y * 1000000)
        
        depot = self.model.add_depot(
            x=scaled_x,
            y=scaled_y,
            name=name
        )
        
        if location_index is not None:
            self.location_mapping[location_index] = depot
            self.depot_mapping[location_index] = depot
        
        return depot
        
    def add_client(self, x: float, y: float, 
                  service_duration: int = 0, tw_early: int = 0, tw_late: int = 86400,
                  required: bool = True, prize: int = 0, name: str = "", 
                  location_index: int = None, job_id: int = None, 
                  pickup: List[int] = None, delivery: List[int] = None) -> int:
        """顧客を追加（集荷・配達対応）"""
        scaled_x = int(x * 1000000)
        scaled_y = int(y * 1000000)
        
        # デフォルト値の設定
        if pickup is None:
            pickup = [0]
        if delivery is None:
            delivery = [0]
        
        client = self.model.add_client(
            x=scaled_x,
            y=scaled_y,
            delivery=delivery,
            pickup=pickup,
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
        
    def add_pickup_delivery_pair(self, pickup_x: float, pickup_y: float,
                                delivery_x: float, delivery_y: float,
                                amount: int, pickup_service: int = 0, delivery_service: int = 0,
                                pickup_tw_early: int = 0, pickup_tw_late: int = 86400,
                                delivery_tw_early: int = 0, delivery_tw_late: int = 86400,
                                pickup_name: str = "", delivery_name: str = "",
                                pickup_location_index: int = None, 
                                delivery_location_index: int = None) -> Tuple[int, int]:
        """Pickup-Deliveryペアを追加"""
        
        # Pickup地点の追加（集荷）
        pickup_client = self.add_client(
            x=pickup_x, y=pickup_y,
            pickup=[amount],  # 集荷量を指定
            delivery=[0],     # 配達量は0
            service_duration=pickup_service,
            tw_early=pickup_tw_early,
            tw_late=pickup_tw_late,
            required=True,
            name=pickup_name,
            location_index=pickup_location_index
        )
        
        # Delivery地点の追加（配達）
        delivery_client = self.add_client(
            x=delivery_x, y=delivery_y,
            pickup=[0],       # 集荷量は0
            delivery=[amount], # 配達量を指定
            service_duration=delivery_service,
            tw_early=delivery_tw_early,
            tw_late=delivery_tw_late,
            required=True,
            name=delivery_name,
            location_index=delivery_location_index
        )
        
        # ペア情報を記録
        self.pickup_delivery_pairs.append({
            'pickup': pickup_client,
            'delivery': delivery_client,
            'amount': amount
        })
        
        return pickup_client, delivery_client
        
    def add_vehicle_type(self, num_available: int = 1, capacity: Union[int, List[int]] = 1000,
                        start_depot=None, end_depot=None, fixed_cost: int = 0, tw_early: int = 0, 
                        tw_late: int = 86400, max_duration: int = None,
                        max_distance: int = None, name: str = "") -> int:
        """車両タイプを追加"""
        # 容量をリスト形式に変換
        if isinstance(capacity, int):
            capacity = [capacity]
            
        vehicle_type = self.model.add_vehicle_type(
            num_available=num_available,
            capacity=capacity,
            start_depot=start_depot,
            end_depot=end_depot,
            fixed_cost=fixed_cost,
            tw_early=tw_early,
            tw_late=tw_late,
            max_duration=max_duration if max_duration else 2147483647,  # max int32
            max_distance=max_distance if max_distance else 2147483647,
            name=name
        )
        
        return vehicle_type
        
    def add_edges_from_matrix(self, duration_matrix: List[List[int]], 
                             distance_matrix: List[List[int]] = None):
        """距離・時間行列からエッジを追加"""
        n = len(duration_matrix)
        
        # location_mappingから順序を保持してlocationsを作成
        locations = []
        for i in range(n):
            if i in self.location_mapping:
                locations.append(self.location_mapping[i])
            else:
                print(f"Warning: No location mapping for index {i}")
                continue
        
        if distance_matrix is None:
            distance_matrix = duration_matrix
            
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    self.model.add_edge(
                        locations[i], 
                        locations[j], 
                        distance=distance_matrix[i][j],
                        duration=duration_matrix[i][j]
                    )

def convert_metro_to_enhanced_pyvrp(model: 'Model') -> EnhancedPyVRPBuilder:
    """既存MetroモデルをEnhanced PyVRPモデルに変換"""
    builder = EnhancedPyVRPBuilder()
    
    # 1. デポの処理
    depot_locations = set()
    depot_coordinates = {}  # location_index -> (x, y)
    
    for vehicle in model.vehicles:
        if vehicle.start_index is not None:
            depot_locations.add(vehicle.start_index)
        if vehicle.end_index is not None:
            depot_locations.add(vehicle.end_index)
        
        # 座標情報の収集
        if vehicle.start and vehicle.start_index is not None:
            depot_coordinates[vehicle.start_index] = (vehicle.start[0], vehicle.start[1])
        if vehicle.end and vehicle.end_index is not None:
            depot_coordinates[vehicle.end_index] = (vehicle.end[0], vehicle.end[1])
    
    # デポを追加
    for depot_idx in depot_locations:
        x, y = depot_coordinates.get(depot_idx, (139.0 + depot_idx * 0.01, 35.0 + depot_idx * 0.01))
        builder.add_depot(x=x, y=y, name=f"depot_{depot_idx}", location_index=depot_idx)
    
    # 2. ジョブの処理
    if model.jobs:
        for job in model.jobs:
            # 座標の取得
            if job.location:
                x, y = job.location[0], job.location[1]
            else:
                x = 139.0 + job.location_index * 0.001 if job.location_index else 139.0
                y = 35.0 + job.location_index * 0.001 if job.location_index else 35.0
            
            # 需要量の処理
            pickup_amount = job.pickup[0] if job.pickup else 0
            delivery_amount = job.delivery[0] if job.delivery else 0
            
            # 時間枠の処理（最初の時間枠を使用）
            tw_early, tw_late = 0, 86400
            if job.time_windows and len(job.time_windows) > 0:
                tw_early, tw_late = job.time_windows[0]
            
            # 優先度をprizeに変換 (基本的に全ての顧客は必須)
            required = True  # デフォルトは必須顧客
            prize = 0  # prizeは使わない (通常のVRPとして扱う)
            
            builder.add_client(
                x=x, y=y,
                service_duration=job.service if job.service else 0,
                tw_early=tw_early,
                tw_late=tw_late,
                required=required,
                prize=prize,
                name=job.description if job.description else f"job_{job.id}",
                location_index=job.location_index,
                job_id=job.id,
                pickup=[pickup_amount],
                delivery=[delivery_amount]
            )
    
    # 3. Shipmentsの処理
    if model.shipments:
        for shipment in model.shipments:
            # Pickup地点の座標
            pickup_x, pickup_y = shipment.pickup.location if shipment.pickup.location else (139.0, 35.0)
            
            # Delivery地点の座標
            delivery_x, delivery_y = shipment.delivery.location if shipment.delivery.location else (139.0, 35.0)
            
            # 輸送量
            amount = shipment.amount[0] if shipment.amount else 0
            
            # 時間枠
            pickup_tw_early, pickup_tw_late = 0, 86400
            if shipment.pickup.time_windows and len(shipment.pickup.time_windows) > 0:
                pickup_tw_early, pickup_tw_late = shipment.pickup.time_windows[0]
                
            delivery_tw_early, delivery_tw_late = 0, 86400
            if shipment.delivery.time_windows and len(shipment.delivery.time_windows) > 0:
                delivery_tw_early, delivery_tw_late = shipment.delivery.time_windows[0]
            
            builder.add_pickup_delivery_pair(
                pickup_x=pickup_x, pickup_y=pickup_y,
                delivery_x=delivery_x, delivery_y=delivery_y,
                amount=amount,
                pickup_service=shipment.pickup.service if shipment.pickup.service else 0,
                delivery_service=shipment.delivery.service if shipment.delivery.service else 0,
                pickup_tw_early=pickup_tw_early,
                pickup_tw_late=pickup_tw_late,
                delivery_tw_early=delivery_tw_early,
                delivery_tw_late=delivery_tw_late,
                pickup_name=f"pickup_{shipment.pickup.id}",
                delivery_name=f"delivery_{shipment.delivery.id}",
                pickup_location_index=shipment.pickup.location_index,
                delivery_location_index=shipment.delivery.location_index
            )
    
    # 4. 車両の処理
    if model.vehicles:
        for vehicle in model.vehicles:
            # 容量
            capacity = vehicle.capacity[0] if vehicle.capacity else 1000
            
            # 時間枠
            tw_early, tw_late = 0, 86400
            if vehicle.time_window:
                tw_early, tw_late = vehicle.time_window
            
            # デポの取得
            start_depot = None
            end_depot = None
            if vehicle.start_index is not None and vehicle.start_index in builder.depot_mapping:
                start_depot = builder.depot_mapping[vehicle.start_index]
            if vehicle.end_index is not None and vehicle.end_index in builder.depot_mapping:
                end_depot = builder.depot_mapping[vehicle.end_index]
            elif start_depot:  # end_indexがない場合はstart_depotと同じにする
                end_depot = start_depot
            
            # 費用
            fixed_cost = vehicle.costs.fixed if vehicle.costs else 0
            
            # 制約
            max_duration = vehicle.max_travel_time
            
            vehicle_type = builder.add_vehicle_type(
                num_available=1,
                capacity=capacity,
                start_depot=start_depot,
                end_depot=end_depot,
                fixed_cost=fixed_cost,
                tw_early=tw_early,
                tw_late=tw_late,
                max_duration=max_duration,
                name=vehicle.description if vehicle.description else f"vehicle_{vehicle.id}"
            )
            
            builder.vehicle_mapping[vehicle.id] = vehicle_type
    
    # 5. 距離・時間行列の処理
    if hasattr(model, 'matrices') and 'car' in model.matrices:
        duration_matrix = model.matrices['car'].durations
        distance_matrix = model.matrices['car'].distances if model.matrices['car'].distances else None
        builder.add_edges_from_matrix(duration_matrix, distance_matrix)
    
    return builder

def optimize_with_enhanced_pyvrp(model: 'Model', max_runtime: int = 60, 
                               max_iterations: int = None, seed: int = 1234,
                               no_improvement_iterations: int = None) -> Tuple[PyVRPSolution, EnhancedPyVRPBuilder]:
    """Enhanced PyVRPを使用した最適化"""
    builder = convert_metro_to_enhanced_pyvrp(model)
    
    # 停止条件の設定
    stop_conditions = []
    
    if max_runtime:
        stop_conditions.append(MaxRuntime(max_runtime))
    
    if max_iterations:
        stop_conditions.append(MaxIterations(max_iterations))
        
    if no_improvement_iterations:
        stop_conditions.append(NoImprovement(no_improvement_iterations))
    
    # デフォルトの停止条件
    if not stop_conditions:
        stop_conditions.append(MaxRuntime(60))
    
    result = builder.model.solve(
        stop=stop_conditions[0],  # 最初の条件を使用
        seed=seed
    )
    
    return result, builder

def convert_enhanced_pyvrp_solution_to_metro(solution: PyVRPSolution, 
                                           builder: EnhancedPyVRPBuilder,
                                           model: 'Model') -> dict:
    """Enhanced PyVRPソリューションを既存Metro形式に変換"""
    
    data = builder.model.data()
    routes = solution.best.routes()
    
    # 未割り当ての計算（訪問されていないクライアントを見つける）
    visited_clients = set()
    for route in routes:
        for client_id in route:
            visited_clients.add(client_id)
    
    # 全クライアントから訪問済みを除外
    # client_mappingの値は整数IDであることを前提とする
    all_clients = set()
    client_id_to_job_mapping = {}  # PyVRP client ID -> original job ID
    
    for job_id, client_obj in builder.client_mapping.items():
        # client_objから実際のIDを取得する方法を探す
        try:
            # client_objが整数の場合
            if isinstance(client_obj, int):
                client_id = client_obj
            else:
                # PyVRPオブジェクトの場合、属性からIDを取得
                client_id = getattr(client_obj, 'idx', None) or getattr(client_obj, 'id', None)
                if client_id is None:
                    # location_mappingを使って逆引き
                    for loc_idx, loc_client in builder.location_mapping.items():
                        if loc_client == client_obj:
                            client_id = loc_idx
                            break
                    if client_id is None:
                        continue
            
            all_clients.add(client_id)
            client_id_to_job_mapping[client_id] = job_id
        except:
            continue
    
    unassigned_clients = all_clients - visited_clients
    
    output_dic = {
        "code": 0,
        "summary": {
            "cost": solution.cost(),
            "routes": len(routes),
            "unassigned": len(unassigned_clients),
            "duration": solution.cost(),
            "service": 0,
            "waiting_time": 0,
            "distance": solution.best.distance() if hasattr(solution.best, 'distance') else solution.cost(),
            "setup": 0,
            "priority": 0,
            "delivery": [0],
            "pickup": [0],
            "violations": []
        },
        "routes": [],
        "unassigned": []
    }
    
    # 未割り当ての処理
    for unassigned_client in unassigned_clients:
        # 対応するジョブIDを取得
        original_job_id = client_id_to_job_mapping.get(unassigned_client, unassigned_client)
        
        # location_indexを逆引き
        original_location_index = None
        for loc_idx, client_obj in builder.location_mapping.items():
            try:
                if isinstance(client_obj, int):
                    client_id = client_obj
                else:
                    client_id = getattr(client_obj, 'idx', None) or getattr(client_obj, 'id', None) or loc_idx
                
                if client_id == unassigned_client:
                    original_location_index = loc_idx
                    break
            except:
                continue
        
        unassigned_data = {
            "id": original_job_id,
            "type": "job",
            "location_index": original_location_index
        }
        output_dic["unassigned"].append(unassigned_data)
    
    # ルート情報の変換
    for route_idx, route in enumerate(routes):
        route_data = {
            "vehicle": route_idx,
            "cost": route.distance_cost() if hasattr(route, 'distance_cost') else 0,
            "duration": route.duration() if hasattr(route, 'duration') else 0,
            "service": route.service_duration() if hasattr(route, 'service_duration') else 0,
            "waiting_time": route.wait_duration() if hasattr(route, 'wait_duration') else 0,
            "distance": route.distance() if hasattr(route, 'distance') else 0,
            "setup": 0,
            "priority": 0,
            "delivery": [0],
            "pickup": [0],
            "violations": [],
            "steps": []
        }
        
        # デポから開始
        start_depot = route.start_depot()
        step = {
            "type": "start",
            "location": [0, 0],  # デポの実際の座標は後で設定
            "location_index": None,
            "arrival": 0,
            "service": 0,
            "waiting_time": 0,
            "duration": 0,
            "load": [0],
            "setup": 0,
            "description": "depot",
            "id": None,
            "job": None
        }
        
        # デポのlocation_indexを逆引き
        for loc_idx, depot_id in builder.depot_mapping.items():
            if depot_id == start_depot:
                step["location_index"] = loc_idx
                break
                
        route_data["steps"].append(step)
        
        # ルートの各クライアントを処理
        current_time = 0
        for client_idx in route:
            try:
                client_data = data.location(client_idx)
                
                step = {
                    "type": "job",
                    "location": [client_data.x / 1000000, client_data.y / 1000000],
                    "location_index": None,
                    "arrival": current_time,
                    "service": client_data.service_duration if hasattr(client_data, 'service_duration') else 0,
                    "waiting_time": 0,
                    "duration": current_time,
                    "load": [0],  # 実際のload計算は複雑なので0で代用
                    "setup": 0,
                    "description": f"client_{client_idx}",
                    "id": client_idx,
                    "job": client_idx
                }
                
                # 元のジョブIDとlocation_indexを逆引き
                for job_id, pyvrp_client_id in builder.client_mapping.items():
                    if pyvrp_client_id == client_idx:
                        step["id"] = job_id
                        step["job"] = job_id
                        step["description"] = f"job_{job_id}"
                        break
                
                for loc_idx, pyvrp_client_id in builder.location_mapping.items():
                    if pyvrp_client_id == client_idx:
                        step["location_index"] = loc_idx
                        break
                
                current_time += step["service"]
                route_data["steps"].append(step)
                
            except Exception as e:
                print(f"Warning: Could not process client {client_idx}: {e}")
                continue
        
        # デポで終了
        end_depot = route.end_depot()
        step = {
            "type": "end",
            "location": [0, 0],  # デポの実際の座標は後で設定
            "location_index": None,
            "arrival": current_time,
            "service": 0,
            "waiting_time": 0,
            "duration": current_time,
            "load": [0],
            "setup": 0,
            "description": "depot",
            "id": None,
            "job": None
        }
        
        # デポのlocation_indexを逆引き
        for loc_idx, depot_id in builder.depot_mapping.items():
            if depot_id == end_depot:
                step["location_index"] = loc_idx
                break
                
        route_data["steps"].append(step)
        
        output_dic["routes"].append(route_data)
    
    return output_dic

def optimize_vrp_with_enhanced_pyvrp(model: 'Model', max_runtime: int = 60, 
                                   max_iterations: int = None, seed: int = 1234,
                                   no_improvement_iterations: int = None) -> Tuple[dict, dict, str]:
    """Enhanced PyVRPを使用してVRP問題を最適化（メイン関数）"""
    try:
        # PyVRPで最適化
        solution, builder = optimize_with_enhanced_pyvrp(
            model, 
            max_runtime=max_runtime,
            max_iterations=max_iterations,
            seed=seed,
            no_improvement_iterations=no_improvement_iterations
        )
        
        # 結果を既存形式に変換
        output_dic = convert_enhanced_pyvrp_solution_to_metro(solution, builder, model)
        
        # 入力データの準備
        input_dic = {
            "jobs": [],
            "shipments": [],
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
        
        # 輸送情報の追加
        if model.shipments:
            for shipment in model.shipments:
                shipment_data = {
                    "pickup": {
                        "location": shipment.pickup.location,
                        "service": shipment.pickup.service,
                        "time_windows": shipment.pickup.time_windows
                    },
                    "delivery": {
                        "location": shipment.delivery.location,
                        "service": shipment.delivery.service,
                        "time_windows": shipment.delivery.time_windows
                    },
                    "amount": shipment.amount,
                    "priority": shipment.priority
                }
                input_dic["shipments"].append(shipment_data)
        
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
        
        # 結果のログ出力
        print(f"PyVRP Optimization completed:")
        print(f"  - Total cost: {solution.cost()}")
        print(f"  - Number of routes: {len(solution.best.routes())}")
        print(f"  - Unassigned clients: {len(output_dic['unassigned'])}")
        
    except Exception as e:
        output_dic = {"code": -1, "error": str(e)}
        input_dic = {}
        error = str(e)
        print(f"PyVRP Optimization failed: {error}")
    
    return input_dic, output_dic, error

print("Enhanced PyVRP implementation module loaded successfully")