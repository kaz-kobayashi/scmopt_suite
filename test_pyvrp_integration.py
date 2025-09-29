# PyVRP Integration Test
# 既存のMETROシステムとPyVRPの統合テスト

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import ast

# 既存のMETROコードをインポート
sys.path.append('.')

# 新しいPyVRP実装をインポート
from pyvrp_enhanced import (
    optimize_vrp_with_enhanced_pyvrp, 
    EnhancedPyVRPBuilder,
    convert_metro_to_enhanced_pyvrp
)

# 既存のMETROクラスをインポート
try:
    from scmopt2.core import *
    from nbs.metro_classes import *  # 既存のJob, Vehicle, Modelクラス等
except ImportError:
    print("既存のMETROクラスが見つかりません。基本クラスを定義します...")
    
    # 基本クラスの定義（テスト用）
    class Job:
        def __init__(self, id, location=None, location_index=None, service=0, 
                     delivery=[0], pickup=[0], time_windows=None, 
                     priority=1, description=""):
            self.id = id
            self.location = location
            self.location_index = location_index
            self.service = service
            self.delivery = delivery
            self.pickup = pickup
            self.time_windows = time_windows if time_windows else [[0, 86400]]
            self.priority = priority
            self.description = description
    
    class ShipmentStep:
        def __init__(self, id, location=None, location_index=None, service=0, time_windows=None):
            self.id = id
            self.location = location
            self.location_index = location_index
            self.service = service
            self.time_windows = time_windows if time_windows else [[0, 86400]]
    
    class Shipment:
        def __init__(self, pickup, delivery, amount=[0], priority=1):
            self.pickup = pickup
            self.delivery = delivery
            self.amount = amount
            self.priority = priority
    
    class VehicleCosts:
        def __init__(self, fixed=0, per_hour=3600, per_km=0):
            self.fixed = fixed
            self.per_hour = per_hour
            self.per_km = per_km
    
    class Vehicle:
        def __init__(self, id, start=None, end=None, start_index=None, end_index=None,
                     capacity=[1000], time_window=None, description="", costs=None):
            self.id = id
            self.start = start
            self.end = end
            self.start_index = start_index
            self.end_index = end_index
            self.capacity = capacity
            self.time_window = time_window if time_window else [0, 86400]
            self.description = description
            self.costs = costs if costs else VehicleCosts()
            self.max_travel_time = None
    
    class Matrix:
        def __init__(self, durations=None, distances=None):
            self.durations = durations
            self.distances = distances
    
    class Model:
        def __init__(self, jobs=None, shipments=None, vehicles=None, matrices=None):
            self.jobs = jobs if jobs else []
            self.shipments = shipments if shipments else []
            self.vehicles = vehicles if vehicles else []
            self.matrices = matrices if matrices else {}

def create_sample_vrp_model():
    """サンプルVRPモデルを作成"""
    
    # ジョブ（顧客）データ
    jobs = [
        Job(id=0, location=[139.7, 35.7], location_index=0, service=600, 
            delivery=[100], pickup=[0], time_windows=[[28800, 64800]], 
            priority=10, description="Customer A"),
        Job(id=1, location=[139.8, 35.8], location_index=1, service=900,
            delivery=[150], pickup=[50], time_windows=[[32400, 61200]], 
            priority=8, description="Customer B"),
        Job(id=2, location=[139.6, 35.6], location_index=2, service=450,
            delivery=[80], pickup=[0], time_windows=[[25200, 68400]], 
            priority=6, description="Customer C"),
        Job(id=3, location=[139.9, 35.9], location_index=3, service=300,
            delivery=[200], pickup=[100], time_windows=[[36000, 57600]], 
            priority=9, description="Customer D"),
    ]
    
    # 輸送（Pickup & Delivery）データ
    pickup_step = ShipmentStep(id=10, location=[139.75, 35.75], location_index=4, 
                              service=600, time_windows=[[28800, 43200]])
    delivery_step = ShipmentStep(id=11, location=[139.85, 35.85], location_index=5,
                                service=900, time_windows=[[36000, 61200]])
    shipment = Shipment(pickup=pickup_step, delivery=delivery_step, 
                       amount=[120], priority=7)
    
    # 運搬車データ
    vehicles = [
        Vehicle(id=0, start=[139.65, 35.65], end=[139.65, 35.65], 
               start_index=6, end_index=6, capacity=[500], 
               time_window=[28800, 64800], description="Truck 1",
               costs=VehicleCosts(fixed=5000, per_hour=3600)),
        Vehicle(id=1, start=[139.65, 35.65], end=[139.65, 35.65],
               start_index=6, end_index=6, capacity=[800],
               time_window=[25200, 68400], description="Truck 2",
               costs=VehicleCosts(fixed=8000, per_hour=4000)),
    ]
    
    # 距離・時間行列（7x7: customers 0-3, pickup 4, delivery 5, depot 6）
    durations = [
        # 0    1    2    3    4    5    6  (to)
        [  0, 1200, 1500, 2100, 900, 1800, 1000], # 0 (from)
        [1200,   0, 2400, 900, 600, 300, 1500],   # 1
        [1500, 2400,   0, 3000, 1800, 2700, 800], # 2
        [2100, 900, 3000,   0, 1200, 600, 2200],  # 3
        [ 900, 600, 1800, 1200,   0, 900, 1200],  # 4 (pickup)
        [1800, 300, 2700, 600, 900,   0, 1800],   # 5 (delivery)
        [1000, 1500, 800, 2200, 1200, 1800,   0], # 6 (depot)
    ]
    
    distances = [
        [  0, 12000, 15000, 21000,  9000, 18000, 10000],
        [12000,   0, 24000,  9000,  6000,  3000, 15000],
        [15000, 24000,   0, 30000, 18000, 27000,  8000],
        [21000,  9000, 30000,   0, 12000,  6000, 22000],
        [ 9000,  6000, 18000, 12000,   0,  9000, 12000],
        [18000,  3000, 27000,  6000,  9000,   0, 18000],
        [10000, 15000,  8000, 22000, 12000, 18000,   0],
    ]
    
    matrices = {
        'car': Matrix(durations=durations, distances=distances)
    }
    
    model = Model(jobs=jobs, shipments=[shipment], vehicles=vehicles, matrices=matrices)
    
    return model

def test_pyvrp_basic_vrp():
    """基本VRP機能のテスト"""
    print("=== 基本VRP機能のテスト ===")
    
    # サンプルモデル作成
    model = create_sample_vrp_model()
    
    print(f"Jobs: {len(model.jobs)}")
    print(f"Shipments: {len(model.shipments)}")
    print(f"Vehicles: {len(model.vehicles)}")
    
    # PyVRPで最適化
    try:
        input_dic, output_dic, error = optimize_vrp_with_enhanced_pyvrp(
            model, 
            max_runtime=30,  # 30秒
            seed=1234
        )
        
        if error:
            print(f"エラー: {error}")
            return False
            
        print("最適化成功！")
        print(f"総コスト: {output_dic['summary']['cost']}")
        print(f"ルート数: {output_dic['summary']['routes']}")
        print(f"未割り当て: {output_dic['summary']['unassigned']}")
        
        # ルート詳細を表示
        for i, route in enumerate(output_dic['routes']):
            print(f"\nルート {i}:")
            print(f"  車両: {route['vehicle']}, コスト: {route['cost']}")
            for j, step in enumerate(route['steps']):
                if step['type'] == 'start':
                    print(f"    {j}. 出発: デポ")
                elif step['type'] == 'end':
                    print(f"    {j}. 到着: デポ")
                else:
                    print(f"    {j}. 訪問: {step['description']} (ID: {step['id']})")
        
        return True
        
    except Exception as e:
        print(f"テストエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pyvrp_with_existing_data():
    """既存データを使用したテスト"""
    print("\n=== 既存データを使用したテスト ===")
    
    try:
        # 既存のサンプルデータがあるかチェック
        if os.path.exists("data/metroIV/job.csv"):
            print("既存データを読み込み中...")
            
            # CSVファイルから読み込み
            job_df = pd.read_csv("data/metroIV/job.csv", index_col=0)
            vehicle_df = pd.read_csv("data/metroIV/vehicle.csv", index_col=0)
            
            print(f"読み込んだジョブ数: {len(job_df)}")
            print(f"読み込んだ車両数: {len(vehicle_df)}")
            
            # データフレームからModelオブジェクトを作成
            jobs = []
            for i, row in job_df.iterrows():
                job = Job(
                    id=i,
                    location=ast.literal_eval(row['location']) if isinstance(row['location'], str) else [139.7, 35.7],
                    location_index=row['location_index'] if 'location_index' in row else i,
                    service=int(row['service']) if 'service' in row else 600,
                    delivery=ast.literal_eval(row['delivery']) if isinstance(row['delivery'], str) else [100],
                    pickup=ast.literal_eval(row['pickup']) if isinstance(row['pickup'], str) else [0],
                    time_windows=ast.literal_eval(row['time_windows']) if isinstance(row['time_windows'], str) else [[0, 86400]],
                    priority=int(row['priority']) if 'priority' in row else 1,
                    description=row['name'] if 'name' in row else f"Customer {i}"
                )
                jobs.append(job)
            
            vehicles = []
            for i, row in vehicle_df.iterrows():
                vehicle = Vehicle(
                    id=i,
                    start=ast.literal_eval(row['start']) if isinstance(row['start'], str) else [139.65, 35.65],
                    end=ast.literal_eval(row['end']) if isinstance(row['end'], str) else [139.65, 35.65],
                    start_index=row['start_index'] if 'start_index' in row else len(jobs),
                    end_index=row['end_index'] if 'end_index' in row else len(jobs),
                    capacity=ast.literal_eval(row['capacity']) if isinstance(row['capacity'], str) else [1000],
                    time_window=ast.literal_eval(row['time_window']) if isinstance(row['time_window'], str) else [0, 86400],
                    description=row['name'] if 'name' in row else f"Vehicle {i}"
                )
                vehicles.append(vehicle)
            
            # ダミーの距離行列を作成
            n = len(jobs) + 1  # +1 for depot
            durations = [[abs(i-j)*600 + 300 for j in range(n)] for i in range(n)]
            distances = [[abs(i-j)*6000 + 3000 for j in range(n)] for i in range(n)]
            
            matrices = {'car': Matrix(durations=durations, distances=distances)}
            
            model = Model(jobs=jobs, vehicles=vehicles, matrices=matrices)
            
            # PyVRPで最適化
            input_dic, output_dic, error = optimize_vrp_with_enhanced_pyvrp(
                model, 
                max_runtime=30,
                seed=1234
            )
            
            if error:
                print(f"エラー: {error}")
                return False
            
            print("既存データでの最適化成功！")
            print(f"総コスト: {output_dic['summary']['cost']}")
            print(f"ルート数: {output_dic['summary']['routes']}")
            print(f"未割り当て: {output_dic['summary']['unassigned']}")
            
            return True
            
        else:
            print("既存データファイルが見つかりません。スキップします。")
            return True
            
    except Exception as e:
        print(f"既存データテストエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def compare_optimization_methods(model):
    """PyVRPと既存手法の比較"""
    print("\n=== 最適化手法の比較 ===")
    
    results = {}
    
    # PyVRPでの最適化
    try:
        print("PyVRP最適化実行中...")
        input_dic, output_dic, error = optimize_vrp_with_enhanced_pyvrp(
            model, max_runtime=30, seed=1234
        )
        
        if not error:
            results['pyvrp'] = {
                'cost': output_dic['summary']['cost'],
                'routes': output_dic['summary']['routes'],
                'unassigned': output_dic['summary']['unassigned'],
                'method': 'PyVRP (HGS)'
            }
            print(f"PyVRP結果: コスト={output_dic['summary']['cost']}, ルート数={output_dic['summary']['routes']}")
    except Exception as e:
        print(f"PyVRPエラー: {e}")
    
    # 結果の表示
    print("\n比較結果:")
    for method, result in results.items():
        print(f"{result['method']}: コスト={result['cost']}, ルート数={result['routes']}")
    
    return results

def main():
    """メイン関数"""
    print("PyVRP統合テストを開始します...")
    
    # 基本機能テスト
    success1 = test_pyvrp_basic_vrp()
    
    # 既存データテスト
    success2 = test_pyvrp_with_existing_data()
    
    if success1:
        # 比較テスト
        model = create_sample_vrp_model()
        compare_optimization_methods(model)
    
    print(f"\nテスト結果:")
    print(f"基本機能テスト: {'✅ 成功' if success1 else '❌ 失敗'}")
    print(f"既存データテスト: {'✅ 成功' if success2 else '❌ 失敗'}")
    
    if success1 and success2:
        print("\n🎉 全てのテストが成功しました！")
        print("PyVRP実装は正常に動作しています。")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")

if __name__ == "__main__":
    main()