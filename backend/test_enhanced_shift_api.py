#!/usr/bin/env python3
"""
Enhanced Shift API Test Script
新機能のテストを含む10shift.ipynb完全実装のテスト
"""

import requests
import json
import sys
import time

BASE_URL = "http://127.0.0.1:8000/api/shift"

def test_generate_sample_data():
    """サンプルデータ生成テスト"""
    print("\n=== サンプルデータ生成テスト ===")
    
    data = {
        "start_date": "2024-01-01",
        "end_date": "2024-01-07", 
        "start_time": "09:00",
        "end_time": "21:00",
        "freq": "1h",
        "job_list": ["レジ打ち", "接客", "商品補充"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate-sample-data", json=data)
        if response.status_code == 200:
            sample_data = response.json()
            print("✓ サンプルデータ生成成功")
            print(f"  - 期間数: {len(sample_data['period_df'])}")
            print(f"  - 日数: {len(sample_data['day_df'])}")
            print(f"  - ジョブ数: {len(sample_data['job_df'])}")
            print(f"  - スタッフ数: {len(sample_data['staff_df'])}")
            print(f"  - 必要人数データ数: {len(sample_data['requirement_df'])}")
            return sample_data
        else:
            print(f"✗ サンプルデータ生成失敗: {response.status_code}")
            print(f"  エラー詳細: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"✗ リクエストエラー: {e}")
        return None

def test_basic_optimization(sample_data):
    """基本最適化テスト"""
    print("\n=== 基本最適化テスト ===")
    
    if not sample_data:
        print("✗ サンプルデータが不足しています")
        return None
        
    # スタッフデータの処理
    processed_staff_df = []
    for staff in sample_data['staff_df']:
        staff_copy = staff.copy()
        # リストに変換
        if isinstance(staff['job_set'], str):
            import ast
            staff_copy['job_set'] = ast.literal_eval(staff['job_set'])
        if isinstance(staff['day_off'], str):
            import ast
            staff_copy['day_off'] = ast.literal_eval(staff['day_off'])
        processed_staff_df.append(staff_copy)
    
    optimization_data = {
        "period_df": sample_data['period_df'],
        "break_df": sample_data['break_df'],
        "day_df": sample_data['day_df'],
        "job_df": sample_data['job_df'],
        "staff_df": processed_staff_df,
        "requirement_df": sample_data['requirement_df'],
        "theta": 1,
        "lb_penalty": 10000,
        "ub_penalty": 0,
        "job_change_penalty": 10,
        "break_penalty": 10000,
        "max_day_penalty": 5000,
        "time_limit": 30,
        "random_seed": 1
    }
    
    try:
        print("最適化実行中...")
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/optimize", json=optimization_data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 基本最適化成功 (実行時間: {end_time - start_time:.2f}秒)")
            print(f"  - ステータス: {result['status']}")
            print(f"  - メッセージ: {result['message']}")
            if 'cost_df' in result:
                total_cost = sum(item['value'] for item in result['cost_df'])
                print(f"  - 総コスト: {total_cost}")
            return result, sample_data
        else:
            print(f"✗ 基本最適化失敗: {response.status_code}")
            print(f"  エラー詳細: {response.text}")
            return None, sample_data
    except requests.exceptions.RequestException as e:
        print(f"✗ リクエストエラー: {e}")
        return None, sample_data

def test_optimization_with_requests(sample_data):
    """日別リクエスト対応最適化テスト"""
    print("\n=== 日別リクエスト対応最適化テスト ===")
    
    if not sample_data:
        print("✗ サンプルデータが不足しています")
        return None
        
    # スタッフデータの処理（リクエスト付き）
    processed_staff_df = []
    for staff in sample_data['staff_df']:
        staff_copy = staff.copy()
        # リストに変換
        if isinstance(staff['job_set'], str):
            import ast
            staff_copy['job_set'] = ast.literal_eval(staff['job_set'])
        if isinstance(staff['day_off'], str):
            import ast
            staff_copy['day_off'] = ast.literal_eval(staff['day_off'])
        
        # 仮想的な日別リクエストを追加
        staff_copy['request'] = {
            0: [2, 3, 4],  # 0日目は2,3,4時間目を希望
            1: [1, 2, 3, 4, 5]  # 1日目は1-5時間目を希望
        }
        processed_staff_df.append(staff_copy)
    
    optimization_data = {
        "period_df": sample_data['period_df'],
        "break_df": sample_data['break_df'],
        "day_df": sample_data['day_df'],
        "job_df": sample_data['job_df'],
        "staff_df": processed_staff_df,
        "requirement_df": sample_data['requirement_df'],
        "theta": 1,
        "lb_penalty": 10000,
        "ub_penalty": 0,
        "job_change_penalty": 10,
        "break_penalty": 10000,
        "max_day_penalty": 5000,
        "time_limit": 30,
        "random_seed": 1
    }
    
    try:
        print("日別リクエスト対応最適化実行中...")
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/optimize-with-requests", json=optimization_data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 日別リクエスト対応最適化成功 (実行時間: {end_time - start_time:.2f}秒)")
            print(f"  - ステータス: {result['status']}")
            print(f"  - メッセージ: {result['message']}")
            if 'cost_df' in result:
                total_cost = sum(item['value'] for item in result['cost_df'])
                print(f"  - 総コスト: {total_cost}")
            return result
        else:
            print(f"✗ 日別リクエスト対応最適化失敗: {response.status_code}")
            print(f"  エラー詳細: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"✗ リクエストエラー: {e}")
        return None

def test_excel_exports(optimization_result, sample_data):
    """Excel出力テスト"""
    print("\n=== Excel出力テスト ===")
    
    if not optimization_result or not sample_data:
        print("✗ 最適化結果またはサンプルデータが不足しています")
        return
    
    # 1. 基本Excel出力テスト
    try:
        response = requests.post(f"{BASE_URL}/export-excel", 
                               json=optimization_result,
                               stream=True)
        if response.status_code == 200:
            print("✓ 基本Excel出力成功")
            # ファイルサイズをチェック
            content_length = len(response.content)
            print(f"  - ファイルサイズ: {content_length} bytes")
        else:
            print(f"✗ 基本Excel出力失敗: {response.status_code}")
    except Exception as e:
        print(f"✗ 基本Excel出力エラー: {e}")
    
    # 2. ガントチャートExcel出力テスト
    try:
        gantt_data = {
            "job_assign": optimization_result.get('job_assign', {}),
            "sampleData": sample_data
        }
        response = requests.post(f"{BASE_URL}/export-gantt-excel", 
                               json=gantt_data,
                               stream=True)
        if response.status_code == 200:
            print("✓ ガントチャートExcel出力成功")
            content_length = len(response.content)
            print(f"  - ファイルサイズ: {content_length} bytes")
        else:
            print(f"✗ ガントチャートExcel出力失敗: {response.status_code}")
            print(f"  エラー詳細: {response.text}")
    except Exception as e:
        print(f"✗ ガントチャートExcel出力エラー: {e}")
    
    # 3. 全シフトExcel出力テスト
    try:
        allshift_data = {
            "optimizationResult": optimization_result,
            "sampleData": sample_data
        }
        response = requests.post(f"{BASE_URL}/export-allshift-excel", 
                               json=allshift_data,
                               stream=True)
        if response.status_code == 200:
            print("✓ 全シフトExcel出力成功")
            content_length = len(response.content)
            print(f"  - ファイルサイズ: {content_length} bytes")
        else:
            print(f"✗ 全シフトExcel出力失敗: {response.status_code}")
            print(f"  エラー詳細: {response.text}")
    except Exception as e:
        print(f"✗ 全シフトExcel出力エラー: {e}")

def test_feasibility_estimation(sample_data):
    """実行可能性推定テスト"""
    print("\n=== 実行可能性推定テスト ===")
    
    if not sample_data:
        print("✗ サンプルデータが不足しています")
        return
    
    try:
        params = {
            'period_data': json.dumps(sample_data['period_df']),
            'day_data': json.dumps(sample_data['day_df']),
            'job_data': json.dumps(sample_data['job_df']),
            'staff_data': json.dumps(sample_data['staff_df']),
            'requirement_data': json.dumps(sample_data['requirement_df'])
        }
        
        response = requests.get(f"{BASE_URL}/estimate-feasibility", params=params)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 実行可能性推定成功")
            print(f"  - グラフデータ生成完了")
        else:
            print(f"✗ 実行可能性推定失敗: {response.status_code}")
            print(f"  エラー詳細: {response.text}")
    except Exception as e:
        print(f"✗ 実行可能性推定エラー: {e}")

def main():
    """メインテスト実行"""
    print("=== Enhanced Shift API Test ===")
    print("10shift.ipynb完全実装機能のテスト")
    
    # 1. サンプルデータ生成
    sample_data = test_generate_sample_data()
    
    # 2. 基本最適化
    optimization_result, sample_data = test_basic_optimization(sample_data)
    
    # 3. 日別リクエスト対応最適化
    request_result = test_optimization_with_requests(sample_data)
    
    # 4. Excel出力テスト
    if optimization_result:
        test_excel_exports(optimization_result, sample_data)
    
    # 5. 実行可能性推定
    test_feasibility_estimation(sample_data)
    
    print("\n=== テスト完了 ===")
    print("すべての新機能が正常に動作することを確認してください。")

if __name__ == "__main__":
    main()