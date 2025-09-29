#!/usr/bin/env python3
"""
シフト最適化APIの動作テスト
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_check():
    """ヘルスチェック"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ ヘルスチェック OK")

def test_generate_sample_data():
    """サンプルデータ生成テスト"""
    data = {
        "start_date": "2024-01-01",
        "end_date": "2024-01-07",
        "start_time": "09:00",
        "end_time": "17:00",
        "freq": "1h",
        "job_list": ["レジ打ち", "接客"]
    }
    
    response = requests.post(f"{BASE_URL}/api/shift/generate-sample-data", json=data)
    assert response.status_code == 200
    
    result = response.json()
    assert "day_df" in result
    assert "period_df" in result
    assert "job_df" in result
    assert "break_df" in result
    assert "staff_df" in result
    assert "requirement_df" in result
    
    print("✅ サンプルデータ生成 OK")
    return result

def test_shift_optimization(sample_data):
    """シフト最適化テスト"""
    
    # スタッフデータの前処理
    staff_data = []
    for staff in sample_data["staff_df"]:
        staff_dict = staff.copy()
        # job_setとday_offを文字列からリストに変換
        if isinstance(staff["job_set"], str):
            staff_dict["job_set"] = json.loads(staff["job_set"])
        if isinstance(staff["day_off"], str):
            staff_dict["day_off"] = json.loads(staff["day_off"])
        staff_data.append(staff_dict)
    
    optimization_request = {
        "period_df": sample_data["period_df"],
        "break_df": sample_data["break_df"],
        "day_df": sample_data["day_df"],
        "job_df": sample_data["job_df"],
        "staff_df": staff_data,
        "requirement_df": sample_data["requirement_df"],
        "theta": 1,
        "lb_penalty": 10000,
        "ub_penalty": 0,
        "job_change_penalty": 10,
        "break_penalty": 10000,
        "max_day_penalty": 5000,
        "time_limit": 10,
        "random_seed": 1
    }
    
    response = requests.post(f"{BASE_URL}/api/shift/optimize", json=optimization_request)
    assert response.status_code == 200
    
    result = response.json()
    assert "status" in result
    assert "cost_df" in result
    assert "staff_df" in result
    assert "job_assign" in result
    
    print("✅ シフト最適化実行 OK")
    print(f"   最適化ステータス: {result['status']}")
    print(f"   メッセージ: {result.get('message', 'なし')}")
    
    return result

def main():
    """テスト実行"""
    print("🚀 シフト最適化API テスト開始")
    
    try:
        # ヘルスチェック
        test_health_check()
        
        # サンプルデータ生成
        sample_data = test_generate_sample_data()
        
        # シフト最適化
        optimization_result = test_shift_optimization(sample_data)
        
        print("🎉 すべてのテストが成功しました！")
        
        # 結果の簡単な確認
        if optimization_result["cost_df"]:
            print("\n📊 コスト内訳:")
            for cost in optimization_result["cost_df"]:
                if cost["value"] > 0:
                    print(f"   {cost['penalty']}: {cost['value']}")
        
        if optimization_result["job_assign"]:
            job_count = len(optimization_result["job_assign"])
            print(f"\n👥 ジョブ割り当て数: {job_count}")
        
        staff_count = len(optimization_result["staff_df"]) if optimization_result["staff_df"] else 0
        print(f"👤 スタッフ数: {staff_count}")
        
    except Exception as e:
        print(f"❌ テスト失敗: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)