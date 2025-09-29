#!/usr/bin/env python3
"""
JobShop API のテストスクリプト
PyJobShop機能をテストします
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import json
from typing import Dict, Any, List

# API設定
API_BASE_URL = "http://127.0.0.1:8000/api"
JOBSHOP_URL = f"{API_BASE_URL}/jobshop"

def test_service_status():
    """サービス状態を確認"""
    print("\n=== JobShop Service Status ===")
    try:
        response = requests.get(f"{JOBSHOP_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✓ PyJobShop Available: {status['pyjobshop_available']}")
            print(f"✓ OR-Tools Available: {status['ortools_available']}")
            print(f"✓ Supported Problem Types: {status['supported_problem_types']}")
            print(f"✓ Supported Objectives: {status['supported_objectives']}")
            return True
        else:
            print(f"✗ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection failed: {str(e)}")
        return False

def test_sample_problem():
    """サンプル問題を取得してテスト"""
    print("\n=== Sample Problem Generation ===")
    try:
        # サンプル問題を取得
        response = requests.get(f"{JOBSHOP_URL}/sample-problem/job_shop")
        if response.status_code != 200:
            print(f"✗ Sample problem failed: {response.status_code}")
            return None
        
        sample_problem = response.json()
        print(f"✓ Generated sample with {len(sample_problem['jobs'])} jobs and {len(sample_problem['machines'])} machines")
        
        # サンプル問題の詳細表示
        for i, job in enumerate(sample_problem['jobs']):
            print(f"  Job {job['id']}: {len(job['operations'])} operations")
            
        return sample_problem
        
    except Exception as e:
        print(f"✗ Sample problem generation failed: {str(e)}")
        return None

def test_job_shop_solving():
    """基本的なジョブショップ問題を解く"""
    print("\n=== Job Shop Solving ===")
    
    # テスト用の簡単な問題を定義
    test_problem = {
        "problem_type": "job_shop",
        "jobs": [
            {
                "id": "J1",
                "name": "Job 1",
                "operations": [
                    {"id": "J1_O1", "job_id": "J1", "machine_id": "M1", "duration": 3, "position_in_job": 0},
                    {"id": "J1_O2", "job_id": "J1", "machine_id": "M2", "duration": 2, "position_in_job": 1},
                    {"id": "J1_O3", "job_id": "J1", "machine_id": "M3", "duration": 2, "position_in_job": 2}
                ],
                "priority": 1,
                "weight": 1.0,
                "release_time": 0
            },
            {
                "id": "J2",
                "name": "Job 2", 
                "operations": [
                    {"id": "J2_O1", "job_id": "J2", "machine_id": "M1", "duration": 2, "position_in_job": 0},
                    {"id": "J2_O2", "job_id": "J2", "machine_id": "M3", "duration": 1, "position_in_job": 1},
                    {"id": "J2_O3", "job_id": "J2", "machine_id": "M2", "duration": 4, "position_in_job": 2}
                ],
                "priority": 1,
                "weight": 1.0,
                "release_time": 0
            },
            {
                "id": "J3",
                "name": "Job 3",
                "operations": [
                    {"id": "J3_O1", "job_id": "J3", "machine_id": "M2", "duration": 4, "position_in_job": 0},
                    {"id": "J3_O2", "job_id": "J3", "machine_id": "M1", "duration": 3, "position_in_job": 1},
                    {"id": "J3_O3", "job_id": "J3", "machine_id": "M3", "duration": 1, "position_in_job": 2}
                ],
                "priority": 1,
                "weight": 1.0,
                "release_time": 0
            }
        ],
        "machines": [
            {"id": "M1", "name": "Machine 1", "capacity": 1, "available_from": 0},
            {"id": "M2", "name": "Machine 2", "capacity": 1, "available_from": 0},
            {"id": "M3", "name": "Machine 3", "capacity": 1, "available_from": 0}
        ],
        "optimization_objective": "makespan",
        "max_solve_time_seconds": 60
    }
    
    try:
        print("Solving job shop problem...")
        response = requests.post(f"{JOBSHOP_URL}/solve", json=test_problem)
        
        if response.status_code == 200:
            solution = response.json()
            print(f"✓ Problem solved successfully!")
            print(f"  Solution Status: {solution['solution_status']}")
            print(f"  Makespan: {solution['metrics']['makespan']}")
            print(f"  Total Completion Time: {solution['metrics']['total_completion_time']}")
            print(f"  Average Machine Utilization: {solution['metrics']['average_machine_utilization']:.2%}")
            print(f"  Solve Time: {solution['metrics']['solve_time_seconds']:.2f} seconds")
            print(f"  Feasible: {solution['metrics']['feasible']}")
            
            # ジョブスケジュールの表示
            print("\n  Job Schedules:")
            for job_schedule in solution['job_schedules']:
                print(f"    {job_schedule['job_id']}: Start={job_schedule['start_time']}, End={job_schedule['completion_time']}")
                
            # マシンスケジュールの表示
            print("\n  Machine Schedules:")
            for machine_schedule in solution['machine_schedules']:
                print(f"    {machine_schedule['machine_id']}: Utilization={machine_schedule['utilization']:.2%}, Operations={len(machine_schedule['operations'])}")
            
            # 改善提案の表示
            if solution.get('improvement_suggestions'):
                print("\n  Improvement Suggestions:")
                for suggestion in solution['improvement_suggestions']:
                    print(f"    - {suggestion}")
            
            return solution
        else:
            print(f"✗ Solving failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Solving failed: {str(e)}")
        return None

def test_flexible_job_shop():
    """フレキシブルジョブショップ問題をテスト"""
    print("\n=== Flexible Job Shop Test ===")
    
    flexible_problem = {
        "problem_type": "flexible_job_shop",
        "jobs": [
            {
                "id": "J1",
                "name": "Flexible Job 1",
                "operations": [
                    {
                        "id": "J1_O1",
                        "job_id": "J1",
                        "duration": 3,
                        "position_in_job": 0,
                        "eligible_machines": ["M1", "M2"]
                    },
                    {
                        "id": "J1_O2", 
                        "job_id": "J1",
                        "duration": 2,
                        "position_in_job": 1,
                        "eligible_machines": ["M2", "M3"]
                    }
                ],
                "priority": 1,
                "weight": 1.0,
                "release_time": 0
            }
        ],
        "machines": [
            {"id": "M1", "name": "Machine 1", "capacity": 1, "available_from": 0},
            {"id": "M2", "name": "Machine 2", "capacity": 1, "available_from": 0},
            {"id": "M3", "name": "Machine 3", "capacity": 1, "available_from": 0}
        ],
        "machine_eligibility": {
            "J1_O1": ["M1", "M2"],
            "J1_O2": ["M2", "M3"]
        },
        "optimization_objective": "makespan",
        "max_solve_time_seconds": 60
    }
    
    try:
        response = requests.post(f"{JOBSHOP_URL}/solve-flexible", json=flexible_problem)
        
        if response.status_code == 200:
            solution = response.json()
            print(f"✓ Flexible job shop solved!")
            print(f"  Makespan: {solution['metrics']['makespan']}")
            print(f"  Solve Time: {solution['metrics']['solve_time_seconds']:.2f} seconds")
            return solution
        else:
            print(f"✗ Flexible solving failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Flexible solving failed: {str(e)}")
        return None

def test_project_scheduling():
    """プロジェクトスケジューリングをテスト"""
    print("\n=== Project Scheduling Test ===")
    
    project_problem = {
        "problem_type": "project_scheduling",
        "activities": [
            {"id": "A1", "name": "Activity 1", "duration": 5},
            {"id": "A2", "name": "Activity 2", "duration": 3},
            {"id": "A3", "name": "Activity 3", "duration": 4}
        ],
        "precedence_relations": [
            {"predecessor": "A1", "successor": "A2"},
            {"predecessor": "A2", "successor": "A3"}
        ],
        "resources": [
            {"id": "R1", "name": "Resource 1", "capacity": 2, "renewable": True}
        ],
        "optimization_objective": "makespan",
        "max_solve_time_seconds": 60
    }
    
    try:
        response = requests.post(f"{JOBSHOP_URL}/solve-project", json=project_problem)
        
        if response.status_code == 200:
            solution = response.json()
            print(f"✓ Project scheduling solved!")
            print(f"  Makespan: {solution['metrics']['makespan']}")
            return solution
        else:
            print(f"✗ Project scheduling failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Project scheduling failed: {str(e)}")
        return None

def test_solution_analysis(solution: Dict[str, Any]):
    """ソリューション分析をテスト"""
    print("\n=== Solution Analysis ===")
    
    try:
        response = requests.post(f"{JOBSHOP_URL}/analyze-solution", json=solution)
        
        if response.status_code == 200:
            analysis = response.json()
            print("✓ Solution analysis completed!")
            
            print("  Performance Metrics:")
            metrics = analysis['performance_metrics']
            for key, value in metrics.items():
                print(f"    {key}: {value}")
            
            print("  Bottleneck Analysis:")
            bottleneck = analysis['bottleneck_analysis']
            print(f"    Bottleneck Machines: {bottleneck['bottleneck_machines']}")
            print(f"    Utilization Variance: {bottleneck['utilization_variance']:.4f}")
            
            return analysis
        else:
            print(f"✗ Analysis failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Analysis failed: {str(e)}")
        return None

def test_batch_solving():
    """バッチ処理をテスト"""
    print("\n=== Batch Solving Test ===")
    
    # 複数の小さな問題を作成
    problems = []
    for i in range(2):
        problem = {
            "problem_type": "job_shop",
            "jobs": [
                {
                    "id": f"J{i}_1",
                    "name": f"Batch Job {i}_1",
                    "operations": [
                        {"id": f"J{i}_1_O1", "job_id": f"J{i}_1", "machine_id": "M1", "duration": 2, "position_in_job": 0},
                        {"id": f"J{i}_1_O2", "job_id": f"J{i}_1", "machine_id": "M2", "duration": 3, "position_in_job": 1}
                    ],
                    "priority": 1,
                    "weight": 1.0,
                    "release_time": 0
                }
            ],
            "machines": [
                {"id": "M1", "name": "Machine 1", "capacity": 1, "available_from": 0},
                {"id": "M2", "name": "Machine 2", "capacity": 1, "available_from": 0}
            ],
            "optimization_objective": "makespan",
            "max_solve_time_seconds": 30
        }
        problems.append(problem)
    
    batch_request = {
        "problems": problems,
        "solver_config": {
            "solver_name": "CP-SAT",
            "time_limit_seconds": 30,
            "num_workers": 1,
            "log_level": 1
        },
        "analysis_config": {
            "include_gantt_chart": True,
            "include_utilization_analysis": True,
            "include_bottleneck_analysis": True,
            "include_critical_path": True,
            "include_improvement_suggestions": True
        }
    }
    
    try:
        response = requests.post(f"{JOBSHOP_URL}/solve-batch", json=batch_request)
        
        if response.status_code == 200:
            batch_result = response.json()
            print("✓ Batch solving completed!")
            print(f"  Total Problems: {batch_result['batch_statistics']['total_problems']}")
            print(f"  Successful Solutions: {batch_result['batch_statistics']['successful_solutions']}")
            print(f"  Average Makespan: {batch_result['comparison_metrics']['avg_makespan']:.1f}")
            print(f"  Average Utilization: {batch_result['comparison_metrics']['avg_utilization']:.2%}")
            return batch_result
        else:
            print(f"✗ Batch solving failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Batch solving failed: {str(e)}")
        return None

def run_comprehensive_test():
    """包括的なテストを実行"""
    print("=" * 50)
    print("JobShop API Comprehensive Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # テスト1: サービス状態確認
    total_tests += 1
    if test_service_status():
        success_count += 1
    
    # テスト2: サンプル問題生成
    total_tests += 1
    sample_problem = test_sample_problem()
    if sample_problem:
        success_count += 1
    
    # テスト3: 基本ジョブショップ
    total_tests += 1
    solution = test_job_shop_solving()
    if solution:
        success_count += 1
        
        # テスト4: ソリューション分析
        total_tests += 1
        if test_solution_analysis(solution):
            success_count += 1
    else:
        total_tests += 1  # ソリューション分析もカウント
    
    # テスト5: フレキシブルジョブショップ
    total_tests += 1
    if test_flexible_job_shop():
        success_count += 1
    
    # テスト6: プロジェクトスケジューリング
    total_tests += 1
    if test_project_scheduling():
        success_count += 1
    
    # テスト7: バッチ処理
    total_tests += 1
    if test_batch_solving():
        success_count += 1
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {success_count}")
    print(f"Failed: {total_tests - success_count}")
    print(f"Success Rate: {success_count/total_tests:.1%}")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! JobShop API is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - success_count} test(s) failed. Please check the API implementation.")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)