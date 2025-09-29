from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import json
import pandas as pd
import io
import logging

from ..models.jobshop_models import (
    JobShopProblem, FlexibleJobShopProblem, FlowShopProblem, HybridFlowShopProblem,
    ProjectSchedulingProblem, JobShopSolution, Job, Machine, Operation, Resource,
    SolverConfig, AnalysisConfig, ProblemType, OptimizationObjective,
    MultiObjectiveProblem, MultiObjectiveWeights, BatchJobShopRequest, BatchJobShopResponse,
    RealTimeSchedulingRequest, JobShopSolveRequest
)
from ..services.jobshop_service import JobShopService

router = APIRouter(tags=["jobshop"])
logger = logging.getLogger(__name__)

# サービスインスタンス
jobshop_service = JobShopService()


@router.get("/status")
async def get_service_status():
    """
    PyJobShop サービスの状態を確認
    """
    return {
        "pyjobshop_available": jobshop_service.pyjobshop_available,
        "ortools_available": jobshop_service.ortools_available,
        "supported_problem_types": [
            "job_shop",
            "flexible_job_shop", 
            "flow_shop",
            "hybrid_flow_shop",
            "project_scheduling"
        ],
        "supported_objectives": [
            "makespan",
            "total_completion_time",
            "total_weighted_completion_time",
            "maximum_lateness",
            "total_tardiness",
            "weighted_tardiness"
        ]
    }


@router.post("/solve", response_model=JobShopSolution)
async def solve_job_shop_problem(request: JobShopSolveRequest):
    """
    ジョブショップ スケジューリング問題を解く
    """
    try:
        solver_config = request.solver_config or SolverConfig()
        analysis_config = request.analysis_config or AnalysisConfig()
            
        logger.info(f"Solving {request.problem.problem_type} with {len(request.problem.jobs)} jobs and {len(request.problem.machines)} machines")
        
        solution = jobshop_service.solve_job_shop(request.problem, solver_config, analysis_config)
        
        return solution
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Job shop solving failed: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"スケジューリング処理に失敗しました: {str(e)} | Traceback: {error_traceback[:200]}...")


@router.post("/solve-flexible", response_model=JobShopSolution)
async def solve_flexible_job_shop(
    problem: FlexibleJobShopProblem,
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None
):
    """
    フレキシブルジョブショップ問題を解く
    """
    try:
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
            
        logger.info(f"Solving flexible job shop with {len(problem.jobs)} jobs and {len(problem.machines)} machines")
        
        solution = jobshop_service.solve_flexible_job_shop(problem)
        
        return solution
        
    except Exception as e:
        logger.error(f"Flexible job shop solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"フレキシブルジョブショップ処理に失敗しました: {str(e)}")


@router.post("/solve-flow", response_model=JobShopSolution)
async def solve_flow_shop(
    problem: FlowShopProblem,
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None
):
    """
    フローショップ問題を解く
    """
    try:
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
            
        logger.info(f"Solving flow shop with {len(problem.jobs)} jobs and {len(problem.machines)} machines")
        
        solution = jobshop_service.solve_flow_shop(problem)
        
        return solution
        
    except Exception as e:
        logger.error(f"Flow shop solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"フローショップ処理に失敗しました: {str(e)}")


@router.post("/solve-hybrid-flow", response_model=JobShopSolution)
async def solve_hybrid_flow_shop(
    problem: HybridFlowShopProblem,
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None
):
    """
    ハイブリッドフローショップ問題を解く
    """
    try:
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
            
        logger.info(f"Solving hybrid flow shop with {len(problem.jobs)} jobs and {len(problem.machines)} machines")
        
        solution = jobshop_service.solve_hybrid_flow_shop(problem)
        
        return solution
        
    except Exception as e:
        logger.error(f"Hybrid flow shop solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ハイブリッドフローショップ処理に失敗しました: {str(e)}")


@router.post("/solve-project", response_model=JobShopSolution)
async def solve_project_scheduling(
    problem: ProjectSchedulingProblem,
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None
):
    """
    プロジェクトスケジューリング問題を解く
    """
    try:
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
            
        logger.info(f"Solving project scheduling with {len(problem.activities)} activities")
        
        solution = jobshop_service.solve_project_scheduling(problem)
        
        return solution
        
    except Exception as e:
        logger.error(f"Project scheduling failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"プロジェクトスケジューリング処理に失敗しました: {str(e)}")


@router.post("/solve-multi-objective", response_model=JobShopSolution)
async def solve_multi_objective_problem(
    problem: MultiObjectiveProblem,
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None
):
    """
    マルチ目的最適化問題を解く
    """
    try:
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
            
        logger.info(f"Solving multi-objective problem with weights: makespan={problem.objective_weights.makespan_weight}, tardiness={problem.objective_weights.tardiness_weight}, completion_time={problem.objective_weights.completion_time_weight}")
        
        solution = jobshop_service.solve_multi_objective(problem, solver_config, analysis_config)
        
        return solution
        
    except Exception as e:
        logger.error(f"Multi-objective solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"マルチ目的最適化処理に失敗しました: {str(e)}")


@router.post("/solve-pareto-analysis", response_model=JobShopSolution)
async def solve_pareto_analysis(
    problem: MultiObjectiveProblem,
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None
):
    """
    Pareto前線分析を実行
    """
    try:
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
            
        # Pareto分析を有効にする
        problem.pareto_analysis = True
        
        logger.info(f"Performing Pareto analysis for multi-objective optimization")
        
        solution = jobshop_service.solve_multi_objective(problem, solver_config, analysis_config)
        
        return solution
        
    except Exception as e:
        logger.error(f"Pareto analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pareto分析処理に失敗しました: {str(e)}")


@router.post("/analyze-advanced", response_model=Dict[str, Any])
async def analyze_advanced_metrics(
    solution: JobShopSolution,
):
    """
    既存の解に対して高度分析を実行
    """
    try:
        logger.info("Performing advanced analysis on existing solution")
        
        # 高度KPI計算
        advanced_kpis = jobshop_service.calculate_advanced_kpis(
            solution.job_schedules, 
            solution.machine_schedules, 
            solution.metrics
        )
        
        # クリティカルパス分析
        critical_path = jobshop_service._find_critical_path(solution.job_schedules)
        
        # ボトルネック分析
        bottleneck_analysis = jobshop_service._identify_bottlenecks(solution.machine_schedules)
        
        # 改善提案生成
        improvement_suggestions = jobshop_service._generate_suggestions(
            solution.machine_schedules,
            solution.metrics,
            solution.job_schedules,
            critical_path,
            bottleneck_analysis
        )
        
        analysis_result = {
            "advanced_kpis": advanced_kpis,
            "critical_path": critical_path,
            "bottleneck_analysis": bottleneck_analysis,
            "improvement_suggestions": improvement_suggestions,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Advanced analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"高度分析処理に失敗しました: {str(e)}")


@router.post("/what-if-analysis", response_model=Dict[str, Any])
async def perform_what_if_analysis(
    problem: JobShopProblem,
    scenarios: List[Dict[str, Any]],
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None
):
    """
    What-if分析を実行（複数シナリオの比較）
    """
    try:
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
        
        logger.info(f"Performing what-if analysis with {len(scenarios)} scenarios")
        
        base_solution = jobshop_service.solve_job_shop(problem, solver_config, analysis_config)
        scenario_results = []
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"Analyzing scenario {i+1}: {scenario.get('name', f'Scenario {i+1}')}")
            
            # シナリオに基づいて問題を変更
            modified_problem = problem.copy(deep=True)
            
            # シナリオ適用ロジック
            if "machine_capacity_change" in scenario:
                for machine_id, multiplier in scenario["machine_capacity_change"].items():
                    for machine in modified_problem.machines:
                        if machine.id == machine_id:
                            machine.capacity = int(machine.capacity * multiplier)
            
            if "job_priority_change" in scenario:
                for job_id, new_priority in scenario["job_priority_change"].items():
                    for job in modified_problem.jobs:
                        if job.id == job_id:
                            job.priority = new_priority
            
            if "duration_change" in scenario:
                for job_id, multiplier in scenario["duration_change"].items():
                    for job in modified_problem.jobs:
                        if job.id == job_id:
                            for op in job.operations:
                                op.duration = int(op.duration * multiplier)
            
            # 変更された問題を解く
            scenario_solution = jobshop_service.solve_job_shop(modified_problem, solver_config, analysis_config)
            
            # 結果の比較
            comparison = {
                "scenario_name": scenario.get("name", f"Scenario {i+1}"),
                "scenario_config": scenario,
                "solution": scenario_solution,
                "improvements": {
                    "makespan_change": scenario_solution.metrics.makespan - base_solution.metrics.makespan,
                    "tardiness_change": scenario_solution.metrics.total_tardiness - base_solution.metrics.total_tardiness,
                    "utilization_change": scenario_solution.metrics.average_machine_utilization - base_solution.metrics.average_machine_utilization,
                },
                "improvement_percentage": {
                    "makespan": ((base_solution.metrics.makespan - scenario_solution.metrics.makespan) / base_solution.metrics.makespan * 100) if base_solution.metrics.makespan > 0 else 0,
                    "tardiness": ((base_solution.metrics.total_tardiness - scenario_solution.metrics.total_tardiness) / max(base_solution.metrics.total_tardiness, 1) * 100),
                    "utilization": ((scenario_solution.metrics.average_machine_utilization - base_solution.metrics.average_machine_utilization) * 100)
                }
            }
            
            scenario_results.append(comparison)
        
        # 最適シナリオの特定
        best_scenario = min(scenario_results, key=lambda x: x["solution"].metrics.makespan)
        
        analysis_result = {
            "base_solution": base_solution,
            "scenario_results": scenario_results,
            "best_scenario": best_scenario["scenario_name"],
            "summary": {
                "total_scenarios": len(scenarios),
                "best_makespan_improvement": best_scenario["improvement_percentage"]["makespan"],
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"What-if analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"What-if分析処理に失敗しました: {str(e)}")


@router.post("/solve-batch", response_model=BatchJobShopResponse)
async def solve_batch_problems(request: BatchJobShopRequest):
    """
    複数の問題を一括で解く
    """
    try:
        solutions = []
        
        for i, problem in enumerate(request.problems):
            logger.info(f"Solving batch problem {i+1}/{len(request.problems)}")
            solution = jobshop_service.solve_job_shop(
                problem, 
                request.solver_config, 
                request.analysis_config
            )
            solutions.append(solution)
        
        # 比較メトリクスの計算
        comparison_metrics = {
            "avg_makespan": sum([s.metrics.makespan for s in solutions]) / len(solutions),
            "avg_utilization": sum([s.metrics.average_machine_utilization for s in solutions]) / len(solutions),
            "avg_solve_time": sum([s.metrics.solve_time_seconds for s in solutions]) / len(solutions),
            "feasible_solutions": sum([1 for s in solutions if s.metrics.feasible])
        }
        
        batch_statistics = {
            "total_problems": len(request.problems),
            "successful_solutions": len([s for s in solutions if s.metrics.feasible]),
            "total_solve_time": sum([s.metrics.solve_time_seconds for s in solutions])
        }
        
        return BatchJobShopResponse(
            solutions=solutions,
            comparison_metrics=comparison_metrics,
            batch_statistics=batch_statistics
        )
        
    except Exception as e:
        logger.error(f"Batch solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"バッチ処理に失敗しました: {str(e)}")


@router.get("/sample-problem/{problem_type}")
async def get_sample_problem(problem_type: str = "job_shop"):
    """
    サンプル問題を取得
    """
    try:
        if problem_type not in ["job_shop", "flexible_job_shop", "flow_shop", "hybrid_flow_shop", "project_scheduling"]:
            raise HTTPException(status_code=400, detail="無効な問題タイプです")
            
        sample_problem = jobshop_service.generate_sample_problem(problem_type)
        
        return sample_problem
        
    except Exception as e:
        logger.error(f"Sample problem generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"サンプル問題の生成に失敗しました: {str(e)}")


@router.post("/upload-problem")
async def upload_problem_file(file: UploadFile = File(...)):
    """
    ファイルからジョブショップ問題をアップロード
    """
    try:
        if not file.filename.endswith(('.json', '.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="サポートされていないファイル形式です")
        
        content = await file.read()
        
        if file.filename.endswith('.json'):
            problem_data = json.loads(content.decode('utf-8'))
            problem = JobShopProblem(**problem_data)
            
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
            problem = _convert_csv_to_problem(df)
            
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(content))
            problem = _convert_excel_to_problem(df)
            
        return problem
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ファイルの処理に失敗しました: {str(e)}")


@router.get("/gantt-chart/{solution_id}")
async def get_gantt_chart_data(solution_id: str):
    """
    ガントチャートデータを取得（将来の拡張用）
    """
    # 実装は後で追加
    return {"message": "ガントチャートデータ機能は開発中です"}


@router.post("/analyze-solution")
async def analyze_solution(solution: JobShopSolution):
    """
    ソリューションの詳細分析
    """
    try:
        analysis = {
            "performance_metrics": {
                "makespan": solution.metrics.makespan,
                "utilization": solution.metrics.average_machine_utilization,
                "tardiness": solution.metrics.total_tardiness,
                "completion_time": solution.metrics.total_completion_time
            },
            "bottleneck_analysis": {
                "bottleneck_machines": solution.bottleneck_machines or [],
                "utilization_variance": _calculate_utilization_variance(solution.machine_schedules)
            },
            "critical_path_analysis": {
                "critical_path": solution.critical_path or [],
                "critical_path_length": solution.metrics.makespan
            },
            "improvement_opportunities": solution.improvement_suggestions or []
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Solution analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ソリューション分析に失敗しました: {str(e)}")


@router.post("/compare-solutions")
async def compare_solutions(solutions: List[JobShopSolution]):
    """
    複数のソリューションを比較
    """
    try:
        if len(solutions) < 2:
            raise HTTPException(status_code=400, detail="比較には最低2つのソリューションが必要です")
        
        comparison = {
            "solutions_count": len(solutions),
            "best_makespan": min([s.metrics.makespan for s in solutions]),
            "best_utilization": max([s.metrics.average_machine_utilization for s in solutions]),
            "best_tardiness": min([s.metrics.total_tardiness for s in solutions]),
            "avg_solve_time": sum([s.metrics.solve_time_seconds for s in solutions]) / len(solutions),
            "detailed_comparison": []
        }
        
        for i, solution in enumerate(solutions):
            comparison["detailed_comparison"].append({
                "solution_index": i,
                "makespan": solution.metrics.makespan,
                "utilization": solution.metrics.average_machine_utilization,
                "tardiness": solution.metrics.total_tardiness,
                "solve_time": solution.metrics.solve_time_seconds,
                "status": solution.solution_status
            })
        
        return comparison
        
    except Exception as e:
        logger.error(f"Solution comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ソリューション比較に失敗しました: {str(e)}")


@router.post("/optimize-parameters")
async def optimize_solver_parameters(
    base_problem: JobShopProblem,
    parameter_ranges: Dict[str, Dict[str, Any]]
):
    """
    ソルバーパラメータの最適化
    """
    try:
        # パラメータの組み合わせをテスト
        best_solution = None
        best_objective = float('inf')
        tested_configs = []
        
        # 簡単な実装：複数の時間制限でテスト
        time_limits = parameter_ranges.get('time_limits', [30, 60, 120, 300])
        
        for time_limit in time_limits:
            config = SolverConfig(time_limit_seconds=time_limit)
            solution = jobshop_service.solve_job_shop(base_problem, config)
            
            tested_configs.append({
                "config": config.dict(),
                "objective_value": solution.metrics.objective_value,
                "solve_time": solution.metrics.solve_time_seconds,
                "feasible": solution.metrics.feasible
            })
            
            if solution.metrics.feasible and solution.metrics.objective_value < best_objective:
                best_solution = solution
                best_objective = solution.metrics.objective_value
        
        return {
            "best_solution": best_solution,
            "best_objective": best_objective,
            "tested_configurations": tested_configs,
            "recommendations": [
                "より長い求解時間で改善の余地があります" if best_objective > time_limits[0] else "現在の設定が最適です"
            ]
        }
        
    except Exception as e:
        logger.error(f"Parameter optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"パラメータ最適化に失敗しました: {str(e)}")


# ヘルパー関数
def _convert_csv_to_problem(df: pd.DataFrame) -> JobShopProblem:
    """
    CSVデータをJobShopProblemに変換
    """
    jobs = []
    machines = []
    
    # 基本的な実装：列名に基づいてデータを解析
    if 'job_id' in df.columns and 'machine_id' in df.columns and 'duration' in df.columns:
        # ジョブデータの作成
        job_groups = df.groupby('job_id')
        
        for job_id, group in job_groups:
            operations = []
            for i, row in group.iterrows():
                operation = Operation(
                    id=f"{job_id}_op_{i}",
                    job_id=job_id,
                    machine_id=row['machine_id'],
                    duration=int(row['duration']),
                    position_in_job=len(operations)
                )
                operations.append(operation)
            
            job = Job(id=job_id, operations=operations)
            jobs.append(job)
        
        # マシンデータの作成
        machine_ids = df['machine_id'].unique()
        for machine_id in machine_ids:
            machine = Machine(id=machine_id, name=f"Machine {machine_id}")
            machines.append(machine)
    
    return JobShopProblem(jobs=jobs, machines=machines)


def _convert_excel_to_problem(df: pd.DataFrame) -> JobShopProblem:
    """
    Excelデータを JobShopProblem に変換
    """
    return _convert_csv_to_problem(df)  # CSVと同じロジックを使用


def _calculate_utilization_variance(machine_schedules: List) -> float:
    """
    マシン稼働率の分散を計算
    """
    if not machine_schedules:
        return 0.0
    
    utilizations = [ms.utilization for ms in machine_schedules]
    mean_util = sum(utilizations) / len(utilizations)
    variance = sum([(u - mean_util) ** 2 for u in utilizations]) / len(utilizations)
    
    return variance