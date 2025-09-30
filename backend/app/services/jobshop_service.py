import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

try:
    from pyjobshop import Model, ProblemData, solve, SolveStatus
    PYJOBSHOP_AVAILABLE = True
except ImportError:
    PYJOBSHOP_AVAILABLE = False
    logging.warning("PyJobShop not available. Job shop scheduling features will be limited.")

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    cp_model = None
    logging.warning("OR-Tools not available. Some scheduling features will be limited.")

from ..models.jobshop_models import (
    JobShopProblem, Job, Machine, Operation, Resource,
    JobShopSolution, JobSchedule, MachineSchedule, ScheduledOperation,
    SolutionMetrics, ResourceUsage, SolverConfig, AnalysisConfig,
    ProblemType, OptimizationObjective, FlexibleJobShopProblem,
    FlowShopProblem, HybridFlowShopProblem, ProjectSchedulingProblem, 
    MultiObjectiveProblem, MultiObjectiveWeights
)


class JobShopService:
    """
    PyJobShopを使用したジョブショップスケジューリングサービス
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pyjobshop_available = PYJOBSHOP_AVAILABLE
        self.ortools_available = ORTOOLS_AVAILABLE
        
        if not self.pyjobshop_available:
            self.logger.warning("PyJobShop not available - using fallback OR-Tools implementation")
    
    def solve_job_shop(
        self,
        problem: JobShopProblem,
        solver_config: Optional[SolverConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None
    ) -> JobShopSolution:
        """
        ジョブショップ問題を解く
        """
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
            
        try:
            if self.pyjobshop_available and problem.problem_type in [ProblemType.job_shop, ProblemType.flexible_job_shop, ProblemType.flow_shop, ProblemType.hybrid_flow_shop]:
                return self._solve_with_pyjobshop(problem, solver_config, analysis_config)
            else:
                return self._solve_with_ortools(problem, solver_config, analysis_config)
        except Exception as e:
            self.logger.error(f"Scheduling failed: {str(e)}")
            return self._create_failed_solution(problem, str(e))
    
    def _solve_with_pyjobshop(
        self,
        problem: JobShopProblem,
        solver_config: SolverConfig,
        analysis_config: AnalysisConfig
    ) -> JobShopSolution:
        """
        PyJobShopを使用してジョブショップ問題を解く
        """
        try:
            # PyJobShop Modelの作成
            model = Model()
            
            # マシンの追加
            pyjobshop_machines = {}
            for machine in problem.machines:
                pyjobshop_machines[machine.id] = model.add_machine(name=machine.name or machine.id)
            
            # ジョブとタスクの追加
            pyjobshop_jobs = {}
            pyjobshop_tasks = {}
            task_to_operation = {}
            ordered_tasks = []  # PyJobShopのタスクを作成順に保持
            
            # 遅延ベースの目的関数の場合、due_dateが必要
            requires_due_dates = problem.optimization_objective in [
                OptimizationObjective.total_tardiness,
                OptimizationObjective.weighted_tardiness
            ]
            
            for job in problem.jobs:
                # due_dateがない場合の推定値（遅延ベース目的関数の場合）
                due_date = job.due_date
                if requires_due_dates and due_date is None:
                    # 全操作の合計時間 + バッファとして推定
                    total_duration = sum(op.duration for op in job.operations)
                    due_date = job.release_time + total_duration + 5  # 5時間のバッファ
                
                # ジョブの追加
                pyjobshop_job = model.add_job(
                    weight=int(job.weight),
                    release_date=job.release_time,
                    due_date=due_date,
                    deadline=job.deadline if job.deadline is not None else 2**31-1,
                    name=job.name or job.id
                )
                pyjobshop_jobs[job.id] = pyjobshop_job
                
                # 各操作をタスクとして追加
                prev_task = None
                for operation in job.operations:
                    task = model.add_task(
                        job=pyjobshop_job,
                        name=operation.id,
                        earliest_start=operation.earliest_start or 0,
                        latest_end=operation.latest_finish or 2**31-1
                    )
                    pyjobshop_tasks[operation.id] = task
                    task_to_operation[task] = operation
                    ordered_tasks.append(task)  # タスクの順序を記録
                    
                    # 操作のモードを追加（どのマシンで処理するか）
                    if operation.machine_id:
                        # 固定マシンの場合
                        machine = pyjobshop_machines[operation.machine_id]
                        model.add_mode(
                            task=task,
                            resources=machine,
                            duration=operation.duration
                        )
                    elif operation.eligible_machines:
                        # フレキシブルな場合（複数のマシンで処理可能）
                        for machine_id in operation.eligible_machines:
                            machine = pyjobshop_machines[machine_id]
                            model.add_mode(
                                task=task,
                                resources=machine,
                                duration=operation.duration
                            )
                    
                    # 順序制約：前のタスクが終わってから次のタスクを開始
                    if prev_task is not None:
                        model.add_end_before_start(prev_task, task, delay=0)
                    
                    prev_task = task
            
            # セットアップ時間の追加（もしあれば）
            if problem.setup_times_included:
                for machine_id, setup_matrix in (problem.transportation_times or {}).items():
                    machine = pyjobshop_machines[machine_id]
                    for task1_id, times in setup_matrix.items():
                        for task2_id, setup_time in times.items():
                            if task1_id in pyjobshop_tasks and task2_id in pyjobshop_tasks:
                                model.add_setup_time(
                                    machine=machine,
                                    task1=pyjobshop_tasks[task1_id],
                                    task2=pyjobshop_tasks[task2_id],
                                    duration=setup_time
                                )
            
            # メンテナンス時間窓の追加
            for machine in problem.machines:
                if machine.maintenance_windows:
                    pyjobshop_machine = pyjobshop_machines[machine.id]
                    for window in machine.maintenance_windows:
                        # メンテナンス期間を不可用期間として設定
                        # PyJobShopでは直接メンテナンスを指定できないので、
                        # ダミータスクを使ってメンテナンス期間をブロック
                        maint_job = model.add_job(weight=0, name=f"Maintenance_{machine.id}_{window['start']}")
                        maint_task = model.add_task(
                            job=maint_job,
                            name=f"Maint_{machine.id}_{window['start']}_{window['end']}",
                            earliest_start=window['start'],
                            latest_end=window['end']
                        )
                        model.add_mode(
                            task=maint_task,
                            resources=pyjobshop_machine,
                            duration=window['end'] - window['start']
                        )
            
            # リソース制約の追加（スキル要件など）
            if problem.resources:
                # リソースをPyJobShopのマシンとして扱う
                for resource in problem.resources:
                    if resource.renewable and resource.capacity == 1:
                        # 単一リソースはマシンとしてモデル化
                        pyjobshop_resource = model.add_machine(name=resource.name or resource.id)
                        # スキル要求を持つタスクにリソースを割り当て
                        for job in problem.jobs:
                            for operation in job.operations:
                                if operation.skill_requirements and resource.id in operation.skill_requirements:
                                    task = pyjobshop_tasks.get(operation.id)
                                    if task:
                                        # 既存のモードにリソースを追加
                                        # 注: PyJobShopでは複数リソースのモデル化が限定的
                                        pass
            
            # 目的関数の設定
            if problem.optimization_objective == OptimizationObjective.makespan:
                model.set_objective(weight_makespan=1)
            elif problem.optimization_objective == OptimizationObjective.total_completion_time:
                model.set_objective(weight_total_flow_time=1)
            elif problem.optimization_objective == OptimizationObjective.total_tardiness:
                model.set_objective(weight_total_tardiness=1)
            elif problem.optimization_objective == OptimizationObjective.weighted_tardiness:
                # PyJobShopは直接的な重み付き遅延をサポートしていないため、
                # 総遅延最小化を使用し、後でカスタム重み付き遅延計算を実行
                model.set_objective(weight_total_tardiness=1)
                self.logger.info("Using total tardiness optimization as approximation for weighted tardiness")
            else:
                model.set_objective(weight_makespan=1)  # デフォルト
            
            # 求解
            start_time = datetime.now()
            result = model.solve(
                solver='ortools',
                time_limit=solver_config.time_limit_seconds,
                display=solver_config.log_level > 0,
                num_workers=solver_config.num_workers if solver_config.num_workers > 1 else None
            )
            solve_time = (datetime.now() - start_time).total_seconds()
            
            if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
                return self._convert_pyjobshop_solution(
                    problem, model, result, solve_time, analysis_config, task_to_operation, ordered_tasks
                )
            else:
                return self._create_failed_solution(
                    problem, f"PyJobShop solver status: {result.status}"
                )
                
        except Exception as e:
            import traceback
            self.logger.error(f"PyJobShop solving failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_failed_solution(problem, str(e))
    
    def _convert_pyjobshop_solution(
        self,
        problem: JobShopProblem,
        model: Any,
        result: Any,
        solve_time: float,
        analysis_config: AnalysisConfig,
        task_to_operation: Dict[Any, Operation],
        ordered_tasks: List[Any]
    ) -> JobShopSolution:
        """
        PyJobShopの結果をJobShopSolutionに変換
        """
        job_schedules = []
        machine_schedules_dict = {machine.id: [] for machine in problem.machines}
        
        # ソリューションから各タスクのスケジュール情報を取得
        solution = result.best
        
        # PyJobShop solution.tasksからTaskDataオブジェクトを処理
        task_schedule_data = {}  # operation_id -> (start_time, end_time, machine_id)
        
        if hasattr(solution, 'tasks') and solution.tasks:
            # 各TaskDataオブジェクトから情報を抽出
            # solution.tasksとordered_tasksの順序が同じであることを想定
            
            for i, task_data in enumerate(solution.tasks):
                if i < len(ordered_tasks):
                    task = ordered_tasks[i]
                    operation = task_to_operation[task]
                    
                    # TaskDataから start, end, resources を取得
                    start_time = int(task_data.start)
                    end_time = int(task_data.end)
                    
                    # リソース（マシン）の特定
                    machine_id = operation.machine_id
                    if hasattr(task_data, 'resources') and task_data.resources:
                        # resources[0] がマシンインデックス
                        machine_idx = task_data.resources[0]
                        if machine_idx < len(problem.machines):
                            machine_id = problem.machines[machine_idx].id
                    
                    task_schedule_data[operation.id] = (start_time, end_time, machine_id)
        
        # ジョブ別にスケジュールを構築
        for job in problem.jobs:
            scheduled_operations = []
            
            for operation in job.operations:
                if operation.id in task_schedule_data:
                    start_time, end_time, machine_id = task_schedule_data[operation.id]
                    duration = end_time - start_time
                    
                    scheduled_op = ScheduledOperation(
                        operation_id=operation.id,
                        job_id=job.id,
                        machine_id=machine_id,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        setup_time=operation.setup_time or 0
                    )
                    scheduled_operations.append(scheduled_op)
                    
                    # マシンスケジュールに追加
                    if machine_id in machine_schedules_dict:
                        machine_schedules_dict[machine_id].append(scheduled_op)
                else:
                    # フォールバック：PyJobShopからデータが取得できない場合
                    self.logger.warning(f"No schedule data found for operation {operation.id}")
                    scheduled_op = ScheduledOperation(
                        operation_id=operation.id,
                        job_id=job.id,
                        machine_id=operation.machine_id or "M1",
                        start_time=0,
                        end_time=operation.duration,
                        duration=operation.duration,
                        setup_time=operation.setup_time or 0
                    )
                    scheduled_operations.append(scheduled_op)
            
            # ジョブスケジュールの作成
            if scheduled_operations:
                scheduled_operations.sort(key=lambda x: x.start_time)
                job_schedule = JobSchedule(
                    job_id=job.id,
                    operations=scheduled_operations,
                    start_time=scheduled_operations[0].start_time,
                    completion_time=scheduled_operations[-1].end_time,
                    tardiness=max(0, scheduled_operations[-1].end_time - (job.due_date or float('inf'))),
                    lateness=scheduled_operations[-1].end_time - (job.due_date or 0) if job.due_date else 0
                )
                job_schedules.append(job_schedule)
        
        # マシンスケジュールの作成
        machine_schedules = []
        for machine in problem.machines:
            ops = sorted(machine_schedules_dict[machine.id], key=lambda x: x.start_time)
            
            if ops:
                total_time = max([op.end_time for op in ops])
                busy_time = sum([op.duration for op in ops])
                utilization = busy_time / total_time if total_time > 0 else 0
            else:
                total_time = 0
                busy_time = 0
                utilization = 0
            
            machine_schedule = MachineSchedule(
                machine_id=machine.id,
                operations=ops,
                utilization=utilization,
                idle_time=total_time - busy_time
            )
            machine_schedules.append(machine_schedule)
        
        # メトリクスの計算 - PyJobShop solution.makespanを使用
        makespan = solution.makespan if hasattr(solution, 'makespan') else max([js.completion_time for js in job_schedules]) if job_schedules else 0
        total_completion_time = sum([js.completion_time for js in job_schedules])
        total_tardiness = sum([js.tardiness for js in job_schedules])
        total_weighted_tardiness = sum([js.tardiness * job.weight for js, job in zip(job_schedules, problem.jobs)])
        avg_utilization = np.mean([ms.utilization for ms in machine_schedules]) if machine_schedules else 0
        
        # 目的関数値の計算
        if problem.optimization_objective == OptimizationObjective.weighted_tardiness:
            objective_value = total_weighted_tardiness
        elif problem.optimization_objective == OptimizationObjective.total_tardiness:
            objective_value = total_tardiness
        elif problem.optimization_objective == OptimizationObjective.total_completion_time:
            objective_value = total_completion_time
        else:
            objective_value = float(result.best_bound) if hasattr(result, 'best_bound') else makespan
        
        metrics = SolutionMetrics(
            makespan=int(makespan),
            total_completion_time=total_completion_time,
            total_tardiness=total_tardiness,
            total_weighted_tardiness=total_weighted_tardiness,
            maximum_lateness=max([js.lateness for js in job_schedules]) if job_schedules else 0,
            average_machine_utilization=avg_utilization,
            objective_value=objective_value,
            solve_time_seconds=solve_time,
            optimality_gap=None,  # PyJobShopは最適性ギャップを直接提供しない
            feasible=True
        )
        
        # ガントチャートデータの生成
        gantt_data = None
        if analysis_config.include_gantt_chart:
            gantt_data = self._generate_gantt_data(job_schedules, machine_schedules)
        
        # 高度分析の実行
        critical_path = None
        bottleneck_analysis = None
        advanced_kpis = None
        
        if analysis_config.include_critical_path:
            critical_path = self._find_critical_path(job_schedules)
            
        if analysis_config.include_bottleneck_analysis:
            bottleneck_analysis = self._identify_bottlenecks(machine_schedules)
            
        # 高度KPI計算
        advanced_kpis = self.calculate_advanced_kpis(job_schedules, machine_schedules, metrics)
        
        # 改善提案生成（全ての分析結果を統合）
        improvement_suggestions = None
        if analysis_config.include_improvement_suggestions:
            improvement_suggestions = self._generate_suggestions(
                machine_schedules, 
                metrics, 
                job_schedules, 
                critical_path, 
                bottleneck_analysis
            )

        return JobShopSolution(
            problem_type=str(problem.problem_type),
            job_schedules=job_schedules,
            machine_schedules=machine_schedules,
            metrics=metrics,
            gantt_chart_data=gantt_data,
            solution_status=str(result.status),
            critical_path=critical_path,
            bottleneck_machines=bottleneck_analysis.get("bottleneck_machines", []) if bottleneck_analysis else None,
            bottleneck_analysis=bottleneck_analysis,
            advanced_kpis=advanced_kpis,
            improvement_suggestions=improvement_suggestions
        )
    
    def _solve_with_ortools(
        self,
        problem: JobShopProblem,
        solver_config: SolverConfig,
        analysis_config: AnalysisConfig
    ) -> JobShopSolution:
        """
        OR-Toolsを使用してジョブショップ問題を解く
        """
        if not self.ortools_available:
            return self._create_failed_solution(problem, "OR-Tools not available")
        
        try:
            model = cp_model.CpModel()
            
            # 変数の定義
            job_data = self._prepare_ortools_data(problem)
            horizon = problem.time_horizon or self._estimate_horizon(problem)
            
            # 作業変数
            all_tasks = {}
            machine_to_intervals = {machine.id: [] for machine in problem.machines}
            
            # 各ジョブの各作業に対する変数を作成
            for job in problem.jobs:
                for i, operation in enumerate(job.operations):
                    machine_id = operation.machine_id or operation.eligible_machines[0] if operation.eligible_machines else problem.machines[0].id
                    
                    start_var = model.NewIntVar(0, horizon, f'start_{job.id}_{i}')
                    duration = operation.duration + (operation.setup_time or 0)
                    end_var = model.NewIntVar(0, horizon, f'end_{job.id}_{i}')
                    interval_var = model.NewIntervalVar(
                        start_var, duration, end_var, f'interval_{job.id}_{i}'
                    )
                    
                    all_tasks[(job.id, i)] = {
                        'start': start_var,
                        'end': end_var,
                        'interval': interval_var,
                        'machine': machine_id,
                        'operation': operation
                    }
                    
                    machine_to_intervals[machine_id].append(interval_var)
            
            # 制約の追加
            self._add_ortools_constraints(model, problem, all_tasks, machine_to_intervals)
            
            # 目的関数の設定
            objective_var = self._set_ortools_objective(
                model, problem, all_tasks, horizon
            )
            
            # 求解
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = solver_config.time_limit_seconds
            solver.parameters.log_search_progress = solver_config.log_level > 0
            
            start_time = datetime.now()
            status = solver.Solve(model)
            solve_time = (datetime.now() - start_time).total_seconds()
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                return self._convert_ortools_solution(
                    problem, solver, all_tasks, solve_time, analysis_config
                )
            else:
                return self._create_failed_solution(
                    problem, f"OR-Tools solver status: {solver.StatusName(status)}"
                )
                
        except Exception as e:
            import traceback
            self.logger.error(f"OR-Tools solving failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_failed_solution(problem, str(e))
    
    def _prepare_ortools_data(self, problem: JobShopProblem) -> Dict[str, Any]:
        """
        OR-Tools用のデータを準備
        """
        jobs_data = []
        for job in problem.jobs:
            job_operations = []
            for operation in job.operations:
                machine_id = operation.machine_id or (
                    operation.eligible_machines[0] if operation.eligible_machines else "default"
                )
                duration = operation.duration + (operation.setup_time or 0)
                job_operations.append((machine_id, duration))
            jobs_data.append(job_operations)
        
        return {
            'jobs_data': jobs_data,
            'machines': [machine.id for machine in problem.machines],
            'num_jobs': len(problem.jobs),
            'num_machines': len(problem.machines)
        }
    
    def _estimate_horizon(self, problem: JobShopProblem) -> int:
        """
        スケジューリング時間枠の推定
        """
        total_duration = 0
        for job in problem.jobs:
            job_duration = sum(op.duration + (op.setup_time or 0) for op in job.operations)
            total_duration += job_duration
        
        # 保守的な見積もり：全作業時間の2倍
        return min(total_duration * 2, 10000)
    
    def _add_ortools_constraints(
        self,
        model: 'cp_model.CpModel',
        problem: JobShopProblem,
        all_tasks: Dict[Tuple[str, int], Dict[str, Any]],
        machine_to_intervals: Dict[str, List]
    ):
        """
        OR-Tools制約を追加
        """
        # 順序制約（ジョブ内の作業順序）
        for job in problem.jobs:
            for i in range(len(job.operations) - 1):
                model.Add(
                    all_tasks[(job.id, i)]['end'] <= all_tasks[(job.id, i + 1)]['start']
                )
        
        # リソース制約（各マシンで同時に実行できる作業は1つまで）
        for machine_id, intervals in machine_to_intervals.items():
            if intervals:
                model.AddNoOverlap(intervals)
        
        # リリース時間制約
        for job in problem.jobs:
            if job.release_time > 0 and job.operations:
                model.Add(
                    all_tasks[(job.id, 0)]['start'] >= job.release_time
                )
        
        # 期限制約
        for job in problem.jobs:
            if job.due_date is not None and job.operations:
                last_op_index = len(job.operations) - 1
                model.Add(
                    all_tasks[(job.id, last_op_index)]['end'] <= job.due_date
                )
    
    def _set_ortools_objective(
        self,
        model: 'cp_model.CpModel',
        problem: JobShopProblem,
        all_tasks: Dict[Tuple[str, int], Dict[str, Any]],
        horizon: int
    ) -> Any:
        """
        OR-Tools目的関数を設定
        """
        if problem.optimization_objective == OptimizationObjective.makespan:
            # メイクスパン最小化
            makespan = model.NewIntVar(0, horizon, 'makespan')
            
            # 全ジョブの完了時間の最大値がメイクスパン
            for job in problem.jobs:
                if job.operations:
                    last_op_index = len(job.operations) - 1
                    model.Add(makespan >= all_tasks[(job.id, last_op_index)]['end'])
            
            model.Minimize(makespan)
            return makespan
            
        elif problem.optimization_objective == OptimizationObjective.total_completion_time:
            # 総完了時間最小化
            total_completion = model.NewIntVar(0, horizon * len(problem.jobs), 'total_completion')
            completion_times = []
            
            for job in problem.jobs:
                if job.operations:
                    last_op_index = len(job.operations) - 1
                    completion_times.append(all_tasks[(job.id, last_op_index)]['end'])
            
            model.Add(total_completion == sum(completion_times))
            model.Minimize(total_completion)
            return total_completion
        
        else:
            # デフォルト：メイクスパン
            makespan = model.NewIntVar(0, horizon, 'makespan')
            for job in problem.jobs:
                if job.operations:
                    last_op_index = len(job.operations) - 1
                    model.Add(makespan >= all_tasks[(job.id, last_op_index)]['end'])
            model.Minimize(makespan)
            return makespan
    
    def _convert_ortools_solution(
        self,
        problem: JobShopProblem,
        solver: 'cp_model.CpSolver',
        all_tasks: Dict[Tuple[str, int], Dict[str, Any]],
        solve_time: float,
        analysis_config: AnalysisConfig
    ) -> JobShopSolution:
        """
        OR-Toolsの結果をJobShopSolutionに変換
        """
        job_schedules = []
        
        for job in problem.jobs:
            operations = []
            for i, operation in enumerate(job.operations):
                task_data = all_tasks[(job.id, i)]
                
                scheduled_op = ScheduledOperation(
                    operation_id=operation.id,
                    job_id=job.id,
                    machine_id=task_data['machine'],
                    start_time=solver.Value(task_data['start']),
                    end_time=solver.Value(task_data['end']),
                    duration=operation.duration,
                    setup_time=operation.setup_time or 0
                )
                operations.append(scheduled_op)
            
            job_schedule = JobSchedule(
                job_id=job.id,
                operations=operations,
                start_time=operations[0].start_time if operations else 0,
                completion_time=operations[-1].end_time if operations else 0,
                tardiness=max(0, (operations[-1].end_time if operations else 0) - (job.due_date or float('inf'))),
                lateness=(operations[-1].end_time if operations else 0) - (job.due_date or 0) if job.due_date else 0
            )
            job_schedules.append(job_schedule)
        
        # マシンスケジュールの作成
        machine_ops = {machine.id: [] for machine in problem.machines}
        for job_schedule in job_schedules:
            for op in job_schedule.operations:
                machine_ops[op.machine_id].append(op)
        
        machine_schedules = []
        for machine in problem.machines:
            ops = sorted(machine_ops[machine.id], key=lambda x: x.start_time)
            total_time = max([op.end_time for op in ops]) if ops else 0
            busy_time = sum([op.duration for op in ops])
            utilization = busy_time / total_time if total_time > 0 else 0
            
            machine_schedule = MachineSchedule(
                machine_id=machine.id,
                operations=ops,
                utilization=utilization,
                idle_time=total_time - busy_time
            )
            machine_schedules.append(machine_schedule)
        
        # メトリクスの計算
        makespan = max([js.completion_time for js in job_schedules]) if job_schedules else 0
        total_completion_time = sum([js.completion_time for js in job_schedules])
        total_tardiness = sum([js.tardiness for js in job_schedules])
        avg_utilization = np.mean([ms.utilization for ms in machine_schedules]) if machine_schedules else 0
        
        metrics = SolutionMetrics(
            makespan=makespan,
            total_completion_time=total_completion_time,
            total_tardiness=total_tardiness,
            total_weighted_tardiness=sum([js.tardiness * job.weight for js, job in zip(job_schedules, problem.jobs)]),
            maximum_lateness=max([js.lateness for js in job_schedules]) if job_schedules else 0,
            average_machine_utilization=avg_utilization,
            objective_value=solver.ObjectiveValue(),
            solve_time_seconds=solve_time,
            optimality_gap=solver.BestObjectiveBound() / solver.ObjectiveValue() - 1 if solver.ObjectiveValue() != 0 else None,
            feasible=True
        )
        
        # ガントチャートデータの生成
        gantt_data = None
        if analysis_config.include_gantt_chart:
            gantt_data = self._generate_gantt_data(job_schedules, machine_schedules)
        
        # 高度分析の実行
        critical_path = None
        bottleneck_analysis = None
        advanced_kpis = None
        
        if analysis_config.include_critical_path:
            critical_path = self._find_critical_path(job_schedules)
            
        if analysis_config.include_bottleneck_analysis:
            bottleneck_analysis = self._identify_bottlenecks(machine_schedules)
            
        # 高度KPI計算
        advanced_kpis = self.calculate_advanced_kpis(job_schedules, machine_schedules, metrics)
        
        # 改善提案生成（全ての分析結果を統合）
        improvement_suggestions = None
        if analysis_config.include_improvement_suggestions:
            improvement_suggestions = self._generate_suggestions(
                machine_schedules, 
                metrics, 
                job_schedules, 
                critical_path, 
                bottleneck_analysis
            )

        return JobShopSolution(
            problem_type=str(problem.problem_type),
            job_schedules=job_schedules,
            machine_schedules=machine_schedules,
            metrics=metrics,
            gantt_chart_data=gantt_data,
            solution_status="OPTIMAL" if solver.ObjectiveValue() == solver.BestObjectiveBound() else "FEASIBLE",
            critical_path=critical_path,
            bottleneck_machines=bottleneck_analysis.get("bottleneck_machines", []) if bottleneck_analysis else None,
            bottleneck_analysis=bottleneck_analysis,
            advanced_kpis=advanced_kpis,
            improvement_suggestions=improvement_suggestions
        )
    
    def _create_failed_solution(self, problem: JobShopProblem, error_msg: str) -> JobShopSolution:
        """
        失敗時のソリューションを作成
        """
        metrics = SolutionMetrics(
            makespan=0,
            total_completion_time=0,
            total_tardiness=0,
            total_weighted_tardiness=0.0,
            maximum_lateness=0,
            average_machine_utilization=0.0,
            objective_value=float('inf'),
            solve_time_seconds=0.0,
            feasible=False
        )
        
        return JobShopSolution(
            problem_type=str(problem.problem_type),
            job_schedules=[],
            machine_schedules=[],
            metrics=metrics,
            solution_status=f"FAILED: {error_msg}",
            improvement_suggestions=[f"Solving failed: {error_msg}"]
        )
    
    def _generate_gantt_data(
        self,
        job_schedules: List[JobSchedule],
        machine_schedules: List[MachineSchedule]
    ) -> Dict[str, Any]:
        """
        ガントチャート用のデータを生成
        """
        gantt_data = {
            'jobs': [],
            'machines': [],
            'timeline': {
                'start': 0,
                'end': 0
            }
        }
        
        # ジョブ別ガントデータ
        for job_schedule in job_schedules:
            job_data = {
                'job_id': job_schedule.job_id,
                'operations': []
            }
            
            for op in job_schedule.operations:
                job_data['operations'].append({
                    'operation_id': op.operation_id,
                    'machine_id': op.machine_id,
                    'start': op.start_time,
                    'end': op.end_time,
                    'duration': op.duration
                })
            
            gantt_data['jobs'].append(job_data)
        
        # マシン別ガントデータ
        for machine_schedule in machine_schedules:
            machine_data = {
                'machine_id': machine_schedule.machine_id,
                'operations': [],
                'utilization': machine_schedule.utilization
            }
            
            for op in machine_schedule.operations:
                machine_data['operations'].append({
                    'operation_id': op.operation_id,
                    'job_id': op.job_id,
                    'start': op.start_time,
                    'end': op.end_time,
                    'duration': op.duration
                })
            
            gantt_data['machines'].append(machine_data)
        
        # タイムラインの設定
        if job_schedules:
            gantt_data['timeline']['end'] = max([js.completion_time for js in job_schedules])
        
        return gantt_data
    
    def _find_critical_path(self, job_schedules: List[JobSchedule]) -> List[str]:
        """
        真のクリティカルパス分析を実行
        ネットワーク分析を用いてクリティカルパスを特定
        """
        if not job_schedules:
            return []
        
        try:
            # 依存関係グラフの構築
            operations = {}
            dependencies = {}
            
            for job_schedule in job_schedules:
                for i, operation in enumerate(job_schedule.operations):
                    op_id = operation.operation_id
                    operations[op_id] = {
                        'start_time': operation.start_time,
                        'end_time': operation.end_time,
                        'duration': operation.end_time - operation.start_time,
                        'job_id': job_schedule.job_id,
                        'position': i
                    }
                    
                    # ジョブ内での依存関係
                    if i > 0:
                        prev_op = job_schedule.operations[i-1]
                        dependencies[op_id] = dependencies.get(op_id, []) + [prev_op.operation_id]
            
            # フォワードパス（最早開始時刻の計算）
            earliest_start = {}
            earliest_finish = {}
            
            def calculate_earliest(op_id):
                if op_id in earliest_start:
                    return earliest_start[op_id], earliest_finish[op_id]
                
                op = operations[op_id]
                max_pred_finish = 0
                
                if op_id in dependencies:
                    for pred_id in dependencies[op_id]:
                        _, pred_finish = calculate_earliest(pred_id)
                        max_pred_finish = max(max_pred_finish, pred_finish)
                
                earliest_start[op_id] = max(max_pred_finish, op['start_time'])
                earliest_finish[op_id] = earliest_start[op_id] + op['duration']
                
                return earliest_start[op_id], earliest_finish[op_id]
            
            for op_id in operations:
                calculate_earliest(op_id)
            
            # バックワードパス（最遅開始時刻の計算）
            project_finish = max(earliest_finish.values())
            latest_start = {}
            latest_finish = {}
            
            # 後続作業を見つける
            successors = {}
            for op_id, deps in dependencies.items():
                for dep_id in deps:
                    successors[dep_id] = successors.get(dep_id, []) + [op_id]
            
            def calculate_latest(op_id):
                if op_id in latest_finish:
                    return latest_start[op_id], latest_finish[op_id]
                
                op = operations[op_id]
                min_succ_start = project_finish
                
                if op_id in successors:
                    for succ_id in successors[op_id]:
                        succ_start, _ = calculate_latest(succ_id)
                        min_succ_start = min(min_succ_start, succ_start)
                
                latest_finish[op_id] = min(min_succ_start, op['end_time'])
                latest_start[op_id] = latest_finish[op_id] - op['duration']
                
                return latest_start[op_id], latest_finish[op_id]
            
            for op_id in reversed(list(operations.keys())):
                calculate_latest(op_id)
            
            # クリティカルパスの特定（スラックがゼロの作業）
            critical_operations = []
            for op_id in operations:
                slack = latest_start[op_id] - earliest_start[op_id]
                if abs(slack) < 0.01:  # 数値誤差を考慮
                    critical_operations.append(op_id)
            
            # クリティカルパスを順序立てて返す
            critical_path = self._order_critical_path(critical_operations, dependencies, operations)
            
            return critical_path
            
        except Exception as e:
            # エラーが発生した場合は従来の方法にフォールバック
            latest_job = max(job_schedules, key=lambda x: x.completion_time)
            return [op.operation_id for op in latest_job.operations]
    
    def _order_critical_path(self, critical_ops: List[str], dependencies: dict, operations: dict) -> List[str]:
        """
        クリティカルパス上の作業を時系列順に並べる
        """
        if not critical_ops:
            return []
        
        # 時刻順にソート
        critical_ops_with_time = [(op_id, operations[op_id]['start_time']) for op_id in critical_ops]
        critical_ops_with_time.sort(key=lambda x: x[1])
        
        return [op_id for op_id, _ in critical_ops_with_time]
    
    def _identify_bottlenecks(self, machine_schedules: List[MachineSchedule]) -> Dict[str, Any]:
        """
        高度なボトルネック分析を実行
        複数の指標を用いてボトルネックを特定し詳細情報を返す
        """
        if not machine_schedules:
            return {"bottleneck_machines": [], "analysis": {}}
        
        try:
            bottleneck_analysis = {
                "utilization_bottlenecks": [],
                "queue_time_bottlenecks": [],
                "setup_time_bottlenecks": [],
                "throughput_bottlenecks": [],
                "overall_bottlenecks": [],
                "bottleneck_scores": {},
                "recommendations": []
            }
            
            # 基本統計の計算
            utilizations = [ms.utilization for ms in machine_schedules]
            avg_utilization = np.mean(utilizations)
            std_utilization = np.std(utilizations)
            
            operation_counts = [len(ms.operations) for ms in machine_schedules]
            avg_operations = np.mean(operation_counts)
            
            for machine_schedule in machine_schedules:
                machine_id = machine_schedule.machine_id
                score = 0
                reasons = []
                
                # 1. 稼働率ベースの分析
                utilization = machine_schedule.utilization
                if utilization > avg_utilization + std_utilization:
                    score += 3
                    reasons.append(f"高稼働率 ({utilization:.1%})")
                    if utilization > 0.9:
                        bottleneck_analysis["utilization_bottlenecks"].append(machine_id)
                
                # 2. 作業数ベースの分析  
                operation_count = len(machine_schedule.operations)
                if operation_count > avg_operations * 1.3:
                    score += 2
                    reasons.append(f"作業数過多 ({operation_count}件)")
                
                # 3. キューイング時間分析
                operations = machine_schedule.operations
                if operations:
                    queue_times = []
                    setup_times = []
                    
                    sorted_ops = sorted(operations, key=lambda x: x.start_time)
                    for i in range(1, len(sorted_ops)):
                        prev_end = sorted_ops[i-1].end_time
                        curr_start = sorted_ops[i].start_time
                        queue_time = curr_start - prev_end
                        
                        if queue_time < 0:  # オーバーラップがある場合
                            score += 1
                            reasons.append("作業オーバーラップ発生")
                        
                        queue_times.append(max(0, queue_time))
                        
                        # セットアップ時間の分析
                        if hasattr(sorted_ops[i], 'setup_time') and sorted_ops[i].setup_time:
                            setup_times.append(sorted_ops[i].setup_time)
                    
                    # キュー時間が長い場合
                    if queue_times and np.mean(queue_times) > 2:
                        score += 2
                        reasons.append(f"長いキュー時間 (平均{np.mean(queue_times):.1f})")
                        bottleneck_analysis["queue_time_bottlenecks"].append(machine_id)
                    
                    # セットアップ時間が多い場合
                    if setup_times and np.sum(setup_times) > np.sum([op.duration for op in operations]) * 0.2:
                        score += 1
                        reasons.append("セットアップ時間過多")
                        bottleneck_analysis["setup_time_bottlenecks"].append(machine_id)
                
                # 4. スループット分析
                if operations:
                    total_duration = sum([op.duration for op in operations])
                    total_time = max([op.end_time for op in operations]) - min([op.start_time for op in operations])
                    throughput = total_duration / total_time if total_time > 0 else 0
                    
                    if throughput < 0.6:  # 60%以下のスループット
                        score += 1
                        reasons.append(f"低スループット ({throughput:.1%})")
                        bottleneck_analysis["throughput_bottlenecks"].append(machine_id)
                
                # スコアに基づくボトルネック判定
                bottleneck_analysis["bottleneck_scores"][machine_id] = {
                    "score": score,
                    "reasons": reasons,
                    "utilization": utilization,
                    "operation_count": operation_count,
                    "is_bottleneck": score >= 3
                }
                
                if score >= 3:
                    bottleneck_analysis["overall_bottlenecks"].append(machine_id)
            
            # 改善提案の生成
            if bottleneck_analysis["overall_bottlenecks"]:
                bottleneck_analysis["recommendations"].extend([
                    "ボトルネックマシンの並列化を検討してください",
                    "作業の再配分により負荷を分散してください",
                    "セットアップ時間の短縮を検討してください"
                ])
            
            if bottleneck_analysis["utilization_bottlenecks"]:
                bottleneck_analysis["recommendations"].append(
                    f"マシン {', '.join(bottleneck_analysis['utilization_bottlenecks'])} の稼働率が非常に高いため、容量増強を検討してください"
                )
            
            return {
                "bottleneck_machines": bottleneck_analysis["overall_bottlenecks"],
                "analysis": bottleneck_analysis
            }
            
        except Exception as e:
            # エラー時は従来の方法にフォールバック
            avg_utilization = np.mean([ms.utilization for ms in machine_schedules])
            bottlenecks = [
                ms.machine_id for ms in machine_schedules 
                if ms.utilization > avg_utilization * 1.2
            ]
            return {
                "bottleneck_machines": bottlenecks,
                "analysis": {"error": str(e)}
            }
    
    def _generate_suggestions(
        self,
        machine_schedules: List[MachineSchedule],
        metrics: SolutionMetrics,
        job_schedules: List[JobSchedule] = None,
        critical_path: List[str] = None,
        bottleneck_analysis: Dict[str, Any] = None
    ) -> List[str]:
        """
        高度な改善提案を生成
        複数の分析結果を統合して具体的な提案を作成
        """
        suggestions = []
        priority_suggestions = []
        
        try:
            # 1. 稼働率に基づく提案
            if metrics.average_machine_utilization < 0.6:
                priority_suggestions.append(
                    f"全体稼働率が低い ({metrics.average_machine_utilization:.1%})。"
                    "作業の並列化や生産計画の見直しを検討してください。"
                )
            elif metrics.average_machine_utilization > 0.95:
                priority_suggestions.append(
                    f"全体稼働率が非常に高い ({metrics.average_machine_utilization:.1%})。"
                    "設備増強や作業時間の最適化を検討してください。"
                )
            
            # 2. 負荷バランスに基づく提案
            if machine_schedules:
                utilizations = [ms.utilization for ms in machine_schedules]
                utilization_std = np.std(utilizations)
                utilization_range = max(utilizations) - min(utilizations)
                
                if utilization_range > 0.4:
                    suggestions.append(
                        f"機械間の負荷バランスが悪い (稼働率格差: {utilization_range:.1%})。"
                        "作業の再配分により負荷を均等化してください。"
                    )
                
                if utilization_std > 0.2:
                    suggestions.append(
                        f"稼働率のばらつきが大きい (標準偏差: {utilization_std:.2f})。"
                        "ボトルネック解消と負荷分散を検討してください。"
                    )
            
            # 3. 遅延に基づく提案
            if metrics.total_tardiness > 0:
                if job_schedules:
                    late_jobs = [js for js in job_schedules if js.tardiness > 0]
                    if late_jobs:
                        avg_tardiness = metrics.total_tardiness / len(late_jobs)
                        priority_suggestions.append(
                            f"{len(late_jobs)}件のジョブで遅延発生 (平均遅延: {avg_tardiness:.1f}時間)。"
                            "優先度の高いジョブの前倒し実行を検討してください。"
                        )
            
            # 4. クリティカルパスに基づく提案
            if critical_path and len(critical_path) > 0:
                suggestions.append(
                    f"クリティカルパス ({len(critical_path)}個の作業) の短縮がプロジェクト全体の改善に最も効果的です。"
                )
                
                if len(critical_path) > len(job_schedules or []) * 0.7:
                    suggestions.append(
                        "クリティカルパス上の作業が多いため、並列化可能な作業を特定してください。"
                    )
            
            # 5. ボトルネック分析に基づく提案
            if bottleneck_analysis and bottleneck_analysis.get("analysis"):
                analysis = bottleneck_analysis["analysis"]
                
                if analysis.get("utilization_bottlenecks"):
                    machines = ", ".join(analysis["utilization_bottlenecks"])
                    priority_suggestions.append(
                        f"高稼働率ボトルネック発生 (マシン: {machines})。"
                        "これらのマシンの容量増強が最優先です。"
                    )
                
                if analysis.get("queue_time_bottlenecks"):
                    machines = ", ".join(analysis["queue_time_bottlenecks"])
                    suggestions.append(
                        f"長いキューイング時間発生 (マシン: {machines})。"
                        "作業順序の最適化を検討してください。"
                    )
                
                if analysis.get("setup_time_bottlenecks"):
                    machines = ", ".join(analysis["setup_time_bottlenecks"])
                    suggestions.append(
                        f"セットアップ時間過多 (マシン: {machines})。"
                        "セットアップ時間短縮やバッチ処理を検討してください。"
                    )
            
            # 6. メイクスパン改善提案
            if metrics.makespan > 0:
                if job_schedules:
                    avg_job_time = sum([js.completion_time - js.start_time for js in job_schedules]) / len(job_schedules)
                    if metrics.makespan > avg_job_time * 1.5:
                        suggestions.append(
                            "メイクスパンが長い。並列処理可能な作業の特定と実行順序の見直しを検討してください。"
                        )
            
            # 7. KPI に基づく総合提案
            if metrics.optimality_gap and metrics.optimality_gap > 0.05:
                suggestions.append(
                    f"最適解からの乖離 ({metrics.optimality_gap:.1%})。"
                    "より長い計算時間での再最適化を検討してください。"
                )
            
            # 8. 成功ケースの評価
            success_indicators = 0
            if metrics.average_machine_utilization >= 0.7 and metrics.average_machine_utilization <= 0.9:
                success_indicators += 1
            if metrics.total_tardiness == 0:
                success_indicators += 1
            if machine_schedules and max([ms.utilization for ms in machine_schedules]) - min([ms.utilization for ms in machine_schedules]) < 0.3:
                success_indicators += 1
            
            if success_indicators >= 2:
                suggestions.append("良好なスケジュールです。現在の設定を維持することを推奨します。")
            
            # 優先度順に並べ替え
            final_suggestions = priority_suggestions + suggestions
            
            return final_suggestions[:10] if final_suggestions else ["分析完了。特別な改善提案はありません。"]
            
        except Exception as e:
            return [f"提案生成エラー: {str(e)}", "基本的なスケジュール見直しを検討してください。"]
    
    def calculate_advanced_kpis(
        self,
        job_schedules: List[JobSchedule],
        machine_schedules: List[MachineSchedule],
        metrics: SolutionMetrics
    ) -> Dict[str, Any]:
        """
        高度なKPI（重要業績指標）を計算
        """
        try:
            kpis = {
                "efficiency_metrics": {},
                "quality_metrics": {},
                "resource_metrics": {},
                "time_metrics": {},
                "cost_metrics": {}
            }
            
            if not job_schedules or not machine_schedules:
                return kpis
            
            # 効率性指標
            total_processing_time = sum([
                sum([op.duration for op in js.operations]) for js in job_schedules
            ])
            total_makespan = metrics.makespan
            
            kpis["efficiency_metrics"] = {
                "schedule_efficiency": total_processing_time / total_makespan if total_makespan > 0 else 0,
                "machine_utilization_variance": np.var([ms.utilization for ms in machine_schedules]),
                "load_balancing_index": 1 - (np.std([ms.utilization for ms in machine_schedules]) / np.mean([ms.utilization for ms in machine_schedules])) if np.mean([ms.utilization for ms in machine_schedules]) > 0 else 0
            }
            
            # 品質指標
            on_time_jobs = sum([1 for js in job_schedules if js.tardiness == 0])
            total_jobs = len(job_schedules)
            
            kpis["quality_metrics"] = {
                "on_time_delivery_rate": on_time_jobs / total_jobs if total_jobs > 0 else 0,
                "average_tardiness": metrics.total_tardiness / total_jobs if total_jobs > 0 else 0,
                "maximum_tardiness": max([js.tardiness for js in job_schedules]) if job_schedules else 0,
                "service_level": on_time_jobs / total_jobs if total_jobs > 0 else 0
            }
            
            # リソース指標
            active_machines = sum([1 for ms in machine_schedules if ms.utilization > 0])
            total_machines = len(machine_schedules)
            
            kpis["resource_metrics"] = {
                "machine_utilization_rate": metrics.average_machine_utilization,
                "active_machine_ratio": active_machines / total_machines if total_machines > 0 else 0,
                "operations_per_machine": np.mean([len(ms.operations) for ms in machine_schedules]),
                "max_operations_per_machine": max([len(ms.operations) for ms in machine_schedules]) if machine_schedules else 0
            }
            
            # 時間指標
            completion_times = [js.completion_time for js in job_schedules]
            start_times = [js.start_time for js in job_schedules]
            
            kpis["time_metrics"] = {
                "average_completion_time": np.mean(completion_times) if completion_times else 0,
                "completion_time_variance": np.var(completion_times) if completion_times else 0,
                "average_flow_time": np.mean([ct - st for ct, st in zip(completion_times, start_times)]) if completion_times and start_times else 0,
                "makespan_efficiency": total_processing_time / total_makespan if total_makespan > 0 else 0
            }
            
            # コスト指標（概算）
            idle_time_cost = sum([ms.idle_time for ms in machine_schedules]) * 100  # 仮の単価
            tardiness_cost = metrics.total_tardiness * 500  # 仮の遅延ペナルティ
            
            kpis["cost_metrics"] = {
                "estimated_idle_cost": idle_time_cost,
                "estimated_tardiness_penalty": tardiness_cost,
                "estimated_total_cost": idle_time_cost + tardiness_cost,
                "cost_per_job": (idle_time_cost + tardiness_cost) / total_jobs if total_jobs > 0 else 0
            }
            
            # 総合スコア計算
            efficiency_score = kpis["efficiency_metrics"]["schedule_efficiency"] * 30
            quality_score = kpis["quality_metrics"]["on_time_delivery_rate"] * 25
            resource_score = kpis["resource_metrics"]["machine_utilization_rate"] * 25
            balance_score = kpis["efficiency_metrics"]["load_balancing_index"] * 20
            
            kpis["overall_score"] = {
                "total_score": efficiency_score + quality_score + resource_score + balance_score,
                "efficiency_score": efficiency_score,
                "quality_score": quality_score,
                "resource_score": resource_score,
                "balance_score": balance_score,
                "grade": self._calculate_grade(efficiency_score + quality_score + resource_score + balance_score)
            }
            
            return kpis
            
        except Exception as e:
            return {"error": str(e), "kpis": {}}
    
    def _calculate_grade(self, score: float) -> str:
        """スコアからグレードを計算"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C+"
        elif score >= 40:
            return "C"
        else:
            return "D"
    
    # 高度な機能
    def solve_flexible_job_shop(self, problem: FlexibleJobShopProblem) -> JobShopSolution:
        """
        フレキシブルジョブショップ問題を解く
        """
        # PyJobShopを使用してフレキシブルジョブショップを解く
        return self.solve_job_shop(problem)
    
    def solve_project_scheduling(self, problem: ProjectSchedulingProblem) -> JobShopSolution:
        """
        プロジェクトスケジューリング問題を解く
        """
        # 基本的な実装：アクティビティをジョブとして扱う
        jobs = []
        machines = [Machine(id="DEFAULT", name="Default Resource")]
        
        for i, activity in enumerate(problem.activities):
            operations = [Operation(
                id=f"op_{i}",
                job_id=f"job_{i}",
                machine_id="DEFAULT",
                duration=activity.get('duration', 1),
                position_in_job=0
            )]
            
            job = Job(
                id=f"job_{i}",
                name=activity.get('name', f"Activity {i}"),
                operations=operations
            )
            jobs.append(job)
        
        base_problem = JobShopProblem(
            problem_type=ProblemType.job_shop,
            jobs=jobs,
            machines=machines,
            optimization_objective=problem.optimization_objective,
            max_solve_time_seconds=problem.max_solve_time_seconds
        )
        
        return self.solve_job_shop(base_problem)
    
    def generate_sample_problem(self, problem_type: str = "job_shop") -> JobShopProblem:
        """
        サンプル問題を生成
        """
        if problem_type == "job_shop":
            return self._generate_sample_job_shop()
        elif problem_type == "flexible_job_shop":
            return self._generate_sample_flexible_job_shop()
        elif problem_type == "flow_shop":
            return self._generate_sample_flow_shop()
        elif problem_type == "hybrid_flow_shop":
            return self._generate_sample_hybrid_flow_shop()
        else:
            return self._generate_sample_job_shop()
    
    def _generate_sample_job_shop(self) -> JobShopProblem:
        """
        サンプルジョブショップ問題を生成
        """
        # 3ジョブ、3マシンの古典的な問題
        machines = [
            Machine(id="M1", name="Machine 1"),
            Machine(id="M2", name="Machine 2"),
            Machine(id="M3", name="Machine 3")
        ]
        
        jobs = [
            Job(
                id="J1",
                name="Job 1",
                operations=[
                    Operation(id="J1_O1", job_id="J1", machine_id="M1", duration=3, position_in_job=0),
                    Operation(id="J1_O2", job_id="J1", machine_id="M2", duration=2, position_in_job=1),
                    Operation(id="J1_O3", job_id="J1", machine_id="M3", duration=2, position_in_job=2)
                ]
            ),
            Job(
                id="J2",
                name="Job 2",
                operations=[
                    Operation(id="J2_O1", job_id="J2", machine_id="M1", duration=2, position_in_job=0),
                    Operation(id="J2_O2", job_id="J2", machine_id="M3", duration=1, position_in_job=1),
                    Operation(id="J2_O3", job_id="J2", machine_id="M2", duration=4, position_in_job=2)
                ]
            ),
            Job(
                id="J3",
                name="Job 3",
                operations=[
                    Operation(id="J3_O1", job_id="J3", machine_id="M2", duration=4, position_in_job=0),
                    Operation(id="J3_O2", job_id="J3", machine_id="M1", duration=3, position_in_job=1),
                    Operation(id="J3_O3", job_id="J3", machine_id="M3", duration=1, position_in_job=2)
                ]
            )
        ]
        
        return JobShopProblem(
            jobs=jobs,
            machines=machines,
            optimization_objective=OptimizationObjective.makespan,
            max_solve_time_seconds=60
        )
    
    def _generate_sample_flexible_job_shop(self) -> FlexibleJobShopProblem:
        """
        サンプルフレキシブルジョブショップ問題を生成
        """
        base_problem = self._generate_sample_job_shop()
        
        machine_eligibility = {
            "J1_O1": ["M1", "M2"],
            "J1_O2": ["M2", "M3"],
            "J1_O3": ["M1", "M3"],
            "J2_O1": ["M1", "M2"],
            "J2_O2": ["M2", "M3"],
            "J2_O3": ["M1", "M3"],
            "J3_O1": ["M1", "M2"],
            "J3_O2": ["M2", "M3"],
            "J3_O3": ["M1", "M3"]
        }
        
        # 各操作に適格マシンを設定
        for job in base_problem.jobs:
            for operation in job.operations:
                operation.machine_id = None  # フレキシブルなので特定のマシンは指定しない
                operation.eligible_machines = machine_eligibility.get(operation.id, ["M1"])
        
        return FlexibleJobShopProblem(
            jobs=base_problem.jobs,
            machines=base_problem.machines,
            optimization_objective=base_problem.optimization_objective,
            max_solve_time_seconds=base_problem.max_solve_time_seconds,
            machine_eligibility=machine_eligibility
        )
    
    def _generate_sample_flow_shop(self) -> FlowShopProblem:
        """
        サンプルフローショップ問題を生成
        """
        # フローショップでは、すべてのジョブが同じ順序でマシンを通る
        machine_sequence = ["M1", "M2", "M3"]
        
        machines = [
            Machine(id="M1", name="Machine 1"),
            Machine(id="M2", name="Machine 2"),
            Machine(id="M3", name="Machine 3")
        ]
        
        jobs = [
            Job(
                id="J1",
                name="Job 1",
                operations=[
                    Operation(id="J1_O1", job_id="J1", machine_id="M1", duration=3, position_in_job=0),
                    Operation(id="J1_O2", job_id="J1", machine_id="M2", duration=2, position_in_job=1),
                    Operation(id="J1_O3", job_id="J1", machine_id="M3", duration=2, position_in_job=2)
                ]
            ),
            Job(
                id="J2",
                name="Job 2",
                operations=[
                    Operation(id="J2_O1", job_id="J2", machine_id="M1", duration=2, position_in_job=0),
                    Operation(id="J2_O2", job_id="J2", machine_id="M2", duration=4, position_in_job=1),
                    Operation(id="J2_O3", job_id="J2", machine_id="M3", duration=1, position_in_job=2)
                ]
            ),
            Job(
                id="J3",
                name="Job 3",
                operations=[
                    Operation(id="J3_O1", job_id="J3", machine_id="M1", duration=4, position_in_job=0),
                    Operation(id="J3_O2", job_id="J3", machine_id="M2", duration=1, position_in_job=1),
                    Operation(id="J3_O3", job_id="J3", machine_id="M3", duration=3, position_in_job=2)
                ]
            )
        ]
        
        return FlowShopProblem(
            jobs=jobs,
            machines=machines,
            machine_sequence=machine_sequence,
            optimization_objective=OptimizationObjective.makespan,
            max_solve_time_seconds=300
        )
    
    def solve_flow_shop(self, problem: FlowShopProblem) -> JobShopSolution:
        """
        フローショップ問題を解く
        """
        # フローショップはジョブショップの特殊ケースとして処理
        return self.solve_job_shop(problem)
    
    def _generate_sample_hybrid_flow_shop(self) -> HybridFlowShopProblem:
        """
        サンプルハイブリッドフローショップ問題を生成
        """
        # 3ステージ: Stage1(2台), Stage2(1台), Stage3(2台)
        stages = [
            {"id": "Stage1", "machines": ["M1_1", "M1_2"], "capacity": 2},
            {"id": "Stage2", "machines": ["M2_1"], "capacity": 1},
            {"id": "Stage3", "machines": ["M3_1", "M3_2"], "capacity": 2}
        ]
        
        stage_sequence = ["Stage1", "Stage2", "Stage3"]
        
        # 各ステージのマシンを定義
        machines = [
            Machine(id="M1_1", name="Stage1 Machine 1"),
            Machine(id="M1_2", name="Stage1 Machine 2"),
            Machine(id="M2_1", name="Stage2 Machine 1"),
            Machine(id="M3_1", name="Stage3 Machine 1"),
            Machine(id="M3_2", name="Stage3 Machine 2")
        ]
        
        jobs = [
            Job(
                id="J1",
                name="Job 1",
                operations=[
                    Operation(
                        id="J1_S1", job_id="J1", machine_id=None, duration=3, position_in_job=0,
                        eligible_machines=["M1_1", "M1_2"]  # Stage1のいずれか
                    ),
                    Operation(
                        id="J1_S2", job_id="J1", machine_id="M2_1", duration=2, position_in_job=1
                    ),
                    Operation(
                        id="J1_S3", job_id="J1", machine_id=None, duration=2, position_in_job=2,
                        eligible_machines=["M3_1", "M3_2"]  # Stage3のいずれか
                    )
                ]
            ),
            Job(
                id="J2",
                name="Job 2",
                operations=[
                    Operation(
                        id="J2_S1", job_id="J2", machine_id=None, duration=2, position_in_job=0,
                        eligible_machines=["M1_1", "M1_2"]
                    ),
                    Operation(
                        id="J2_S2", job_id="J2", machine_id="M2_1", duration=4, position_in_job=1
                    ),
                    Operation(
                        id="J2_S3", job_id="J2", machine_id=None, duration=1, position_in_job=2,
                        eligible_machines=["M3_1", "M3_2"]
                    )
                ]
            ),
            Job(
                id="J3",
                name="Job 3",
                operations=[
                    Operation(
                        id="J3_S1", job_id="J3", machine_id=None, duration=4, position_in_job=0,
                        eligible_machines=["M1_1", "M1_2"]
                    ),
                    Operation(
                        id="J3_S2", job_id="J3", machine_id="M2_1", duration=1, position_in_job=1
                    ),
                    Operation(
                        id="J3_S3", job_id="J3", machine_id=None, duration=3, position_in_job=2,
                        eligible_machines=["M3_1", "M3_2"]
                    )
                ]
            )
        ]
        
        return HybridFlowShopProblem(
            jobs=jobs,
            machines=machines,
            stages=stages,
            stage_sequence=stage_sequence,
            optimization_objective=OptimizationObjective.makespan,
            max_solve_time_seconds=300
        )
    
    def solve_hybrid_flow_shop(self, problem: HybridFlowShopProblem) -> JobShopSolution:
        """
        ハイブリッドフローショップ問題を解く
        """
        # ハイブリッドフローショップはフレキシブルジョブショップの特殊ケースとして処理
        return self.solve_job_shop(problem)
    
    def solve_multi_objective(
        self,
        problem: 'MultiObjectiveProblem',
        solver_config: Optional[SolverConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None
    ) -> JobShopSolution:
        """
        マルチ目的最適化を実行
        複数の目的関数を重み付きで最適化
        """
        if solver_config is None:
            solver_config = SolverConfig()
        if analysis_config is None:
            analysis_config = AnalysisConfig()
        
        try:
            # Pareto解析が要求された場合
            if problem.pareto_analysis:
                return self._solve_pareto_analysis(problem, solver_config, analysis_config)
            else:
                # 重み付き単一目的最適化
                return self._solve_weighted_objective(problem, solver_config, analysis_config)
                
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed: {str(e)}")
            return self._create_failed_solution(problem, str(e))
    
    def _solve_weighted_objective(
        self,
        problem: 'MultiObjectiveProblem',
        solver_config: SolverConfig,
        analysis_config: AnalysisConfig
    ) -> JobShopSolution:
        """
        重み付き単一目的最適化を実行
        """
        weights = problem.objective_weights
        
        # 各目的関数の重みが0でない場合、対応する目的関数で複数回解く
        solutions = []
        objective_values = []
        
        # 1. Makespan最小化
        if weights.makespan_weight > 0:
            problem_dict = problem.dict(exclude={'objective_weights', 'pareto_analysis', 'optimization_objective'})
            makespan_problem = JobShopProblem(
                **problem_dict,
                optimization_objective=OptimizationObjective.makespan
            )
            makespan_solution = self.solve_job_shop(makespan_problem, solver_config, analysis_config)
            solutions.append(('makespan', makespan_solution))
            objective_values.append(makespan_solution.metrics.makespan * weights.makespan_weight)
        
        # 2. Total tardiness最小化
        if weights.tardiness_weight > 0:
            problem_dict = problem.dict(exclude={'objective_weights', 'pareto_analysis', 'optimization_objective'})
            tardiness_problem = JobShopProblem(
                **problem_dict,
                optimization_objective=OptimizationObjective.total_tardiness
            )
            tardiness_solution = self.solve_job_shop(tardiness_problem, solver_config, analysis_config)
            solutions.append(('tardiness', tardiness_solution))
            objective_values.append(tardiness_solution.metrics.total_tardiness * weights.tardiness_weight)
        
        # 3. Total completion time最小化
        if weights.completion_time_weight > 0:
            problem_dict = problem.dict(exclude={'objective_weights', 'pareto_analysis', 'optimization_objective'})
            completion_problem = JobShopProblem(
                **problem_dict,
                optimization_objective=OptimizationObjective.total_completion_time
            )
            completion_solution = self.solve_job_shop(completion_problem, solver_config, analysis_config)
            solutions.append(('completion_time', completion_solution))
            objective_values.append(completion_solution.metrics.total_completion_time * weights.completion_time_weight)
        
        # 最も重みが大きい目的関数のソリューションをベースとして使用
        if not solutions:
            # 重みがすべて0の場合、makespan最小化をデフォルトとする
            problem_dict = problem.dict(exclude={'objective_weights', 'pareto_analysis', 'optimization_objective'})
            default_problem = JobShopProblem(
                **problem_dict,
                optimization_objective=OptimizationObjective.makespan
            )
            return self.solve_job_shop(default_problem, solver_config, analysis_config)
        
        # 重み付き目的関数値が最小のソリューションを選択
        best_idx = objective_values.index(min(objective_values))
        best_objective, best_solution = solutions[best_idx]
        
        # 複合目的関数値を計算して設定
        composite_objective_value = sum(objective_values)
        best_solution.metrics.objective_value = composite_objective_value
        
        # 改善提案にマルチ目的最適化の情報を追加
        multi_obj_suggestions = [
            f"使用された主要目的関数: {best_objective}",
            f"複合目的関数値: {composite_objective_value:.2f}",
            f"重み設定: makespan={weights.makespan_weight}, tardiness={weights.tardiness_weight}, completion_time={weights.completion_time_weight}"
        ]
        
        if best_solution.improvement_suggestions:
            best_solution.improvement_suggestions.extend(multi_obj_suggestions)
        else:
            best_solution.improvement_suggestions = multi_obj_suggestions
        
        return best_solution
    
    def _solve_pareto_analysis(
        self,
        problem: 'MultiObjectiveProblem',
        solver_config: SolverConfig,
        analysis_config: AnalysisConfig
    ) -> JobShopSolution:
        """
        Pareto前線分析を実行
        """
        # Pareto分析：複数の重み組み合わせで解を求める
        pareto_solutions = []
        weight_combinations = [
            {'makespan': 1.0, 'tardiness': 0.0, 'completion_time': 0.0},
            {'makespan': 0.0, 'tardiness': 1.0, 'completion_time': 0.0},
            {'makespan': 0.0, 'tardiness': 0.0, 'completion_time': 1.0},
            {'makespan': 0.5, 'tardiness': 0.5, 'completion_time': 0.0},
            {'makespan': 0.5, 'tardiness': 0.0, 'completion_time': 0.5},
            {'makespan': 0.0, 'tardiness': 0.5, 'completion_time': 0.5},
            {'makespan': 0.33, 'tardiness': 0.33, 'completion_time': 0.34}
        ]
        
        for weight_combo in weight_combinations:
            # 重みを設定して解く
            temp_weights = MultiObjectiveWeights(
                makespan_weight=weight_combo['makespan'],
                tardiness_weight=weight_combo['tardiness'],
                completion_time_weight=weight_combo['completion_time']
            )
            problem_dict = problem.dict(exclude={'objective_weights', 'pareto_analysis'})
            temp_problem = MultiObjectiveProblem(
                **problem_dict,
                objective_weights=temp_weights,
                pareto_analysis=False
            )
            
            solution = self._solve_weighted_objective(temp_problem, solver_config, analysis_config)
            pareto_solutions.append((weight_combo, solution))
        
        # Pareto支配関係をチェックして非支配解を選択
        pareto_frontier = self._find_pareto_frontier(pareto_solutions)
        
        # 最もバランスの取れた解（等重みの解）を返す
        balanced_solution = next(
            (sol for weights, sol in pareto_frontier if weights.get('makespan') == 0.33),
            pareto_frontier[0][1] if pareto_frontier else pareto_solutions[0][1]
        )
        
        # Pareto分析の結果を改善提案に追加
        pareto_suggestions = [
            f"Pareto前線分析を実行: {len(pareto_frontier)}個の非支配解を発見",
            "複数の目的関数のトレードオフを考慮した最適化を実行",
            f"最適なバランス解を選択: makespan={balanced_solution.metrics.makespan}, tardiness={balanced_solution.metrics.total_tardiness}"
        ]
        
        if balanced_solution.improvement_suggestions:
            balanced_solution.improvement_suggestions.extend(pareto_suggestions)
        else:
            balanced_solution.improvement_suggestions = pareto_suggestions
        
        return balanced_solution
    
    def _find_pareto_frontier(self, solutions: List[Tuple[Dict, JobShopSolution]]) -> List[Tuple[Dict, JobShopSolution]]:
        """
        Pareto前線を求める（非支配解を抽出）
        """
        pareto_solutions = []
        
        for i, (weights_i, sol_i) in enumerate(solutions):
            is_dominated = False
            
            for j, (weights_j, sol_j) in enumerate(solutions):
                if i != j:
                    # sol_jがsol_iを支配するかチェック
                    if (sol_j.metrics.makespan <= sol_i.metrics.makespan and
                        sol_j.metrics.total_tardiness <= sol_i.metrics.total_tardiness and
                        sol_j.metrics.total_completion_time <= sol_i.metrics.total_completion_time and
                        (sol_j.metrics.makespan < sol_i.metrics.makespan or
                         sol_j.metrics.total_tardiness < sol_i.metrics.total_tardiness or
                         sol_j.metrics.total_completion_time < sol_i.metrics.total_completion_time)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_solutions.append((weights_i, sol_i))
        
        return pareto_solutions

# Create global instance
jobshop_service = JobShopService()