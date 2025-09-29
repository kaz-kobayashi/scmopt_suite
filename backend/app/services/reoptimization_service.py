"""
スケジュール再最適化アルゴリズムサービス
"""
import json
import copy
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..models.jobshop_models import (
    Job, Machine, Operation, JobShopProblem, JobShopSolution,
    SolverConfig, AnalysisConfig
)
from ..models.realtime_models import (
    ScheduleEvent, ScheduleEventType, RealtimeScheduleManager,
    ReoptimizationRequest, ReoptimizationResult
)
from ..services.jobshop_service import JobShopService

logger = logging.getLogger(__name__)


class ReoptimizationStrategy(str, Enum):
    """再最適化戦略"""
    COMPLETE = "complete"                    # 完全再最適化
    INCREMENTAL = "incremental"              # 増分最適化
    LOCAL_SEARCH = "local_search"           # 局所探索
    TIME_WINDOW = "time_window"             # 時間窓制約
    PRIORITY_FOCUSED = "priority_focused"    # 優先度重点
    BOTTLENECK_FOCUSED = "bottleneck_focused"  # ボトルネック重点


class ReoptimizationTrigger(str, Enum):
    """再最適化トリガー"""
    MANUAL = "manual"                       # 手動
    EVENT_THRESHOLD = "event_threshold"     # イベント閾値
    PERFORMANCE_DEGRADATION = "performance_degradation"  # 性能劣化
    CRITICAL_EVENT = "critical_event"       # 緊急イベント
    SCHEDULED = "scheduled"                 # スケジュール実行


class ReoptimizationService:
    """スケジュール再最適化サービス"""
    
    def __init__(self):
        self.jobshop_service = JobShopService()
        self.reoptimization_history: List[ReoptimizationResult] = []
    
    def execute_reoptimization(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        strategy: ReoptimizationStrategy = ReoptimizationStrategy.INCREMENTAL,
        trigger: ReoptimizationTrigger = ReoptimizationTrigger.MANUAL,
        time_limit: int = 60,
        preserve_completed: bool = True
    ) -> ReoptimizationResult:
        """再最適化の実行"""
        
        start_time = datetime.now()
        logger.info(f"Starting reoptimization with strategy: {strategy}")
        
        try:
            # イベント分析
            event_analysis = self._analyze_events(events)
            
            # 再最適化戦略の選択
            if strategy == ReoptimizationStrategy.INCREMENTAL:
                new_solution = self._incremental_reoptimization(
                    current_solution, events, event_analysis, time_limit, preserve_completed
                )
            elif strategy == ReoptimizationStrategy.COMPLETE:
                new_solution = self._complete_reoptimization(
                    current_solution, events, time_limit, preserve_completed
                )
            elif strategy == ReoptimizationStrategy.LOCAL_SEARCH:
                new_solution = self._local_search_reoptimization(
                    current_solution, events, event_analysis, time_limit
                )
            elif strategy == ReoptimizationStrategy.TIME_WINDOW:
                new_solution = self._time_window_reoptimization(
                    current_solution, events, event_analysis, time_limit
                )
            elif strategy == ReoptimizationStrategy.PRIORITY_FOCUSED:
                new_solution = self._priority_focused_reoptimization(
                    current_solution, events, event_analysis, time_limit
                )
            elif strategy == ReoptimizationStrategy.BOTTLENECK_FOCUSED:
                new_solution = self._bottleneck_focused_reoptimization(
                    current_solution, events, event_analysis, time_limit
                )
            else:
                raise ValueError(f"Unknown reoptimization strategy: {strategy}")
            
            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()
            
            # 結果分析
            changes_summary = self._analyze_changes(current_solution, new_solution)
            impact_metrics = self._calculate_impact_metrics(current_solution, new_solution)
            
            result = ReoptimizationResult(
                schedule_id=current_solution.problem_id if hasattr(current_solution, 'problem_id') else "unknown",
                success=new_solution is not None,
                new_solution=new_solution,
                optimization_time=optimization_time,
                changes_summary=changes_summary,
                impact_metrics=impact_metrics,
                error_message=None
            )
            
            self.reoptimization_history.append(result)
            logger.info(f"Reoptimization completed in {optimization_time:.2f}s")
            
            return result
        
        except Exception as e:
            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()
            
            result = ReoptimizationResult(
                schedule_id=current_solution.problem_id if hasattr(current_solution, 'problem_id') else "unknown",
                success=False,
                new_solution=None,
                optimization_time=optimization_time,
                changes_summary={},
                impact_metrics={},
                error_message=str(e)
            )
            
            logger.error(f"Reoptimization failed: {str(e)}")
            return result
    
    def _analyze_events(self, events: List[ScheduleEvent]) -> Dict[str, Any]:
        """イベント分析"""
        analysis = {
            "event_count": len(events),
            "event_types": {},
            "impact_levels": {},
            "critical_events": [],
            "affected_jobs": set(),
            "affected_machines": set(),
            "max_delay": 0,
            "total_disruption": 0
        }
        
        for event in events:
            # イベントタイプ別カウント
            event_type = event.event_type
            analysis["event_types"][event_type] = analysis["event_types"].get(event_type, 0) + 1
            
            # 影響レベル別カウント
            impact_level = event.impact_level
            analysis["impact_levels"][impact_level] = analysis["impact_levels"].get(impact_level, 0) + 1
            
            # クリティカルイベント
            if impact_level in ["critical", "high"]:
                analysis["critical_events"].append(event)
            
            # 影響を受けるリソース
            if event.event_type in [ScheduleEventType.JOB_DELAY, ScheduleEventType.JOB_COMPLETION]:
                analysis["affected_jobs"].add(event.target_id)
            elif event.event_type == ScheduleEventType.MACHINE_BREAKDOWN:
                analysis["affected_machines"].add(event.target_id)
            
            # 遅延分析
            if event.event_type == ScheduleEventType.JOB_DELAY:
                delay = event.event_data.get("estimated_delay", 0)
                analysis["max_delay"] = max(analysis["max_delay"], delay)
                analysis["total_disruption"] += delay
        
        # 変換
        analysis["affected_jobs"] = list(analysis["affected_jobs"])
        analysis["affected_machines"] = list(analysis["affected_machines"])
        
        return analysis
    
    def _incremental_reoptimization(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        event_analysis: Dict[str, Any],
        time_limit: int,
        preserve_completed: bool
    ) -> Optional[JobShopSolution]:
        """増分再最適化（影響を受ける部分のみ最適化）"""
        
        logger.info("Executing incremental reoptimization")
        
        # 影響を受けるジョブとマシンを特定
        affected_jobs = set(event_analysis["affected_jobs"])
        affected_machines = set(event_analysis["affected_machines"])
        
        # 完了済みジョブを除外
        if preserve_completed:
            completed_jobs = self._get_completed_jobs(current_solution)
            affected_jobs -= completed_jobs
        
        # 影響を受ける操作の特定
        affected_operations = self._get_affected_operations(
            current_solution, affected_jobs, affected_machines
        )
        
        # 部分問題の構築
        partial_problem = self._build_partial_problem(
            current_solution, affected_operations, events
        )
        
        if not partial_problem:
            logger.warning("No partial problem to optimize")
            return current_solution
        
        # 部分最適化の実行
        solver_config = SolverConfig(time_limit_seconds=time_limit // 2)
        analysis_config = AnalysisConfig()
        
        partial_solution = self.jobshop_service.solve_job_shop(
            partial_problem, solver_config, analysis_config
        )
        
        if partial_solution:
            # 部分解を元の解に統合
            return self._integrate_partial_solution(
                current_solution, partial_solution, affected_operations
            )
        else:
            logger.warning("Partial optimization failed, using complete reoptimization")
            return self._complete_reoptimization(current_solution, events, time_limit, preserve_completed)
    
    def _complete_reoptimization(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        time_limit: int,
        preserve_completed: bool
    ) -> Optional[JobShopSolution]:
        """完全再最適化"""
        
        logger.info("Executing complete reoptimization")
        
        # 現在の状態に基づいて新しい問題を構築
        updated_problem = self._build_updated_problem(current_solution, events, preserve_completed)
        
        # 完全最適化の実行
        solver_config = SolverConfig(time_limit_seconds=time_limit)
        analysis_config = AnalysisConfig()
        
        return self.jobshop_service.solve_job_shop(
            updated_problem, solver_config, analysis_config
        )
    
    def _local_search_reoptimization(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        event_analysis: Dict[str, Any],
        time_limit: int
    ) -> Optional[JobShopSolution]:
        """局所探索による再最適化"""
        
        logger.info("Executing local search reoptimization")
        
        improved_solution = copy.deepcopy(current_solution)
        
        # 複数の局所探索手法を適用
        search_methods = [
            self._job_swap_search,
            self._operation_rescheduling_search,
            self._machine_reassignment_search
        ]
        
        for method in search_methods:
            try:
                candidate = method(improved_solution, event_analysis, time_limit // len(search_methods))
                if candidate and self._is_better_solution(candidate, improved_solution):
                    improved_solution = candidate
            except Exception as e:
                logger.error(f"Local search method failed: {str(e)}")
        
        return improved_solution
    
    def _time_window_reoptimization(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        event_analysis: Dict[str, Any],
        time_limit: int
    ) -> Optional[JobShopSolution]:
        """時間窓制約による再最適化"""
        
        logger.info("Executing time window reoptimization")
        
        # 現在時刻以降の操作のみを対象とする
        current_time = datetime.now()
        
        # 未来の操作を特定
        future_operations = self._get_future_operations(current_solution, current_time)
        
        # 時間窓制約付き問題の構築
        windowed_problem = self._build_time_windowed_problem(
            current_solution, future_operations, events
        )
        
        # 最適化実行
        solver_config = SolverConfig(time_limit_seconds=time_limit)
        analysis_config = AnalysisConfig()
        
        windowed_solution = self.jobshop_service.solve_job_shop(
            windowed_problem, solver_config, analysis_config
        )
        
        if windowed_solution:
            # 時間窓解を元の解に統合
            return self._integrate_windowed_solution(
                current_solution, windowed_solution, current_time
            )
        
        return current_solution
    
    def _priority_focused_reoptimization(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        event_analysis: Dict[str, Any],
        time_limit: int
    ) -> Optional[JobShopSolution]:
        """優先度重点の再最適化"""
        
        logger.info("Executing priority focused reoptimization")
        
        # 優先度の高いジョブを特定
        high_priority_jobs = self._get_high_priority_jobs(current_solution, events)
        
        # 優先度重視の制約を追加した問題の構築
        priority_problem = self._build_priority_focused_problem(
            current_solution, high_priority_jobs, events
        )
        
        # 優先度重視の目的関数で最適化
        solver_config = SolverConfig(time_limit_seconds=time_limit)
        analysis_config = AnalysisConfig()
        
        return self.jobshop_service.solve_job_shop(
            priority_problem, solver_config, analysis_config
        )
    
    def _bottleneck_focused_reoptimization(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        event_analysis: Dict[str, Any],
        time_limit: int
    ) -> Optional[JobShopSolution]:
        """ボトルネック重点の再最適化"""
        
        logger.info("Executing bottleneck focused reoptimization")
        
        # ボトルネックマシンを特定
        bottleneck_machines = self._identify_bottleneck_machines(current_solution)
        
        # ボトルネック解消に焦点を当てた問題の構築
        bottleneck_problem = self._build_bottleneck_focused_problem(
            current_solution, bottleneck_machines, events
        )
        
        # ボトルネック解消の目的関数で最適化
        solver_config = SolverConfig(time_limit_seconds=time_limit)
        analysis_config = AnalysisConfig()
        
        return self.jobshop_service.solve_job_shop(
            bottleneck_problem, solver_config, analysis_config
        )
    
    def _get_completed_jobs(self, solution: JobShopSolution) -> set:
        """完了済みジョブの取得"""
        completed = set()
        current_time = datetime.now()
        
        for job_schedule in solution.job_schedules:
            # 全操作が完了している場合
            if all(op.end_time <= current_time.timestamp() for op in job_schedule.operations):
                completed.add(job_schedule.job_id)
        
        return completed
    
    def _get_affected_operations(
        self, 
        solution: JobShopSolution, 
        affected_jobs: set, 
        affected_machines: set
    ) -> List[Any]:
        """影響を受ける操作の特定"""
        affected_operations = []
        
        for job_schedule in solution.job_schedules:
            if job_schedule.job_id in affected_jobs:
                affected_operations.extend(job_schedule.operations)
        
        for machine_schedule in solution.machine_schedules:
            if machine_schedule.machine_id in affected_machines:
                affected_operations.extend(machine_schedule.operations)
        
        return affected_operations
    
    def _build_partial_problem(
        self,
        current_solution: JobShopSolution,
        affected_operations: List[Any],
        events: List[ScheduleEvent]
    ) -> Optional[JobShopProblem]:
        """部分問題の構築"""
        # 簡略化された実装
        # 実際には、影響を受ける操作のみを含む部分問題を構築
        
        affected_job_ids = set(op.job_id for op in affected_operations)
        
        jobs = []
        machines = []
        
        # 影響を受けるジョブのみを含める
        for job_schedule in current_solution.job_schedules:
            if job_schedule.job_id in affected_job_ids:
                # ジョブの再構築
                operations = []
                for i, op in enumerate(job_schedule.operations):
                    operations.append(Operation(
                        id=op.operation_id,
                        job_id=op.job_id,
                        machine_id=op.machine_id,
                        duration=op.duration if op.duration else 10,
                        position_in_job=i,
                        setup_time=getattr(op, 'setup_time', 0)
                    ))
                
                jobs.append(Job(
                    id=job_schedule.job_id,
                    name=f"Job {job_schedule.job_id}",
                    priority=getattr(job_schedule, 'priority', 1),
                    weight=getattr(job_schedule, 'weight', 1.0),
                    release_time=getattr(job_schedule, 'release_time', 0),
                    operations=operations
                ))
        
        # マシンの構築
        for machine_schedule in current_solution.machine_schedules:
            machines.append(Machine(
                id=machine_schedule.machine_id,
                name=f"Machine {machine_schedule.machine_id}",
                capacity=1,
                available_from=0
            ))
        
        if not jobs:
            return None
        
        return JobShopProblem(
            problem_type="job_shop",
            jobs=jobs,
            machines=machines,
            optimization_objective="makespan"
        )
    
    def _build_updated_problem(
        self,
        current_solution: JobShopSolution,
        events: List[ScheduleEvent],
        preserve_completed: bool
    ) -> JobShopProblem:
        """更新された問題の構築"""
        
        jobs = []
        machines = []
        
        # 完了済みジョブの特定
        completed_jobs = set()
        if preserve_completed:
            completed_jobs = self._get_completed_jobs(current_solution)
        
        # ジョブの再構築（完了済み以外）
        for job_schedule in current_solution.job_schedules:
            if job_schedule.job_id not in completed_jobs:
                operations = []
                for i, op in enumerate(job_schedule.operations):
                    # イベントによる変更を反映
                    duration = self._get_updated_duration(op, events)
                    
                    operations.append(Operation(
                        id=op.operation_id,
                        job_id=op.job_id,
                        machine_id=op.machine_id,
                        duration=duration,
                        position_in_job=i,
                        setup_time=getattr(op, 'setup_time', 0)
                    ))
                
                jobs.append(Job(
                    id=job_schedule.job_id,
                    name=f"Job {job_schedule.job_id}",
                    priority=getattr(job_schedule, 'priority', 1),
                    weight=getattr(job_schedule, 'weight', 1.0),
                    release_time=getattr(job_schedule, 'release_time', 0),
                    operations=operations
                ))
        
        # マシンの再構築
        for machine_schedule in current_solution.machine_schedules:
            # 故障イベントを反映
            availability = self._get_machine_availability(machine_schedule.machine_id, events)
            
            machines.append(Machine(
                id=machine_schedule.machine_id,
                name=f"Machine {machine_schedule.machine_id}",
                capacity=1,
                available_from=availability.get('available_from', 0),
                available_until=availability.get('available_until')
            ))
        
        # 緊急ジョブの追加
        for event in events:
            if event.event_type == ScheduleEventType.URGENT_JOB_ADD:
                urgent_job_data = event.event_data.get('urgent_job', {})
                if urgent_job_data:
                    jobs.append(Job(**urgent_job_data))
        
        return JobShopProblem(
            problem_type="job_shop",
            jobs=jobs,
            machines=machines,
            optimization_objective="makespan"
        )
    
    def _get_updated_duration(self, operation: Any, events: List[ScheduleEvent]) -> int:
        """イベントを反映した操作時間の取得"""
        duration = getattr(operation, 'duration', 10)
        
        for event in events:
            if (event.event_type == ScheduleEventType.JOB_DELAY and 
                event.event_data.get('operation_id') == operation.operation_id):
                duration = event.event_data.get('new_duration', duration)
        
        return duration
    
    def _get_machine_availability(self, machine_id: str, events: List[ScheduleEvent]) -> Dict[str, Any]:
        """マシンの利用可能性の取得"""
        availability = {'available_from': 0}
        
        for event in events:
            if (event.event_type == ScheduleEventType.MACHINE_BREAKDOWN and 
                event.target_id == machine_id):
                # 故障期間を反映
                repair_time = event.event_data.get('estimated_repair_time')
                if repair_time:
                    availability['available_from'] = repair_time
        
        return availability
    
    def _integrate_partial_solution(
        self,
        original_solution: JobShopSolution,
        partial_solution: JobShopSolution,
        affected_operations: List[Any]
    ) -> JobShopSolution:
        """部分解の統合"""
        # 簡略化された実装
        # 実際には、部分解を元の解に慎重に統合する必要がある
        return partial_solution if partial_solution else original_solution
    
    def _is_better_solution(self, candidate: JobShopSolution, current: JobShopSolution) -> bool:
        """解の優劣判定"""
        if not candidate.metrics or not current.metrics:
            return False
        
        return candidate.metrics.makespan < current.metrics.makespan
    
    def _analyze_changes(self, old_solution: JobShopSolution, new_solution: JobShopSolution) -> Dict[str, Any]:
        """変更分析"""
        if not new_solution:
            return {"status": "no_change"}
        
        changes = {
            "status": "optimized",
            "job_changes": 0,
            "machine_changes": 0,
            "time_improvement": 0
        }
        
        if old_solution.metrics and new_solution.metrics:
            changes["time_improvement"] = old_solution.metrics.makespan - new_solution.metrics.makespan
        
        return changes
    
    def _calculate_impact_metrics(self, old_solution: JobShopSolution, new_solution: JobShopSolution) -> Dict[str, float]:
        """影響メトリクスの計算"""
        if not new_solution or not new_solution.metrics:
            return {}
        
        metrics = {
            "new_makespan": new_solution.metrics.makespan,
            "improvement_ratio": 0.0
        }
        
        if old_solution.metrics:
            metrics["old_makespan"] = old_solution.metrics.makespan
            if old_solution.metrics.makespan > 0:
                metrics["improvement_ratio"] = (
                    (old_solution.metrics.makespan - new_solution.metrics.makespan) / 
                    old_solution.metrics.makespan * 100
                )
        
        return metrics
    
    # 追加のヘルパーメソッド（簡略化された実装）
    def _job_swap_search(self, solution: JobShopSolution, event_analysis: Dict, time_limit: int):
        """ジョブ交換探索"""
        return solution  # 簡略化
    
    def _operation_rescheduling_search(self, solution: JobShopSolution, event_analysis: Dict, time_limit: int):
        """操作再スケジューリング探索"""
        return solution  # 簡略化
    
    def _machine_reassignment_search(self, solution: JobShopSolution, event_analysis: Dict, time_limit: int):
        """マシン再割り当て探索"""
        return solution  # 簡略化
    
    def _get_future_operations(self, solution: JobShopSolution, current_time: datetime):
        """未来の操作取得"""
        return []  # 簡略化
    
    def _build_time_windowed_problem(self, solution: JobShopSolution, operations: List, events: List):
        """時間窓問題構築"""
        return self._build_updated_problem(solution, events, True)  # 簡略化
    
    def _integrate_windowed_solution(self, original: JobShopSolution, windowed: JobShopSolution, current_time: datetime):
        """時間窓解統合"""
        return windowed if windowed else original  # 簡略化
    
    def _get_high_priority_jobs(self, solution: JobShopSolution, events: List):
        """高優先度ジョブ取得"""
        return []  # 簡略化
    
    def _build_priority_focused_problem(self, solution: JobShopSolution, priority_jobs: List, events: List):
        """優先度重点問題構築"""
        return self._build_updated_problem(solution, events, True)  # 簡略化
    
    def _identify_bottleneck_machines(self, solution: JobShopSolution):
        """ボトルネックマシン特定"""
        return []  # 簡略化
    
    def _build_bottleneck_focused_problem(self, solution: JobShopSolution, bottlenecks: List, events: List):
        """ボトルネック重点問題構築"""
        return self._build_updated_problem(solution, events, True)  # 簡略化


# サービスのシングルトンインスタンス
reoptimization_service = ReoptimizationService()