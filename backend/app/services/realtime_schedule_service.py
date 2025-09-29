"""
リアルタイムスケジュール管理サービス
"""
import json
import os
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import uuid4
import logging

from ..models.realtime_models import (
    ScheduleEvent,
    ScheduleEventType,
    ScheduleUpdate,
    RealtimeScheduleManager,
    ScheduleMonitor,
    ScheduleAlert,
    ReoptimizationRequest,
    ReoptimizationResult,
    ScheduleSnapshot,
    RealtimeStats,
    JobDelayEvent,
    MachineBreakdownEvent,
    UrgentJobEvent
)
from ..models.jobshop_models import JobShopSolution, JobShopProblem
from ..services.jobshop_service import JobShopService

logger = logging.getLogger(__name__)


class RealtimeScheduleService:
    """リアルタイムスケジュール管理サービス"""
    
    def __init__(self, storage_path: str = "data/realtime"):
        self.storage_path = storage_path
        self.schedules_file = os.path.join(storage_path, "active_schedules.json")
        self.events_file = os.path.join(storage_path, "events.json")
        self.alerts_file = os.path.join(storage_path, "alerts.json")
        self.snapshots_path = os.path.join(storage_path, "snapshots")
        
        self.jobshop_service = JobShopService()
        self.active_schedules: Dict[str, RealtimeScheduleManager] = {}
        self.event_processors: Dict[ScheduleEventType, callable] = {
            ScheduleEventType.JOB_DELAY: self._process_job_delay,
            ScheduleEventType.MACHINE_BREAKDOWN: self._process_machine_breakdown,
            ScheduleEventType.URGENT_JOB_ADD: self._process_urgent_job,
            ScheduleEventType.JOB_COMPLETION: self._process_job_completion,
            ScheduleEventType.PRIORITY_CHANGE: self._process_priority_change
        }
        
        self._ensure_storage_directory()
        self._load_active_schedules()
    
    def _ensure_storage_directory(self):
        """ストレージディレクトリの作成"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.snapshots_path, exist_ok=True)
        
        if not os.path.exists(self.schedules_file):
            self._save_schedules({})
        if not os.path.exists(self.events_file):
            self._save_events([])
        if not os.path.exists(self.alerts_file):
            self._save_alerts([])
    
    def _load_active_schedules(self):
        """アクティブスケジュールの読み込み"""
        try:
            with open(self.schedules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for schedule_id, schedule_data in data.items():
                    self.active_schedules[schedule_id] = RealtimeScheduleManager(**schedule_data)
        except (FileNotFoundError, json.JSONDecodeError):
            self.active_schedules = {}
    
    def _save_schedules(self, schedules: Dict[str, RealtimeScheduleManager]):
        """スケジュールの保存"""
        data = {}
        for schedule_id, schedule in schedules.items():
            data[schedule_id] = schedule.dict()
        
        with open(self.schedules_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_events(self, events: List[ScheduleEvent]):
        """イベントの保存"""
        data = [event.dict() for event in events]
        with open(self.events_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_alerts(self, alerts: List[ScheduleAlert]):
        """アラートの保存"""
        data = [alert.dict() for alert in alerts]
        with open(self.alerts_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def create_realtime_schedule(
        self, 
        solution: JobShopSolution,
        monitor_config: Optional[ScheduleMonitor] = None
    ) -> str:
        """リアルタイムスケジュールの作成"""
        schedule_id = str(uuid4())
        
        if monitor_config is None:
            monitor_config = ScheduleMonitor(schedule_id=schedule_id)
        
        realtime_manager = RealtimeScheduleManager(
            schedule_id=schedule_id,
            current_solution=solution,
            last_update=datetime.now(),
            status="active"
        )
        
        self.active_schedules[schedule_id] = realtime_manager
        self._save_schedules(self.active_schedules)
        
        # 初期スナップショットの保存
        self._create_snapshot(schedule_id, "initial_creation")
        
        logger.info(f"Created realtime schedule: {schedule_id}")
        return schedule_id
    
    def add_event(self, schedule_id: str, event: ScheduleEvent) -> bool:
        """イベントの追加"""
        if schedule_id not in self.active_schedules:
            logger.error(f"Schedule not found: {schedule_id}")
            return False
        
        schedule_manager = self.active_schedules[schedule_id]
        schedule_manager.active_events.append(event)
        schedule_manager.last_update = datetime.now()
        
        # イベントの重要度に応じた処理
        if event.impact_level == "critical" or event.auto_reoptimize:
            asyncio.create_task(self._process_event_async(schedule_id, event))
        
        # アラートの生成
        self._generate_alert_if_needed(schedule_id, event)
        
        self._save_schedules(self.active_schedules)
        
        # WebSocket通知
        asyncio.create_task(self._notify_event_via_websocket(schedule_id, event))
        
        logger.info(f"Added event {event.id} to schedule {schedule_id}")
        return True
    
    async def _process_event_async(self, schedule_id: str, event: ScheduleEvent):
        """イベントの非同期処理"""
        try:
            processor = self.event_processors.get(event.event_type)
            if processor:
                await processor(schedule_id, event)
            
            # 再最適化の検討
            schedule_manager = self.active_schedules[schedule_id]
            if (schedule_manager.auto_reoptimize and 
                len(schedule_manager.active_events) >= schedule_manager.reoptimization_threshold):
                await self._trigger_reoptimization(schedule_id)
        
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {str(e)}")
    
    async def _process_job_delay(self, schedule_id: str, event: ScheduleEvent):
        """作業遅延イベントの処理"""
        delay_data = JobDelayEvent(**event.event_data)
        schedule_manager = self.active_schedules[schedule_id]
        
        # 現在のソリューションを更新
        current_solution = schedule_manager.current_solution
        
        # 対象ジョブの遅延を反映
        for job_schedule in current_solution.job_schedules:
            if job_schedule.job_id == delay_data.job_id:
                for operation in job_schedule.operations:
                    if operation.operation_id == delay_data.operation_id:
                        # 遅延を反映した時間調整
                        delay = delay_data.new_duration - delay_data.original_duration
                        operation.end_time += delay
                        
                        # 後続作業への影響を計算
                        self._propagate_delay(current_solution, operation, delay)
                        break
                break
        
        # クリティカルパスへの影響を分析
        if current_solution.critical_path and delay_data.operation_id in current_solution.critical_path:
            self._create_critical_alert(schedule_id, f"クリティカルパス作業 {delay_data.operation_id} に遅延発生")
    
    async def _process_machine_breakdown(self, schedule_id: str, event: ScheduleEvent):
        """機械故障イベントの処理"""
        breakdown_data = MachineBreakdownEvent(**event.event_data)
        schedule_manager = self.active_schedules[schedule_id]
        
        # 故障マシンの作業を他のマシンに再配分
        affected_operations = breakdown_data.affected_operations
        current_solution = schedule_manager.current_solution
        
        # 緊急再最適化が必要
        self._create_critical_alert(
            schedule_id, 
            f"マシン {breakdown_data.machine_id} 故障により緊急再最適化が必要"
        )
        
        # 自動再最適化をトリガー
        await self._trigger_reoptimization(schedule_id, reopt_type="emergency")
    
    async def _process_urgent_job(self, schedule_id: str, event: ScheduleEvent):
        """緊急ジョブ追加イベントの処理"""
        urgent_data = UrgentJobEvent(**event.event_data)
        schedule_manager = self.active_schedules[schedule_id]
        
        # 現在の問題に緊急ジョブを追加
        current_solution = schedule_manager.current_solution
        
        # 緊急ジョブの優先度に応じてスケジュール調整
        if urgent_data.preempt_existing:
            self._create_critical_alert(
                schedule_id,
                f"緊急ジョブ {urgent_data.urgent_job.id} により既存作業の中断が必要"
            )
        
        # 再最適化をトリガー
        await self._trigger_reoptimization(schedule_id, urgent_job=urgent_data.urgent_job)
    
    async def _process_job_completion(self, schedule_id: str, event: ScheduleEvent):
        """作業完了イベントの処理"""
        schedule_manager = self.active_schedules[schedule_id]
        job_id = event.event_data.get("job_id")
        
        # 完了ジョブを記録
        current_solution = schedule_manager.current_solution
        for job_schedule in current_solution.job_schedules:
            if job_schedule.job_id == job_id:
                job_schedule.status = "completed"
                break
        
        # 統計更新
        self._update_realtime_stats(schedule_id)
    
    async def _process_priority_change(self, schedule_id: str, event: ScheduleEvent):
        """優先度変更イベントの処理"""
        job_id = event.event_data.get("job_id")
        new_priority = event.event_data.get("new_priority")
        
        # 優先度変更による影響を分析
        impact_analysis = self._analyze_priority_impact(schedule_id, job_id, new_priority)
        
        if impact_analysis.get("requires_reoptimization"):
            await self._trigger_reoptimization(schedule_id)
    
    def _propagate_delay(self, solution: JobShopSolution, delayed_operation, delay_minutes: int):
        """遅延の後続作業への伝播"""
        # 同じマシンの後続作業を調整
        machine_id = delayed_operation.machine_id
        
        for machine_schedule in solution.machine_schedules:
            if machine_schedule.machine_id == machine_id:
                operations = sorted(machine_schedule.operations, key=lambda x: x.start_time)
                
                for i, operation in enumerate(operations):
                    if operation.operation_id == delayed_operation.operation_id:
                        # 後続の全作業を遅延時間分シフト
                        for j in range(i + 1, len(operations)):
                            operations[j].start_time += delay_minutes
                            operations[j].end_time += delay_minutes
                        break
                break
    
    async def _trigger_reoptimization(
        self, 
        schedule_id: str, 
        reopt_type: str = "incremental",
        urgent_job: Optional[Any] = None
    ):
        """再最適化のトリガー"""
        try:
            schedule_manager = self.active_schedules[schedule_id]
            current_solution = schedule_manager.current_solution
            
            # 現在の状況に基づいて新しい問題を構築
            updated_problem = self._build_updated_problem(
                current_solution, 
                schedule_manager.active_events,
                urgent_job
            )
            
            # 再最適化実行
            reoptimization_request = ReoptimizationRequest(
                schedule_id=schedule_id,
                trigger_events=[e.id for e in schedule_manager.active_events],
                reoptimization_type=reopt_type,
                preserve_completed_jobs=True
            )
            
            new_solution = await self._execute_reoptimization(updated_problem, reoptimization_request)
            
            if new_solution:
                # 新しいソリューションを適用
                schedule_manager.current_solution = new_solution
                
                # 処理済みイベントを履歴に移動
                schedule_manager.event_history.extend(schedule_manager.active_events)
                schedule_manager.active_events = []
                
                # スナップショット作成
                self._create_snapshot(schedule_id, f"reoptimization_{reopt_type}")
                
                self._save_schedules(self.active_schedules)
                
                logger.info(f"Reoptimization completed for schedule {schedule_id}")
            
        except Exception as e:
            logger.error(f"Reoptimization failed for schedule {schedule_id}: {str(e)}")
    
    def _build_updated_problem(self, current_solution, active_events, urgent_job=None):
        """更新された問題の構築"""
        # 既存の問題設定を基に新しい問題を構築
        # これは簡略化された実装で、実際にはより複雑な処理が必要
        
        jobs = []
        machines = []
        
        # 現在の設定から基本構造を復元
        for job_schedule in current_solution.job_schedules:
            if job_schedule.status != "completed":  # 未完了ジョブのみ
                jobs.append({
                    "id": job_schedule.job_id,
                    "operations": job_schedule.operations
                })
        
        for machine_schedule in current_solution.machine_schedules:
            machines.append({
                "id": machine_schedule.machine_id,
                "capacity": 1  # デフォルト値
            })
        
        # 緊急ジョブの追加
        if urgent_job:
            jobs.append(urgent_job.dict())
        
        return {
            "problem_type": "job_shop",
            "jobs": jobs,
            "machines": machines,
            "optimization_objective": "minimize_makespan"
        }
    
    async def _execute_reoptimization(self, problem_data, reopt_request):
        """再最適化の実行"""
        try:
            # 高度な再最適化サービスを使用
            from ..services.reoptimization_service import reoptimization_service, ReoptimizationStrategy
            
            # 現在のスケジュール情報を取得
            schedule_manager = self.active_schedules[reopt_request.schedule_id]
            
            # 再最適化戦略を決定
            strategy = ReoptimizationStrategy.INCREMENTAL
            if reopt_request.reoptimization_type == "full":
                strategy = ReoptimizationStrategy.COMPLETE
            elif reopt_request.reoptimization_type == "emergency":
                strategy = ReoptimizationStrategy.PRIORITY_FOCUSED
            
            # 再最適化実行
            result = reoptimization_service.execute_reoptimization(
                current_solution=schedule_manager.current_solution,
                events=schedule_manager.active_events,
                strategy=strategy,
                time_limit=reopt_request.time_limit,
                preserve_completed=reopt_request.preserve_completed_jobs
            )
            
            return result.new_solution if result.success else None
            
        except Exception as e:
            logger.error(f"Reoptimization execution failed: {str(e)}")
            return None
    
    def _generate_alert_if_needed(self, schedule_id: str, event: ScheduleEvent):
        """必要に応じてアラートを生成"""
        alert_needed = False
        severity = "info"
        
        if event.impact_level == "critical":
            alert_needed = True
            severity = "critical"
        elif event.event_type in [ScheduleEventType.MACHINE_BREAKDOWN, ScheduleEventType.URGENT_JOB_ADD]:
            alert_needed = True
            severity = "error"
        elif event.event_type == ScheduleEventType.JOB_DELAY:
            delay_minutes = event.event_data.get("estimated_delay", 0)
            if delay_minutes > 30:  # 30分以上の遅延
                alert_needed = True
                severity = "warning"
        
        if alert_needed:
            alert = ScheduleAlert(
                id=str(uuid4()),
                schedule_id=schedule_id,
                alert_type=str(event.event_type),
                severity=severity,
                message=event.description,
                timestamp=datetime.now(),
                related_event_id=event.id
            )
            
            # アラート保存とWebSocket通知
            asyncio.create_task(self._notify_alert_via_websocket(schedule_id, alert))
            logger.warning(f"Alert generated: {alert.message}")
    
    def _create_critical_alert(self, schedule_id: str, message: str):
        """クリティカルアラートの作成"""
        alert = ScheduleAlert(
            id=str(uuid4()),
            schedule_id=schedule_id,
            alert_type="critical_event",
            severity="critical",
            message=message,
            timestamp=datetime.now()
        )
        
        # WebSocket通知
        asyncio.create_task(self._notify_alert_via_websocket(schedule_id, alert))
        logger.critical(f"Critical alert: {message}")
    
    def _create_snapshot(self, schedule_id: str, reason: str):
        """スケジュールスナップショットの作成"""
        if schedule_id not in self.active_schedules:
            return
        
        schedule_manager = self.active_schedules[schedule_id]
        snapshot = ScheduleSnapshot(
            schedule_id=schedule_id,
            timestamp=datetime.now(),
            solution=schedule_manager.current_solution,
            active_events=schedule_manager.active_events,
            performance_metrics=self._calculate_performance_metrics(schedule_id),
            snapshot_reason=reason
        )
        
        # スナップショットファイルに保存
        snapshot_file = os.path.join(
            self.snapshots_path, 
            f"{schedule_id}_{snapshot.timestamp.isoformat()}.json"
        )
        
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot.dict(), f, ensure_ascii=False, indent=2, default=str)
    
    def _calculate_performance_metrics(self, schedule_id: str) -> Dict[str, float]:
        """パフォーマンスメトリクスの計算"""
        if schedule_id not in self.active_schedules:
            return {}
        
        schedule_manager = self.active_schedules[schedule_id]
        solution = schedule_manager.current_solution
        
        return {
            "makespan": solution.metrics.makespan if solution.metrics else 0,
            "total_tardiness": solution.metrics.total_tardiness if solution.metrics else 0,
            "average_utilization": solution.metrics.average_machine_utilization if solution.metrics else 0,
            "active_events_count": len(schedule_manager.active_events),
            "reoptimizations_count": len(schedule_manager.event_history)
        }
    
    def _update_realtime_stats(self, schedule_id: str):
        """リアルタイム統計の更新"""
        # 実装は簡略化
        logger.info(f"Updated realtime stats for schedule {schedule_id}")
    
    def _analyze_priority_impact(self, schedule_id: str, job_id: str, new_priority: int) -> Dict[str, Any]:
        """優先度変更の影響分析"""
        # 簡略化された実装
        return {"requires_reoptimization": new_priority <= 2}
    
    async def _notify_event_via_websocket(self, schedule_id: str, event: ScheduleEvent):
        """WebSocket経由でイベントを通知"""
        try:
            from ..routers.websocket import notify_event_added
            await notify_event_added(schedule_id, event)
        except Exception as e:
            logger.error(f"Failed to notify event via WebSocket: {str(e)}")
    
    async def _notify_stats_via_websocket(self, schedule_id: str):
        """WebSocket経由で統計情報を通知"""
        try:
            stats = self.get_realtime_stats(schedule_id)
            if stats:
                from ..routers.websocket import notify_stats_update
                await notify_stats_update(schedule_id, stats)
        except Exception as e:
            logger.error(f"Failed to notify stats via WebSocket: {str(e)}")
    
    async def _notify_alert_via_websocket(self, schedule_id: str, alert: ScheduleAlert):
        """WebSocket経由でアラートを通知"""
        try:
            from ..routers.websocket import notify_alert
            await notify_alert(schedule_id, alert)
        except Exception as e:
            logger.error(f"Failed to notify alert via WebSocket: {str(e)}")
    
    def get_schedule_status(self, schedule_id: str) -> Optional[RealtimeScheduleManager]:
        """スケジュール状況の取得"""
        return self.active_schedules.get(schedule_id)
    
    def get_realtime_stats(self, schedule_id: str) -> Optional[RealtimeStats]:
        """リアルタイム統計の取得"""
        if schedule_id not in self.active_schedules:
            return None
        
        schedule_manager = self.active_schedules[schedule_id]
        solution = schedule_manager.current_solution
        
        # 統計計算
        completed_jobs = sum(1 for js in solution.job_schedules if getattr(js, 'status', '') == 'completed')
        active_jobs = len(solution.job_schedules) - completed_jobs
        delayed_jobs = sum(1 for js in solution.job_schedules if js.tardiness > 0)
        
        machine_utilization = {}
        for ms in solution.machine_schedules:
            machine_utilization[ms.machine_id] = ms.utilization
        
        return RealtimeStats(
            schedule_id=schedule_id,
            current_time=datetime.now(),
            completed_jobs=completed_jobs,
            active_jobs=active_jobs,
            delayed_jobs=delayed_jobs,
            machine_utilization=machine_utilization,
            critical_path_status="active" if solution.critical_path else "unknown",
            estimated_completion=datetime.now() + timedelta(minutes=solution.metrics.makespan) if solution.metrics else datetime.now(),
            kpi_metrics=self._calculate_performance_metrics(schedule_id)
        )


# サービスのシングルトンインスタンス
realtime_schedule_service = RealtimeScheduleService()