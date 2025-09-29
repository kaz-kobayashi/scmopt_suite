"""
リアルタイムスケジュール管理APIエンドポイント
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from uuid import uuid4

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
from ..models.jobshop_models import JobShopSolution
from ..services.realtime_schedule_service import realtime_schedule_service

router = APIRouter(prefix="/realtime", tags=["リアルタイムスケジュール"])


@router.post("/schedules", response_model=str)
async def create_realtime_schedule(
    solution: JobShopSolution,
    monitor_config: Optional[ScheduleMonitor] = None
):
    """新しいリアルタイムスケジュールを作成"""
    try:
        schedule_id = realtime_schedule_service.create_realtime_schedule(solution, monitor_config)
        return schedule_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create realtime schedule: {str(e)}")


@router.get("/schedules/{schedule_id}", response_model=RealtimeScheduleManager)
async def get_schedule_status(schedule_id: str):
    """スケジュールの現在状況を取得"""
    schedule = realtime_schedule_service.get_schedule_status(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.get("/schedules/{schedule_id}/stats", response_model=RealtimeStats)
async def get_realtime_stats(schedule_id: str):
    """リアルタイム統計を取得"""
    stats = realtime_schedule_service.get_realtime_stats(schedule_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return stats


@router.post("/schedules/{schedule_id}/events", response_model=bool)
async def add_event(schedule_id: str, event_data: dict):
    """スケジュールイベントを追加"""
    try:
        # イベントデータからScheduleEventを構築
        event = ScheduleEvent(
            id=str(uuid4()),
            event_type=ScheduleEventType(event_data["event_type"]),
            timestamp=datetime.now(),
            target_id=event_data["target_id"],
            description=event_data.get("description", ""),
            event_data=event_data.get("event_data", {}),
            impact_level=event_data.get("impact_level", "medium"),
            auto_reoptimize=event_data.get("auto_reoptimize", True)
        )
        
        success = realtime_schedule_service.add_event(schedule_id, event)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return success
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add event: {str(e)}")


@router.post("/schedules/{schedule_id}/events/job-delay", response_model=bool)
async def add_job_delay_event(
    schedule_id: str,
    job_id: str,
    operation_id: str,
    original_duration: int,
    new_duration: int,
    delay_reason: str,
    impact_level: str = "medium"
):
    """作業遅延イベントを追加"""
    try:
        delay_data = JobDelayEvent(
            job_id=job_id,
            operation_id=operation_id,
            original_duration=original_duration,
            new_duration=new_duration,
            delay_reason=delay_reason,
            estimated_delay=new_duration - original_duration
        )
        
        event = ScheduleEvent(
            id=str(uuid4()),
            event_type=ScheduleEventType.JOB_DELAY,
            timestamp=datetime.now(),
            target_id=job_id,
            description=f"Job {job_id} delayed: {delay_reason}",
            event_data=delay_data.dict(),
            impact_level=impact_level,
            auto_reoptimize=True
        )
        
        success = realtime_schedule_service.add_event(schedule_id, event)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return success
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add job delay event: {str(e)}")


@router.post("/schedules/{schedule_id}/events/machine-breakdown", response_model=bool)
async def add_machine_breakdown_event(
    schedule_id: str,
    machine_id: str,
    affected_operations: List[str],
    severity: str = "medium",
    estimated_repair_time: Optional[datetime] = None,
    repair_notes: Optional[str] = None
):
    """機械故障イベントを追加"""
    try:
        breakdown_data = MachineBreakdownEvent(
            machine_id=machine_id,
            breakdown_time=datetime.now(),
            estimated_repair_time=estimated_repair_time,
            affected_operations=affected_operations,
            severity=severity,
            repair_notes=repair_notes
        )
        
        impact_level = "critical" if severity == "high" else "high"
        
        event = ScheduleEvent(
            id=str(uuid4()),
            event_type=ScheduleEventType.MACHINE_BREAKDOWN,
            timestamp=datetime.now(),
            target_id=machine_id,
            description=f"Machine {machine_id} breakdown - {severity} severity",
            event_data=breakdown_data.dict(),
            impact_level=impact_level,
            auto_reoptimize=True
        )
        
        success = realtime_schedule_service.add_event(schedule_id, event)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return success
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add machine breakdown event: {str(e)}")


@router.post("/schedules/{schedule_id}/events/urgent-job", response_model=bool)
async def add_urgent_job_event(
    schedule_id: str,
    urgent_job: dict,
    priority_level: int = 1,
    deadline: Optional[datetime] = None,
    preempt_existing: bool = False
):
    """緊急ジョブ追加イベントを追加"""
    try:
        from ..models.jobshop_models import Job
        
        urgent_job_obj = Job(**urgent_job)
        
        urgent_data = UrgentJobEvent(
            urgent_job=urgent_job_obj,
            priority_level=priority_level,
            deadline=deadline,
            preempt_existing=preempt_existing
        )
        
        impact_level = "critical" if priority_level == 1 else "high"
        
        event = ScheduleEvent(
            id=str(uuid4()),
            event_type=ScheduleEventType.URGENT_JOB_ADD,
            timestamp=datetime.now(),
            target_id=urgent_job_obj.id,
            description=f"Urgent job {urgent_job_obj.id} added with priority {priority_level}",
            event_data=urgent_data.dict(),
            impact_level=impact_level,
            auto_reoptimize=True
        )
        
        success = realtime_schedule_service.add_event(schedule_id, event)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return success
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add urgent job event: {str(e)}")


@router.post("/schedules/{schedule_id}/events/job-completion", response_model=bool)
async def add_job_completion_event(
    schedule_id: str,
    job_id: str,
    actual_completion_time: Optional[datetime] = None
):
    """作業完了イベントを追加"""
    try:
        completion_time = actual_completion_time or datetime.now()
        
        event = ScheduleEvent(
            id=str(uuid4()),
            event_type=ScheduleEventType.JOB_COMPLETION,
            timestamp=datetime.now(),
            target_id=job_id,
            description=f"Job {job_id} completed",
            event_data={"job_id": job_id, "completion_time": completion_time.isoformat()},
            impact_level="low",
            auto_reoptimize=False
        )
        
        success = realtime_schedule_service.add_event(schedule_id, event)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return success
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add job completion event: {str(e)}")


@router.post("/schedules/{schedule_id}/events/priority-change", response_model=bool)
async def add_priority_change_event(
    schedule_id: str,
    job_id: str,
    old_priority: int,
    new_priority: int,
    reason: Optional[str] = None
):
    """優先度変更イベントを追加"""
    try:
        event = ScheduleEvent(
            id=str(uuid4()),
            event_type=ScheduleEventType.PRIORITY_CHANGE,
            timestamp=datetime.now(),
            target_id=job_id,
            description=f"Job {job_id} priority changed from {old_priority} to {new_priority}",
            event_data={
                "job_id": job_id,
                "old_priority": old_priority,
                "new_priority": new_priority,
                "reason": reason
            },
            impact_level="medium" if abs(new_priority - old_priority) > 2 else "low",
            auto_reoptimize=new_priority <= 2  # 高優先度の場合は自動再最適化
        )
        
        success = realtime_schedule_service.add_event(schedule_id, event)
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return success
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add priority change event: {str(e)}")


@router.post("/schedules/{schedule_id}/reoptimize", response_model=ReoptimizationResult)
async def trigger_manual_reoptimization(
    schedule_id: str,
    reoptimization_type: str = "incremental",
    time_limit: int = 60,
    preserve_completed_jobs: bool = True,
    strategy: Optional[str] = None
):
    """手動で再最適化をトリガー"""
    try:
        schedule = realtime_schedule_service.get_schedule_status(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # 高度な再最適化サービスを使用
        from ..services.reoptimization_service import reoptimization_service, ReoptimizationStrategy, ReoptimizationTrigger
        
        # 戦略の決定
        if strategy:
            reopt_strategy = ReoptimizationStrategy(strategy)
        else:
            # デフォルト戦略の選択
            if reoptimization_type == "full":
                reopt_strategy = ReoptimizationStrategy.COMPLETE
            elif reoptimization_type == "emergency":
                reopt_strategy = ReoptimizationStrategy.PRIORITY_FOCUSED
            elif any(event.impact_level == "critical" for event in schedule.active_events):
                reopt_strategy = ReoptimizationStrategy.PRIORITY_FOCUSED
            else:
                reopt_strategy = ReoptimizationStrategy.INCREMENTAL
        
        # 再最適化実行
        result = reoptimization_service.execute_reoptimization(
            current_solution=schedule.current_solution,
            events=schedule.active_events,
            strategy=reopt_strategy,
            trigger=ReoptimizationTrigger.MANUAL,
            time_limit=time_limit,
            preserve_completed=preserve_completed_jobs
        )
        
        if result.success and result.new_solution:
            # スケジュールを更新
            schedule.current_solution = result.new_solution
            
            # 処理済みイベントを履歴に移動
            schedule.event_history.extend(schedule.active_events)
            schedule.active_events = []
            
            # スケジュールを保存
            realtime_schedule_service._save_schedules(realtime_schedule_service.active_schedules)
            
            # スナップショット作成
            realtime_schedule_service._create_snapshot(schedule_id, f"manual_reoptimization_{reopt_strategy}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger reoptimization: {str(e)}")


@router.get("/schedules/{schedule_id}/events", response_model=List[ScheduleEvent])
async def get_schedule_events(
    schedule_id: str,
    include_history: bool = False,
    event_type: Optional[str] = None,
    limit: int = 100
):
    """スケジュールのイベント履歴を取得"""
    try:
        schedule = realtime_schedule_service.get_schedule_status(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        events = []
        
        # アクティブイベント
        events.extend(schedule.active_events)
        
        # 履歴も含める場合
        if include_history:
            events.extend(schedule.event_history)
        
        # イベントタイプでフィルタ
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # 最新順にソート
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 制限
        return events[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get events: {str(e)}")


@router.get("/schedules", response_model=Dict[str, RealtimeScheduleManager])
async def get_all_active_schedules():
    """全アクティブスケジュールを取得"""
    try:
        return realtime_schedule_service.active_schedules
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schedules: {str(e)}")


@router.delete("/schedules/{schedule_id}")
async def deactivate_schedule(schedule_id: str):
    """スケジュールを非アクティブ化"""
    try:
        if schedule_id not in realtime_schedule_service.active_schedules:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # スケジュールを非アクティブ化
        schedule = realtime_schedule_service.active_schedules[schedule_id]
        schedule.status = "completed"
        
        realtime_schedule_service._save_schedules(realtime_schedule_service.active_schedules)
        
        return {"message": "Schedule deactivated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deactivate schedule: {str(e)}")


@router.get("/schedules/{schedule_id}/snapshots", response_model=List[str])
async def get_schedule_snapshots(schedule_id: str):
    """スケジュールのスナップショット一覧を取得"""
    try:
        import os
        import glob
        
        snapshot_pattern = os.path.join(
            realtime_schedule_service.snapshots_path,
            f"{schedule_id}_*.json"
        )
        
        snapshot_files = glob.glob(snapshot_pattern)
        snapshot_files.sort(reverse=True)  # 最新順
        
        return [os.path.basename(f) for f in snapshot_files]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get snapshots: {str(e)}")


@router.get("/schedules/{schedule_id}/snapshots/{snapshot_file}", response_model=ScheduleSnapshot)
async def get_snapshot(schedule_id: str, snapshot_file: str):
    """特定のスナップショットを取得"""
    try:
        import os
        import json
        
        snapshot_path = os.path.join(
            realtime_schedule_service.snapshots_path,
            snapshot_file
        )
        
        if not os.path.exists(snapshot_path):
            raise HTTPException(status_code=404, detail="Snapshot not found")
        
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            snapshot_data = json.load(f)
        
        return ScheduleSnapshot(**snapshot_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot: {str(e)}")