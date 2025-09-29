"""
リアルタイムスケジュール更新関連のPydanticモデル
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

from .jobshop_models import Job, Machine, Operation, JobShopSolution


class ScheduleEventType(str, Enum):
    """スケジュールイベントの種類"""
    JOB_DELAY = "job_delay"              # 作業遅延
    JOB_COMPLETION = "job_completion"    # 作業完了
    MACHINE_BREAKDOWN = "machine_breakdown"  # 機械故障
    MACHINE_REPAIR = "machine_repair"    # 機械修理完了
    URGENT_JOB_ADD = "urgent_job_add"    # 緊急ジョブ追加
    JOB_CANCEL = "job_cancel"           # ジョブキャンセル
    PRIORITY_CHANGE = "priority_change"  # 優先度変更
    RESOURCE_CHANGE = "resource_change"  # リソース変更


class ScheduleEvent(BaseModel):
    """スケジュール変更イベント"""
    id: str
    event_type: ScheduleEventType
    timestamp: datetime
    target_id: str  # 対象となるジョブID、マシンIDなど
    description: str
    event_data: Dict[str, Any]
    impact_level: str = "medium"  # low, medium, high, critical
    auto_reoptimize: bool = True
    processed: bool = False


class JobDelayEvent(BaseModel):
    """作業遅延イベント"""
    job_id: str
    operation_id: str
    original_duration: int
    new_duration: int
    delay_reason: str
    estimated_delay: int


class MachineBreakdownEvent(BaseModel):
    """機械故障イベント"""
    machine_id: str
    breakdown_time: datetime
    estimated_repair_time: Optional[datetime] = None
    affected_operations: List[str]
    severity: str = "medium"  # low, medium, high
    repair_notes: Optional[str] = None


class UrgentJobEvent(BaseModel):
    """緊急ジョブ追加イベント"""
    urgent_job: Job
    priority_level: int = 1  # 1が最高優先度
    deadline: Optional[datetime] = None
    preempt_existing: bool = False


class ScheduleUpdate(BaseModel):
    """スケジュール更新情報"""
    original_solution: JobShopSolution
    events: List[ScheduleEvent]
    updated_solution: Optional[JobShopSolution] = None
    reoptimization_required: bool = True
    update_strategy: str = "incremental"  # incremental, full_reoptimization
    impact_analysis: Optional[Dict[str, Any]] = None


class RealtimeScheduleManager(BaseModel):
    """リアルタイムスケジュール管理"""
    schedule_id: str
    current_solution: JobShopSolution
    active_events: List[ScheduleEvent] = []
    event_history: List[ScheduleEvent] = []
    last_update: datetime
    status: str = "active"  # active, paused, completed
    auto_reoptimize: bool = True
    reoptimization_threshold: int = 3  # イベント数の閾値


class ScheduleMonitor(BaseModel):
    """スケジュール監視設定"""
    schedule_id: str
    monitor_jobs: bool = True
    monitor_machines: bool = True
    monitor_delays: bool = True
    alert_thresholds: Dict[str, Any] = {
        "delay_threshold": 30,  # 分
        "utilization_threshold": 0.95,
        "critical_path_delay": 15
    }
    notification_settings: Dict[str, bool] = {
        "email_alerts": False,
        "browser_notifications": True,
        "dashboard_alerts": True
    }


class ScheduleAlert(BaseModel):
    """スケジュールアラート"""
    id: str
    schedule_id: str
    alert_type: str
    severity: str  # info, warning, error, critical
    message: str
    timestamp: datetime
    acknowledged: bool = False
    auto_action_taken: bool = False
    related_event_id: Optional[str] = None


class ReoptimizationRequest(BaseModel):
    """再最適化リクエスト"""
    schedule_id: str
    trigger_events: List[str]
    reoptimization_type: str = "incremental"  # incremental, full
    constraints_override: Optional[Dict[str, Any]] = None
    time_limit: int = 60  # 秒
    preserve_completed_jobs: bool = True


class ReoptimizationResult(BaseModel):
    """再最適化結果"""
    schedule_id: str
    success: bool
    new_solution: Optional[JobShopSolution] = None
    optimization_time: float
    changes_summary: Dict[str, Any]
    impact_metrics: Dict[str, float]
    error_message: Optional[str] = None


class ScheduleSnapshot(BaseModel):
    """スケジュールスナップショット"""
    schedule_id: str
    timestamp: datetime
    solution: JobShopSolution
    active_events: List[ScheduleEvent]
    performance_metrics: Dict[str, float]
    snapshot_reason: str  # scheduled, event_triggered, manual


class ScheduleComparison(BaseModel):
    """スケジュール比較結果"""
    original_schedule_id: str
    updated_schedule_id: str
    changes: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    recommendations: List[str]


# イベント処理戦略
class EventProcessingStrategy(BaseModel):
    """イベント処理戦略"""
    strategy_type: str  # immediate, batched, scheduled
    batch_size: int = 5
    processing_interval: int = 300  # 秒
    priority_threshold: int = 2  # この優先度以上は即座に処理


# リアルタイム統計
class RealtimeStats(BaseModel):
    """リアルタイム統計"""
    schedule_id: str
    current_time: datetime
    completed_jobs: int
    active_jobs: int
    delayed_jobs: int
    machine_utilization: Dict[str, float]
    critical_path_status: str
    estimated_completion: datetime
    kpi_metrics: Dict[str, float]


# 予測モデル
class SchedulePrediction(BaseModel):
    """スケジュール予測"""
    schedule_id: str
    prediction_horizon: int  # 時間（分）
    predicted_delays: List[Dict[str, Any]]
    predicted_bottlenecks: List[str]
    risk_factors: List[Dict[str, Any]]
    confidence_score: float
    generated_at: datetime