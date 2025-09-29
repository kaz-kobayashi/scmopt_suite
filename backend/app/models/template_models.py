"""
スケジュールテンプレート関連のPydanticモデル
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from .jobshop_models import JobShopProblem, Machine, Job


class ScheduleTemplate(BaseModel):
    """スケジュールテンプレートモデル"""
    id: str
    name: str
    description: Optional[str] = None
    category: str = "general"  # general, manufacturing, project, etc.
    problem_template: JobShopProblem
    default_solver_config: Optional[Dict[str, Any]] = None
    default_analysis_config: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    is_public: bool = True
    created_by: Optional[str] = None


class TemplateCreateRequest(BaseModel):
    """テンプレート作成リクエスト"""
    name: str
    description: Optional[str] = None
    category: str = "general"
    problem_template: JobShopProblem
    default_solver_config: Optional[Dict[str, Any]] = None
    default_analysis_config: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    is_public: bool = True


class TemplateUpdateRequest(BaseModel):
    """テンプレート更新リクエスト"""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    problem_template: Optional[JobShopProblem] = None
    default_solver_config: Optional[Dict[str, Any]] = None
    default_analysis_config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None


class TemplateSearchRequest(BaseModel):
    """テンプレート検索リクエスト"""
    query: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    limit: int = 20
    offset: int = 0


class TemplateListResponse(BaseModel):
    """テンプレート一覧レスポンス"""
    templates: List[ScheduleTemplate]
    total_count: int
    page: int
    per_page: int


class TemplateUsageStats(BaseModel):
    """テンプレート使用統計"""
    template_id: str
    usage_count: int
    last_used: Optional[datetime] = None
    average_solve_time: Optional[float] = None
    success_rate: float = 0.0


class PredefinedTemplate(BaseModel):
    """事前定義テンプレート（システム標準）"""
    name: str
    description: str
    category: str
    machine_count: int
    job_count: int
    avg_operations_per_job: int
    complexity_level: str  # simple, medium, complex
    use_case: str
    template_data: JobShopProblem


# 事前定義されたテンプレートの例
PREDEFINED_TEMPLATES = [
    {
        "name": "小規模製造業テンプレート",
        "description": "3台の機械で5つのジョブを処理する基本的な製造スケジュール",
        "category": "manufacturing",
        "machine_count": 3,
        "job_count": 5,
        "avg_operations_per_job": 3,
        "complexity_level": "simple",
        "use_case": "小規模製造工場での日次生産計画"
    },
    {
        "name": "中規模プロジェクトテンプレート", 
        "description": "5台の機械で10のジョブを処理するプロジェクト管理向け",
        "category": "project",
        "machine_count": 5,
        "job_count": 10,
        "avg_operations_per_job": 4,
        "complexity_level": "medium",
        "use_case": "中規模プロジェクトのタスクスケジューリング"
    },
    {
        "name": "複雑フローショップテンプレート",
        "description": "8台の機械で15のジョブを処理する複雑なフローライン",
        "category": "manufacturing",
        "machine_count": 8, 
        "job_count": 15,
        "avg_operations_per_job": 6,
        "complexity_level": "complex",
        "use_case": "複雑な製造フローラインの最適化"
    }
]