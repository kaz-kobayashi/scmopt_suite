"""
スケジュールテンプレート管理API
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging

from ..models.template_models import (
    ScheduleTemplate,
    TemplateCreateRequest,
    TemplateUpdateRequest, 
    TemplateSearchRequest,
    TemplateListResponse,
    TemplateUsageStats
)
from ..models.jobshop_models import JobShopSolution, SolverConfig, AnalysisConfig
from ..services.template_service import template_service
from ..services.jobshop_service import jobshop_service

router = APIRouter(prefix="/templates", tags=["templates"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=ScheduleTemplate)
async def create_template(request: TemplateCreateRequest):
    """
    新しいスケジュールテンプレートを作成
    """
    try:
        logger.info(f"Creating new template: {request.name}")
        template = template_service.create_template(request)
        return template
    except Exception as e:
        logger.error(f"Template creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート作成に失敗しました: {str(e)}")


@router.get("/{template_id}", response_model=ScheduleTemplate)
async def get_template(template_id: str):
    """
    指定IDのテンプレートを取得
    """
    try:
        template = template_service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート取得に失敗しました: {str(e)}")


@router.put("/{template_id}", response_model=ScheduleTemplate)
async def update_template(template_id: str, request: TemplateUpdateRequest):
    """
    テンプレートを更新
    """
    try:
        logger.info(f"Updating template: {template_id}")
        template = template_service.update_template(template_id, request)
        if not template:
            raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート更新に失敗しました: {str(e)}")


@router.delete("/{template_id}")
async def delete_template(template_id: str):
    """
    テンプレートを削除
    """
    try:
        logger.info(f"Deleting template: {template_id}")
        success = template_service.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=404, detail="テンプレートが見つからないか、削除できません")
        return {"message": "テンプレートが削除されました"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート削除に失敗しました: {str(e)}")


@router.post("/search", response_model=TemplateListResponse)
async def search_templates(request: TemplateSearchRequest):
    """
    テンプレートを検索
    """
    try:
        logger.info(f"Searching templates with query: {request.query}")
        result = template_service.search_templates(request)
        return result
    except Exception as e:
        logger.error(f"Template search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート検索に失敗しました: {str(e)}")


@router.get("/", response_model=TemplateListResponse)
async def list_templates(
    category: Optional[str] = None,
    is_public: Optional[bool] = None,
    limit: int = 20,
    offset: int = 0
):
    """
    テンプレート一覧を取得
    """
    try:
        search_request = TemplateSearchRequest(
            category=category,
            is_public=is_public,
            limit=limit,
            offset=offset
        )
        result = template_service.search_templates(search_request)
        return result
    except Exception as e:
        logger.error(f"Template listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート一覧取得に失敗しました: {str(e)}")


@router.get("/categories/list", response_model=List[str])
async def get_categories():
    """
    利用可能なテンプレートカテゴリを取得
    """
    try:
        categories = template_service.get_categories()
        return categories
    except Exception as e:
        logger.error(f"Category listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"カテゴリ取得に失敗しました: {str(e)}")


@router.get("/popular/list", response_model=List[ScheduleTemplate])
async def get_popular_templates(limit: int = 10):
    """
    人気テンプレートを取得
    """
    try:
        templates = template_service.get_popular_templates(limit)
        return templates
    except Exception as e:
        logger.error(f"Popular templates retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"人気テンプレート取得に失敗しました: {str(e)}")


@router.post("/{template_id}/duplicate", response_model=ScheduleTemplate)
async def duplicate_template(template_id: str, new_name: str):
    """
    テンプレートを複製
    """
    try:
        logger.info(f"Duplicating template: {template_id} -> {new_name}")
        template = template_service.duplicate_template(template_id, new_name)
        if not template:
            raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template duplication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート複製に失敗しました: {str(e)}")


@router.post("/{template_id}/solve", response_model=JobShopSolution)
async def solve_from_template(
    template_id: str,
    solver_config: Optional[SolverConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None,
    override_problem: Optional[dict] = None
):
    """
    テンプレートから問題を解く
    """
    try:
        logger.info(f"Solving problem from template: {template_id}")
        
        # テンプレートの取得
        template = template_service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
        
        # 設定の準備（テンプレートのデフォルト設定を使用）
        if solver_config is None and template.default_solver_config:
            solver_config = SolverConfig(**template.default_solver_config)
        elif solver_config is None:
            solver_config = SolverConfig()
        
        if analysis_config is None and template.default_analysis_config:
            analysis_config = AnalysisConfig(**template.default_analysis_config)
        elif analysis_config is None:
            analysis_config = AnalysisConfig()
        
        # 問題の準備
        problem = template.problem_template
        
        # 問題の上書きがある場合は適用
        if override_problem:
            problem_dict = problem.dict()
            problem_dict.update(override_problem)
            from ..models.jobshop_models import JobShopProblem
            problem = JobShopProblem(**problem_dict)
        
        # 解決時間の記録開始
        start_time = datetime.now()
        
        # 問題を解く
        solution = jobshop_service.solve_job_shop(problem, solver_config, analysis_config)
        
        # 解決時間の計算
        solve_time = (datetime.now() - start_time).total_seconds()
        
        # 使用統計の更新
        success = solution.solution_status in ["OPTIMAL", "FEASIBLE"]
        template_service.increment_usage(template_id, solve_time, success)
        
        return solution
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレートからの問題解決に失敗しました: {str(e)}")


@router.get("/{template_id}/stats", response_model=TemplateUsageStats)
async def get_template_stats(template_id: str):
    """
    テンプレートの使用統計を取得
    """
    try:
        stats = template_service.get_usage_stats(template_id)
        if not stats:
            # テンプレートは存在するが統計がない場合のデフォルト
            template = template_service.get_template(template_id)
            if not template:
                raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
            
            stats = TemplateUsageStats(
                template_id=template_id,
                usage_count=template.usage_count,
                last_used=None,
                average_solve_time=None,
                success_rate=0.0
            )
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレート統計取得に失敗しました: {str(e)}")


@router.post("/{template_id}/export")
async def export_template(template_id: str):
    """
    テンプレートをエクスポート（JSON形式）
    """
    try:
        template = template_service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="テンプレートが見つかりません")
        
        # エクスポート用のデータ準備
        export_data = {
            "template": template.dict(),
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        from fastapi.responses import JSONResponse
        
        return JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f"attachment; filename=template_{template_id}.json"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレートエクスポートに失敗しました: {str(e)}")


@router.post("/import", response_model=ScheduleTemplate)
async def import_template(import_data: dict):
    """
    テンプレートをインポート
    """
    try:
        logger.info("Importing template from data")
        
        if "template" not in import_data:
            raise HTTPException(status_code=400, detail="無効なインポートデータです")
        
        template_data = import_data["template"]
        
        # インポート用のリクエストを作成
        request = TemplateCreateRequest(
            name=template_data.get("name", "インポートされたテンプレート"),
            description=template_data.get("description", "外部からインポートされたテンプレート"),
            category=template_data.get("category", "general"),
            problem_template=template_data["problem_template"],
            default_solver_config=template_data.get("default_solver_config"),
            default_analysis_config=template_data.get("default_analysis_config"),
            tags=template_data.get("tags", []) + ["インポート"],
            is_public=False  # インポートは非公開に
        )
        
        template = template_service.create_template(request)
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テンプレートインポートに失敗しました: {str(e)}")