from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import tempfile
import os
from datetime import datetime, date

from app.services.shift_service import ShiftService

router = APIRouter()

class PeriodData(BaseModel):
    id: int
    description: str

class BreakData(BaseModel):
    period: int
    break_time: int

class DayData(BaseModel):
    id: int
    day: str
    day_of_week: str
    day_type: str

class JobData(BaseModel):
    id: int
    description: str

class StaffData(BaseModel):
    name: str
    wage_per_period: int
    max_period: int
    max_day: int
    job_set: List[int]
    day_off: List[int]
    start: int
    end: int
    request: Optional[Dict[int, List[int]]] = None

class RequirementData(BaseModel):
    day_type: str
    job: int
    period: int
    requirement: int

class ShiftOptimizationRequest(BaseModel):
    period_df: List[PeriodData]
    break_df: List[BreakData]
    day_df: List[DayData]
    job_df: List[JobData]
    staff_df: List[StaffData]
    requirement_df: List[RequirementData]
    theta: int = Field(default=1, description="開始直後・終了直前の休憩禁止期間数")
    lb_penalty: int = Field(default=10000, description="必要人数下限逸脱ペナルティ")
    ub_penalty: int = Field(default=0, description="必要人数上限逸脱ペナルティ")
    job_change_penalty: int = Field(default=10, description="ジョブ変更ペナルティ")
    break_penalty: int = Field(default=10000, description="休憩時間逸脱ペナルティ")
    max_day_penalty: int = Field(default=5000, description="最大勤務日数超過ペナルティ")
    time_limit: int = Field(default=30, description="計算時間上限(秒)")
    random_seed: int = Field(default=1, description="乱数シード")

class DataGenerationRequest(BaseModel):
    start_date: str
    end_date: str
    start_time: str = Field(default="09:00", description="開始時刻")
    end_time: str = Field(default="21:00", description="終了時刻")
    freq: str = Field(default="1h", description="時間刻み")
    job_list: List[str] = Field(default=["レジ打ち", "接客"], description="ジョブリスト")

@router.post("/optimize")
async def optimize_shift(request: ShiftOptimizationRequest):
    """シフト最適化を実行"""
    try:
        service = ShiftService()
        
        # データフレームの作成
        period_df = pd.DataFrame([p.dict() for p in request.period_df])
        break_df = pd.DataFrame([b.dict() for b in request.break_df])
        day_df = pd.DataFrame([d.dict() for d in request.day_df])
        job_df = pd.DataFrame([j.dict() for j in request.job_df])
        
        # スタッフデータの処理
        staff_data = []
        for staff in request.staff_df:
            staff_dict = staff.dict()
            staff_dict['job_set'] = staff.job_set
            staff_dict['day_off'] = staff.day_off
            if staff.request:
                staff_dict['request'] = staff.request
            else:
                staff_dict['request'] = None
            staff_data.append(staff_dict)
        staff_df = pd.DataFrame(staff_data)
        
        requirement_df = pd.DataFrame([r.dict() for r in request.requirement_df])
        
        # 最適化実行
        result = service.shift_scheduling(
            period_df=period_df,
            break_df=break_df,
            day_df=day_df,
            job_df=job_df,
            staff_df=staff_df,
            requirement_df=requirement_df,
            theta=request.theta,
            lb_penalty=request.lb_penalty,
            ub_penalty=request.ub_penalty,
            job_change_penalty=request.job_change_penalty,
            break_penalty=request.break_penalty,
            max_day_penalty=request.max_day_penalty,
            time_limit=request.time_limit,
            random_seed=request.random_seed
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"最適化エラー: {str(e)}")

@router.post("/generate-sample-data")
async def generate_sample_data(request: DataGenerationRequest):
    """サンプルデータの生成"""
    try:
        service = ShiftService()
        
        # データ生成
        data = service.generate_sample_data(
            start_date=request.start_date,
            end_date=request.end_date,
            start_time=request.start_time,
            end_time=request.end_time,
            freq=request.freq,
            job_list=request.job_list
        )
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"データ生成エラー: {str(e)}")

@router.post("/export-excel")
async def export_excel(request: Dict[str, Any]):
    """結果をExcel形式でエクスポート"""
    try:
        service = ShiftService()
        
        # Excelファイル生成
        excel_path = service.create_excel_report(request)
        
        return FileResponse(
            path=excel_path,
            filename=f"shift_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel出力エラー: {str(e)}")

@router.post("/optimize-with-requests")
async def optimize_shift_with_requests(request: ShiftOptimizationRequest):
    """日別希望勤務時間を考慮したシフト最適化を実行"""
    try:
        service = ShiftService()
        
        # データフレームの作成
        period_df = pd.DataFrame([p.dict() for p in request.period_df])
        break_df = pd.DataFrame([b.dict() for b in request.break_df])
        day_df = pd.DataFrame([d.dict() for d in request.day_df])
        job_df = pd.DataFrame([j.dict() for j in request.job_df])
        
        # スタッフデータの処理
        staff_data = []
        for staff in request.staff_df:
            staff_dict = staff.dict()
            staff_dict['job_set'] = staff.job_set
            staff_dict['day_off'] = staff.day_off
            if staff.request:
                staff_dict['request'] = staff.request
            else:
                staff_dict['request'] = None
            staff_data.append(staff_dict)
        staff_df = pd.DataFrame(staff_data)
        
        requirement_df = pd.DataFrame([r.dict() for r in request.requirement_df])
        
        # shift_scheduling2を使用（日別リクエスト対応版）
        result = service.shift_scheduling2(
            period_df=period_df,
            break_df=break_df,
            day_df=day_df,
            job_df=job_df,
            staff_df=staff_df,
            requirement_df=requirement_df,
            theta=request.theta,
            lb_penalty=request.lb_penalty,
            ub_penalty=request.ub_penalty,
            job_change_penalty=request.job_change_penalty,
            break_penalty=request.break_penalty,
            max_day_penalty=request.max_day_penalty,
            time_limit=request.time_limit,
            random_seed=request.random_seed
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"最適化エラー: {str(e)}")

@router.post("/export-gantt-excel")
async def export_gantt_excel(request: Dict[str, Any]):
    """ガントチャートをExcel形式でエクスポート"""
    try:
        service = ShiftService()
        
        # サンプルデータから必要な情報を取得
        sample_data = request.get('sampleData', {})
        period_df = pd.DataFrame(sample_data.get('period_df', []))
        day_df = pd.DataFrame(sample_data.get('day_df', []))
        job_df = pd.DataFrame(sample_data.get('job_df', []))
        staff_df = pd.DataFrame(sample_data.get('staff_df', []))
        requirement_df = pd.DataFrame(sample_data.get('requirement_df', []))
        
        # Excel ガントチャートファイル生成
        excel_path = service.make_gannt_excel(
            request.get('job_assign', {}),
            period_df, day_df, job_df, staff_df, requirement_df
        )
        
        return FileResponse(
            path=excel_path,
            filename=f"shift_gantt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ガントチャートExcel出力エラー: {str(e)}")

@router.post("/export-allshift-excel")
async def export_allshift_excel(request: Dict[str, Any]):
    """全シフト詳細をExcel形式でエクスポート"""
    try:
        service = ShiftService()
        
        # サンプルデータから必要な情報を取得
        sample_data = request.get('sampleData', {})
        day_df = pd.DataFrame(sample_data.get('day_df', []))
        period_df = pd.DataFrame(sample_data.get('period_df', []))
        
        # optimizationResultからstaff_dfを取得
        optimization_result = request.get('optimizationResult', {})
        staff_data = optimization_result.get('staff_df', [])
        
        # Excel 全シフトファイル生成
        excel_path = service.make_allshift_excel(
            staff_data, day_df, period_df
        )
        
        return FileResponse(
            path=excel_path,
            filename=f"all_shifts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"全シフトExcel出力エラー: {str(e)}")

@router.get("/estimate-feasibility")
async def estimate_feasibility(
    period_data: str,
    day_data: str,
    job_data: str,
    staff_data: str,
    requirement_data: str,
    days: Optional[str] = None
):
    """実行可能性の推定"""
    try:
        service = ShiftService()
        
        # JSONデータをDataFrameに変換
        period_df = pd.read_json(period_data)
        day_df = pd.read_json(day_data)
        job_df = pd.read_json(job_data)
        staff_df = pd.read_json(staff_data)
        requirement_df = pd.read_json(requirement_data)
        
        days_list = None
        if days:
            days_list = json.loads(days)
        
        # 実行可能性推定
        fig_json = service.estimate_requirement(
            day_df, period_df, job_df, staff_df, requirement_df, days_list
        )
        
        return {"feasibility_chart": fig_json}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"実行可能性推定エラー: {str(e)}")