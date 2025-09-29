"""
データインポート・エクスポートAPIエンドポイント
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from fastapi.responses import StreamingResponse
import io
import logging

from ..models.jobshop_models import Job, Machine, JobShopProblem, JobShopSolution
from ..models.realtime_models import ScheduleEvent
from ..models.vrp_unified_models import VRPProblemData, ClientModel, DepotModel, VehicleTypeModel
from ..services.data_import_export_service import data_import_export_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["データインポート・エクスポート"])


# ======== VRP インポートエンドポイント ========

@router.post("/import/vrp/clients", response_model=List[ClientModel])
async def import_vrp_clients(file: UploadFile = File(...)):
    """VRP顧客データをインポート"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        clients = await data_import_export_service.import_vrp_clients_from_file(file)
        return clients
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"VRP clients import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VRP顧客インポートに失敗しました: {str(e)}")


@router.post("/import/vrp/depots", response_model=List[DepotModel])
async def import_vrp_depots(file: UploadFile = File(...)):
    """VRPデポデータをインポート"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        depots = await data_import_export_service.import_vrp_depots_from_file(file)
        return depots
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"VRP depots import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VRPデポインポートに失敗しました: {str(e)}")


@router.post("/import/vrp/vehicles", response_model=List[VehicleTypeModel])
async def import_vrp_vehicles(file: UploadFile = File(...)):
    """VRP車両データをインポート"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        vehicles = await data_import_export_service.import_vrp_vehicles_from_file(file)
        return vehicles
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"VRP vehicles import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VRP車両インポートに失敗しました: {str(e)}")


@router.post("/import/vrp/problem", response_model=VRPProblemData)
async def import_vrp_problem(file: UploadFile = File(...)):
    """VRP問題データをインポート"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        problem = await data_import_export_service.import_vrp_problem_from_file(file)
        return problem
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"VRP problem import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VRP問題インポートに失敗しました: {str(e)}")


@router.post("/import")
async def import_data(file: UploadFile = File(...)):
    """汎用データインポート（VRP対応）"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        # ファイル名から推測してVRP顧客データとして処理
        problem = await data_import_export_service.import_vrp_problem_from_file(file)
        
        return {
            "success": True,
            "data": problem.dict(),
            "summary": {
                "clients_imported": len(problem.clients),
                "depots_imported": len(problem.depots),
                "vehicle_types_imported": len(problem.vehicle_types)
            }
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Data import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"データインポートに失敗しました: {str(e)}")


# ======== インポートエンドポイント ========

@router.post("/import/jobs", response_model=List[Job])
async def import_jobs(file: UploadFile = File(...)):
    """ジョブデータをインポート"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        jobs = await data_import_export_service.import_jobs_from_file(file)
        return jobs
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Job import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ジョブインポートに失敗しました: {str(e)}")


@router.post("/import/machines", response_model=List[Machine])
async def import_machines(file: UploadFile = File(...)):
    """マシンデータをインポート"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        machines = await data_import_export_service.import_machines_from_file(file)
        return machines
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Machine import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"マシンインポートに失敗しました: {str(e)}")


@router.post("/import/problem", response_model=JobShopProblem)
async def import_problem(file: UploadFile = File(...)):
    """問題定義をインポート"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        problem = await data_import_export_service.import_problem_from_file(file)
        return problem
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Problem import error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"問題インポートに失敗しました: {str(e)}")


# ======== エクスポートエンドポイント ========

@router.post("/export/solution/csv")
async def export_solution_csv(solution: JobShopSolution):
    """ソリューションをCSVでエクスポート"""
    try:
        csv_data = data_import_export_service.export_solution_to_csv(solution)
        
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=solution_{solution.problem_id or 'unknown'}.csv"}
        )
    
    except Exception as e:
        logger.error(f"Solution CSV export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSVエクスポートに失敗しました: {str(e)}")


@router.post("/export/solution/excel")
async def export_solution_excel(solution: JobShopSolution):
    """ソリューションをExcelでエクスポート"""
    try:
        excel_data = data_import_export_service.export_solution_to_excel(solution)
        
        return Response(
            content=excel_data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=solution_{solution.problem_id or 'unknown'}.xlsx"}
        )
    
    except Exception as e:
        logger.error(f"Solution Excel export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excelエクスポートに失敗しました: {str(e)}")


@router.post("/export/solution/json")
async def export_solution_json(solution: JobShopSolution):
    """ソリューションをJSONでエクスポート"""
    try:
        json_data = data_import_export_service.export_solution_to_json(solution)
        
        return StreamingResponse(
            io.StringIO(json_data),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=solution_{solution.problem_id or 'unknown'}.json"}
        )
    
    except Exception as e:
        logger.error(f"Solution JSON export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JSONエクスポートに失敗しました: {str(e)}")


@router.post("/export/problem/json")
async def export_problem_json(problem: JobShopProblem):
    """問題定義をJSONでエクスポート"""
    try:
        json_data = data_import_export_service.export_problem_to_json(problem)
        
        return StreamingResponse(
            io.StringIO(json_data),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=problem_{problem.problem_type or 'jobshop'}.json"}
        )
    
    except Exception as e:
        logger.error(f"Problem JSON export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"問題JSONエクスポートに失敗しました: {str(e)}")


@router.post("/export/events/csv")
async def export_events_csv(events: List[ScheduleEvent]):
    """イベントをCSVでエクスポート"""
    try:
        csv_data = data_import_export_service.export_events_to_csv(events)
        
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=schedule_events.csv"}
        )
    
    except Exception as e:
        logger.error(f"Events CSV export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"イベントCSVエクスポートに失敗しました: {str(e)}")


# ======== テンプレートエンドポイント ========

@router.get("/templates/jobs/csv")
async def get_jobs_csv_template():
    """ジョブCSVテンプレートを取得"""
    template_data = """job_id,job_name,priority,weight,release_time,due_date,operation_id,machine_id,duration,position,setup_time,eligible_machines,skill_requirements
J001,Job 1,1,1.0,0,100,OP001,M001,10,0,2,"M001,M002",
J001,Job 1,1,1.0,0,100,OP002,M002,15,1,1,M002,
J002,Job 2,2,1.5,0,120,OP003,M001,20,0,3,"M001,M003",cutting
J002,Job 2,2,1.5,0,120,OP004,M003,12,1,2,M003,assembly"""
    
    return StreamingResponse(
        io.StringIO(template_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=jobs_template.csv"}
    )


@router.get("/templates/machines/csv")
async def get_machines_csv_template():
    """マシンCSVテンプレートを取得"""
    template_data = """machine_id,machine_name,capacity,available_from,available_until,skills,maintenance
M001,Machine 1,1,0,1000,"cutting,drilling","100-120,500-520"
M002,Machine 2,1,0,1000,"assembly,welding","200-210"
M003,Machine 3,2,0,1000,"painting,finishing","""
    
    return StreamingResponse(
        io.StringIO(template_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=machines_template.csv"}
    )


@router.get("/templates/vrp/clients/csv")
async def get_vrp_clients_csv_template():
    """VRP顧客CSVテンプレートを取得"""
    template_data = """x,y,delivery,pickup,service_duration,tw_early,tw_late,prize,required,priority
139.4000,35.7000,5,0,10,480,720,0,true,1
139.6000,35.6500,7,0,12,540,780,0,true,1
139.3000,35.8000,4,0,8,480,660,0,true,1
139.5000,35.5000,6,2,15,600,900,50,false,2
139.2000,35.9000,5,0,11,480,720,0,true,1"""
    
    return StreamingResponse(
        io.StringIO(template_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vrp_clients_template.csv"}
    )


@router.get("/templates/vrp/depots/csv")
async def get_vrp_depots_csv_template():
    """VRPデポCSVテンプレートを取得"""
    template_data = """x,y,tw_early,tw_late
139.4500,35.7500,480,1080
139.7000,35.5500,480,1080"""
    
    return StreamingResponse(
        io.StringIO(template_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vrp_depots_template.csv"}
    )


@router.get("/templates/vrp/vehicles/csv")
async def get_vrp_vehicles_csv_template():
    """VRP車両CSVテンプレートを取得"""
    template_data = """num_available,capacity,start_depot,end_depot,fixed_cost,tw_early,tw_late,max_duration,max_distance
2,100,0,0,100,480,1080,600,200000
1,150,1,1,120,480,1080,720,250000"""
    
    return StreamingResponse(
        io.StringIO(template_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vrp_vehicles_template.csv"}
    )


@router.get("/formats")
async def get_supported_formats():
    """サポートされているファイル形式を取得"""
    return {
        "import_formats": {
            "jobs": ["csv", "xlsx", "xls", "json"],
            "machines": ["csv", "xlsx", "xls", "json"],
            "problem": ["json", "xlsx", "xls"],
            "vrp_clients": ["csv", "xlsx", "xls", "json"],
            "vrp_depots": ["csv", "xlsx", "xls", "json"],
            "vrp_vehicles": ["csv", "xlsx", "xls", "json"],
            "vrp_problem": ["json", "xlsx", "xls"]
        },
        "export_formats": {
            "solution": ["csv", "xlsx", "json"],
            "problem": ["json"],
            "events": ["csv"]
        },
        "templates": ["jobs_csv", "machines_csv", "vrp_clients_csv", "vrp_depots_csv", "vrp_vehicles_csv"],
        "file_size_limit": "10MB"
    }


# ======== バリデーションエンドポイント ========

@router.post("/validate/jobs")
async def validate_jobs_file(file: UploadFile = File(...)):
    """ジョブファイルのバリデーション"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        # ファイル形式チェック
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in data_import_export_service.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"サポートされていないファイル形式: {file_extension}"
            )
        
        # ファイルサイズチェック（10MB制限）
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="ファイルサイズが10MBを超えています")
        
        # ファイルポインタをリセット
        await file.seek(0)
        
        # インポート試行（エラーチェック）
        try:
            jobs = await data_import_export_service.import_jobs_from_file(file)
            
            validation_result = {
                "valid": True,
                "job_count": len(jobs),
                "warnings": [],
                "errors": []
            }
            
            # 基本的なバリデーション
            for job in jobs:
                if len(job.operations) == 0:
                    validation_result["warnings"].append(f"ジョブ {job.id} に操作がありません")
                
                if not job.name:
                    validation_result["warnings"].append(f"ジョブ {job.id} に名前がありません")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "job_count": 0,
                "warnings": [],
                "errors": [str(e)]
            }
    
    except Exception as e:
        logger.error(f"Jobs validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"バリデーションに失敗しました: {str(e)}")


@router.post("/validate/machines")
async def validate_machines_file(file: UploadFile = File(...)):
    """マシンファイルのバリデーション"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        # ファイル形式チェック
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in data_import_export_service.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"サポートされていないファイル形式: {file_extension}"
            )
        
        # ファイルサイズチェック（10MB制限）
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="ファイルサイズが10MBを超えています")
        
        # ファイルポインタをリセット
        await file.seek(0)
        
        # インポート試行
        try:
            machines = await data_import_export_service.import_machines_from_file(file)
            
            validation_result = {
                "valid": True,
                "machine_count": len(machines),
                "warnings": [],
                "errors": []
            }
            
            # 基本的なバリデーション
            for machine in machines:
                if machine.capacity <= 0:
                    validation_result["warnings"].append(f"マシン {machine.id} の容量が0以下です")
                
                if not machine.name:
                    validation_result["warnings"].append(f"マシン {machine.id} に名前がありません")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "machine_count": 0,
                "warnings": [],
                "errors": [str(e)]
            }
    
    except Exception as e:
        logger.error(f"Machines validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"バリデーションに失敗しました: {str(e)}")