"""
CSV/Excelデータインポート・エクスポートサービス
"""
import csv
import json
import io
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from fastapi import UploadFile
import tempfile
import os

from ..models.jobshop_models import (
    Job, Machine, Operation, JobShopProblem, JobShopSolution
)
from ..models.realtime_models import ScheduleEvent
from ..models.vrp_unified_models import VRPProblemData, ClientModel, DepotModel, VehicleTypeModel

logger = logging.getLogger(__name__)


class DataImportExportService:
    """データインポート・エクスポートサービス"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json']
    
    # ======== インポート機能 ========
    
    async def import_jobs_from_file(self, file: UploadFile) -> List[Job]:
        """ファイルからジョブデータをインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'csv':
                return await self._import_jobs_from_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                return await self._import_jobs_from_excel(file)
            elif file_extension == 'json':
                return await self._import_jobs_from_json(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import jobs: {str(e)}")
            raise
    
    async def import_machines_from_file(self, file: UploadFile) -> List[Machine]:
        """ファイルからマシンデータをインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'csv':
                return await self._import_machines_from_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                return await self._import_machines_from_excel(file)
            elif file_extension == 'json':
                return await self._import_machines_from_json(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import machines: {str(e)}")
            raise
    
    async def import_problem_from_file(self, file: UploadFile) -> JobShopProblem:
        """ファイルから問題定義をインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'json':
                return await self._import_problem_from_json(file)
            elif file_extension in ['csv', 'xlsx', 'xls']:
                return await self._import_problem_from_structured_file(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import problem: {str(e)}")
            raise
    
    # CSV インポート
    
    async def _import_jobs_from_csv(self, file: UploadFile) -> List[Job]:
        """CSVからジョブをインポート"""
        content = await file.read()
        csv_data = content.decode('utf-8')
        
        jobs = []
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        job_operations = {}  # job_id -> operations mapping
        
        for row in csv_reader:
            job_id = row.get('job_id', row.get('ジョブID'))
            operation_id = row.get('operation_id', row.get('操作ID'))
            
            if not job_id:
                continue
            
            # Operation data
            operation_data = {
                'id': operation_id or f"{job_id}_op_{len(job_operations.get(job_id, []))}",
                'job_id': job_id,
                'machine_id': row.get('machine_id', row.get('マシンID')),
                'duration': int(row.get('duration', row.get('処理時間', 10))),
                'position_in_job': int(row.get('position', row.get('位置', 0))),
                'setup_time': int(row.get('setup_time', row.get('段取時間', 0))),
                'eligible_machines': self._parse_list_field(row.get('eligible_machines', row.get('利用可能マシン', ''))),
                'skill_requirements': self._parse_list_field(row.get('skill_requirements', row.get('必要スキル', '')))
            }
            
            if job_id not in job_operations:
                job_operations[job_id] = []
            
            job_operations[job_id].append(Operation(**operation_data))
        
        # Create Job objects
        job_info = {}
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        for row in csv_reader:
            job_id = row.get('job_id', row.get('ジョブID'))
            if not job_id or job_id in job_info:
                continue
            
            job_info[job_id] = {
                'id': job_id,
                'name': row.get('job_name', row.get('ジョブ名', f"Job {job_id}")),
                'priority': int(row.get('priority', row.get('優先度', 1))),
                'weight': float(row.get('weight', row.get('重み', 1.0))),
                'release_time': int(row.get('release_time', row.get('リリース時間', 0))),
                'due_date': int(row.get('due_date', row.get('期限', 0))) or None
            }
        
        for job_id, operations in job_operations.items():
            job_data = job_info.get(job_id, {'id': job_id, 'name': f"Job {job_id}"})
            job_data['operations'] = sorted(operations, key=lambda x: x.position_in_job)
            jobs.append(Job(**job_data))
        
        return jobs
    
    async def _import_machines_from_csv(self, file: UploadFile) -> List[Machine]:
        """CSVからマシンをインポート"""
        content = await file.read()
        csv_data = content.decode('utf-8')
        
        machines = []
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        for row in csv_reader:
            machine_id = row.get('machine_id', row.get('マシンID'))
            if not machine_id:
                continue
            
            machine_data = {
                'id': machine_id,
                'name': row.get('machine_name', row.get('マシン名', f"Machine {machine_id}")),
                'capacity': int(row.get('capacity', row.get('容量', 1))),
                'available_from': int(row.get('available_from', row.get('利用開始時間', 0))),
                'available_until': int(row.get('available_until', row.get('利用終了時間', 0))) or None,
                'skills': self._parse_list_field(row.get('skills', row.get('スキル', ''))),
                'maintenance_windows': self._parse_maintenance_windows(row.get('maintenance', row.get('メンテナンス', '')))
            }
            
            machines.append(Machine(**machine_data))
        
        return machines
    
    # VRP Data Import Methods
    
    async def import_vrp_clients_from_file(self, file: UploadFile) -> List[ClientModel]:
        """ファイルからVRP顧客データをインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'csv':
                return await self._import_vrp_clients_from_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                return await self._import_vrp_clients_from_excel(file)
            elif file_extension == 'json':
                return await self._import_vrp_clients_from_json(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import VRP clients: {str(e)}")
            raise
    
    async def import_vrp_depots_from_file(self, file: UploadFile) -> List[DepotModel]:
        """ファイルからVRPデポデータをインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'csv':
                return await self._import_vrp_depots_from_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                return await self._import_vrp_depots_from_excel(file)
            elif file_extension == 'json':
                return await self._import_vrp_depots_from_json(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import VRP depots: {str(e)}")
            raise
    
    async def import_vrp_vehicles_from_file(self, file: UploadFile) -> List[VehicleTypeModel]:
        """ファイルからVRP車両データをインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'csv':
                return await self._import_vrp_vehicles_from_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                return await self._import_vrp_vehicles_from_excel(file)
            elif file_extension == 'json':
                return await self._import_vrp_vehicles_from_json(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import VRP vehicles: {str(e)}")
            raise
    
    async def import_vrp_problem_from_file(self, file: UploadFile) -> VRPProblemData:
        """ファイルからVRP問題データをインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'json':
                return await self._import_vrp_problem_from_json(file)
            elif file_extension in ['csv', 'xlsx', 'xls']:
                return await self._import_vrp_problem_from_structured_file(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import VRP problem: {str(e)}")
            raise
    
    # VRP CSV Import Methods
    
    async def _import_vrp_clients_from_csv(self, file: UploadFile) -> List[ClientModel]:
        """CSVからVRP顧客をインポート"""
        content = await file.read()
        csv_data = content.decode('utf-8')
        
        clients = []
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        for row in csv_reader:
            # Required fields
            if not (row.get('x') and row.get('y')):
                continue
            
            client_data = {
                'x': int(float(row.get('x', 0))),
                'y': int(float(row.get('y', 0))),
                'delivery': int(float(row.get('delivery', 0))),
                'pickup': int(float(row.get('pickup', 0))),
                'service_duration': int(float(row.get('service_duration', 10))),
                'tw_early': int(float(row.get('tw_early', 0))) if row.get('tw_early') else None,
                'tw_late': int(float(row.get('tw_late', 1440))) if row.get('tw_late') else None,
                'release_time': int(float(row.get('release_time', 0))) if row.get('release_time') else None,
                'prize': int(float(row.get('prize', 0))) if row.get('prize') else None,
                'required': str(row.get('required', 'true')).lower() in ['true', '1', 'yes'],
                'priority': int(float(row.get('priority', 1))) if row.get('priority') else None
            }
            
            clients.append(ClientModel(**client_data))
        
        return clients
    
    async def _import_vrp_depots_from_csv(self, file: UploadFile) -> List[DepotModel]:
        """CSVからVRPデポをインポート"""
        content = await file.read()
        csv_data = content.decode('utf-8')
        
        depots = []
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        for row in csv_reader:
            if not (row.get('x') and row.get('y')):
                continue
            
            depot_data = {
                'x': int(float(row.get('x', 0))),
                'y': int(float(row.get('y', 0))),
                'tw_early': int(float(row.get('tw_early', 0))) if row.get('tw_early') else None,
                'tw_late': int(float(row.get('tw_late', 1440))) if row.get('tw_late') else None
            }
            
            depots.append(DepotModel(**depot_data))
        
        return depots
    
    async def _import_vrp_vehicles_from_csv(self, file: UploadFile) -> List[VehicleTypeModel]:
        """CSVからVRP車両タイプをインポート"""
        content = await file.read()
        csv_data = content.decode('utf-8')
        
        vehicles = []
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        for row in csv_reader:
            if not row.get('num_available'):
                continue
            
            vehicle_data = {
                'num_available': int(float(row.get('num_available', 1))),
                'capacity': int(float(row.get('capacity', 100))),
                'start_depot': int(float(row.get('start_depot', 0))),
                'end_depot': int(float(row.get('end_depot', 0))) if row.get('end_depot') else None,
                'fixed_cost': int(float(row.get('fixed_cost', 0))) if row.get('fixed_cost') else None,
                'tw_early': int(float(row.get('tw_early', 0))) if row.get('tw_early') else None,
                'tw_late': int(float(row.get('tw_late', 1440))) if row.get('tw_late') else None,
                'max_duration': int(float(row.get('max_duration', 0))) if row.get('max_duration') else None,
                'max_distance': int(float(row.get('max_distance', 0))) if row.get('max_distance') else None
            }
            
            vehicles.append(VehicleTypeModel(**vehicle_data))
        
        return vehicles
    
    # VRP JSON Import Methods
    
    async def _import_vrp_clients_from_json(self, file: UploadFile) -> List[ClientModel]:
        """JSONからVRP顧客をインポート"""
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        if isinstance(data, list):
            return [ClientModel(**client_data) for client_data in data]
        elif isinstance(data, dict) and 'clients' in data:
            return [ClientModel(**client_data) for client_data in data['clients']]
        else:
            raise ValueError("Invalid JSON format for VRP clients")
    
    async def _import_vrp_depots_from_json(self, file: UploadFile) -> List[DepotModel]:
        """JSONからVRPデポをインポート"""
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        if isinstance(data, list):
            return [DepotModel(**depot_data) for depot_data in data]
        elif isinstance(data, dict) and 'depots' in data:
            return [DepotModel(**depot_data) for depot_data in data['depots']]
        else:
            raise ValueError("Invalid JSON format for VRP depots")
    
    async def _import_vrp_vehicles_from_json(self, file: UploadFile) -> List[VehicleTypeModel]:
        """JSONからVRP車両をインポート"""
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        if isinstance(data, list):
            return [VehicleTypeModel(**vehicle_data) for vehicle_data in data]
        elif isinstance(data, dict) and 'vehicle_types' in data:
            return [VehicleTypeModel(**vehicle_data) for vehicle_data in data['vehicle_types']]
        else:
            raise ValueError("Invalid JSON format for VRP vehicles")
    
    async def _import_vrp_problem_from_json(self, file: UploadFile) -> VRPProblemData:
        """JSONからVRP問題定義をインポート"""
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        return VRPProblemData(**data)
    
    # VRP Excel Import Methods
    
    async def _import_vrp_clients_from_excel(self, file: UploadFile) -> List[ClientModel]:
        """ExcelからVRP顧客をインポート"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_excel(tmp_file_path)
            return self._process_vrp_clients_dataframe(df)
        finally:
            os.unlink(tmp_file_path)
    
    async def _import_vrp_depots_from_excel(self, file: UploadFile) -> List[DepotModel]:
        """ExcelからVRPデポをインポート"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_excel(tmp_file_path)
            return self._process_vrp_depots_dataframe(df)
        finally:
            os.unlink(tmp_file_path)
    
    async def _import_vrp_vehicles_from_excel(self, file: UploadFile) -> List[VehicleTypeModel]:
        """ExcelからVRP車両をインポート"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_excel(tmp_file_path)
            return self._process_vrp_vehicles_dataframe(df)
        finally:
            os.unlink(tmp_file_path)
    
    async def _import_vrp_problem_from_structured_file(self, file: UploadFile) -> VRPProblemData:
        """構造化ファイル（CSV/Excel）からVRP問題をインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'csv':
                # CSVの場合は顧客データとして処理
                clients = await self._import_vrp_clients_from_csv(file)
                
                # デフォルトのデポと車両タイプを生成
                depots = [DepotModel(x=1394500, y=357500)]  # Central Tokyo
                vehicle_types = [VehicleTypeModel(num_available=2, capacity=100, start_depot=0)]
                
                return VRPProblemData(
                    clients=clients,
                    depots=depots,
                    vehicle_types=vehicle_types
                )
            
            elif file_extension in ['xlsx', 'xls']:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                try:
                    excel_file = pd.ExcelFile(tmp_file_path)
                    
                    clients = []
                    depots = []
                    vehicle_types = []
                    
                    # Read different sheets
                    if 'Clients' in excel_file.sheet_names or '顧客' in excel_file.sheet_names:
                        clients_sheet = 'Clients' if 'Clients' in excel_file.sheet_names else '顧客'
                        df_clients = pd.read_excel(tmp_file_path, sheet_name=clients_sheet)
                        clients = self._process_vrp_clients_dataframe(df_clients)
                    
                    if 'Depots' in excel_file.sheet_names or 'デポ' in excel_file.sheet_names:
                        depots_sheet = 'Depots' if 'Depots' in excel_file.sheet_names else 'デポ'
                        df_depots = pd.read_excel(tmp_file_path, sheet_name=depots_sheet)
                        depots = self._process_vrp_depots_dataframe(df_depots)
                    
                    if 'Vehicles' in excel_file.sheet_names or '車両' in excel_file.sheet_names:
                        vehicles_sheet = 'Vehicles' if 'Vehicles' in excel_file.sheet_names else '車両'
                        df_vehicles = pd.read_excel(tmp_file_path, sheet_name=vehicles_sheet)
                        vehicle_types = self._process_vrp_vehicles_dataframe(df_vehicles)
                    
                    # Default values if not provided
                    if not depots:
                        depots = [DepotModel(x=1394500, y=357500)]
                    if not vehicle_types:
                        vehicle_types = [VehicleTypeModel(num_available=2, capacity=100, start_depot=0)]
                    
                    return VRPProblemData(
                        clients=clients,
                        depots=depots,
                        vehicle_types=vehicle_types
                    )
                
                finally:
                    os.unlink(tmp_file_path)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import VRP problem from structured file: {str(e)}")
            raise
    
    def _process_vrp_clients_dataframe(self, df: pd.DataFrame) -> List[ClientModel]:
        """DataFrameからVRP顧客を処理"""
        clients = []
        
        for _, row in df.iterrows():
            if pd.isna(row.get('x')) or pd.isna(row.get('y')):
                continue
            
            client_data = {
                'x': int(float(row.get('x', 0))),
                'y': int(float(row.get('y', 0))),
                'delivery': int(float(row.get('delivery', 0))),
                'pickup': int(float(row.get('pickup', 0))),
                'service_duration': int(float(row.get('service_duration', 10))),
                'tw_early': int(float(row.get('tw_early', 0))) if not pd.isna(row.get('tw_early')) else None,
                'tw_late': int(float(row.get('tw_late', 1440))) if not pd.isna(row.get('tw_late')) else None,
                'prize': int(float(row.get('prize', 0))) if not pd.isna(row.get('prize')) else None,
                'required': str(row.get('required', 'true')).lower() in ['true', '1', 'yes']
            }
            
            clients.append(ClientModel(**client_data))
        
        return clients
    
    def _process_vrp_depots_dataframe(self, df: pd.DataFrame) -> List[DepotModel]:
        """DataFrameからVRPデポを処理"""
        depots = []
        
        for _, row in df.iterrows():
            if pd.isna(row.get('x')) or pd.isna(row.get('y')):
                continue
            
            depot_data = {
                'x': int(float(row.get('x', 0))),
                'y': int(float(row.get('y', 0))),
                'tw_early': int(float(row.get('tw_early', 0))) if not pd.isna(row.get('tw_early')) else None,
                'tw_late': int(float(row.get('tw_late', 1440))) if not pd.isna(row.get('tw_late')) else None
            }
            
            depots.append(DepotModel(**depot_data))
        
        return depots
    
    def _process_vrp_vehicles_dataframe(self, df: pd.DataFrame) -> List[VehicleTypeModel]:
        """DataFrameからVRP車両タイプを処理"""
        vehicles = []
        
        for _, row in df.iterrows():
            if pd.isna(row.get('num_available')):
                continue
            
            vehicle_data = {
                'num_available': int(float(row.get('num_available', 1))),
                'capacity': int(float(row.get('capacity', 100))),
                'start_depot': int(float(row.get('start_depot', 0))),
                'end_depot': int(float(row.get('end_depot', 0))) if not pd.isna(row.get('end_depot')) else None,
                'fixed_cost': int(float(row.get('fixed_cost', 0))) if not pd.isna(row.get('fixed_cost')) else None,
                'tw_early': int(float(row.get('tw_early', 0))) if not pd.isna(row.get('tw_early')) else None,
                'tw_late': int(float(row.get('tw_late', 1440))) if not pd.isna(row.get('tw_late')) else None,
                'max_duration': int(float(row.get('max_duration', 0))) if not pd.isna(row.get('max_duration')) else None,
                'max_distance': int(float(row.get('max_distance', 0))) if not pd.isna(row.get('max_distance')) else None
            }
            
            vehicles.append(VehicleTypeModel(**vehicle_data))
        
        return vehicles
    
    # Excel インポート
    
    async def _import_jobs_from_excel(self, file: UploadFile) -> List[Job]:
        """Excelからジョブをインポート"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_excel(tmp_file_path)
            return self._process_jobs_dataframe(df)
        finally:
            os.unlink(tmp_file_path)
    
    async def _import_machines_from_excel(self, file: UploadFile) -> List[Machine]:
        """Excelからマシンをインポート"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_excel(tmp_file_path)
            return self._process_machines_dataframe(df)
        finally:
            os.unlink(tmp_file_path)
    
    # JSON インポート
    
    async def _import_jobs_from_json(self, file: UploadFile) -> List[Job]:
        """JSONからジョブをインポート"""
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        if isinstance(data, list):
            return [Job(**job_data) for job_data in data]
        elif isinstance(data, dict) and 'jobs' in data:
            return [Job(**job_data) for job_data in data['jobs']]
        else:
            raise ValueError("Invalid JSON format for jobs")
    
    async def _import_machines_from_json(self, file: UploadFile) -> List[Machine]:
        """JSONからマシンをインポート"""
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        if isinstance(data, list):
            return [Machine(**machine_data) for machine_data in data]
        elif isinstance(data, dict) and 'machines' in data:
            return [Machine(**machine_data) for machine_data in data['machines']]
        else:
            raise ValueError("Invalid JSON format for machines")
    
    async def _import_problem_from_json(self, file: UploadFile) -> JobShopProblem:
        """JSONから問題定義をインポート"""
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        return JobShopProblem(**data)
    
    # ======== エクスポート機能 ========
    
    def export_solution_to_csv(self, solution: JobShopSolution) -> str:
        """ソリューションをCSVに出力"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'job_id', 'operation_id', 'machine_id', 'start_time', 'end_time', 
            'duration', 'setup_time', 'tardiness', 'priority'
        ])
        
        # Job schedules
        for job_schedule in solution.job_schedules:
            for operation in job_schedule.operations:
                writer.writerow([
                    job_schedule.job_id,
                    operation.operation_id,
                    operation.machine_id,
                    operation.start_time,
                    operation.end_time,
                    operation.end_time - operation.start_time if operation.end_time else 0,
                    getattr(operation, 'setup_time', 0),
                    job_schedule.tardiness,
                    getattr(job_schedule, 'priority', 1)
                ])
        
        return output.getvalue()
    
    def export_solution_to_excel(self, solution: JobShopSolution) -> bytes:
        """ソリューションをExcelに出力"""
        # Job Schedule データ
        job_data = []
        for job_schedule in solution.job_schedules:
            for operation in job_schedule.operations:
                job_data.append({
                    'ジョブID': job_schedule.job_id,
                    '操作ID': operation.operation_id,
                    'マシンID': operation.machine_id,
                    '開始時間': operation.start_time,
                    '終了時間': operation.end_time,
                    '処理時間': operation.end_time - operation.start_time if operation.end_time else 0,
                    '段取時間': getattr(operation, 'setup_time', 0),
                    '遅延時間': job_schedule.tardiness,
                    '優先度': getattr(job_schedule, 'priority', 1)
                })
        
        # Machine Schedule データ
        machine_data = []
        for machine_schedule in solution.machine_schedules:
            machine_data.append({
                'マシンID': machine_schedule.machine_id,
                '稼働率': machine_schedule.utilization,
                'アイドル時間': machine_schedule.idle_time,
                '操作数': len(machine_schedule.operations),
                'メイクスパン': machine_schedule.end_time
            })
        
        # メトリクスデータ
        metrics_data = []
        if solution.metrics:
            metrics_data.append({
                '項目': 'メイクスパン',
                '値': solution.metrics.makespan,
                '単位': '分'
            })
            metrics_data.append({
                '項目': '総遅延時間',
                '値': solution.metrics.total_tardiness,
                '単位': '分'
            })
            metrics_data.append({
                '項目': '平均稼働率',
                '値': solution.metrics.average_machine_utilization,
                '単位': '%'
            })
        
        # Excel書き出し
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                pd.DataFrame(job_data).to_excel(writer, sheet_name='ジョブスケジュール', index=False)
                pd.DataFrame(machine_data).to_excel(writer, sheet_name='マシンスケジュール', index=False)
                pd.DataFrame(metrics_data).to_excel(writer, sheet_name='メトリクス', index=False)
            
            with open(tmp_file.name, 'rb') as f:
                excel_data = f.read()
        
        os.unlink(tmp_file.name)
        return excel_data
    
    def export_problem_to_json(self, problem: JobShopProblem) -> str:
        """問題定義をJSONに出力"""
        return problem.json(ensure_ascii=False, indent=2)
    
    def export_solution_to_json(self, solution: JobShopSolution) -> str:
        """ソリューションをJSONに出力"""
        return solution.json(ensure_ascii=False, indent=2)
    
    def export_events_to_csv(self, events: List[ScheduleEvent]) -> str:
        """イベントをCSVに出力"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'event_id', 'event_type', 'timestamp', 'target_id', 
            'description', 'impact_level', 'auto_reoptimize', 'processed'
        ])
        
        # Events
        for event in events:
            writer.writerow([
                event.id,
                event.event_type,
                event.timestamp.isoformat(),
                event.target_id,
                event.description,
                event.impact_level,
                event.auto_reoptimize,
                event.processed
            ])
        
        return output.getvalue()
    
    # ======== ヘルパーメソッド ========
    
    def _get_file_extension(self, filename: Optional[str]) -> str:
        """ファイル拡張子を取得"""
        if not filename:
            raise ValueError("Filename is required")
        
        return filename.lower().split('.')[-1]
    
    def _parse_list_field(self, field_value: str) -> List[str]:
        """カンマ区切りの文字列をリストに変換"""
        if not field_value:
            return []
        
        return [item.strip() for item in field_value.split(',') if item.strip()]
    
    def _parse_maintenance_windows(self, maintenance_str: str) -> List[Dict[str, Any]]:
        """メンテナンス窓の解析"""
        if not maintenance_str:
            return []
        
        windows = []
        try:
            # Format: "start1-end1,start2-end2"
            for window_str in maintenance_str.split(','):
                if '-' in window_str:
                    start, end = window_str.strip().split('-')
                    windows.append({
                        'start_time': int(start.strip()),
                        'end_time': int(end.strip())
                    })
        except Exception as e:
            logger.warning(f"Failed to parse maintenance windows: {e}")
        
        return windows
    
    def _process_jobs_dataframe(self, df: pd.DataFrame) -> List[Job]:
        """DataFrameからジョブを処理"""
        jobs = []
        job_operations = {}
        
        for _, row in df.iterrows():
            job_id = str(row.get('job_id', row.get('ジョブID', '')))
            if not job_id:
                continue
            
            operation_data = {
                'id': str(row.get('operation_id', row.get('操作ID', f"{job_id}_op_{len(job_operations.get(job_id, []))}"))),
                'job_id': job_id,
                'machine_id': str(row.get('machine_id', row.get('マシンID', ''))),
                'duration': int(row.get('duration', row.get('処理時間', 10))),
                'position_in_job': int(row.get('position', row.get('位置', 0))),
                'setup_time': int(row.get('setup_time', row.get('段取時間', 0)))
            }
            
            if job_id not in job_operations:
                job_operations[job_id] = []
            
            job_operations[job_id].append(Operation(**operation_data))
        
        # Create jobs
        job_info = {}
        for _, row in df.iterrows():
            job_id = str(row.get('job_id', row.get('ジョブID', '')))
            if not job_id or job_id in job_info:
                continue
            
            job_info[job_id] = {
                'id': job_id,
                'name': str(row.get('job_name', row.get('ジョブ名', f"Job {job_id}"))),
                'priority': int(row.get('priority', row.get('優先度', 1))),
                'weight': float(row.get('weight', row.get('重み', 1.0))),
                'release_time': int(row.get('release_time', row.get('リリース時間', 0)))
            }
        
        for job_id, operations in job_operations.items():
            job_data = job_info.get(job_id, {'id': job_id, 'name': f"Job {job_id}"})
            job_data['operations'] = sorted(operations, key=lambda x: x.position_in_job)
            jobs.append(Job(**job_data))
        
        return jobs
    
    def _process_machines_dataframe(self, df: pd.DataFrame) -> List[Machine]:
        """DataFrameからマシンを処理"""
        machines = []
        
        for _, row in df.iterrows():
            machine_id = str(row.get('machine_id', row.get('マシンID', '')))
            if not machine_id:
                continue
            
            machine_data = {
                'id': machine_id,
                'name': str(row.get('machine_name', row.get('マシン名', f"Machine {machine_id}"))),
                'capacity': int(row.get('capacity', row.get('容量', 1))),
                'available_from': int(row.get('available_from', row.get('利用開始時間', 0))),
                'skills': self._parse_list_field(str(row.get('skills', row.get('スキル', ''))))
            }
            
            machines.append(Machine(**machine_data))
        
        return machines
    
    async def _import_problem_from_structured_file(self, file: UploadFile) -> JobShopProblem:
        """構造化ファイル（CSV/Excel）から問題をインポート"""
        try:
            file_extension = self._get_file_extension(file.filename)
            
            if file_extension == 'csv':
                # CSVの場合、複数のセクションを想定
                content = await file.read()
                csv_data = content.decode('utf-8')
                
                # 簡略化: 全てジョブデータとして扱う
                jobs = await self._import_jobs_from_csv(file)
                
                # デフォルトマシンを生成
                machines = self._generate_default_machines_from_jobs(jobs)
                
                return JobShopProblem(
                    problem_type="job_shop",
                    jobs=jobs,
                    machines=machines,
                    optimization_objective="makespan"
                )
            
            elif file_extension in ['xlsx', 'xls']:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                try:
                    # 複数シートを読み込み
                    excel_file = pd.ExcelFile(tmp_file_path)
                    
                    jobs = []
                    machines = []
                    
                    if 'Jobs' in excel_file.sheet_names or 'ジョブ' in excel_file.sheet_names:
                        jobs_sheet = 'Jobs' if 'Jobs' in excel_file.sheet_names else 'ジョブ'
                        df_jobs = pd.read_excel(tmp_file_path, sheet_name=jobs_sheet)
                        jobs = self._process_jobs_dataframe(df_jobs)
                    
                    if 'Machines' in excel_file.sheet_names or 'マシン' in excel_file.sheet_names:
                        machines_sheet = 'Machines' if 'Machines' in excel_file.sheet_names else 'マシン'
                        df_machines = pd.read_excel(tmp_file_path, sheet_name=machines_sheet)
                        machines = self._process_machines_dataframe(df_machines)
                    
                    if not machines:
                        machines = self._generate_default_machines_from_jobs(jobs)
                    
                    return JobShopProblem(
                        problem_type="job_shop",
                        jobs=jobs,
                        machines=machines,
                        optimization_objective="makespan"
                    )
                    
                finally:
                    os.unlink(tmp_file_path)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Failed to import problem from structured file: {str(e)}")
            raise
    
    def _generate_default_machines_from_jobs(self, jobs: List[Job]) -> List[Machine]:
        """ジョブから必要なマシンを生成"""
        machine_ids = set()
        
        for job in jobs:
            for operation in job.operations:
                if operation.machine_id:
                    machine_ids.add(operation.machine_id)
                if operation.eligible_machines:
                    machine_ids.update(operation.eligible_machines)
        
        machines = []
        for machine_id in machine_ids:
            machines.append(Machine(
                id=machine_id,
                name=f"Machine {machine_id}",
                capacity=1,
                available_from=0
            ))
        
        return machines


# サービスのシングルトンインスタンス
data_import_export_service = DataImportExportService()