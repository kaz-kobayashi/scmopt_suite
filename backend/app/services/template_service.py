"""
スケジュールテンプレート管理サービス
"""
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from ..models.template_models import (
    ScheduleTemplate, 
    TemplateCreateRequest,
    TemplateUpdateRequest, 
    TemplateSearchRequest,
    TemplateListResponse,
    TemplateUsageStats,
    PredefinedTemplate,
    PREDEFINED_TEMPLATES
)
from ..models.jobshop_models import JobShopProblem, Machine, Job, Operation


class TemplateService:
    """スケジュールテンプレート管理サービス"""
    
    def __init__(self, storage_path: str = "data/templates"):
        self.storage_path = storage_path
        self.templates_file = os.path.join(storage_path, "templates.json")
        self.usage_stats_file = os.path.join(storage_path, "usage_stats.json")
        self._ensure_storage_directory()
        self._initialize_predefined_templates()
    
    def _ensure_storage_directory(self):
        """ストレージディレクトリの作成"""
        os.makedirs(self.storage_path, exist_ok=True)
        
        # テンプレートファイルが存在しない場合は初期化
        if not os.path.exists(self.templates_file):
            self._save_templates({})
        
        # 使用統計ファイルが存在しない場合は初期化
        if not os.path.exists(self.usage_stats_file):
            self._save_usage_stats({})
    
    def _load_templates(self) -> Dict[str, ScheduleTemplate]:
        """テンプレートファイルの読み込み"""
        try:
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                templates = {}
                for template_id, template_data in data.items():
                    templates[template_id] = ScheduleTemplate(**template_data)
                return templates
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_templates(self, templates: Dict[str, ScheduleTemplate]):
        """テンプレートファイルの保存"""
        data = {}
        for template_id, template in templates.items():
            data[template_id] = template.dict()
        
        with open(self.templates_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def _load_usage_stats(self) -> Dict[str, TemplateUsageStats]:
        """使用統計の読み込み"""
        try:
            with open(self.usage_stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats = {}
                for template_id, stats_data in data.items():
                    stats[template_id] = TemplateUsageStats(**stats_data)
                return stats
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_usage_stats(self, stats: Dict[str, TemplateUsageStats]):
        """使用統計の保存"""
        data = {}
        for template_id, stat in stats.items():
            data[template_id] = stat.dict()
        
        with open(self.usage_stats_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def _initialize_predefined_templates(self):
        """事前定義テンプレートの初期化"""
        templates = self._load_templates()
        
        for template_data in PREDEFINED_TEMPLATES:
            template_id = f"predefined_{template_data['name'].lower().replace(' ', '_')}"
            
            # 既に存在する場合はスキップ
            if template_id in templates:
                continue
            
            # サンプル問題データの生成
            problem = self._generate_sample_problem(
                template_data["machine_count"],
                template_data["job_count"],
                template_data["avg_operations_per_job"]
            )
            
            template = ScheduleTemplate(
                id=template_id,
                name=template_data["name"],
                description=template_data["description"],
                category=template_data["category"],
                problem_template=problem,
                default_solver_config={
                    "solver_type": "pyjobshop",
                    "time_limit_seconds": 30,
                    "optimization_objective": "minimize_makespan"
                },
                default_analysis_config={
                    "include_critical_path": True,
                    "include_bottleneck_analysis": True,
                    "include_improvement_suggestions": True,
                    "include_gantt_chart": True
                },
                tags=[template_data["complexity_level"], template_data["category"]],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                usage_count=0,
                is_public=True,
                created_by="system"
            )
            
            templates[template_id] = template
        
        self._save_templates(templates)
    
    def _generate_sample_problem(self, machine_count: int, job_count: int, avg_operations: int) -> JobShopProblem:
        """サンプル問題データの生成"""
        import random
        
        # マシンの生成
        machines = []
        for i in range(machine_count):
            machines.append(Machine(
                id=f"M{i+1}",
                name=f"マシン{i+1}",
                capacity=1,
                available_from=0,
                available_until=480  # 8時間稼働
            ))
        
        # ジョブの生成
        jobs = []
        for i in range(job_count):
            operations = []
            num_operations = max(1, avg_operations + random.randint(-1, 1))
            
            for j in range(num_operations):
                # ランダムにマシンを選択
                machine_id = f"M{random.randint(1, machine_count)}"
                duration = random.randint(10, 60)  # 10-60分の作業時間
                setup_time = random.randint(0, 10) if random.random() > 0.5 else 0
                
                operations.append(Operation(
                    id=f"J{i+1}_O{j+1}",
                    job_id=f"J{i+1}",
                    machine_id=machine_id,
                    duration=duration,
                    position_in_job=j,
                    setup_time=setup_time
                ))
            
            due_date = random.randint(200, 400) if random.random() > 0.3 else None
            
            jobs.append(Job(
                id=f"J{i+1}",
                name=f"ジョブ{i+1}",
                priority=random.randint(1, 5),
                weight=random.uniform(0.5, 2.0),
                release_time=random.randint(0, 30),
                due_date=due_date,
                operations=operations
            ))
        
        return JobShopProblem(
            problem_type="job_shop",
            jobs=jobs,
            machines=machines,
            optimization_objective="makespan",
            time_horizon=500,
            allow_preemption=False,
            setup_times_included=True
        )
    
    def create_template(self, request: TemplateCreateRequest) -> ScheduleTemplate:
        """新しいテンプレートの作成"""
        templates = self._load_templates()
        
        template_id = str(uuid4())
        template = ScheduleTemplate(
            id=template_id,
            name=request.name,
            description=request.description,
            category=request.category,
            problem_template=request.problem_template,
            default_solver_config=request.default_solver_config,
            default_analysis_config=request.default_analysis_config,
            tags=request.tags,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            usage_count=0,
            is_public=request.is_public,
            created_by="user"
        )
        
        templates[template_id] = template
        self._save_templates(templates)
        
        return template
    
    def get_template(self, template_id: str) -> Optional[ScheduleTemplate]:
        """テンプレートの取得"""
        templates = self._load_templates()
        return templates.get(template_id)
    
    def update_template(self, template_id: str, request: TemplateUpdateRequest) -> Optional[ScheduleTemplate]:
        """テンプレートの更新"""
        templates = self._load_templates()
        
        if template_id not in templates:
            return None
        
        template = templates[template_id]
        
        # 更新可能なフィールドの更新
        if request.name is not None:
            template.name = request.name
        if request.description is not None:
            template.description = request.description
        if request.category is not None:
            template.category = request.category
        if request.problem_template is not None:
            template.problem_template = request.problem_template
        if request.default_solver_config is not None:
            template.default_solver_config = request.default_solver_config
        if request.default_analysis_config is not None:
            template.default_analysis_config = request.default_analysis_config
        if request.tags is not None:
            template.tags = request.tags
        if request.is_public is not None:
            template.is_public = request.is_public
        
        template.updated_at = datetime.now()
        
        templates[template_id] = template
        self._save_templates(templates)
        
        return template
    
    def delete_template(self, template_id: str) -> bool:
        """テンプレートの削除"""
        templates = self._load_templates()
        
        if template_id not in templates:
            return False
        
        # システムテンプレートは削除不可
        if templates[template_id].created_by == "system":
            return False
        
        del templates[template_id]
        self._save_templates(templates)
        
        # 使用統計も削除
        stats = self._load_usage_stats()
        if template_id in stats:
            del stats[template_id]
            self._save_usage_stats(stats)
        
        return True
    
    def search_templates(self, request: TemplateSearchRequest) -> TemplateListResponse:
        """テンプレートの検索"""
        templates = self._load_templates()
        filtered_templates = []
        
        for template in templates.values():
            # 公開設定でのフィルタリング
            if request.is_public is not None and template.is_public != request.is_public:
                continue
            
            # カテゴリでのフィルタリング
            if request.category and template.category != request.category:
                continue
            
            # タグでのフィルタリング
            if request.tags:
                if not any(tag in template.tags for tag in request.tags):
                    continue
            
            # クエリでのフィルタリング（名前・説明での検索）
            if request.query:
                query_lower = request.query.lower()
                if (query_lower not in template.name.lower() and 
                    query_lower not in (template.description or "").lower()):
                    continue
            
            filtered_templates.append(template)
        
        # 使用回数でソート（人気順）
        filtered_templates.sort(key=lambda t: t.usage_count, reverse=True)
        
        # ページネーション
        total_count = len(filtered_templates)
        start = request.offset
        end = start + request.limit
        paginated_templates = filtered_templates[start:end]
        
        return TemplateListResponse(
            templates=paginated_templates,
            total_count=total_count,
            page=(request.offset // request.limit) + 1,
            per_page=request.limit
        )
    
    def increment_usage(self, template_id: str, solve_time: Optional[float] = None, success: bool = True):
        """テンプレート使用回数の更新"""
        # テンプレート使用回数の更新
        templates = self._load_templates()
        if template_id in templates:
            templates[template_id].usage_count += 1
            self._save_templates(templates)
        
        # 使用統計の更新
        stats = self._load_usage_stats()
        
        if template_id not in stats:
            stats[template_id] = TemplateUsageStats(
                template_id=template_id,
                usage_count=1,
                last_used=datetime.now(),
                average_solve_time=solve_time,
                success_rate=1.0 if success else 0.0
            )
        else:
            stat = stats[template_id]
            stat.usage_count += 1
            stat.last_used = datetime.now()
            
            # 平均解決時間の更新
            if solve_time is not None:
                if stat.average_solve_time is None:
                    stat.average_solve_time = solve_time
                else:
                    stat.average_solve_time = (stat.average_solve_time + solve_time) / 2
            
            # 成功率の更新
            if success:
                stat.success_rate = (stat.success_rate * (stat.usage_count - 1) + 1.0) / stat.usage_count
            else:
                stat.success_rate = (stat.success_rate * (stat.usage_count - 1)) / stat.usage_count
        
        self._save_usage_stats(stats)
    
    def get_usage_stats(self, template_id: str) -> Optional[TemplateUsageStats]:
        """テンプレート使用統計の取得"""
        stats = self._load_usage_stats()
        return stats.get(template_id)
    
    def get_popular_templates(self, limit: int = 10) -> List[ScheduleTemplate]:
        """人気テンプレートの取得"""
        templates = self._load_templates()
        sorted_templates = sorted(
            templates.values(), 
            key=lambda t: t.usage_count, 
            reverse=True
        )
        return sorted_templates[:limit]
    
    def get_categories(self) -> List[str]:
        """利用可能なカテゴリの取得"""
        templates = self._load_templates()
        categories = set(template.category for template in templates.values())
        return sorted(list(categories))
    
    def duplicate_template(self, template_id: str, new_name: str) -> Optional[ScheduleTemplate]:
        """テンプレートの複製"""
        original = self.get_template(template_id)
        if not original:
            return None
        
        # 複製用のリクエストを作成
        request = TemplateCreateRequest(
            name=new_name,
            description=f"{original.description} (複製)" if original.description else "複製されたテンプレート",
            category=original.category,
            problem_template=original.problem_template,
            default_solver_config=original.default_solver_config,
            default_analysis_config=original.default_analysis_config,
            tags=original.tags + ["複製"],
            is_public=False  # 複製は非公開に
        )
        
        return self.create_template(request)


# サービスのシングルトンインスタンス
template_service = TemplateService()