"""
Batch processing models for large-scale VRP operations
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime

class BatchStatus(str, Enum):
    """Batch processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class BatchPriority(str, Enum):
    """Batch processing priority"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class BatchType(str, Enum):
    """Types of batch operations"""
    BULK_OPTIMIZATION = "bulk_optimization"
    PARAMETER_SWEEP = "parameter_sweep"
    SCENARIO_ANALYSIS = "scenario_analysis"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    PERIODIC_REOPTIMIZATION = "periodic_reoptimization"

class BatchProblemItem(BaseModel):
    """Single problem item in a batch"""
    item_id: str = Field(description="Unique identifier for this problem item")
    problem_data: Dict[str, Any] = Field(description="VRP problem data")
    priority: BatchPriority = Field(BatchPriority.NORMAL, description="Item priority within batch")
    timeout_seconds: Optional[int] = Field(300, description="Timeout for this specific item")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")
    dependencies: List[str] = Field([], description="IDs of items that must complete first")

class ParameterSweepConfig(BaseModel):
    """Configuration for parameter sweep batch"""
    parameter_ranges: Dict[str, List[Union[int, float, str]]] = Field(description="Parameter values to test")
    base_problem: Dict[str, Any] = Field(description="Base problem configuration")
    metrics_to_track: List[str] = Field(["total_distance", "total_cost", "vehicle_count"], description="Metrics to compare")
    
class ScenarioAnalysisConfig(BaseModel):
    """Configuration for scenario analysis batch"""
    scenarios: List[Dict[str, Any]] = Field(description="List of scenario configurations")
    scenario_names: List[str] = Field(description="Human-readable scenario names")
    comparison_metrics: List[str] = Field(description="Metrics to compare across scenarios")
    
class BatchProcessingConfig(BaseModel):
    """Batch processing configuration"""
    batch_id: str = Field(description="Unique batch identifier")
    batch_type: BatchType = Field(description="Type of batch processing")
    batch_name: Optional[str] = Field(None, description="Human-readable batch name")
    description: Optional[str] = Field(None, description="Batch description")
    priority: BatchPriority = Field(BatchPriority.NORMAL, description="Overall batch priority")
    
    # Processing settings
    max_concurrent_jobs: int = Field(3, ge=1, le=10, description="Maximum concurrent processing jobs")
    retry_failed_items: bool = Field(True, description="Retry failed items")
    max_retries: int = Field(2, ge=0, le=5, description="Maximum retry attempts")
    continue_on_failure: bool = Field(True, description="Continue batch even if some items fail")
    
    # Resource limits
    total_timeout_seconds: Optional[int] = Field(None, description="Total batch timeout")
    memory_limit_mb: Optional[int] = Field(None, description="Memory limit per job")
    cpu_limit: Optional[float] = Field(None, description="CPU limit per job")
    
    # Notification settings
    progress_notification_interval: int = Field(10, description="Progress notification interval (percentage)")
    email_notifications: List[str] = Field([], description="Email addresses for notifications")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for batch completion")
    
    # Specialized configurations
    parameter_sweep: Optional[ParameterSweepConfig] = Field(None, description="Parameter sweep configuration")
    scenario_analysis: Optional[ScenarioAnalysisConfig] = Field(None, description="Scenario analysis configuration")

class BatchJobRequest(BaseModel):
    """Request for batch job processing"""
    config: BatchProcessingConfig = Field(description="Batch processing configuration")
    items: List[BatchProblemItem] = Field(description="Items to process in batch")
    
    @validator('items')
    def validate_items(cls, v, values):
        if len(v) == 0:
            raise ValueError("Batch must contain at least one item")
        
        # Check for duplicate item IDs
        item_ids = [item.item_id for item in v]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Duplicate item IDs found in batch")
        
        # Validate dependencies
        for item in v:
            for dep_id in item.dependencies:
                if dep_id not in item_ids:
                    raise ValueError(f"Dependency {dep_id} not found in batch items")
        
        return v

class BatchItemResult(BaseModel):
    """Result of processing a single batch item"""
    item_id: str = Field(description="Item identifier")
    status: BatchStatus = Field(description="Item processing status")
    solution: Optional[Dict[str, Any]] = Field(None, description="VRP solution if successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time_seconds: float = Field(description="Time taken to process this item")
    retry_count: int = Field(0, description="Number of retries attempted")
    metadata: Dict[str, Any] = Field({}, description="Additional result metadata")
    
    # Performance metrics
    memory_used_mb: Optional[float] = Field(None, description="Peak memory usage")
    cpu_time_seconds: Optional[float] = Field(None, description="CPU time used")
    
class BatchExecutionSummary(BaseModel):
    """Summary of batch execution"""
    total_items: int = Field(description="Total number of items in batch")
    completed_items: int = Field(description="Number of successfully completed items")
    failed_items: int = Field(description="Number of failed items")
    cancelled_items: int = Field(description="Number of cancelled items")
    
    # Timing
    start_time: datetime = Field(description="Batch start time")
    end_time: Optional[datetime] = Field(None, description="Batch end time")
    total_duration_seconds: Optional[float] = Field(None, description="Total processing time")
    average_item_duration_seconds: Optional[float] = Field(None, description="Average time per item")
    
    # Resource usage
    peak_memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage across all jobs")
    total_cpu_time_seconds: Optional[float] = Field(None, description="Total CPU time used")
    
    # Performance statistics
    success_rate: float = Field(description="Success rate percentage")
    throughput_items_per_hour: Optional[float] = Field(None, description="Processing throughput")

class BatchResult(BaseModel):
    """Complete batch processing result"""
    batch_id: str = Field(description="Batch identifier")
    status: BatchStatus = Field(description="Overall batch status")
    summary: BatchExecutionSummary = Field(description="Execution summary")
    item_results: List[BatchItemResult] = Field(description="Individual item results")
    
    # Analysis results
    performance_analysis: Optional[Dict[str, Any]] = Field(None, description="Performance analysis results")
    comparison_results: Optional[Dict[str, Any]] = Field(None, description="Comparison results if applicable")
    recommendations: List[str] = Field([], description="Optimization recommendations")
    
    # Export information
    export_urls: Dict[str, str] = Field({}, description="URLs for downloadable results")
    report_url: Optional[str] = Field(None, description="URL for detailed report")

class BatchProgress(BaseModel):
    """Batch processing progress information"""
    batch_id: str = Field(description="Batch identifier")
    status: BatchStatus = Field(description="Current batch status")
    progress_percentage: float = Field(ge=0, le=100, description="Progress percentage")
    
    # Item counts
    total_items: int = Field(description="Total items in batch")
    queued_items: int = Field(description="Items waiting to be processed")
    processing_items: int = Field(description="Currently processing items")
    completed_items: int = Field(description="Successfully completed items")
    failed_items: int = Field(description="Failed items")
    
    # Timing estimates
    elapsed_time_seconds: float = Field(description="Time elapsed since batch start")
    estimated_remaining_seconds: Optional[float] = Field(None, description="Estimated time remaining")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Current activity
    current_activities: List[str] = Field([], description="Current processing activities")
    last_completed_item: Optional[str] = Field(None, description="ID of last completed item")
    
    # Resource utilization
    cpu_utilization_percent: Optional[float] = Field(None, description="Current CPU utilization")
    memory_utilization_percent: Optional[float] = Field(None, description="Current memory utilization")

class BatchListFilter(BaseModel):
    """Filters for listing batches"""
    status: Optional[BatchStatus] = Field(None, description="Filter by batch status")
    batch_type: Optional[BatchType] = Field(None, description="Filter by batch type")
    priority: Optional[BatchPriority] = Field(None, description="Filter by priority")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    name_contains: Optional[str] = Field(None, description="Filter by name substring")

class BatchMetrics(BaseModel):
    """Batch processing system metrics"""
    # Queue statistics
    total_queued_batches: int = Field(description="Total batches in queue")
    total_processing_batches: int = Field(description="Batches currently processing")
    average_queue_wait_time_seconds: float = Field(description="Average queue wait time")
    
    # Processing statistics
    batches_completed_today: int = Field(description="Batches completed today")
    items_processed_today: int = Field(description="Items processed today")
    average_batch_duration_seconds: float = Field(description="Average batch processing time")
    system_throughput_items_per_hour: float = Field(description="System throughput")
    
    # Resource utilization
    current_cpu_utilization: float = Field(description="Current system CPU utilization")
    current_memory_utilization: float = Field(description="Current system memory utilization")
    active_worker_count: int = Field(description="Number of active worker processes")
    
    # Quality metrics
    overall_success_rate: float = Field(description="Overall batch success rate")
    average_solution_quality_score: Optional[float] = Field(None, description="Average solution quality")

class BatchTemplate(BaseModel):
    """Template for common batch operations"""
    template_id: str = Field(description="Template identifier")
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    batch_type: BatchType = Field(description="Batch operation type")
    default_config: BatchProcessingConfig = Field(description="Default configuration")
    parameter_schema: Dict[str, Any] = Field(description="Schema for template parameters")
    example_usage: Dict[str, Any] = Field(description="Example usage of template")