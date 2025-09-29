"""
Pydantic models for async VRP processing
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime

class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(str, Enum):
    """Job priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class VRPJobRequest(BaseModel):
    """Request model for async VRP job submission"""
    problem_data: Dict[str, Any] = Field(description="VRP problem data in unified format")
    job_id: Optional[str] = Field(None, description="Optional custom job ID")
    priority: JobPriority = Field(JobPriority.NORMAL, description="Job priority level")
    callback_url: Optional[str] = Field(None, description="Optional webhook URL for job completion notification")
    timeout_seconds: Optional[int] = Field(600, description="Maximum processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the job")

class BatchVRPJobRequest(BaseModel):
    """Request model for batch VRP job submission"""
    batch_problems: List[Dict[str, Any]] = Field(description="List of VRP problem data")
    batch_id: Optional[str] = Field(None, description="Optional custom batch ID")
    priority: JobPriority = Field(JobPriority.NORMAL, description="Batch priority level")
    callback_url: Optional[str] = Field(None, description="Optional webhook URL for batch completion notification")
    timeout_seconds: Optional[int] = Field(1800, description="Maximum processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the batch")

class JobProgress(BaseModel):
    """Job progress information"""
    job_id: str = Field(description="Job identifier")
    stage: str = Field(description="Current processing stage")
    progress: int = Field(description="Progress percentage (0-100)")
    message: str = Field(description="Current status message")
    timestamp: float = Field(description="Last update timestamp")

class VRPJobResult(BaseModel):
    """Result model for completed VRP jobs"""
    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(description="Final job status")
    solution: Optional[Dict[str, Any]] = Field(None, description="VRP solution data")
    runtime_seconds: Optional[float] = Field(None, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    timestamp: float = Field(description="Completion timestamp")
    task_id: Optional[str] = Field(None, description="Celery task ID")

class BatchVRPJobResult(BaseModel):
    """Result model for completed batch VRP jobs"""
    batch_id: str = Field(description="Batch identifier")
    status: JobStatus = Field(description="Final batch status")
    total_problems: int = Field(description="Total number of problems in batch")
    successful_problems: int = Field(description="Number of successfully solved problems")
    failed_problems: int = Field(description="Number of failed problems")
    results: List[Dict[str, Any]] = Field(description="Individual problem results")
    runtime_seconds: Optional[float] = Field(None, description="Total processing time in seconds")
    timestamp: float = Field(description="Completion timestamp")
    task_id: Optional[str] = Field(None, description="Celery task ID")

class JobStatusResponse(BaseModel):
    """Response model for job status queries"""
    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(description="Current job status")
    progress: Optional[JobProgress] = Field(None, description="Progress information if job is processing")
    result: Optional[Union[VRPJobResult, BatchVRPJobResult]] = Field(None, description="Job result if completed")
    created_at: float = Field(description="Job creation timestamp")
    started_at: Optional[float] = Field(None, description="Job start timestamp")
    completed_at: Optional[float] = Field(None, description="Job completion timestamp")

class JobListResponse(BaseModel):
    """Response model for listing jobs"""
    jobs: List[JobStatusResponse] = Field(description="List of jobs")
    total_count: int = Field(description="Total number of jobs")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of jobs per page")
    has_next: bool = Field(description="Whether there are more pages")

class WebSocketMessage(BaseModel):
    """WebSocket message model for real-time updates"""
    message_type: str = Field(description="Type of message (job_update, job_completed, etc.)")
    job_id: str = Field(description="Job identifier")
    data: Dict[str, Any] = Field(description="Message payload")
    timestamp: float = Field(description="Message timestamp")

class JobStatistics(BaseModel):
    """Statistics about job processing"""
    total_jobs: int = Field(description="Total number of jobs processed")
    completed_jobs: int = Field(description="Number of completed jobs")
    failed_jobs: int = Field(description="Number of failed jobs")
    pending_jobs: int = Field(description="Number of pending jobs")
    processing_jobs: int = Field(description="Number of currently processing jobs")
    average_runtime_seconds: Optional[float] = Field(None, description="Average processing time")
    success_rate: float = Field(description="Success rate percentage")
    queue_length: int = Field(description="Current queue length")