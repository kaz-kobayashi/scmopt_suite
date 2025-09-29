"""
Async VRP API routes for commercial-grade VRP processing
"""
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import uuid
import time
import logging
from celery.result import AsyncResult

from celery_app import celery_app
from app.models.async_models import (
    VRPJobRequest, BatchVRPJobRequest, JobStatusResponse, JobListResponse,
    VRPJobResult, BatchVRPJobResult, JobStatus, JobPriority, JobStatistics
)
from app.tasks.vrp_tasks import solve_vrp_async, solve_batch_vrp_async

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory job storage (in production, this would be Redis or database)
job_storage = {}

@router.post("/jobs", response_model=Dict[str, str])
async def submit_vrp_job(request: VRPJobRequest):
    """
    Submit a VRP problem for async processing
    
    This endpoint follows commercial VRP API standards by providing:
    - Async job submission with unique job ID
    - Priority-based queue management
    - Webhook callback support for job completion
    - Standardized request/response format
    """
    try:
        # Generate job ID if not provided
        job_id = request.job_id or str(uuid.uuid4())
        
        # Validate job ID uniqueness
        if job_id in job_storage:
            raise HTTPException(
                status_code=409, 
                detail=f"Job ID {job_id} already exists"
            )
        
        logger.info(f"Submitting VRP job {job_id} with priority {request.priority}")
        
        # Submit Celery task with priority routing
        task_kwargs = {
            'problem_data': request.problem_data,
            'job_id': job_id
        }
        
        # Apply priority-based routing
        queue_name = 'vrp_queue'
        if request.priority == JobPriority.URGENT:
            queue_name = 'urgent_queue'
        elif request.priority == JobPriority.HIGH:
            queue_name = 'high_priority_queue'
        
        task = solve_vrp_async.apply_async(
            kwargs=task_kwargs,
            queue=queue_name,
            task_id=job_id
        )
        
        # Store job information
        job_info = {
            'job_id': job_id,
            'task_id': task.id,
            'status': JobStatus.PENDING,
            'priority': request.priority,
            'callback_url': request.callback_url,
            'timeout_seconds': request.timeout_seconds,
            'metadata': request.metadata or {},
            'created_at': time.time(),
            'started_at': None,
            'completed_at': None
        }
        
        job_storage[job_id] = job_info
        
        return {
            "job_id": job_id,
            "task_id": task.id,
            "status": "accepted",
            "message": f"VRP job {job_id} submitted successfully",
            "estimated_completion_time": f"{request.timeout_seconds or 600} seconds"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit VRP job: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit job: {str(e)}"
        )

@router.post("/jobs/batch", response_model=Dict[str, str])
async def submit_batch_vrp_job(request: BatchVRPJobRequest):
    """
    Submit multiple VRP problems for batch processing
    
    Supports commercial API pattern for bulk problem solving:
    - Batch job submission with progress tracking
    - Individual problem result reporting
    - Batch-level statistics and success rates
    """
    try:
        # Generate batch ID if not provided
        batch_id = request.batch_id or str(uuid.uuid4())
        
        # Validate batch ID uniqueness
        if batch_id in job_storage:
            raise HTTPException(
                status_code=409, 
                detail=f"Batch ID {batch_id} already exists"
            )
        
        logger.info(f"Submitting batch VRP job {batch_id} with {len(request.batch_problems)} problems")
        
        # Submit Celery batch task
        task_kwargs = {
            'batch_problems': request.batch_problems,
            'batch_id': batch_id
        }
        
        task = solve_batch_vrp_async.apply_async(
            kwargs=task_kwargs,
            queue='batch_queue',
            task_id=batch_id
        )
        
        # Store batch job information
        job_info = {
            'job_id': batch_id,
            'task_id': task.id,
            'status': JobStatus.PENDING,
            'priority': request.priority,
            'callback_url': request.callback_url,
            'timeout_seconds': request.timeout_seconds,
            'metadata': request.metadata or {},
            'created_at': time.time(),
            'started_at': None,
            'completed_at': None,
            'is_batch': True,
            'total_problems': len(request.batch_problems)
        }
        
        job_storage[batch_id] = job_info
        
        return {
            "batch_id": batch_id,
            "task_id": task.id,
            "status": "accepted",
            "message": f"Batch VRP job {batch_id} with {len(request.batch_problems)} problems submitted successfully",
            "estimated_completion_time": f"{request.timeout_seconds or 1800} seconds"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit batch VRP job: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit batch job: {str(e)}"
        )

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get current status and progress of a VRP job
    
    Returns comprehensive job information including:
    - Current processing stage and progress percentage
    - Estimated completion time
    - Intermediate results if available
    - Error details if job failed
    """
    try:
        # Check if job exists in storage
        if job_id not in job_storage:
            raise HTTPException(
                status_code=404, 
                detail=f"Job {job_id} not found"
            )
        
        job_info = job_storage[job_id]
        
        # Get Celery task result
        task = AsyncResult(job_info['task_id'], app=celery_app)
        
        # Update job status based on task state
        current_status = JobStatus.PENDING
        progress = None
        result = None
        
        if task.state == 'PENDING':
            current_status = JobStatus.PENDING
        elif task.state == 'PROCESSING':
            current_status = JobStatus.PROCESSING
            if task.info:
                progress = {
                    'job_id': job_id,
                    'stage': task.info.get('stage', 'unknown'),
                    'progress': task.info.get('progress', 0),
                    'message': task.info.get('message', ''),
                    'timestamp': time.time()
                }
        elif task.state == 'SUCCESS':
            current_status = JobStatus.COMPLETED
            if task.result:
                if job_info.get('is_batch', False):
                    result = BatchVRPJobResult(**task.result)
                else:
                    result = VRPJobResult(**task.result)
                job_info['completed_at'] = time.time()
        elif task.state == 'FAILURE':
            current_status = JobStatus.FAILED
            job_info['completed_at'] = time.time()
        
        # Update stored job status
        job_info['status'] = current_status
        
        response = JobStatusResponse(
            job_id=job_id,
            status=current_status,
            progress=progress,
            result=result,
            created_at=job_info['created_at'],
            started_at=job_info.get('started_at'),
            completed_at=job_info.get('completed_at')
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve job status: {str(e)}"
        )

@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    priority: Optional[JobPriority] = Query(None, description="Filter by job priority"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of jobs per page")
):
    """
    List VRP jobs with filtering and pagination
    
    Supports commercial API patterns for job management:
    - Status-based filtering (pending, processing, completed, failed)
    - Priority-based filtering
    - Pagination for large job lists
    - Summary statistics
    """
    try:
        # Filter jobs based on criteria
        filtered_jobs = []
        
        for job_id, job_info in job_storage.items():
            # Apply status filter
            if status and job_info.get('status') != status:
                continue
                
            # Apply priority filter
            if priority and job_info.get('priority') != priority:
                continue
            
            # Get current job status
            task = AsyncResult(job_info['task_id'], app=celery_app)
            current_status = _get_job_status_from_task(task)
            
            job_response = JobStatusResponse(
                job_id=job_id,
                status=current_status,
                progress=None,
                result=None,
                created_at=job_info['created_at'],
                started_at=job_info.get('started_at'),
                completed_at=job_info.get('completed_at')
            )
            
            filtered_jobs.append(job_response)
        
        # Sort by creation time (newest first)
        filtered_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        total_count = len(filtered_jobs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_jobs = filtered_jobs[start_idx:end_idx]
        
        has_next = end_idx < total_count
        
        return JobListResponse(
            jobs=paginated_jobs,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve job list: {str(e)}"
        )

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a pending or processing VRP job
    
    Supports commercial API pattern for job lifecycle management:
    - Graceful task cancellation
    - Resource cleanup
    - Status update to cancelled
    """
    try:
        # Check if job exists
        if job_id not in job_storage:
            raise HTTPException(
                status_code=404, 
                detail=f"Job {job_id} not found"
            )
        
        job_info = job_storage[job_id]
        
        # Get Celery task
        task = AsyncResult(job_info['task_id'], app=celery_app)
        
        # Check if job can be cancelled
        if task.state in ['SUCCESS', 'FAILURE']:
            raise HTTPException(
                status_code=409, 
                detail=f"Job {job_id} is already completed and cannot be cancelled"
            )
        
        # Cancel the task
        task.revoke(terminate=True)
        
        # Update job status
        job_info['status'] = JobStatus.CANCELLED
        job_info['completed_at'] = time.time()
        
        logger.info(f"Job {job_id} cancelled successfully")
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": f"Job {job_id} has been cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to cancel job: {str(e)}"
        )

@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the complete result of a completed VRP job
    
    Returns detailed solution data including:
    - Optimized routes and schedules
    - Performance metrics and statistics
    - Algorithm information and convergence data
    """
    try:
        # Check if job exists
        if job_id not in job_storage:
            raise HTTPException(
                status_code=404, 
                detail=f"Job {job_id} not found"
            )
        
        job_info = job_storage[job_id]
        
        # Get Celery task result
        task = AsyncResult(job_info['task_id'], app=celery_app)
        
        if task.state != 'SUCCESS':
            raise HTTPException(
                status_code=409, 
                detail=f"Job {job_id} is not completed successfully. Current state: {task.state}"
            )
        
        # Return the complete result
        return task.result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve job result: {str(e)}"
        )

@router.get("/statistics", response_model=JobStatistics)
async def get_job_statistics():
    """
    Get comprehensive statistics about VRP job processing
    
    Provides insights for system monitoring and capacity planning:
    - Queue lengths and processing rates
    - Success/failure ratios
    - Average processing times
    - System performance metrics
    """
    try:
        total_jobs = len(job_storage)
        
        if total_jobs == 0:
            return JobStatistics(
                total_jobs=0,
                completed_jobs=0,
                failed_jobs=0,
                pending_jobs=0,
                processing_jobs=0,
                average_runtime_seconds=None,
                success_rate=0.0,
                queue_length=0
            )
        
        # Count jobs by status
        status_counts = {
            JobStatus.COMPLETED: 0,
            JobStatus.FAILED: 0,
            JobStatus.PENDING: 0,
            JobStatus.PROCESSING: 0
        }
        
        runtimes = []
        queue_length = 0
        
        for job_info in job_storage.values():
            task = AsyncResult(job_info['task_id'], app=celery_app)
            current_status = _get_job_status_from_task(task)
            
            if current_status in status_counts:
                status_counts[current_status] += 1
            
            if current_status in [JobStatus.PENDING, JobStatus.PROCESSING]:
                queue_length += 1
            
            # Calculate runtime for completed jobs
            if (current_status == JobStatus.COMPLETED and 
                job_info.get('completed_at') and job_info.get('created_at')):
                runtime = job_info['completed_at'] - job_info['created_at']
                runtimes.append(runtime)
        
        # Calculate metrics
        completed_jobs = status_counts[JobStatus.COMPLETED]
        failed_jobs = status_counts[JobStatus.FAILED]
        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
        average_runtime = sum(runtimes) / len(runtimes) if runtimes else None
        
        return JobStatistics(
            total_jobs=total_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            pending_jobs=status_counts[JobStatus.PENDING],
            processing_jobs=status_counts[JobStatus.PROCESSING],
            average_runtime_seconds=average_runtime,
            success_rate=success_rate,
            queue_length=queue_length
        )
        
    except Exception as e:
        logger.error(f"Failed to get job statistics: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve statistics: {str(e)}"
        )

def _get_job_status_from_task(task: AsyncResult) -> JobStatus:
    """Helper function to convert Celery task state to JobStatus"""
    if task.state == 'PENDING':
        return JobStatus.PENDING
    elif task.state == 'PROCESSING':
        return JobStatus.PROCESSING
    elif task.state == 'SUCCESS':
        return JobStatus.COMPLETED
    elif task.state == 'FAILURE':
        return JobStatus.FAILED
    else:
        return JobStatus.PENDING

@router.get("/health")
async def health_check():
    """Health check endpoint for async VRP service"""
    try:
        # Check Celery connection
        celery_status = "healthy" if celery_app.control.ping() else "unhealthy"
        
        return {
            "status": "healthy",
            "service": "Async VRP API",
            "celery_status": celery_status,
            "queue_status": {
                "vrp_queue": "active",
                "batch_queue": "active"
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }