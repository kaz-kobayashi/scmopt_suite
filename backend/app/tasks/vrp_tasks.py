"""
Celery tasks for async VRP processing
"""
from celery import current_task
from celery_app import celery_app
import logging
import time
import traceback
from typing import Dict, Any, List
import json

from app.services.pyvrp_unified_service import PyVRPUnifiedService
from app.models.vrp_unified_models import VRPProblemData, UnifiedVRPSolution
from app.models.async_models import JobStatus, VRPJobResult

logger = logging.getLogger(__name__)

# Initialize PyVRP service
pyvrp_service = PyVRPUnifiedService()

@celery_app.task(bind=True, name='app.tasks.vrp_tasks.solve_vrp_async')
def solve_vrp_async(self, problem_data: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Async task for solving VRP problems
    
    Args:
        problem_data: VRP problem data in dictionary format
        job_id: Unique job identifier
        
    Returns:
        Dictionary containing solution result
    """
    try:
        # Update task status
        current_task.update_state(
            state='PROCESSING',
            meta={
                'job_id': job_id,
                'stage': 'initializing',
                'progress': 0,
                'message': 'Starting VRP optimization...'
            }
        )
        
        logger.info(f"Starting async VRP solve for job {job_id}")
        start_time = time.time()
        
        # Parse problem data
        current_task.update_state(
            state='PROCESSING',
            meta={
                'job_id': job_id,
                'stage': 'parsing',
                'progress': 10,
                'message': 'Parsing problem data...'
            }
        )
        
        # Convert dict to Pydantic model
        vrp_problem = VRPProblemData(**problem_data)
        
        # Update progress
        current_task.update_state(
            state='PROCESSING',
            meta={
                'job_id': job_id,
                'stage': 'solving',
                'progress': 20,
                'message': 'Running VRP optimization algorithm...'
            }
        )
        
        # Solve VRP
        solution = pyvrp_service.solve(vrp_problem)
        
        # Update progress
        current_task.update_state(
            state='PROCESSING',
            meta={
                'job_id': job_id,
                'stage': 'finalizing',
                'progress': 90,
                'message': 'Finalizing solution...'
            }
        )
        
        runtime = time.time() - start_time
        
        # Prepare result
        result = {
            'job_id': job_id,
            'status': 'completed',
            'solution': solution.dict() if solution else None,
            'runtime_seconds': runtime,
            'timestamp': time.time(),
            'task_id': self.request.id
        }
        
        # Final success state
        current_task.update_state(
            state='SUCCESS',
            meta={
                'job_id': job_id,
                'stage': 'completed',
                'progress': 100,
                'message': 'VRP optimization completed successfully',
                'result': result
            }
        )
        
        logger.info(f"VRP solve completed for job {job_id} in {runtime:.2f}s")
        return result
        
    except Exception as e:
        error_msg = f"VRP solve failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Update task state to FAILURE
        current_task.update_state(
            state='FAILURE',
            meta={
                'job_id': job_id,
                'stage': 'failed',
                'progress': 0,
                'message': error_msg,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        # Re-raise the exception to mark task as failed
        raise Exception(error_msg)

@celery_app.task(bind=True, name='app.tasks.vrp_tasks.solve_batch_vrp_async')
def solve_batch_vrp_async(self, batch_problems: List[Dict[str, Any]], batch_id: str) -> Dict[str, Any]:
    """
    Async task for solving multiple VRP problems in batch
    
    Args:
        batch_problems: List of VRP problem data dictionaries
        batch_id: Unique batch identifier
        
    Returns:
        Dictionary containing batch results
    """
    try:
        current_task.update_state(
            state='PROCESSING',
            meta={
                'batch_id': batch_id,
                'stage': 'initializing',
                'progress': 0,
                'message': f'Starting batch processing of {len(batch_problems)} problems...',
                'total_problems': len(batch_problems),
                'completed_problems': 0
            }
        )
        
        logger.info(f"Starting batch VRP solve for batch {batch_id} with {len(batch_problems)} problems")
        start_time = time.time()
        
        results = []
        total_problems = len(batch_problems)
        
        for i, problem_data in enumerate(batch_problems):
            try:
                # Update progress
                progress = int((i / total_problems) * 90)  # Reserve 10% for finalization
                current_task.update_state(
                    state='PROCESSING',
                    meta={
                        'batch_id': batch_id,
                        'stage': 'solving',
                        'progress': progress,
                        'message': f'Solving problem {i+1} of {total_problems}...',
                        'total_problems': total_problems,
                        'completed_problems': i
                    }
                )
                
                # Parse and solve individual problem
                vrp_problem = VRPProblemData(**problem_data)
                solution = pyvrp_service.solve(vrp_problem)
                
                problem_result = {
                    'problem_index': i,
                    'problem_id': problem_data.get('problem_id', f'problem_{i}'),
                    'status': 'completed',
                    'solution': solution.dict() if solution else None,
                    'timestamp': time.time()
                }
                
                results.append(problem_result)
                
            except Exception as e:
                # Handle individual problem failure
                logger.error(f"Problem {i} failed in batch {batch_id}: {str(e)}")
                
                problem_result = {
                    'problem_index': i,
                    'problem_id': problem_data.get('problem_id', f'problem_{i}'),
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': time.time()
                }
                
                results.append(problem_result)
        
        # Finalization
        current_task.update_state(
            state='PROCESSING',
            meta={
                'batch_id': batch_id,
                'stage': 'finalizing',
                'progress': 95,
                'message': 'Finalizing batch results...',
                'total_problems': total_problems,
                'completed_problems': total_problems
            }
        )
        
        runtime = time.time() - start_time
        successful_problems = sum(1 for r in results if r['status'] == 'completed')
        failed_problems = total_problems - successful_problems
        
        # Prepare batch result
        batch_result = {
            'batch_id': batch_id,
            'status': 'completed',
            'total_problems': total_problems,
            'successful_problems': successful_problems,
            'failed_problems': failed_problems,
            'results': results,
            'runtime_seconds': runtime,
            'timestamp': time.time(),
            'task_id': self.request.id
        }
        
        # Final success state
        current_task.update_state(
            state='SUCCESS',
            meta={
                'batch_id': batch_id,
                'stage': 'completed',
                'progress': 100,
                'message': f'Batch processing completed. {successful_problems}/{total_problems} problems solved successfully',
                'result': batch_result
            }
        )
        
        logger.info(f"Batch VRP solve completed for batch {batch_id}: {successful_problems}/{total_problems} successful in {runtime:.2f}s")
        return batch_result
        
    except Exception as e:
        error_msg = f"Batch VRP solve failed for batch {batch_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Update task state to FAILURE
        current_task.update_state(
            state='FAILURE',
            meta={
                'batch_id': batch_id,
                'stage': 'failed',
                'progress': 0,
                'message': error_msg,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        # Re-raise the exception to mark task as failed
        raise Exception(error_msg)

@celery_app.task(name='app.tasks.vrp_tasks.cleanup_expired_jobs')
def cleanup_expired_jobs():
    """
    Periodic task to clean up expired job results
    """
    try:
        # This would typically interact with Redis or database to clean up old jobs
        # For now, just log the cleanup attempt
        logger.info("Running periodic cleanup of expired VRP jobs")
        return {"status": "completed", "message": "Cleanup task executed"}
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {str(e)}")
        raise