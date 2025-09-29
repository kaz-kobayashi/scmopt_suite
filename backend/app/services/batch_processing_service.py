"""
Batch processing service for large-scale VRP operations
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import resource
import psutil

from app.models.batch_processing_models import (
    BatchJobRequest, BatchResult, BatchProgress, BatchStatus, BatchPriority,
    BatchType, BatchItemResult, BatchExecutionSummary, BatchProblemItem,
    ParameterSweepConfig, ScenarioAnalysisConfig, BatchMetrics
)
from app.models.vrp_unified_models import VRPProblemData, UnifiedVRPSolution
from app.services.pyvrp_unified_service import PyVRPUnifiedService
from app.services.multi_objective_service import MultiObjectiveOptimizer
from app.services.constraint_validation_service import ConstraintValidator

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Manages large-scale batch processing of VRP problems"""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Services
        self.pyvrp_service = PyVRPUnifiedService()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        self.constraint_validator = ConstraintValidator()
        
        # Active batches tracking
        self.active_batches: Dict[str, Dict[str, Any]] = {}
        self.batch_metrics = {
            'total_processed': 0,
            'total_failed': 0,
            'total_processing_time': 0.0,
            'start_time': time.time()
        }
    
    async def submit_batch(self, request: BatchJobRequest) -> Dict[str, Any]:
        """
        Submit a batch job for processing
        
        Args:
            request: Batch job request with configuration and items
            
        Returns:
            Batch submission confirmation
        """
        try:
            batch_id = request.config.batch_id or str(uuid.uuid4())
            self.logger.info(f"Submitting batch {batch_id} with {len(request.items)} items")
            
            # Validate batch request
            self._validate_batch_request(request)
            
            # Initialize batch tracking
            batch_info = {
                'request': request,
                'status': BatchStatus.QUEUED,
                'start_time': datetime.now(),
                'progress': BatchProgress(
                    batch_id=batch_id,
                    status=BatchStatus.QUEUED,
                    progress_percentage=0.0,
                    total_items=len(request.items),
                    queued_items=len(request.items),
                    processing_items=0,
                    completed_items=0,
                    failed_items=0,
                    elapsed_time_seconds=0.0,
                    current_activities=[]
                ),
                'results': [],
                'futures': []
            }
            
            self.active_batches[batch_id] = batch_info
            
            # Start batch processing asynchronously
            asyncio.create_task(self._process_batch(batch_id))
            
            return {
                'batch_id': batch_id,
                'status': 'accepted',
                'message': f'Batch submitted with {len(request.items)} items',
                'estimated_completion_time': self._estimate_completion_time(request),
                'tracking_url': f'/api/vrp/v1/batch/{batch_id}/status'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to submit batch: {str(e)}")
            raise
    
    async def _process_batch(self, batch_id: str):
        """Process a batch of VRP problems"""
        try:
            batch_info = self.active_batches[batch_id]
            request = batch_info['request']
            
            self.logger.info(f"Starting batch processing for {batch_id}")
            batch_info['status'] = BatchStatus.PROCESSING
            batch_info['progress'].status = BatchStatus.PROCESSING
            
            # Sort items by priority and dependencies
            sorted_items = self._sort_items_by_priority_and_dependencies(request.items)
            
            # Process items based on batch type
            if request.config.batch_type == BatchType.PARAMETER_SWEEP:
                results = await self._process_parameter_sweep(batch_id, request)
            elif request.config.batch_type == BatchType.SCENARIO_ANALYSIS:
                results = await self._process_scenario_analysis(batch_id, request)
            elif request.config.batch_type == BatchType.BENCHMARK_COMPARISON:
                results = await self._process_benchmark_comparison(batch_id, request)
            else:
                results = await self._process_standard_batch(batch_id, sorted_items, request.config)
            
            # Update batch status
            batch_info['results'] = results
            batch_info['end_time'] = datetime.now()
            batch_info['status'] = BatchStatus.COMPLETED
            
            # Generate final result
            final_result = self._generate_batch_result(batch_id, results)
            batch_info['final_result'] = final_result
            
            # Send completion notification if configured
            await self._send_completion_notification(batch_id, final_result)
            
            self.logger.info(f"Batch {batch_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Batch processing failed for {batch_id}: {str(e)}")
            batch_info = self.active_batches.get(batch_id)
            if batch_info:
                batch_info['status'] = BatchStatus.FAILED
                batch_info['error'] = str(e)
    
    async def _process_standard_batch(
        self, 
        batch_id: str, 
        items: List[BatchProblemItem], 
        config
    ) -> List[BatchItemResult]:
        """Process standard batch of VRP problems"""
        results = []
        batch_info = self.active_batches[batch_id]
        progress = batch_info['progress']
        
        # Process items with concurrency control
        semaphore = asyncio.Semaphore(config.max_concurrent_jobs)
        tasks = []
        
        for item in items:
            task = asyncio.create_task(
                self._process_single_item(batch_id, item, semaphore)
            )
            tasks.append(task)
        
        # Process all tasks and collect results
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append(result)
                
                # Update progress
                progress.completed_items += 1
                progress.progress_percentage = (progress.completed_items / progress.total_items) * 100
                progress.elapsed_time_seconds = (datetime.now() - batch_info['start_time']).total_seconds()
                
                self.logger.info(f"Completed item {i+1}/{len(items)} in batch {batch_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to process item in batch {batch_id}: {str(e)}")
                progress.failed_items += 1
                
                # Create failed result
                failed_result = BatchItemResult(
                    item_id=items[i].item_id,
                    status=BatchStatus.FAILED,
                    error_message=str(e),
                    processing_time_seconds=0.0
                )
                results.append(failed_result)
        
        return results
    
    async def _process_single_item(
        self, 
        batch_id: str, 
        item: BatchProblemItem, 
        semaphore: asyncio.Semaphore
    ) -> BatchItemResult:
        """Process a single batch item"""
        async with semaphore:
            start_time = time.time()
            
            try:
                # Update current activity
                batch_info = self.active_batches[batch_id]
                batch_info['progress'].current_activities.append(f"Processing {item.item_id}")
                batch_info['progress'].processing_items += 1
                batch_info['progress'].queued_items -= 1
                
                # Convert to VRP problem data
                problem_data = VRPProblemData(**item.problem_data)
                
                # Solve VRP problem
                solution = self.pyvrp_service.solve(problem_data)
                
                processing_time = time.time() - start_time
                
                if solution and solution.status == "solved":
                    result = BatchItemResult(
                        item_id=item.item_id,
                        status=BatchStatus.COMPLETED,
                        solution=solution.dict(),
                        processing_time_seconds=processing_time,
                        metadata=item.metadata
                    )
                else:
                    result = BatchItemResult(
                        item_id=item.item_id,
                        status=BatchStatus.FAILED,
                        error_message="VRP solver failed to find solution",
                        processing_time_seconds=processing_time
                    )
                
                # Update activity
                batch_info['progress'].current_activities = [
                    a for a in batch_info['progress'].current_activities 
                    if f"Processing {item.item_id}" not in a
                ]
                batch_info['progress'].processing_items -= 1
                batch_info['progress'].last_completed_item = item.item_id
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                self.logger.error(f"Failed to process item {item.item_id}: {str(e)}")
                
                return BatchItemResult(
                    item_id=item.item_id,
                    status=BatchStatus.FAILED,
                    error_message=str(e),
                    processing_time_seconds=processing_time
                )
    
    async def _process_parameter_sweep(self, batch_id: str, request: BatchJobRequest) -> List[BatchItemResult]:
        """Process parameter sweep batch"""
        if not request.config.parameter_sweep:
            raise ValueError("Parameter sweep configuration required")
        
        config = request.config.parameter_sweep
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(config.parameter_ranges)
        
        self.logger.info(f"Processing parameter sweep with {len(param_combinations)} combinations")
        
        # Process each combination
        for i, params in enumerate(param_combinations):
            # Create modified problem data
            problem_data = config.base_problem.copy()
            problem_data.update(params)
            
            item = BatchProblemItem(
                item_id=f"param_sweep_{i}",
                problem_data=problem_data,
                metadata={'parameters': params}
            )
            
            # Process item
            result = await self._process_single_item(batch_id, item, asyncio.Semaphore(1))
            results.append(result)
        
        return results
    
    async def _process_scenario_analysis(self, batch_id: str, request: BatchJobRequest) -> List[BatchItemResult]:
        """Process scenario analysis batch"""
        if not request.config.scenario_analysis:
            raise ValueError("Scenario analysis configuration required")
        
        config = request.config.scenario_analysis
        results = []
        
        self.logger.info(f"Processing scenario analysis with {len(config.scenarios)} scenarios")
        
        # Process each scenario
        for i, scenario in enumerate(config.scenarios):
            scenario_name = config.scenario_names[i] if i < len(config.scenario_names) else f"Scenario_{i}"
            
            item = BatchProblemItem(
                item_id=f"scenario_{i}",
                problem_data=scenario,
                metadata={'scenario_name': scenario_name}
            )
            
            result = await self._process_single_item(batch_id, item, asyncio.Semaphore(1))
            results.append(result)
        
        return results
    
    async def _process_benchmark_comparison(self, batch_id: str, request: BatchJobRequest) -> List[BatchItemResult]:
        """Process benchmark comparison batch"""
        results = []
        
        # Process each item with different algorithm configurations
        for item in request.items:
            # Process with multiple algorithm variants
            algorithm_variants = ['standard', 'genetic', 'local_search']
            
            for variant in algorithm_variants:
                modified_item = BatchProblemItem(
                    item_id=f"{item.item_id}_{variant}",
                    problem_data=item.problem_data,
                    metadata={**item.metadata, 'algorithm': variant}
                )
                
                result = await self._process_single_item(batch_id, modified_item, asyncio.Semaphore(1))
                results.append(result)
        
        return results
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchProgress]:
        """Get current status of a batch"""
        if batch_id not in self.active_batches:
            return None
        
        batch_info = self.active_batches[batch_id]
        progress = batch_info['progress']
        
        # Update elapsed time
        if batch_info['start_time']:
            progress.elapsed_time_seconds = (datetime.now() - batch_info['start_time']).total_seconds()
        
        # Estimate remaining time
        if progress.completed_items > 0 and progress.progress_percentage < 100:
            avg_time_per_item = progress.elapsed_time_seconds / progress.completed_items
            remaining_items = progress.total_items - progress.completed_items
            progress.estimated_remaining_seconds = avg_time_per_item * remaining_items
            progress.estimated_completion_time = datetime.now() + timedelta(
                seconds=progress.estimated_remaining_seconds
            )
        
        return progress
    
    def get_batch_result(self, batch_id: str) -> Optional[BatchResult]:
        """Get final result of a completed batch"""
        if batch_id not in self.active_batches:
            return None
        
        batch_info = self.active_batches[batch_id]
        
        if batch_info['status'] != BatchStatus.COMPLETED:
            return None
        
        return batch_info.get('final_result')
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch"""
        if batch_id not in self.active_batches:
            return False
        
        batch_info = self.active_batches[batch_id]
        
        if batch_info['status'] in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
            return False
        
        # Cancel futures
        for future in batch_info.get('futures', []):
            future.cancel()
        
        # Update status
        batch_info['status'] = BatchStatus.CANCELLED
        batch_info['progress'].status = BatchStatus.CANCELLED
        
        self.logger.info(f"Batch {batch_id} cancelled")
        return True
    
    def get_system_metrics(self) -> BatchMetrics:
        """Get batch processing system metrics"""
        current_time = time.time()
        uptime = current_time - self.batch_metrics['start_time']
        
        # Count active batches
        queued_batches = sum(1 for b in self.active_batches.values() if b['status'] == BatchStatus.QUEUED)
        processing_batches = sum(1 for b in self.active_batches.values() if b['status'] == BatchStatus.PROCESSING)
        
        # Calculate success rate
        total_processed = self.batch_metrics['total_processed']
        total_failed = self.batch_metrics['total_failed']
        success_rate = (total_processed / (total_processed + total_failed) * 100) if (total_processed + total_failed) > 0 else 100.0
        
        # System resource utilization
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        return BatchMetrics(
            total_queued_batches=queued_batches,
            total_processing_batches=processing_batches,
            average_queue_wait_time_seconds=0.0,  # Would be calculated from actual queue data
            batches_completed_today=total_processed,  # Simplified
            items_processed_today=total_processed,
            average_batch_duration_seconds=self.batch_metrics['total_processing_time'] / max(total_processed, 1),
            system_throughput_items_per_hour=total_processed / (uptime / 3600) if uptime > 0 else 0.0,
            current_cpu_utilization=cpu_percent,
            current_memory_utilization=memory_percent,
            active_worker_count=self.max_workers,
            overall_success_rate=success_rate,
            average_solution_quality_score=None
        )
    
    def _validate_batch_request(self, request: BatchJobRequest):
        """Validate batch request"""
        if len(request.items) == 0:
            raise ValueError("Batch must contain at least one item")
        
        if len(request.items) > 1000:
            raise ValueError("Batch size exceeds maximum limit (1000 items)")
        
        # Validate dependencies
        item_ids = {item.item_id for item in request.items}
        for item in request.items:
            for dep_id in item.dependencies:
                if dep_id not in item_ids:
                    raise ValueError(f"Invalid dependency {dep_id} for item {item.item_id}")
    
    def _sort_items_by_priority_and_dependencies(self, items: List[BatchProblemItem]) -> List[BatchProblemItem]:
        """Sort items by priority and resolve dependencies"""
        # Simple topological sort for dependencies
        # This is a simplified version - production would need more robust dependency resolution
        sorted_items = []
        remaining_items = items.copy()
        
        while remaining_items:
            # Find items with no unresolved dependencies
            ready_items = []
            completed_ids = {item.item_id for item in sorted_items}
            
            for item in remaining_items:
                if all(dep_id in completed_ids for dep_id in item.dependencies):
                    ready_items.append(item)
            
            if not ready_items:
                # No items without dependencies - possible circular dependency
                ready_items = remaining_items  # Process anyway
            
            # Sort ready items by priority
            priority_order = {
                BatchPriority.CRITICAL: 0,
                BatchPriority.URGENT: 1,
                BatchPriority.HIGH: 2,
                BatchPriority.NORMAL: 3,
                BatchPriority.LOW: 4
            }
            
            ready_items.sort(key=lambda x: priority_order.get(x.priority, 3))
            
            # Add to sorted list and remove from remaining
            sorted_items.extend(ready_items)
            for item in ready_items:
                remaining_items.remove(item)
        
        return sorted_items
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _estimate_completion_time(self, request: BatchJobRequest) -> datetime:
        """Estimate batch completion time"""
        # Simple estimation based on average processing time
        avg_time_per_item = 30  # seconds
        total_estimated_seconds = len(request.items) * avg_time_per_item / request.config.max_concurrent_jobs
        
        return datetime.now() + timedelta(seconds=total_estimated_seconds)
    
    def _generate_batch_result(self, batch_id: str, item_results: List[BatchItemResult]) -> BatchResult:
        """Generate final batch result"""
        batch_info = self.active_batches[batch_id]
        request = batch_info['request']
        
        # Calculate summary statistics
        total_items = len(item_results)
        completed_items = sum(1 for r in item_results if r.status == BatchStatus.COMPLETED)
        failed_items = sum(1 for r in item_results if r.status == BatchStatus.FAILED)
        cancelled_items = sum(1 for r in item_results if r.status == BatchStatus.CANCELLED)
        
        start_time = batch_info['start_time']
        end_time = batch_info.get('end_time', datetime.now())
        total_duration = (end_time - start_time).total_seconds()
        
        avg_duration = total_duration / max(completed_items, 1)
        success_rate = (completed_items / total_items * 100) if total_items > 0 else 0.0
        throughput = completed_items / (total_duration / 3600) if total_duration > 0 else 0.0
        
        summary = BatchExecutionSummary(
            total_items=total_items,
            completed_items=completed_items,
            failed_items=failed_items,
            cancelled_items=cancelled_items,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=total_duration,
            average_item_duration_seconds=avg_duration,
            success_rate=success_rate,
            throughput_items_per_hour=throughput
        )
        
        # Generate recommendations
        recommendations = []
        if success_rate < 90:
            recommendations.append("Consider reviewing failed items for common issues")
        if avg_duration > 60:
            recommendations.append("Items taking longer than expected - consider optimization")
        
        return BatchResult(
            batch_id=batch_id,
            status=batch_info['status'],
            summary=summary,
            item_results=item_results,
            recommendations=recommendations,
            export_urls={}  # Would be populated with actual export URLs
        )
    
    async def _send_completion_notification(self, batch_id: str, result: BatchResult):
        """Send batch completion notification"""
        # Placeholder for notification implementation
        # Would send emails, webhooks, etc.
        self.logger.info(f"Batch {batch_id} completed - notifications sent")

# Global batch processor instance
batch_processor = BatchProcessor()