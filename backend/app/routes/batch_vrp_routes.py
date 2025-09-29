"""
Batch VRP processing routes for large-scale operations
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.models.batch_processing_models import (
    BatchJobRequest, BatchResult, BatchProgress, BatchStatus, BatchType,
    BatchPriority, BatchListFilter, BatchMetrics, BatchTemplate
)
from app.services.batch_processing_service import batch_processor

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/batch", response_model=Dict[str, str])
async def submit_batch_job(request: BatchJobRequest, background_tasks: BackgroundTasks):
    """
    Submit a batch job for processing
    
    This endpoint provides large-scale batch processing capabilities for VRP:
    - Bulk optimization of multiple VRP problems
    - Parameter sweep analysis across different configurations
    - Scenario analysis for strategic planning
    - Benchmark comparisons between algorithms
    - Resource-controlled concurrent processing
    
    Features commercial-grade batch processing:
    - Priority-based job scheduling
    - Dependency management between batch items
    - Progress tracking and real-time monitoring
    - Automatic retry of failed items
    - Resource limits and timeout controls
    """
    try:
        logger.info(f"Submitting batch job with {len(request.items)} items")
        
        # Validate request
        if len(request.items) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size exceeds maximum limit of 1000 items"
            )
        
        # Submit batch for processing
        result = await batch_processor.submit_batch(request)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit batch job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit batch job: {str(e)}"
        )

@router.get("/batch/{batch_id}/status", response_model=BatchProgress)
async def get_batch_status(batch_id: str):
    """
    Get current status and progress of a batch job
    
    Returns comprehensive progress information including:
    - Overall batch status and completion percentage
    - Item-level progress (queued, processing, completed, failed)
    - Time estimates and completion predictions
    - Current processing activities
    - Resource utilization metrics
    
    Essential for monitoring long-running batch operations
    and providing users with accurate progress updates.
    """
    try:
        progress = batch_processor.get_batch_status(batch_id)
        
        if not progress:
            raise HTTPException(
                status_code=404,
                detail=f"Batch {batch_id} not found"
            )
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch status for {batch_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve batch status: {str(e)}"
        )

@router.get("/batch/{batch_id}/result", response_model=BatchResult)
async def get_batch_result(batch_id: str):
    """
    Get complete results of a finished batch job
    
    Returns detailed batch execution results including:
    - Individual item results with solutions or error details
    - Execution summary with timing and success metrics
    - Performance analysis across all processed items
    - Comparison results for parameter sweeps or scenarios
    - Downloadable export URLs for results
    - Optimization recommendations based on analysis
    
    Only available for completed batch jobs.
    """
    try:
        result = batch_processor.get_batch_result(batch_id)
        
        if not result:
            # Check if batch exists but isn't completed
            progress = batch_processor.get_batch_status(batch_id)
            if progress:
                if progress.status in [BatchStatus.QUEUED, BatchStatus.PROCESSING]:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Batch {batch_id} is still processing. Current status: {progress.status}"
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Batch {batch_id} failed or was cancelled"
                    )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Batch {batch_id} not found"
                )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch result for {batch_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve batch result: {str(e)}"
        )

@router.delete("/batch/{batch_id}")
async def cancel_batch(batch_id: str):
    """
    Cancel a running batch job
    
    Gracefully cancels a batch job that is queued or currently processing:
    - Stops processing of new items in the batch
    - Allows currently processing items to complete
    - Updates batch status to cancelled
    - Preserves results of completed items
    - Releases allocated resources
    
    Cannot cancel already completed or failed batches.
    """
    try:
        success = batch_processor.cancel_batch(batch_id)
        
        if not success:
            # Check if batch exists
            progress = batch_processor.get_batch_status(batch_id)
            if not progress:
                raise HTTPException(
                    status_code=404,
                    detail=f"Batch {batch_id} not found"
                )
            else:
                raise HTTPException(
                    status_code=409,
                    detail=f"Batch {batch_id} cannot be cancelled (status: {progress.status})"
                )
        
        return {
            "batch_id": batch_id,
            "status": "cancelled",
            "message": f"Batch {batch_id} has been cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel batch {batch_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel batch: {str(e)}"
        )

@router.get("/batch/list")
async def list_batches(
    status: Optional[BatchStatus] = Query(None, description="Filter by batch status"),
    batch_type: Optional[BatchType] = Query(None, description="Filter by batch type"),
    priority: Optional[BatchPriority] = Query(None, description="Filter by priority"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of batches to return"),
    offset: int = Query(0, ge=0, description="Number of batches to skip")
):
    """
    List batch jobs with filtering and pagination
    
    Provides comprehensive batch job management interface:
    - Filter by status, type, and priority
    - Pagination for large batch lists
    - Summary information for each batch
    - Quick access to batch status and results
    
    Essential for batch job monitoring and management
    in production environments.
    """
    try:
        # Get all active batches (in production, this would query a database)
        all_batches = list(batch_processor.active_batches.items())
        
        # Apply filters
        filtered_batches = []
        for batch_id, batch_info in all_batches:
            # Status filter
            if status and batch_info['status'] != status:
                continue
                
            # Type filter
            if batch_type and batch_info['request'].config.batch_type != batch_type:
                continue
                
            # Priority filter  
            if priority and batch_info['request'].config.priority != priority:
                continue
            
            # Create summary
            batch_summary = {
                "batch_id": batch_id,
                "batch_name": batch_info['request'].config.batch_name,
                "batch_type": batch_info['request'].config.batch_type,
                "priority": batch_info['request'].config.priority,
                "status": batch_info['status'],
                "total_items": len(batch_info['request'].items),
                "completed_items": batch_info['progress'].completed_items,
                "failed_items": batch_info['progress'].failed_items,
                "progress_percentage": batch_info['progress'].progress_percentage,
                "created_at": batch_info['start_time'].isoformat(),
                "estimated_completion": batch_info['progress'].estimated_completion_time.isoformat() if batch_info['progress'].estimated_completion_time else None
            }
            
            filtered_batches.append(batch_summary)
        
        # Sort by creation time (newest first)
        filtered_batches.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Apply pagination
        total_count = len(filtered_batches)
        paginated_batches = filtered_batches[offset:offset + limit]
        
        return {
            "batches": paginated_batches,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to list batches: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve batch list: {str(e)}"
        )

@router.get("/batch/metrics", response_model=BatchMetrics)
async def get_batch_metrics():
    """
    Get batch processing system metrics
    
    Returns comprehensive system performance metrics:
    - Queue statistics and wait times
    - Processing throughput and completion rates
    - Resource utilization (CPU, memory)
    - Success rates and quality metrics
    - Active worker information
    
    Critical for system monitoring, capacity planning,
    and performance optimization in production environments.
    """
    try:
        metrics = batch_processor.get_system_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get batch metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )

@router.post("/batch/templates/parameter-sweep")
async def create_parameter_sweep_batch(
    base_problem: Dict[str, Any],
    parameter_ranges: Dict[str, List],
    batch_name: Optional[str] = None,
    priority: BatchPriority = BatchPriority.NORMAL
):
    """
    Create a parameter sweep batch job
    
    Automatically generates a batch job that tests different parameter
    combinations on a base VRP problem:
    - Systematic exploration of parameter space
    - Automated generation of parameter combinations
    - Performance comparison across configurations
    - Statistical analysis of parameter impact
    
    Useful for algorithm tuning and sensitivity analysis.
    """
    try:
        # Validate parameter ranges
        if not parameter_ranges:
            raise HTTPException(
                status_code=400,
                detail="Parameter ranges must be specified"
            )
        
        # Generate parameter combinations
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        
        if len(combinations) > 500:
            raise HTTPException(
                status_code=400,
                detail=f"Too many parameter combinations ({len(combinations)}). Maximum is 500."
            )
        
        # Create batch items
        items = []
        for i, combination in enumerate(combinations):
            param_dict = dict(zip(param_names, combination))
            
            # Create modified problem
            problem_data = base_problem.copy()
            problem_data.update(param_dict)
            
            item = {
                "item_id": f"param_sweep_{i}",
                "problem_data": problem_data,
                "metadata": {"parameters": param_dict}
            }
            items.append(item)
        
        # Create batch configuration
        config = {
            "batch_id": f"param_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "batch_type": BatchType.PARAMETER_SWEEP,
            "batch_name": batch_name or f"Parameter Sweep - {len(combinations)} combinations",
            "priority": priority,
            "max_concurrent_jobs": min(4, len(combinations)),
            "parameter_sweep": {
                "parameter_ranges": parameter_ranges,
                "base_problem": base_problem,
                "metrics_to_track": ["total_distance", "total_cost", "vehicle_count"]
            }
        }
        
        # Create batch request
        batch_request = BatchJobRequest(
            config=config,
            items=items
        )
        
        # Submit batch
        result = await batch_processor.submit_batch(batch_request)
        
        return {
            **result,
            "parameter_combinations": len(combinations),
            "parameters_tested": list(parameter_ranges.keys()),
            "analysis_url": f"/api/vrp/v1/batch/{result['batch_id']}/analysis"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create parameter sweep batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create parameter sweep: {str(e)}"
        )

@router.post("/batch/templates/scenario-analysis")
async def create_scenario_analysis_batch(
    scenarios: List[Dict[str, Any]],
    scenario_names: List[str],
    batch_name: Optional[str] = None,
    priority: BatchPriority = BatchPriority.NORMAL
):
    """
    Create a scenario analysis batch job
    
    Compares multiple business scenarios with different VRP configurations:
    - Strategic planning support with "what-if" analysis
    - Comparison of different operational strategies
    - Risk analysis across various scenarios
    - Decision support with quantitative comparisons
    
    Essential for strategic decision making in logistics operations.
    """
    try:
        # Validate scenarios
        if len(scenarios) != len(scenario_names):
            raise HTTPException(
                status_code=400,
                detail="Number of scenarios must match number of scenario names"
            )
        
        if len(scenarios) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 scenarios allowed per batch"
            )
        
        # Create batch items
        items = []
        for i, (scenario, name) in enumerate(zip(scenarios, scenario_names)):
            item = {
                "item_id": f"scenario_{i}",
                "problem_data": scenario,
                "metadata": {"scenario_name": name, "scenario_index": i}
            }
            items.append(item)
        
        # Create batch configuration
        config = {
            "batch_id": f"scenario_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "batch_type": BatchType.SCENARIO_ANALYSIS,
            "batch_name": batch_name or f"Scenario Analysis - {len(scenarios)} scenarios",
            "priority": priority,
            "max_concurrent_jobs": min(4, len(scenarios)),
            "scenario_analysis": {
                "scenarios": scenarios,
                "scenario_names": scenario_names,
                "comparison_metrics": ["total_distance", "total_cost", "vehicle_count", "service_level"]
            }
        }
        
        # Create batch request
        batch_request = BatchJobRequest(
            config=config,
            items=items
        )
        
        # Submit batch
        result = await batch_processor.submit_batch(batch_request)
        
        return {
            **result,
            "scenarios_count": len(scenarios),
            "scenario_names": scenario_names,
            "comparison_url": f"/api/vrp/v1/batch/{result['batch_id']}/comparison"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create scenario analysis batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create scenario analysis: {str(e)}"
        )

@router.get("/batch/templates")
async def get_batch_templates():
    """
    Get available batch processing templates
    
    Returns predefined templates for common batch operations:
    - Parameter sweep templates for algorithm tuning
    - Scenario analysis templates for strategic planning
    - Benchmark comparison templates for performance testing
    - Sensitivity analysis templates for robustness testing
    
    Templates provide quick setup for common batch processing patterns
    and ensure best practices are followed.
    """
    templates = {
        "parameter_sweep": {
            "name": "Parameter Sweep",
            "description": "Systematic exploration of parameter combinations",
            "batch_type": "parameter_sweep",
            "example_parameters": {
                "vehicle_capacity": [1000, 1500, 2000],
                "max_routes": [5, 10, 15],
                "time_limit": [30, 60, 120]
            },
            "use_cases": [
                "Algorithm tuning and optimization",
                "Sensitivity analysis of key parameters",
                "Performance comparison across settings"
            ]
        },
        "scenario_analysis": {
            "name": "Scenario Analysis",
            "description": "Compare different business scenarios",
            "batch_type": "scenario_analysis",
            "example_scenarios": [
                "Current operations",
                "Increased demand (+20%)",
                "Reduced fleet (-2 vehicles)",
                "Extended service area"
            ],
            "use_cases": [
                "Strategic planning and decision support",
                "Risk analysis and contingency planning",
                "Business case development"
            ]
        },
        "benchmark_comparison": {
            "name": "Benchmark Comparison",
            "description": "Compare algorithm performance",
            "batch_type": "benchmark_comparison",
            "algorithms": ["genetic", "local_search", "hybrid"],
            "use_cases": [
                "Algorithm selection and validation",
                "Performance benchmarking",
                "Quality assurance testing"
            ]
        },
        "sensitivity_analysis": {
            "name": "Sensitivity Analysis",
            "description": "Analyze solution robustness to data changes",
            "batch_type": "sensitivity_analysis",
            "perturbation_types": ["demand_variation", "travel_time_variation", "capacity_changes"],
            "use_cases": [
                "Solution robustness evaluation",
                "Risk assessment",
                "Data quality impact analysis"
            ]
        }
    }
    
    return {
        "templates": templates,
        "batch_types": [bt.value for bt in BatchType],
        "priority_levels": [bp.value for bp in BatchPriority],
        "usage_guidelines": [
            "Choose template based on your analysis objective",
            "Start with smaller batches to validate setup",
            "Use appropriate priority levels for time-sensitive analysis",
            "Monitor resource utilization for large batches"
        ]
    }

@router.get("/batch/{batch_id}/analysis")
async def get_batch_analysis(batch_id: str):
    """
    Get detailed analysis of completed batch results
    
    Provides advanced analysis and insights from batch processing:
    - Statistical analysis of results across batch items
    - Parameter impact analysis for parameter sweeps
    - Scenario comparison with recommendations
    - Performance trends and outlier detection
    - Visualization data for charts and graphs
    
    Essential for extracting actionable insights from batch processing results.
    """
    try:
        # Get batch result
        result = batch_processor.get_batch_result(batch_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Batch {batch_id} not found or not completed"
            )
        
        # Perform analysis based on batch type
        analysis = {}
        
        # Extract metrics from results
        successful_results = [r for r in result.item_results if r.status == BatchStatus.COMPLETED]
        
        if not successful_results:
            return {
                "batch_id": batch_id,
                "analysis": "No successful results to analyze",
                "recommendations": ["Review failed items for common issues"]
            }
        
        # Basic statistical analysis
        metrics = {}
        for result_item in successful_results:
            if result_item.solution:
                solution_data = result_item.solution
                
                for metric in ["total_distance", "total_cost", "vehicle_count"]:
                    if metric not in metrics:
                        metrics[metric] = []
                    
                    if metric in solution_data:
                        metrics[metric].append(solution_data[metric])
        
        # Calculate statistics
        stats = {}
        for metric, values in metrics.items():
            if values:
                stats[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
                }
        
        analysis = {
            "batch_id": batch_id,
            "successful_items": len(successful_results),
            "statistical_summary": stats,
            "performance_insights": _generate_performance_insights(stats),
            "recommendations": _generate_batch_recommendations(result, stats)
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze batch {batch_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze batch: {str(e)}"
        )

def _generate_performance_insights(stats: Dict[str, Dict]) -> List[str]:
    """Generate performance insights from statistics"""
    insights = []
    
    for metric, stat_data in stats.items():
        if stat_data['std'] / stat_data['mean'] > 0.2:  # High variation
            insights.append(f"{metric} shows high variation (CV: {stat_data['std']/stat_data['mean']:.2f})")
        
        if metric == "total_distance":
            if stat_data['mean'] > 1000:
                insights.append("Routes are relatively long - consider more vehicles or depots")
        
        if metric == "vehicle_count":
            if stat_data['std'] > 1:
                insights.append("Vehicle usage varies significantly across scenarios")
    
    return insights

def _generate_batch_recommendations(result: BatchResult, stats: Dict[str, Dict]) -> List[str]:
    """Generate recommendations based on batch results"""
    recommendations = []
    
    if result.summary.success_rate < 90:
        recommendations.append("Review failed items for data quality issues")
    
    if result.summary.average_item_duration_seconds > 120:
        recommendations.append("Consider optimizing solver parameters for faster processing")
    
    if stats.get("total_distance", {}).get("std", 0) > stats.get("total_distance", {}).get("mean", 0) * 0.3:
        recommendations.append("High distance variation suggests opportunity for route optimization")
    
    return recommendations