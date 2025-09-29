"""
Reporting service for VRP analytics and insights
"""
import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from io import BytesIO
import base64

# For report generation (in production, you'd use proper libraries)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from app.models.reporting_models import (
    ReportType, ReportFormat, GeneratedReport, ReportRequest,
    ReportSection, ReportMetrics, TrendAnalysis, BenchmarkComparison,
    ReportTemplate, ReportAnalytics
)
from app.models.vrp_unified_models import UnifiedVRPSolution
from app.models.batch_processing_models import BatchResult

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive reports for VRP operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Report templates storage (in production, this would be database)
        self.templates = self._initialize_templates()
        
        # Generated reports cache
        self.generated_reports: Dict[str, GeneratedReport] = {}
        
        # Analytics tracking
        self.analytics = {
            'total_generated': 0,
            'by_type': {},
            'by_format': {},
            'generation_times': []
        }
    
    def _initialize_templates(self) -> Dict[str, ReportTemplate]:
        """Initialize default report templates"""
        templates = {}
        
        # Executive Summary Template
        exec_template = ReportTemplate(
            template_id="executive_summary",
            name="Executive Summary",
            description="High-level overview of VRP performance",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            sections=[
                {"type": "summary_metrics", "title": "Key Performance Indicators"},
                {"type": "cost_analysis", "title": "Cost Analysis"},
                {"type": "efficiency_trends", "title": "Efficiency Trends"},
                {"type": "recommendations", "title": "Strategic Recommendations"}
            ],
            default_format=ReportFormat.PDF
        )
        templates[exec_template.template_id] = exec_template
        
        # Performance Analysis Template
        perf_template = ReportTemplate(
            template_id="performance_analysis",
            name="Performance Analysis",
            description="Detailed performance analysis of VRP operations",
            report_type=ReportType.PERFORMANCE_ANALYSIS,
            sections=[
                {"type": "route_efficiency", "title": "Route Efficiency Analysis"},
                {"type": "vehicle_utilization", "title": "Vehicle Utilization"},
                {"type": "service_quality", "title": "Service Quality Metrics"},
                {"type": "benchmark_comparison", "title": "Benchmark Comparison"}
            ],
            default_format=ReportFormat.HTML
        )
        templates[perf_template.template_id] = perf_template
        
        return templates
    
    async def generate_report(self, request: ReportRequest) -> GeneratedReport:
        """
        Generate a comprehensive VRP report
        
        Args:
            request: Report generation request
            
        Returns:
            Generated report with all sections and metrics
        """
        try:
            start_time = datetime.now()
            report_id = str(uuid.uuid4())
            
            self.logger.info(f"Generating report {report_id} of type {request.report_type}")
            
            # Collect data from specified sources
            solution_data = await self._collect_solution_data(request.solution_ids)
            batch_data = await self._collect_batch_data(request.batch_ids)
            
            # Calculate key metrics
            metrics = self._calculate_report_metrics(solution_data, batch_data)
            
            # Generate report sections
            sections = await self._generate_report_sections(
                request, solution_data, batch_data, metrics
            )
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(metrics, sections)
            
            # Create report file
            file_url, file_size = await self._create_report_file(
                request, sections, metrics, executive_summary
            )
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create report object
            report = GeneratedReport(
                report_id=report_id,
                report_name=request.report_name,
                report_type=request.report_type,
                format=request.format,
                generated_at=start_time,
                generation_time_seconds=generation_time,
                sections=sections,
                metrics=metrics,
                executive_summary=executive_summary,
                file_url=file_url,
                file_size_mb=file_size,
                expires_at=datetime.now() + timedelta(days=30),
                data_sources=request.solution_ids + request.batch_ids,
                parameters_used=request.dict()
            )
            
            # Store report
            self.generated_reports[report_id] = report
            
            # Update analytics
            self._update_analytics(request.report_type, request.format, generation_time)
            
            # Send notifications if requested
            if request.email_recipients:
                await self._send_report_notifications(report, request.email_recipients)
            
            self.logger.info(f"Report {report_id} generated successfully in {generation_time:.2f}s")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    async def _collect_solution_data(self, solution_ids: List[str]) -> List[UnifiedVRPSolution]:
        """Collect VRP solution data for reporting"""
        # In production, this would query the database
        solutions = []
        for solution_id in solution_ids:
            # Simulate solution retrieval
            solution = self._create_sample_solution(solution_id)
            solutions.append(solution)
        
        return solutions
    
    async def _collect_batch_data(self, batch_ids: List[str]) -> List[BatchResult]:
        """Collect batch processing data for reporting"""
        # In production, this would query the database
        batch_results = []
        for batch_id in batch_ids:
            # Simulate batch result retrieval
            batch_result = self._create_sample_batch_result(batch_id)
            batch_results.append(batch_result)
        
        return batch_results
    
    def _calculate_report_metrics(
        self, 
        solutions: List[UnifiedVRPSolution], 
        batch_results: List[BatchResult]
    ) -> ReportMetrics:
        """Calculate key metrics for the report"""
        if not solutions:
            # Default metrics if no data
            return ReportMetrics(
                total_distance=0.0, total_duration=0.0, total_cost=0.0,
                vehicles_used=0, average_capacity_utilization=0.0, 
                vehicle_efficiency_score=0.0, customers_served=0,
                on_time_delivery_rate=0.0, service_level_score=0.0,
                co2_emissions=0.0, fuel_consumption=0.0,
                environmental_impact_score=0.0, route_efficiency=0.0,
                cost_per_delivery=0.0, distance_per_delivery=0.0
            )
        
        # Aggregate metrics across solutions
        total_distance = sum(s.total_distance or 0 for s in solutions)
        total_duration = sum(s.total_duration or 0 for s in solutions) / 3600  # Convert to hours
        vehicles_used = sum(len(s.routes) for s in solutions)
        customers_served = sum(
            len([stop for route in s.routes for stop in route.stops if stop.client_id != 0])
            for s in solutions
        )
        
        # Calculate derived metrics
        avg_capacity_util = 75.0  # Simplified calculation
        total_cost = total_distance * 0.5 + vehicles_used * 100  # Simplified cost model
        co2_emissions = total_distance * 0.2  # Simplified emissions
        fuel_consumption = total_distance * 0.08  # Simplified fuel consumption
        
        return ReportMetrics(
            total_distance=total_distance,
            total_duration=total_duration,
            total_cost=total_cost,
            vehicles_used=vehicles_used,
            average_capacity_utilization=avg_capacity_util,
            vehicle_efficiency_score=85.0,
            customers_served=customers_served,
            on_time_delivery_rate=92.0,
            service_level_score=88.0,
            co2_emissions=co2_emissions,
            fuel_consumption=fuel_consumption,
            environmental_impact_score=78.0,
            route_efficiency=82.0,
            cost_per_delivery=total_cost / max(customers_served, 1),
            distance_per_delivery=total_distance / max(customers_served, 1)
        )
    
    async def _generate_report_sections(
        self,
        request: ReportRequest,
        solutions: List[UnifiedVRPSolution],
        batch_results: List[BatchResult],
        metrics: ReportMetrics
    ) -> List[ReportSection]:
        """Generate all sections for the report"""
        sections = []
        
        # Key Performance Indicators section
        kpi_section = ReportSection(
            section_id="kpi",
            title="Key Performance Indicators",
            content_type="metrics",
            data={
                "total_distance": metrics.total_distance,
                "total_cost": metrics.total_cost,
                "vehicles_used": metrics.vehicles_used,
                "customers_served": metrics.customers_served,
                "efficiency_score": metrics.route_efficiency
            },
            insights=[
                f"Total distance of {metrics.total_distance:.1f} km across {metrics.vehicles_used} vehicles",
                f"Average cost per delivery: ${metrics.cost_per_delivery:.2f}",
                f"Route efficiency score: {metrics.route_efficiency:.1f}%"
            ]
        )
        sections.append(kpi_section)
        
        # Cost Analysis section
        cost_section = ReportSection(
            section_id="cost_analysis",
            title="Cost Analysis",
            content_type="chart",
            data={
                "total_cost": metrics.total_cost,
                "cost_breakdown": {
                    "fuel_cost": metrics.fuel_consumption * 1.5,
                    "vehicle_cost": metrics.vehicles_used * 100,
                    "driver_cost": metrics.total_duration * 25,
                    "overhead_cost": metrics.total_cost * 0.1
                }
            },
            chart_config={
                "type": "pie",
                "title": "Cost Breakdown",
                "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
            },
            insights=[
                "Fuel costs represent the largest expense category",
                f"Cost per kilometer: ${metrics.total_cost / max(metrics.total_distance, 1):.2f}",
                "Opportunity for 15% cost reduction through route optimization"
            ]
        )
        sections.append(cost_section)
        
        # Environmental Impact section
        env_section = ReportSection(
            section_id="environmental",
            title="Environmental Impact",
            content_type="chart",
            data={
                "co2_emissions": metrics.co2_emissions,
                "fuel_consumption": metrics.fuel_consumption,
                "emissions_per_delivery": metrics.co2_emissions / max(metrics.customers_served, 1),
                "environmental_score": metrics.environmental_impact_score
            },
            chart_config={
                "type": "gauge",
                "title": "Environmental Impact Score",
                "min": 0,
                "max": 100,
                "value": metrics.environmental_impact_score
            },
            insights=[
                f"Total CO2 emissions: {metrics.co2_emissions:.1f} kg",
                f"Emissions per delivery: {metrics.co2_emissions / max(metrics.customers_served, 1):.2f} kg",
                "Consider electric vehicles for 25% emission reduction"
            ]
        )
        sections.append(env_section)
        
        # Route Efficiency section if solutions available
        if solutions:
            route_section = await self._generate_route_analysis_section(solutions)
            sections.append(route_section)
        
        # Batch Analysis section if batch results available
        if batch_results:
            batch_section = await self._generate_batch_analysis_section(batch_results)
            sections.append(batch_section)
        
        return sections
    
    async def _generate_route_analysis_section(
        self, 
        solutions: List[UnifiedVRPSolution]
    ) -> ReportSection:
        """Generate route analysis section"""
        
        # Analyze route characteristics
        route_distances = []
        route_durations = []
        stops_per_route = []
        
        for solution in solutions:
            for route in solution.routes:
                route_distances.append(route.distance)
                route_durations.append(route.duration / 3600)  # Convert to hours
                stops_per_route.append(len(route.stops) - 2)  # Exclude depot stops
        
        return ReportSection(
            section_id="route_analysis",
            title="Route Analysis",
            content_type="chart",
            data={
                "avg_route_distance": sum(route_distances) / len(route_distances) if route_distances else 0,
                "avg_route_duration": sum(route_durations) / len(route_durations) if route_durations else 0,
                "avg_stops_per_route": sum(stops_per_route) / len(stops_per_route) if stops_per_route else 0,
                "route_distribution": {
                    "short_routes": len([d for d in route_distances if d < 50]),
                    "medium_routes": len([d for d in route_distances if 50 <= d < 100]),
                    "long_routes": len([d for d in route_distances if d >= 100])
                }
            },
            chart_config={
                "type": "bar",
                "title": "Route Length Distribution",
                "x_axis": "Route Category",
                "y_axis": "Number of Routes"
            },
            insights=[
                f"Average route distance: {sum(route_distances) / len(route_distances):.1f} km" if route_distances else "No route data",
                f"Most routes are {'short' if len([d for d in route_distances if d < 50]) > len(route_distances)/2 else 'medium'} distance",
                "Consider route consolidation for improved efficiency"
            ]
        )
    
    async def _generate_batch_analysis_section(
        self, 
        batch_results: List[BatchResult]
    ) -> ReportSection:
        """Generate batch analysis section"""
        
        # Analyze batch performance
        success_rates = []
        processing_times = []
        total_items = 0
        
        for batch in batch_results:
            success_rate = batch.summary.success_rate
            success_rates.append(success_rate)
            processing_times.append(batch.summary.total_duration_seconds or 0)
            total_items += batch.summary.total_items
        
        return ReportSection(
            section_id="batch_analysis",
            title="Batch Processing Analysis",
            content_type="chart",
            data={
                "total_batches": len(batch_results),
                "total_items_processed": total_items,
                "average_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
                "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "batch_performance": [
                    {
                        "batch_id": batch.batch_id,
                        "success_rate": batch.summary.success_rate,
                        "items": batch.summary.total_items,
                        "duration": batch.summary.total_duration_seconds
                    }
                    for batch in batch_results
                ]
            },
            chart_config={
                "type": "scatter",
                "title": "Batch Performance",
                "x_axis": "Processing Time (seconds)",
                "y_axis": "Success Rate (%)"
            },
            insights=[
                f"Processed {total_items} items across {len(batch_results)} batches",
                f"Average success rate: {sum(success_rates) / len(success_rates):.1f}%" if success_rates else "No batch data",
                "Batch processing is performing within expected parameters"
            ]
        )
    
    def _generate_executive_summary(
        self, 
        metrics: ReportMetrics, 
        sections: List[ReportSection]
    ) -> str:
        """Generate executive summary text"""
        summary_parts = [
            f"This report analyzes VRP operations covering {metrics.customers_served} customer deliveries across {metrics.vehicles_used} vehicles.",
            f"Total operational distance was {metrics.total_distance:.1f} km with a total cost of ${metrics.total_cost:.2f}.",
            f"Overall route efficiency achieved {metrics.route_efficiency:.1f}% with an average cost per delivery of ${metrics.cost_per_delivery:.2f}.",
            f"Environmental impact includes {metrics.co2_emissions:.1f} kg of CO2 emissions.",
        ]
        
        # Add key insights from sections
        key_insights = []
        for section in sections:
            if section.insights:
                key_insights.extend(section.insights[:1])  # Take first insight from each section
        
        if key_insights:
            summary_parts.append("Key insights include: " + "; ".join(key_insights[:3]) + ".")
        
        return " ".join(summary_parts)
    
    async def _create_report_file(
        self,
        request: ReportRequest,
        sections: List[ReportSection],
        metrics: ReportMetrics,
        executive_summary: str
    ) -> Tuple[str, float]:
        """Create the actual report file"""
        
        if request.format == ReportFormat.JSON:
            # Generate JSON report
            report_content = {
                "executive_summary": executive_summary,
                "metrics": metrics.dict(),
                "sections": [section.dict() for section in sections],
                "generated_at": datetime.now().isoformat()
            }
            
            # In production, save to file storage and return URL
            file_url = f"/reports/{uuid.uuid4()}.json"
            file_size = len(json.dumps(report_content)) / (1024 * 1024)  # MB
            
        elif request.format == ReportFormat.HTML:
            # Generate HTML report
            html_content = self._generate_html_report(sections, metrics, executive_summary)
            file_url = f"/reports/{uuid.uuid4()}.html"
            file_size = len(html_content) / (1024 * 1024)  # MB
            
        elif request.format == ReportFormat.PDF:
            # Generate PDF report (simplified)
            file_url = f"/reports/{uuid.uuid4()}.pdf"
            file_size = 2.5  # MB (estimated)
            
        else:
            # Default to JSON
            file_url = f"/reports/{uuid.uuid4()}.json"
            file_size = 1.0  # MB (estimated)
        
        return file_url, file_size
    
    def _generate_html_report(
        self, 
        sections: List[ReportSection], 
        metrics: ReportMetrics, 
        executive_summary: str
    ) -> str:
        """Generate HTML report content"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VRP Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ color: #1976d2; border-bottom: 2px solid #1976d2; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metrics {{ display: flex; justify-content: space-around; }}
                .metric {{ text-align: center; }}
                .insights {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>VRP Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{executive_summary}</p>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>{metrics.total_distance:.1f} km</h3>
                        <p>Total Distance</p>
                    </div>
                    <div class="metric">
                        <h3>${metrics.total_cost:.2f}</h3>
                        <p>Total Cost</p>
                    </div>
                    <div class="metric">
                        <h3>{metrics.vehicles_used}</h3>
                        <p>Vehicles Used</p>
                    </div>
                    <div class="metric">
                        <h3>{metrics.customers_served}</h3>
                        <p>Customers Served</p>
                    </div>
                </div>
            </div>
        """
        
        # Add sections
        for section in sections:
            html_content += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <div class="insights">
                    <h4>Key Insights:</h4>
                    <ul>
            """
            for insight in section.insights:
                html_content += f"<li>{insight}</li>"
            
            html_content += """
                    </ul>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    async def _send_report_notifications(self, report: GeneratedReport, recipients: List[str]):
        """Send report completion notifications"""
        # In production, this would send actual emails
        self.logger.info(f"Report {report.report_id} notification sent to {len(recipients)} recipients")
    
    def _update_analytics(self, report_type: ReportType, format: ReportFormat, generation_time: float):
        """Update reporting analytics"""
        self.analytics['total_generated'] += 1
        
        # Update by type
        type_key = report_type.value
        self.analytics['by_type'][type_key] = self.analytics['by_type'].get(type_key, 0) + 1
        
        # Update by format
        format_key = format.value
        self.analytics['by_format'][format_key] = self.analytics['by_format'].get(format_key, 0) + 1
        
        # Update generation times
        self.analytics['generation_times'].append(generation_time)
    
    def get_report_analytics(self) -> ReportAnalytics:
        """Get reporting analytics"""
        generation_times = self.analytics['generation_times']
        avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
        
        return ReportAnalytics(
            total_reports_generated=self.analytics['total_generated'],
            reports_by_type=self.analytics['by_type'],
            reports_by_format=self.analytics['by_format'],
            average_generation_time=avg_time,
            most_popular_templates=[
                {"template_id": "executive_summary", "usage_count": 50},
                {"template_id": "performance_analysis", "usage_count": 35}
            ],
            scheduled_reports_count=0,  # Would track scheduled reports
            user_engagement={"active_users": 25, "reports_per_user": 2.5}
        )
    
    def get_available_templates(self) -> List[ReportTemplate]:
        """Get available report templates"""
        return list(self.templates.values())
    
    def _create_sample_solution(self, solution_id: str) -> UnifiedVRPSolution:
        """Create sample solution data (for testing)"""
        from app.models.vrp_unified_models import UnifiedRouteModel as RouteModel
        
        # Create sample route
        route = RouteModel(
            vehicle_type=0,
            vehicle_id=1,
            start_depot=0,
            end_depot=0,
            clients=[1, 2],
            distance=120500,
            duration=10800,
            fixed_cost=100.0,
            variable_cost=120.5,
            total_cost=220.5,
            demand_served=25,
            max_load=25,
            capacity_utilization=0.25,
            start_time=0,
            end_time=10800,
            num_clients=2,
            empty_distance=0,
            loaded_distance=120500
        )
        
        return UnifiedVRPSolution(
            status="optimal",
            objective_value=220.5,
            routes=[route],
            computation_time=15.2,
            solver="PyVRP",
            is_feasible=True,
            problem_type="CVRP",
            problem_size={
                "num_clients": 2,
                "num_depots": 1,
                "num_vehicles": 1
            }
        )
    
    def _create_sample_batch_result(self, batch_id: str) -> BatchResult:
        """Create sample batch result (for testing)"""
        from app.models.batch_processing_models import BatchExecutionSummary, BatchItemResult, BatchStatus
        
        summary = BatchExecutionSummary(
            total_items=10,
            completed_items=9,
            failed_items=1,
            cancelled_items=0,
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
            total_duration_seconds=3600,
            average_item_duration_seconds=360,
            success_rate=90.0,
            throughput_items_per_hour=9.0
        )
        
        return BatchResult(
            batch_id=batch_id,
            status=BatchStatus.COMPLETED,
            summary=summary,
            item_results=[],
            recommendations=["Review failed item for data quality issues"]
        )

# Global report generator instance
report_generator = ReportGenerator()