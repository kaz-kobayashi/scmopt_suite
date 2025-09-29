"""
Reporting and analytics models for VRP operations
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime, date

class ReportType(str, Enum):
    """Types of VRP reports"""
    OPTIMIZATION_SUMMARY = "optimization_summary"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    COST_ANALYSIS = "cost_analysis"
    ROUTE_EFFICIENCY = "route_efficiency"
    VEHICLE_UTILIZATION = "vehicle_utilization"
    CUSTOMER_SERVICE = "customer_service"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    TREND_ANALYSIS = "trend_analysis"
    EXECUTIVE_SUMMARY = "executive_summary"

class ReportFormat(str, Enum):
    """Report output formats"""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    POWERPOINT = "powerpoint"

class ReportFrequency(str, Enum):
    """Report generation frequency"""
    ON_DEMAND = "on_demand"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class ChartType(str, Enum):
    """Types of charts for reports"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"
    MAP = "map"

class ReportSection(BaseModel):
    """Individual section of a report"""
    section_id: str = Field(description="Unique section identifier")
    title: str = Field(description="Section title")
    content_type: str = Field(description="Type of content (chart, table, text, map)")
    data: Dict[str, Any] = Field(description="Section data")
    chart_config: Optional[Dict[str, Any]] = Field(None, description="Chart configuration if applicable")
    description: Optional[str] = Field(None, description="Section description")
    insights: List[str] = Field([], description="Key insights for this section")

class ReportTemplate(BaseModel):
    """Template for report generation"""
    template_id: str = Field(description="Unique template identifier")
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    report_type: ReportType = Field(description="Type of report")
    sections: List[Dict[str, Any]] = Field(description="Section configurations")
    default_format: ReportFormat = Field(description="Default output format")
    parameters: Dict[str, Any] = Field({}, description="Template parameters")
    created_by: Optional[str] = Field(None, description="Template creator")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

class ReportRequest(BaseModel):
    """Request for report generation"""
    template_id: Optional[str] = Field(None, description="Template to use")
    report_type: ReportType = Field(description="Type of report to generate")
    report_name: str = Field(description="Name for the report")
    format: ReportFormat = Field(ReportFormat.PDF, description="Output format")
    
    # Data source configuration
    solution_ids: List[str] = Field([], description="VRP solution IDs to include")
    batch_ids: List[str] = Field([], description="Batch job IDs to include")
    date_range: Optional[Dict[str, date]] = Field(None, description="Date range filter")
    
    # Report customization
    sections_to_include: List[str] = Field([], description="Specific sections to include")
    custom_sections: List[ReportSection] = Field([], description="Custom report sections")
    branding: Dict[str, Any] = Field({}, description="Company branding configuration")
    
    # Delivery options
    email_recipients: List[str] = Field([], description="Email addresses for delivery")
    schedule: Optional[ReportFrequency] = Field(None, description="Scheduled report frequency")
    webhook_url: Optional[str] = Field(None, description="Webhook for report completion")

class ReportMetrics(BaseModel):
    """Key metrics for VRP reports"""
    # Route metrics
    total_distance: float = Field(description="Total distance across all routes")
    total_duration: float = Field(description="Total duration in hours")
    total_cost: float = Field(description="Total operational cost")
    
    # Vehicle metrics
    vehicles_used: int = Field(description="Number of vehicles utilized")
    average_capacity_utilization: float = Field(description="Average vehicle capacity utilization")
    vehicle_efficiency_score: float = Field(description="Overall vehicle efficiency score")
    
    # Service metrics
    customers_served: int = Field(description="Number of customers served")
    on_time_delivery_rate: float = Field(description="On-time delivery percentage")
    service_level_score: float = Field(description="Overall service level score")
    
    # Environmental metrics
    co2_emissions: float = Field(description="Total CO2 emissions in kg")
    fuel_consumption: float = Field(description="Total fuel consumption in liters")
    environmental_impact_score: float = Field(description="Environmental impact score")
    
    # Efficiency metrics
    route_efficiency: float = Field(description="Route efficiency percentage")
    cost_per_delivery: float = Field(description="Cost per delivery")
    distance_per_delivery: float = Field(description="Distance per delivery")

class TrendAnalysis(BaseModel):
    """Trend analysis data"""
    metric_name: str = Field(description="Name of the metric")
    time_series: List[Dict[str, Any]] = Field(description="Time series data points")
    trend_direction: str = Field(description="Trend direction (improving, declining, stable)")
    trend_strength: float = Field(description="Strength of trend (0-1)")
    forecast: Optional[List[Dict[str, Any]]] = Field(None, description="Forecast data points")
    insights: List[str] = Field(description="Trend insights")

class BenchmarkComparison(BaseModel):
    """Benchmark comparison data"""
    benchmark_name: str = Field(description="Name of benchmark")
    current_performance: Dict[str, float] = Field(description="Current performance metrics")
    benchmark_performance: Dict[str, float] = Field(description="Benchmark performance metrics")
    performance_gaps: Dict[str, float] = Field(description="Performance gaps (+ is better)")
    improvement_opportunities: List[str] = Field(description="Areas for improvement")
    
class GeneratedReport(BaseModel):
    """Generated report information"""
    report_id: str = Field(description="Unique report identifier")
    report_name: str = Field(description="Report name")
    report_type: ReportType = Field(description="Type of report")
    format: ReportFormat = Field(description="Output format")
    
    # Generation info
    generated_at: datetime = Field(description="Generation timestamp")
    generated_by: Optional[str] = Field(None, description="User who requested the report")
    generation_time_seconds: float = Field(description="Time taken to generate report")
    
    # Content info
    sections: List[ReportSection] = Field(description="Report sections")
    metrics: ReportMetrics = Field(description="Key metrics included")
    executive_summary: str = Field(description="Executive summary")
    
    # File info
    file_url: str = Field(description="URL to download report file")
    file_size_mb: float = Field(description="File size in MB")
    expires_at: Optional[datetime] = Field(None, description="File expiration time")
    
    # Metadata
    data_sources: List[str] = Field(description="Data sources used")
    parameters_used: Dict[str, Any] = Field(description="Parameters used for generation")

class ReportSchedule(BaseModel):
    """Scheduled report configuration"""
    schedule_id: str = Field(description="Unique schedule identifier")
    report_request: ReportRequest = Field(description="Report configuration")
    frequency: ReportFrequency = Field(description="Generation frequency")
    next_run: datetime = Field(description="Next scheduled run")
    last_run: Optional[datetime] = Field(None, description="Last successful run")
    is_active: bool = Field(True, description="Whether schedule is active")
    created_by: Optional[str] = Field(None, description="Schedule creator")
    
class ReportAnalytics(BaseModel):
    """Analytics about report usage"""
    total_reports_generated: int = Field(description="Total reports generated")
    reports_by_type: Dict[ReportType, int] = Field(description="Report count by type")
    reports_by_format: Dict[ReportFormat, int] = Field(description="Report count by format")
    average_generation_time: float = Field(description="Average generation time")
    most_popular_templates: List[Dict[str, Any]] = Field(description="Most used templates")
    scheduled_reports_count: int = Field(description="Number of scheduled reports")
    user_engagement: Dict[str, Any] = Field(description="User engagement metrics")