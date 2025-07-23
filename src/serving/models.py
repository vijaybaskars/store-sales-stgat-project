"""
Pydantic models for Phase 6 FastAPI backend
Defines request/response schemas for the store sales forecasting API
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """API health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "Phase 6.0"


class EvaluationCase(BaseModel):
    """Evaluation case information"""
    case_id: str
    store_nbr: int
    family: str
    quality_score: float
    selection_metrics: Dict[str, Any]


class EvaluationCasesResponse(BaseModel):
    """Response for listing available evaluation cases"""
    cases: List[EvaluationCase]
    total_cases: int
    metadata: Dict[str, Any]


class PatternAnalysisResponse(BaseModel):
    """Pattern analysis results"""
    store_nbr: int
    family: str
    coefficient_variation: float
    pattern_type: str  # "REGULAR" or "VOLATILE"
    confidence_score: float
    zero_sales_percentage: float
    seasonal_strength: float
    trend_strength: float
    raw_metrics: Dict[str, Any]
    recommended_model_type: str  # "NEURAL" or "TRADITIONAL"


class PredictionRequest(BaseModel):
    """Request for generating predictions"""
    forecast_horizon: int = Field(default=15, ge=1, le=30)
    include_analysis: bool = Field(default=True)


class PredictionResponse(BaseModel):
    """Prediction results with comprehensive metrics"""
    store_nbr: int
    family: str
    pattern_analysis: PatternAnalysisResponse
    selected_model_type: str
    selected_model_name: str
    test_rmsle: float
    test_mae: float
    test_mape: float
    predictions: List[float]
    actuals: List[float]
    model_params: Dict[str, Any]
    fit_time: float
    improvement_vs_traditional_baseline: float
    improvement_vs_neural_baseline: float
    beats_traditional: bool
    beats_neural: bool
    performance_summary: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelPerformanceMetrics(BaseModel):
    """Model performance comparison metrics"""
    phase_2_traditional_baseline: float = 0.4755
    phase_3_neural_baseline: float = 0.5466
    phase_6_current_result: float
    improvement_vs_phase_2: float
    improvement_vs_phase_3: float
    champion_status: str  # "CHAMPION", "COMPETITIVE", "UNDERPERFORMING"


class DashboardSummary(BaseModel):
    """Summary data for dashboard display"""
    total_evaluation_cases: int
    cases_evaluated: int
    average_rmsle: float
    neural_routing_count: int
    traditional_routing_count: int
    champion_cases: int  # Cases that beat both baselines
    performance_distribution: Dict[str, int]
    top_performing_cases: List[Dict[str, Any]]