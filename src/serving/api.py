"""
FastAPI backend for Phase 6 Store Sales Forecasting
Provides REST API endpoints for the Pattern-Based Selection methodology
"""

import sys
from pathlib import Path
from typing import List
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Path as PathParam, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from serving.models import (
    HealthResponse, EvaluationCasesResponse, PatternAnalysisResponse,
    PredictionRequest, PredictionResponse, ErrorResponse, DashboardSummary
)
from serving.data_service import data_service
from serving.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Store Sales Forecasting API - Phase 6",
    description="FastAPI backend for Pattern-Based Selection forecasting methodology",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).dict()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Validate configuration
        validation = config.validate_paths()
        
        # Check if key paths exist
        missing_paths = [name for name, info in validation.items() if not info["exists"]]
        
        if missing_paths:
            logger.warning(f"Missing paths: {missing_paths}")
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="Phase 6.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/validate")
async def validate_configuration():
    """Validate system configuration and paths"""
    try:
        validation = config.validate_paths()
        return {
            "config_validation": validation,
            "api_config": {
                "host": config.api_host,
                "port": config.api_port,
                "project_root": str(config.project_root)
            },
            "model_config": {
                "pattern_threshold": config.pattern_threshold,
                "forecast_horizon": config.forecast_horizon,
                "traditional_baseline": config.traditional_baseline,
                "neural_baseline": config.neural_baseline
            }
        }
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cases", response_model=EvaluationCasesResponse)
async def get_evaluation_cases():
    """Get all available evaluation cases"""
    try:
        cases = data_service.get_evaluation_cases()
        
        metadata = {
            "total_cases": len(cases),
            "data_source": str(config.evaluation_cases_path),
            "pattern_threshold": config.pattern_threshold
        }
        
        return EvaluationCasesResponse(
            cases=cases,
            total_cases=len(cases),
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Failed to get evaluation cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{store_nbr}/{family}", response_model=PatternAnalysisResponse)
async def analyze_pattern(
    store_nbr: int = PathParam(..., description="Store number"),
    family: str = PathParam(..., description="Product family")
):
    """Analyze time series pattern for store-family combination"""
    try:
        # Validate store-family combination
        if not data_service.validate_store_family(store_nbr, family):
            raise HTTPException(
                status_code=404,
                detail=f"Store {store_nbr} - Family '{family}' not found in evaluation cases"
            )
        
        analysis = data_service.analyze_pattern(store_nbr, family)
        
        logger.info(f"Pattern analysis completed for store {store_nbr}, family {family}")
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/{store_nbr}/{family}", response_model=PredictionResponse)
async def generate_prediction(
    store_nbr: int = PathParam(..., description="Store number"),
    family: str = PathParam(..., description="Product family"),
    request: PredictionRequest = PredictionRequest()
):
    """Generate predictions using adaptive model selection"""
    try:
        # Validate store-family combination
        if not data_service.validate_store_family(store_nbr, family):
            raise HTTPException(
                status_code=404,
                detail=f"Store {store_nbr} - Family '{family}' not found in evaluation cases"
            )
        
        # Validate forecast horizon
        if not (1 <= request.forecast_horizon <= 30):
            raise HTTPException(
                status_code=400,
                detail="Forecast horizon must be between 1 and 30 days"
            )
        
        logger.info(f"Generating prediction for store {store_nbr}, family {family}, horizon {request.forecast_horizon}")
        
        prediction = data_service.generate_prediction(
            store_nbr, family, request.forecast_horizon
        )
        
        logger.info(f"Prediction completed: RMSLE={prediction.test_rmsle:.4f}, Model={prediction.selected_model_name}")
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary():
    """Get summary statistics for dashboard display"""
    try:
        summary = data_service.get_dashboard_summary()
        logger.info("Dashboard summary generated successfully")
        return summary
    except Exception as e:
        logger.error(f"Dashboard summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/performance")
async def get_model_performance():
    """Get model performance comparison across all evaluation cases"""
    try:
        cases = data_service.get_evaluation_cases()
        
        performance_summary = {
            "methodology": "Pattern-Based Selection (Phase 3.6)",
            "total_evaluation_cases": len(cases),
            "baselines": {
                "traditional_phase_2": config.traditional_baseline,
                "neural_phase_3": config.neural_baseline
            },
            "champion_result": 0.4190,  # Phase 3.6 champion RMSLE
            "improvement_vs_traditional": ((config.traditional_baseline - 0.4190) / config.traditional_baseline) * 100,
            "improvement_vs_neural": ((config.neural_baseline - 0.4190) / config.neural_baseline) * 100,
            "model_routing": {
                "threshold": config.pattern_threshold,
                "regular_patterns": "Neural Models (LSTM variants)",
                "volatile_patterns": "Traditional Models (ARIMA, Exponential Smoothing)"
            }
        }
        
        return performance_summary
        
    except Exception as e:
        logger.error(f"Performance summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Store Sales Forecasting API - Phase 6",
        "methodology": "Pattern-Based Selection with Adaptive Model Routing",
        "version": "6.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "cases": "/cases",
            "analysis": "/analysis/{store_nbr}/{family}",
            "predict": "/predict/{store_nbr}/{family}",
            "dashboard": "/dashboard/summary",
            "performance": "/models/performance"
        },
        "streamlit_dashboard": f"http://{config.streamlit_host}:{config.streamlit_port}"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI server...")
    logger.info(f"API URL: {config.api_url}")
    logger.info(f"Project root: {config.project_root}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info"
    )