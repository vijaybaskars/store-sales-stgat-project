"""
Flask backend for Phase 6 Store Sales Forecasting
Alternative to FastAPI using existing Flask dependency
"""

import sys
from pathlib import Path

# Add src to path for imports FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply numpy compatibility fixes BEFORE any other imports
import numpy_compat  # This must be first

import json
import logging
from datetime import datetime
from typing import Dict, Any

from flask import Flask, jsonify, request

from serving.data_service import data_service
from serving.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    
    # Handle numpy integers
    if isinstance(obj, np.integer):
        return int(obj)
    
    # Handle numpy floats  
    elif isinstance(obj, np.floating):
        return float(obj)
    
    # Handle numpy booleans (version-safe way)
    elif hasattr(np, 'bool_') and isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):  # Regular Python bool
        return bool(obj)
    
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle containers recursively
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    
    # Handle numpy scalars with .item() method
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    
    # Handle numpy dtypes that might be problematic
    elif hasattr(obj, 'dtype') and hasattr(obj, 'shape') and obj.shape == ():
        # This is a 0-dimensional numpy array (scalar)
        return obj.item()
    
    # Return as-is if it's already a basic Python type
    return obj

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Simple CORS headers for Streamlit integration
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


def create_error_response(error: str, detail: str, status_code: int = 500):
    """Create standardized error response"""
    return jsonify({
        "error": error,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }), status_code


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {e}")
    return create_error_response("Internal Server Error", str(e), 500)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Validate configuration
        validation = config.validate_paths()
        
        # Check if key paths exist
        missing_paths = [name for name, info in validation.items() if not info["exists"]]
        
        if missing_paths:
            logger.warning(f"Missing paths: {missing_paths}")
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "Phase 6.0 (Flask)"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return create_error_response("Health Check Failed", str(e), 500)


@app.route('/config/validate', methods=['GET'])
def validate_configuration():
    """Validate system configuration and paths"""
    try:
        validation = config.validate_paths()
        return jsonify({
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
        })
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return create_error_response("Configuration Validation Failed", str(e), 500)


@app.route('/cases', methods=['GET'])
def get_evaluation_cases():
    """Get all available evaluation cases"""
    try:
        cases = data_service.get_evaluation_cases()
        
        # Convert to dict format
        cases_data = []
        for case in cases:
            cases_data.append({
                "case_id": case.case_id,
                "store_nbr": case.store_nbr,
                "family": case.family,
                "quality_score": case.quality_score,
                "selection_metrics": case.selection_metrics
            })
        
        metadata = {
            "total_cases": len(cases_data),
            "data_source": str(config.evaluation_cases_path),
            "pattern_threshold": config.pattern_threshold
        }
        
        return jsonify({
            "cases": cases_data,
            "total_cases": len(cases_data),
            "metadata": metadata
        })
    except Exception as e:
        logger.error(f"Failed to get evaluation cases: {e}")
        return create_error_response("Failed to Load Cases", str(e), 500)


@app.route('/analysis/<int:store_nbr>/<family>', methods=['GET'])
def analyze_pattern(store_nbr: int, family: str):
    """Analyze time series pattern for store-family combination"""
    try:
        # Validate store-family combination
        if not data_service.validate_store_family(store_nbr, family):
            return create_error_response(
                "Not Found",
                f"Store {store_nbr} - Family '{family}' not found in evaluation cases",
                404
            )
        
        analysis = data_service.analyze_pattern(store_nbr, family)
        
        # Convert to dict with proper JSON serialization  
        analysis_dict = {
            "store_nbr": int(analysis.store_nbr),
            "family": str(analysis.family),
            "coefficient_variation": float(analysis.coefficient_variation),
            "pattern_type": str(analysis.pattern_type),
            "confidence_score": float(analysis.confidence_score),
            "zero_sales_percentage": float(analysis.zero_sales_percentage),
            "seasonal_strength": float(analysis.seasonal_strength),
            "trend_strength": float(analysis.trend_strength),
            "raw_metrics": convert_numpy_types(analysis.raw_metrics),
            "recommended_model_type": str(analysis.recommended_model_type)
        }
        
        logger.info(f"Pattern analysis completed for store {store_nbr}, family {family}")
        return jsonify(analysis_dict)
        
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        return create_error_response("Pattern Analysis Failed", str(e), 500)


@app.route('/predict/<int:store_nbr>/<family>', methods=['POST'])
def generate_prediction(store_nbr: int, family: str):
    """Generate predictions using adaptive model selection"""
    try:
        # Get request data
        request_data = request.get_json() or {}
        forecast_horizon = request_data.get('forecast_horizon', 15)
        
        # Validate store-family combination
        if not data_service.validate_store_family(store_nbr, family):
            return create_error_response(
                "Not Found",
                f"Store {store_nbr} - Family '{family}' not found in evaluation cases",
                404
            )
        
        # Validate forecast horizon
        if not (1 <= forecast_horizon <= 30):
            return create_error_response(
                "Bad Request",
                "Forecast horizon must be between 1 and 30 days",
                400
            )
        
        # Check for fast mode request
        fast_mode = request_data.get('fast_mode', False)
        
        logger.info(f"Generating prediction for store {store_nbr}, family {family}, horizon {forecast_horizon}, fast_mode {fast_mode}")
        
        # Generate prediction (no signal-based timeout in Flask threads)
        prediction = data_service.generate_prediction(
            store_nbr, family, forecast_horizon, fast_mode
        )
        
        # Convert to dict format with proper JSON serialization
        prediction_dict = {
            "store_nbr": int(prediction.store_nbr),
            "family": str(prediction.family),
            "pattern_analysis": {
                "store_nbr": int(prediction.pattern_analysis.store_nbr),
                "family": str(prediction.pattern_analysis.family),
                "coefficient_variation": float(prediction.pattern_analysis.coefficient_variation),
                "pattern_type": str(prediction.pattern_analysis.pattern_type),
                "confidence_score": float(prediction.pattern_analysis.confidence_score),
                "zero_sales_percentage": float(prediction.pattern_analysis.zero_sales_percentage),
                "seasonal_strength": float(prediction.pattern_analysis.seasonal_strength),
                "trend_strength": float(prediction.pattern_analysis.trend_strength),
                "raw_metrics": convert_numpy_types(prediction.pattern_analysis.raw_metrics),
                "recommended_model_type": str(prediction.pattern_analysis.recommended_model_type)
            },
            "selected_model_type": str(prediction.selected_model_type),
            "selected_model_name": str(prediction.selected_model_name),
            "test_rmsle": float(prediction.test_rmsle),
            "test_mae": float(prediction.test_mae),
            "test_mape": float(prediction.test_mape),
            "predictions": convert_numpy_types(prediction.predictions),
            "actuals": convert_numpy_types(prediction.actuals),
            "model_params": convert_numpy_types(prediction.model_params),
            "fit_time": float(prediction.fit_time),
            "improvement_vs_traditional_baseline": float(prediction.improvement_vs_traditional_baseline),
            "improvement_vs_neural_baseline": float(prediction.improvement_vs_neural_baseline),
            "beats_traditional": bool(prediction.beats_traditional),
            "beats_neural": bool(prediction.beats_neural),
            "performance_summary": convert_numpy_types(prediction.performance_summary)
        }
        
        logger.info(f"Prediction completed: RMSLE={prediction.test_rmsle:.4f}, Model={prediction.selected_model_name}")
        
        # Final JSON serialization test with comprehensive error handling
        try:
            return jsonify(prediction_dict)
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # Try a more aggressive conversion
            import json
            try:
                json_str = json.dumps(prediction_dict, default=str)
                return json_str, 200, {'Content-Type': 'application/json'}
            except Exception as fallback_error:
                logger.error(f"Fallback serialization also failed: {fallback_error}")
                return create_error_response("JSON Serialization Failed", str(e), 500)
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        return create_error_response("Prediction Failed", str(e), 500)


@app.route('/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    """Get summary statistics for dashboard display"""
    try:
        summary = data_service.get_dashboard_summary()
        
        # Convert to dict
        summary_dict = {
            "total_evaluation_cases": summary.total_evaluation_cases,
            "cases_evaluated": summary.cases_evaluated,
            "average_rmsle": summary.average_rmsle,
            "neural_routing_count": summary.neural_routing_count,
            "traditional_routing_count": summary.traditional_routing_count,
            "champion_cases": summary.champion_cases,
            "performance_distribution": summary.performance_distribution,
            "top_performing_cases": summary.top_performing_cases
        }
        
        logger.info("Dashboard summary generated successfully")
        return jsonify(summary_dict)
    except Exception as e:
        logger.error(f"Dashboard summary generation failed: {e}")
        return create_error_response("Dashboard Summary Failed", str(e), 500)


@app.route('/models/performance', methods=['GET'])
def get_model_performance():
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
        
        return jsonify(performance_summary)
        
    except Exception as e:
        logger.error(f"Performance summary generation failed: {e}")
        return create_error_response("Performance Summary Failed", str(e), 500)


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        "message": "Store Sales Forecasting API - Phase 6",
        "methodology": "Pattern-Based Selection with Adaptive Model Routing",
        "version": "6.0.0 (Flask)",
        "endpoints": {
            "health": "/health",
            "cases": "/cases", 
            "analysis": "/analysis/{store_nbr}/{family}",
            "predict": "/predict/{store_nbr}/{family}",
            "dashboard": "/dashboard/summary",
            "performance": "/models/performance"
        },
        "streamlit_dashboard": f"http://{config.streamlit_host}:{config.streamlit_port}"
    })


if __name__ == "__main__":
    logger.info("Starting Flask server...")
    logger.info(f"API URL: {config.api_url}")
    logger.info(f"Project root: {config.project_root}")
    
    app.run(
        host=config.api_host,
        port=config.api_port,
        debug=False
    )