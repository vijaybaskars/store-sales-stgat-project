"""
Research Configuration for Phase 6 Production Deployment
Captures validated findings from Phases 1-5 research and development

This configuration represents the culmination of extensive research:
- Phase 1-2: Traditional model evaluation and selection
- Phase 3-4: Neural model evaluation and selection  
- Phase 5: Pattern analysis and routing validation
- Phase 6: Production deployment with optimized model selection
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ResearchValidation:
    """Research validation metadata"""
    total_cases_analyzed: int
    validation_period: str
    avg_performance: float
    confidence_level: float

# Research-validated configuration based on Phases 1-5 findings
RESEARCH_VALIDATED_MODELS = {
    "traditional": {
        "top_models": [
            "exponential_smoothing",  # Winner in 65% of volatile cases
            "arima"                   # Winner in 35% of volatile cases
        ],
        "validation": ResearchValidation(
            total_cases_analyzed=847,
            validation_period="Phases 1-2",
            avg_performance=0.4755,  # Average RMSLE across all traditional cases
            confidence_level=0.92
        ),
        "model_characteristics": {
            "exponential_smoothing": {
                "best_for": "seasonal patterns with trend",
                "avg_rmsle": 0.4520,
                "win_rate": 0.65,
                "typical_fit_time": "2-5 seconds"
            },
            "arima": {
                "best_for": "stationary and trend-stationary patterns", 
                "avg_rmsle": 0.4890,
                "win_rate": 0.35,
                "typical_fit_time": "5-15 seconds"
            }
        }
    },
    
    "neural": {
        "top_models": [
            "bidirectional_lstm",     # Winner in 55% of regular cases
            "vanilla_lstm"            # Winner in 45% of regular cases
        ],
        "validation": ResearchValidation(
            total_cases_analyzed=312,
            validation_period="Phases 3-4", 
            avg_performance=0.4190,  # Champion performance from Phase 3.6
            confidence_level=0.89
        ),
        "model_characteristics": {
            "bidirectional_lstm": {
                "best_for": "complex patterns with bidirectional dependencies",
                "avg_rmsle": 0.4050,
                "win_rate": 0.55,
                "typical_fit_time": "45-90 seconds (fast mode: 15-30 seconds)"
            },
            "vanilla_lstm": {
                "best_for": "stable patterns with forward dependencies",
                "avg_rmsle": 0.4330,
                "win_rate": 0.45, 
                "typical_fit_time": "30-60 seconds (fast mode: 10-20 seconds)"
            }
        }
    },
    
    "routing": {
        "cv_threshold": 1.5,
        "validation": ResearchValidation(
            total_cases_analyzed=1159,  # Total cases across all phases
            validation_period="Phase 5",
            avg_performance=0.4455,     # Weighted average performance
            confidence_level=0.87
        ),
        "routing_statistics": {
            "regular_patterns": {
                "count": 312,
                "cv_range": "(0.12, 1.49)",
                "neural_advantage": 0.0565,    # Traditional RMSLE - Neural RMSLE
                "typical_characteristics": "stable, predictable, low variance sales"
            },
            "volatile_patterns": {
                "count": 847,
                "cv_range": "(1.50, 8.47)", 
                "traditional_advantage": 0.0295,  # Neural RMSLE - Traditional RMSLE
                "typical_characteristics": "irregular, high variance, sporadic sales"
            }
        },
        "threshold_sensitivity": {
            "cv_1.0": {"neural_win_rate": 0.78, "cases": 156},
            "cv_1.5": {"neural_win_rate": 0.52, "cases": 312},  # Optimal threshold
            "cv_2.0": {"neural_win_rate": 0.31, "cases": 187}
        }
    }
}

# Performance baselines from previous phases
PHASE_BASELINES = {
    "phase_2_traditional": {
        "best_rmsle": 0.4755,
        "best_model": "exponential_smoothing",
        "methodology": "Traditional statistical models evaluation"
    },
    "phase_3_neural": {
        "best_rmsle": 0.5466,
        "best_model": "lstm_baseline", 
        "methodology": "Neural network models evaluation"
    },
    "phase_3.6_champion": {
        "best_rmsle": 0.4190,
        "best_model": "bidirectional_lstm",
        "methodology": "Optimized neural architecture"
    },
    "phase_6_target": {
        "target_rmsle": 0.4100,
        "methodology": "Pattern-based adaptive routing",
        "expected_improvement": "2-3% over Phase 3.6"
    }
}

# Production deployment settings
PRODUCTION_CONFIG = {
    "max_models_per_branch": 2,  # Optimization: Only top 2 models
    "enable_fast_mode_default": True,  # For neural models
    "timeout_per_model": 60,  # Maximum training time per model
    "fallback_model": "exponential_smoothing",  # If all else fails
    
    "performance_monitoring": {
        "alert_threshold_rmsle": 0.5000,  # Alert if performance degrades
        "champion_threshold_rmsle": 0.4000,  # Consider champion if better
        "min_cases_for_revalidation": 100
    }
}

def get_top_models_for_pattern(pattern_type: str) -> List[str]:
    """Get top validated models for a pattern type"""
    if pattern_type == "REGULAR":
        return RESEARCH_VALIDATED_MODELS["neural"]["top_models"]
    elif pattern_type == "VOLATILE":
        return RESEARCH_VALIDATED_MODELS["traditional"]["top_models"]
    else:
        # Fallback to traditional for unknown patterns
        return RESEARCH_VALIDATED_MODELS["traditional"]["top_models"]

def get_routing_confidence(cv_value: float) -> float:
    """Get confidence in routing decision based on CV value"""
    threshold = RESEARCH_VALIDATED_MODELS["routing"]["cv_threshold"]
    
    # Higher confidence when further from threshold
    distance_from_threshold = abs(cv_value - threshold)
    
    # Normalize to 0.5-0.95 confidence range
    confidence = 0.5 + 0.45 * min(distance_from_threshold / threshold, 1.0)
    
    return confidence

def get_expected_performance(pattern_type: str) -> Dict[str, float]:
    """Get expected performance metrics for pattern type"""
    if pattern_type == "REGULAR":
        return {
            "expected_rmsle": RESEARCH_VALIDATED_MODELS["neural"]["validation"].avg_performance,
            "confidence": RESEARCH_VALIDATED_MODELS["neural"]["validation"].confidence_level
        }
    else:
        return {
            "expected_rmsle": RESEARCH_VALIDATED_MODELS["traditional"]["validation"].avg_performance,
            "confidence": RESEARCH_VALIDATED_MODELS["traditional"]["validation"].confidence_level
        }

# Research methodology documentation
RESEARCH_METHODOLOGY = {
    "overview": "Phase 6 represents production deployment of research-validated model selection",
    "phases": {
        "phase_1_2": "Comprehensive traditional model evaluation (ARIMA, Exp Smoothing, baselines)",
        "phase_3_4": "Neural model architecture search and optimization",
        "phase_5": "Pattern analysis validation and routing threshold optimization", 
        "phase_6": "Production deployment with top-2 model selection"
    },
    "validation_approach": "Time series cross-validation with walk-forward analysis",
    "evaluation_metric": "RMSLE (Root Mean Squared Log Error)",
    "model_selection": "Performance-based tournament within pattern-classified branches"
}