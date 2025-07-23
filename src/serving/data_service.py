"""
Data service layer for Phase 6 serving
Handles data loading, caching, and integration with existing models
"""

import sys
from pathlib import Path

# Add src to path for imports FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply numpy compatibility fixes BEFORE any other imports
import numpy_compat  # This must be first

import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import logging

from data.evaluation_cases import EvaluationCaseManager, load_evaluation_cases
from models.pattern_selection import PatternBasedSelector
from serving.config import config
from serving.models import (
    EvaluationCase, PatternAnalysisResponse, PredictionResponse,
    ModelPerformanceMetrics, DashboardSummary
)

logger = logging.getLogger(__name__)


class Phase6DataService:
    """Data service for Phase 6 serving layer"""
    
    def __init__(self):
        self.config = config
        self._case_manager: Optional[EvaluationCaseManager] = None
        self._pattern_selector: Optional[PatternBasedSelector] = None
        self._evaluation_cases: Optional[List[Dict]] = None
        self._sales_data: Optional[pd.DataFrame] = None
        
    @property
    def case_manager(self) -> EvaluationCaseManager:
        """Lazy-loaded case manager"""
        if self._case_manager is None:
            self._case_manager = EvaluationCaseManager(str(self.config.evaluation_cases_path))
        return self._case_manager
    
    @property 
    def pattern_selector(self) -> PatternBasedSelector:
        """Lazy-loaded pattern selector"""
        if self._pattern_selector is None:
            self._pattern_selector = PatternBasedSelector(
                evaluation_case_manager=self.case_manager,
                pattern_threshold=self.config.pattern_threshold
            )
        return self._pattern_selector
    
    @property
    def sales_data(self) -> pd.DataFrame:
        """Lazy-loaded sales data"""
        if self._sales_data is None:
            try:
                self._sales_data = pd.read_csv(self.config.data_path)
                self._sales_data['date'] = pd.to_datetime(self._sales_data['date'])
                logger.info(f"Loaded sales data: {len(self._sales_data):,} records")
            except Exception as e:
                logger.error(f"Failed to load sales data: {e}")
                # Create empty DataFrame as fallback
                self._sales_data = pd.DataFrame()
        return self._sales_data
    
    def get_evaluation_cases(self) -> List[EvaluationCase]:
        """Get all evaluation cases"""
        try:
            cases = load_evaluation_cases(str(self.config.evaluation_cases_path))
            
            evaluation_cases = []
            for case in cases:
                evaluation_cases.append(EvaluationCase(
                    case_id=case.get('case_id', f"store_{case['store_nbr']}_family_{case['family']}"),
                    store_nbr=case['store_nbr'],
                    family=case['family'],
                    quality_score=case.get('selection_metrics', {}).get('quality_score', 0.0),
                    selection_metrics=case.get('selection_metrics', {})
                ))
            
            return evaluation_cases
            
        except Exception as e:
            logger.error(f"Failed to load evaluation cases: {e}")
            return []
    
    def analyze_pattern(self, store_nbr: int, family: str) -> PatternAnalysisResponse:
        """Analyze time series pattern for store-family combination"""
        try:
            pattern_analysis = self.pattern_selector.analyze_pattern(store_nbr, family)
            
            recommended_model = self.pattern_selector.select_optimal_model(pattern_analysis)
            
            return PatternAnalysisResponse(
                store_nbr=pattern_analysis.store_nbr,
                family=pattern_analysis.family,
                coefficient_variation=pattern_analysis.coefficient_variation,
                pattern_type=pattern_analysis.pattern_type,
                confidence_score=pattern_analysis.confidence_score,
                zero_sales_percentage=pattern_analysis.zero_sales_percentage,
                seasonal_strength=pattern_analysis.seasonal_strength,
                trend_strength=pattern_analysis.trend_strength,
                raw_metrics=pattern_analysis.raw_metrics,
                recommended_model_type=recommended_model
            )
            
        except Exception as e:
            logger.error(f"Pattern analysis failed for store {store_nbr}, family {family}: {e}")
            raise
    
    def generate_prediction(self, store_nbr: int, family: str, 
                          forecast_horizon: int = 15, fast_mode: bool = False) -> PredictionResponse:
        """Generate prediction using adaptive model selection"""
        try:
            logger.info(f"Starting prediction for Store {store_nbr}, Family {family} (fast_mode: {fast_mode})")
            
            # Use faster configuration for demo purposes if requested
            if fast_mode:
                logger.info("Using fast mode - simplified model training")
            
            # Pass fast_mode and production_mode to pattern selector for optimization
            result = self.pattern_selector.evaluate_case_adaptive(
                store_nbr, family, forecast_horizon, fast_mode=fast_mode, production_mode=True
            )
            
            if result is None:
                raise ValueError("Prediction generation failed")
            
            # Convert pattern analysis
            pattern_response = PatternAnalysisResponse(
                store_nbr=result.pattern_analysis.store_nbr,
                family=result.pattern_analysis.family,
                coefficient_variation=result.pattern_analysis.coefficient_variation,
                pattern_type=result.pattern_analysis.pattern_type,
                confidence_score=result.pattern_analysis.confidence_score,
                zero_sales_percentage=result.pattern_analysis.zero_sales_percentage,
                seasonal_strength=result.pattern_analysis.seasonal_strength,
                trend_strength=result.pattern_analysis.trend_strength,
                raw_metrics=result.pattern_analysis.raw_metrics,
                recommended_model_type=result.selected_model_type
            )
            
            # Performance summary
            performance_summary = {
                "rmsle_rank": self._calculate_rmsle_rank(result.test_rmsle),
                "baseline_comparison": {
                    "traditional": {
                        "baseline": self.config.traditional_baseline,
                        "improvement": result.improvement_vs_traditional_baseline,
                        "beats_baseline": result.beats_traditional
                    },
                    "neural": {
                        "baseline": self.config.neural_baseline,
                        "improvement": result.improvement_vs_neural_baseline,
                        "beats_baseline": result.beats_neural
                    }
                },
                "model_selection": {
                    "selected_type": result.selected_model_type,
                    "selected_name": result.selected_model_name,
                    "pattern_confidence": pattern_response.confidence_score
                }
            }
            
            return PredictionResponse(
                store_nbr=result.pattern_analysis.store_nbr,
                family=result.pattern_analysis.family,
                pattern_analysis=pattern_response,
                selected_model_type=result.selected_model_type,
                selected_model_name=result.selected_model_name,
                test_rmsle=result.test_rmsle,
                test_mae=result.test_mae,
                test_mape=result.test_mape,
                predictions=result.predictions,
                actuals=result.actuals,
                model_params=result.model_params,
                fit_time=result.fit_time,
                improvement_vs_traditional_baseline=result.improvement_vs_traditional_baseline,
                improvement_vs_neural_baseline=result.improvement_vs_neural_baseline,
                beats_traditional=result.beats_traditional,
                beats_neural=result.beats_neural,
                performance_summary=performance_summary
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for store {store_nbr}, family {family}: {e}")
            raise
    
    def get_dashboard_summary(self) -> DashboardSummary:
        """Get summary statistics for dashboard"""
        try:
            cases = self.get_evaluation_cases()
            
            # Initialize counters
            total_cases = len(cases)
            neural_routing = 0
            traditional_routing = 0
            champion_cases = 0
            
            # Performance distribution
            rmsle_ranges = {"< 0.4": 0, "0.4-0.45": 0, "0.45-0.5": 0, "> 0.5": 0}
            
            # Top performing cases (placeholder)
            top_cases = []
            
            # If we have cached results, use them
            try:
                with open(self.config.pattern_results_path, 'r') as f:
                    cached_results = json.load(f)
                    
                if 'performance_summary' in cached_results:
                    summary = cached_results['performance_summary']
                    champion_cases = summary.get('cases_beat_both', 0)
                    
                if 'routing_decisions' in cached_results:
                    routing = cached_results['routing_decisions']
                    neural_routing = routing.get('NEURAL', 0)
                    traditional_routing = routing.get('TRADITIONAL', 0)
                    
            except (FileNotFoundError, json.JSONDecodeError):
                # Use default values if no cached results
                pass
            
            return DashboardSummary(
                total_evaluation_cases=total_cases,
                cases_evaluated=total_cases,
                average_rmsle=0.4190,  # Phase 3.6 champion result
                neural_routing_count=neural_routing,
                traditional_routing_count=traditional_routing,
                champion_cases=champion_cases,
                performance_distribution=rmsle_ranges,
                top_performing_cases=top_cases[:5]
            )
            
        except Exception as e:
            logger.error(f"Dashboard summary generation failed: {e}")
            # Return minimal summary
            return DashboardSummary(
                total_evaluation_cases=0,
                cases_evaluated=0,
                average_rmsle=0.0,
                neural_routing_count=0,
                traditional_routing_count=0,
                champion_cases=0,
                performance_distribution={"< 0.4": 0, "0.4-0.45": 0, "0.45-0.5": 0, "> 0.5": 0},
                top_performing_cases=[]
            )
    
    def _calculate_rmsle_rank(self, rmsle: float) -> str:
        """Calculate performance rank based on RMSLE"""
        if rmsle < 0.4:
            return "Excellent"
        elif rmsle < 0.45:
            return "Very Good"
        elif rmsle < 0.5:
            return "Good"
        else:
            return "Needs Improvement"
    
    def validate_store_family(self, store_nbr: int, family: str) -> bool:
        """Validate if store-family combination exists in evaluation cases"""
        cases = self.get_evaluation_cases()
        return any(case.store_nbr == store_nbr and case.family == family for case in cases)


# Global data service instance
data_service = Phase6DataService()