"""
Pattern-Based Model Selection for Store Sales Forecasting
Integrates with existing Phase 2 & Phase 3 infrastructure
Implements intelligent model routing based on time series patterns
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import time
from pathlib import Path

# Import existing model implementations
from .traditional import TraditionalBaselines, ModelResults
try:
    from .neural.neural_baselines_fixed import NeuralBaselines, NeuralModelResults
    NEURAL_AVAILABLE = True
except ImportError:
    print("⚠️ Neural models not available - using traditional models only")
    NEURAL_AVAILABLE = False

warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)
    
@dataclass
class PatternAnalysis:
    """Container for time series pattern analysis"""
    store_nbr: int
    family: str
    coefficient_variation: float
    zero_sales_percentage: float
    seasonal_strength: float
    trend_strength: float
    pattern_type: str  # "REGULAR" or "VOLATILE"
    confidence_score: float
    raw_metrics: Dict

@dataclass
class AdaptiveResults:
    """Container for adaptive model selection results"""
    pattern_analysis: PatternAnalysis
    selected_model_type: str  # "NEURAL" or "TRADITIONAL"
    selected_model_name: str
    test_rmsle: float
    test_mae: float
    test_mape: float
    predictions: List[float]
    actuals: List[float]
    model_params: Dict
    fit_time: float
    improvement_vs_neural_baseline: float
    improvement_vs_traditional_baseline: float
    beats_neural: bool
    beats_traditional: bool

class PatternBasedSelector:
    """
    Intelligent model selector that analyzes time series patterns
    and routes cases to optimal models (Neural vs Traditional)
    """
    
    def __init__(self, evaluation_case_manager=None, pattern_threshold: float = 1.5):
        """
        Initialize with existing infrastructure
        
        Args:
            evaluation_case_manager: Existing EvaluationCaseManager instance
            pattern_threshold: Coefficient of Variation threshold for classification
        """
        if evaluation_case_manager is None:
            import sys
            sys.path.insert(0, '.')
            from data import EvaluationCaseManager
            self.case_manager = EvaluationCaseManager()
        else:
            self.case_manager = evaluation_case_manager
            
        self.pattern_threshold = pattern_threshold
        self.results = {}
        self.pattern_cache = {}
        
        # Initialize model evaluators
        self.traditional_evaluator = TraditionalBaselines(evaluation_case_manager)
        
        # Check both global availability and configuration setting
        from serving.config import config
        neural_enabled = NEURAL_AVAILABLE and config.enable_neural_models
        
        if neural_enabled:
            self.neural_evaluator = NeuralBaselines(evaluation_case_manager)
            print("✅ Pattern-Based Selector initialized with Neural + Traditional models")
        else:
            self.neural_evaluator = None
            if not config.enable_neural_models:
                print("⚠️  Neural models disabled in configuration - using Traditional models only")
            else:
                print("✅ Pattern-Based Selector initialized with Traditional models only")
        
        print(f"   Pattern threshold: CV = {pattern_threshold}")
        if neural_enabled:
            print(f"   REGULAR patterns (CV < {pattern_threshold}) → Neural models")
            print(f"   VOLATILE patterns (CV ≥ {pattern_threshold}) → Traditional models")
        else:
            print(f"   ALL patterns → Traditional models (neural disabled)")
    
    def analyze_pattern(self, store_nbr: int, family: str) -> PatternAnalysis:
        """
        Analyze time series pattern characteristics for model selection
        
        Args:
            store_nbr: Store number
            family: Product family
            
        Returns:
            PatternAnalysis with classification and metrics
        """
        
        # Check cache first
        cache_key = f"{store_nbr}_{family}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        print(f"🔍 Analyzing pattern: Store {store_nbr}, {family}")
        
        try:
            # Load data using existing infrastructure
            train_series, test_series = self.traditional_evaluator.prepare_case_data(store_nbr, family)
            
            # Calculate pattern metrics
            sales_values = train_series.values
            
            # 1. Coefficient of Variation (primary metric)
            mean_sales = np.mean(sales_values)
            std_sales = np.std(sales_values)
            coefficient_variation = std_sales / mean_sales if mean_sales > 0 else float('inf')
            
            # 2. Zero sales percentage
            zero_sales_pct = (sales_values == 0).sum() / len(sales_values) * 100
            
            # 3. Seasonal strength (simplified)
            try:
                # Weekly seasonality check
                weekly_means = []
                for day_of_week in range(7):
                    day_values = sales_values[day_of_week::7]
                    if len(day_values) > 0:
                        weekly_means.append(np.mean(day_values))
                seasonal_strength = np.std(weekly_means) / np.mean(weekly_means) if len(weekly_means) > 0 else 0
            except:
                seasonal_strength = 0
            
            # 4. Trend strength (simplified)
            try:
                x = np.arange(len(sales_values))
                trend_coef = np.polyfit(x, sales_values, 1)[0]
                trend_strength = abs(trend_coef) / mean_sales if mean_sales > 0 else 0
            except:
                trend_strength = 0
            
            # Pattern classification
            if coefficient_variation < self.pattern_threshold:
                pattern_type = "REGULAR"
                confidence_score = min(1.0, (self.pattern_threshold - coefficient_variation) / self.pattern_threshold)
            else:
                pattern_type = "VOLATILE" 
                confidence_score = min(1.0, (coefficient_variation - self.pattern_threshold) / coefficient_variation)
            
            # Raw metrics for analysis
            raw_metrics = {
                'mean_sales': mean_sales,
                'std_sales': std_sales,
                'min_sales': np.min(sales_values),
                'max_sales': np.max(sales_values),
                'median_sales': np.median(sales_values),
                'skewness': float(pd.Series(sales_values).skew()),
                'kurtosis': float(pd.Series(sales_values).kurtosis()),
                'train_length': len(sales_values),
                'test_length': len(test_series)
            }
            
            pattern_analysis = PatternAnalysis(
                store_nbr=store_nbr,
                family=family,
                coefficient_variation=coefficient_variation,
                zero_sales_percentage=zero_sales_pct,
                seasonal_strength=seasonal_strength,
                trend_strength=trend_strength,
                pattern_type=pattern_type,
                confidence_score=confidence_score,
                raw_metrics=raw_metrics
            )
            
            # Cache result
            self.pattern_cache[cache_key] = pattern_analysis
            
            print(f"   CV: {coefficient_variation:.3f} → {pattern_type} (confidence: {confidence_score:.3f})")
            
            return pattern_analysis
            
        except Exception as e:
            print(f"   ❌ Pattern analysis failed: {e}")
            # Return default analysis
            return PatternAnalysis(
                store_nbr=store_nbr, family=family, coefficient_variation=float('inf'),
                zero_sales_percentage=100, seasonal_strength=0, trend_strength=0,
                pattern_type="VOLATILE", confidence_score=0, raw_metrics={}
            )
    
    def select_optimal_model(self, pattern_analysis: PatternAnalysis) -> str:
        """
        Select optimal model type based on pattern analysis
        
        Args:
            pattern_analysis: Pattern characteristics
            
        Returns:
            "NEURAL" or "TRADITIONAL"
        """
        
        # Simple rule-based selection
        if pattern_analysis.pattern_type == "REGULAR" and NEURAL_AVAILABLE and self.neural_evaluator is not None:
            return "NEURAL"
        else:
            return "TRADITIONAL"
    
    def evaluate_case_adaptive(self, store_nbr: int, family: str, 
                              forecast_horizon: int = 15, fast_mode: bool = False, production_mode: bool = True) -> Optional[AdaptiveResults]:
        """
        Evaluate single case using adaptive model selection
        
        Args:
            store_nbr: Store number
            family: Product family
            forecast_horizon: Prediction horizon
            fast_mode: Use faster training parameters for neural models
            production_mode: Use only top 2 research-validated models per branch
            
        Returns:
            AdaptiveResults with comprehensive evaluation
        """
        
        print(f"\n🎯 Adaptive Evaluation: Store {store_nbr} - {family}")
        if production_mode:
            print("   🏭 Production mode: Using research-validated top 2 models")
        start_time = time.time()
        
        try:
            # Step 1: Analyze pattern
            pattern_analysis = self.analyze_pattern(store_nbr, family)
            
            # Add research-backed confidence information
            from serving.research_config import get_routing_confidence, get_expected_performance
            routing_confidence = get_routing_confidence(pattern_analysis.coefficient_variation)
            expected_perf = get_expected_performance(pattern_analysis.pattern_type)
            
            print(f"   📊 Pattern: {pattern_analysis.pattern_type} (CV: {pattern_analysis.coefficient_variation:.3f})")
            print(f"   🎯 Routing confidence: {routing_confidence:.1%}")
            print(f"   📈 Expected RMSLE: {expected_perf['expected_rmsle']:.4f} (±{1-expected_perf['confidence']:.2f})")
            
            # Step 2: Select optimal model
            selected_model_type = self.select_optimal_model(pattern_analysis)
            
            print(f"   Selected: {selected_model_type} model")
            if fast_mode:
                print(f"   ⚡ Fast mode enabled - using optimized training parameters")
            
            # Step 3: Run evaluation with selected model
            if selected_model_type == "NEURAL" and self.neural_evaluator:
                try:
                    print(f"   🧠 Running neural evaluation...")
                    case_results = self.neural_evaluator.evaluate_case(store_nbr, family, forecast_horizon, fast_mode=fast_mode, production_mode=production_mode)
                    
                    # Get best neural result
                    if case_results:
                        best_result = min(case_results.items(), key=lambda x: x[1].test_rmsle)
                        selected_model_name, result = best_result
                        
                        # Convert to compatible format
                        test_rmsle = result.test_rmsle
                        test_mae = result.test_mae
                        test_mape = result.test_mape
                        predictions = result.predictions
                        actuals = result.actuals
                        model_params = result.model_params
                        model_fit_time = result.fit_time
                        print(f"   ✅ Neural model completed: {selected_model_name} (RMSLE: {test_rmsle:.4f})")
                    else:
                        print("   ❌ Neural evaluation returned empty results - falling back to traditional")
                        selected_model_type = "TRADITIONAL"
                        case_results = None
                except Exception as neural_error:
                    print(f"   ❌ Neural evaluation crashed: {neural_error}")
                    print("   🔄 Falling back to traditional models...")
                    selected_model_type = "TRADITIONAL"
                    case_results = None
            
            if selected_model_type == "TRADITIONAL" or not case_results:
                case_results = self.traditional_evaluator.evaluate_case(store_nbr, family, forecast_horizon, production_mode=production_mode)
                
                if case_results:
                    # Get best traditional result
                    best_result = min(case_results.items(), key=lambda x: x[1].test_rmsle)
                    selected_model_name, result = best_result
                    
                    test_rmsle = result.test_rmsle
                    test_mae = result.test_mae
                    test_mape = result.test_mape
                    predictions = result.predictions
                    actuals = result.actuals
                    model_params = result.model_params
                    model_fit_time = result.fit_time
                else:
                    print("   ❌ Traditional evaluation also failed")
                    return None
            
            # Step 4: Calculate improvement metrics (vs Phase 2 & 3 baselines)
            traditional_baseline = 0.4755  # From Phase 2 results
            neural_baseline = 0.5466  # From Phase 3 results
            
            improvement_vs_traditional = (traditional_baseline - test_rmsle) / traditional_baseline * 100
            improvement_vs_neural = (neural_baseline - test_rmsle) / neural_baseline * 100
            
            beats_traditional = test_rmsle < traditional_baseline
            beats_neural = test_rmsle < neural_baseline
            
            total_time = time.time() - start_time
            
            adaptive_result = AdaptiveResults(
                pattern_analysis=pattern_analysis,
                selected_model_type=selected_model_type,
                selected_model_name=selected_model_name,
                test_rmsle=test_rmsle,
                test_mae=test_mae,
                test_mape=test_mape,
                predictions=predictions,
                actuals=actuals,
                model_params=model_params,
                fit_time=model_fit_time,
                improvement_vs_neural_baseline=improvement_vs_neural,
                improvement_vs_traditional_baseline=improvement_vs_traditional,
                beats_neural=beats_neural,
                beats_traditional=beats_traditional
            )
            
            print(f"   ✅ RMSLE: {test_rmsle:.4f}")
            print(f"      vs Traditional: {improvement_vs_traditional:+.1f}% {'✅' if beats_traditional else '❌'}")
            print(f"      vs Neural: {improvement_vs_neural:+.1f}% {'✅' if beats_neural else '❌'}")
            print(f"      Total time: {total_time:.1f}s")
            
            return adaptive_result
            
        except Exception as e:
            print(f"   ❌ Adaptive evaluation failed: {type(e).__name__}: {e}")
            print(f"   📍 Error occurred during pattern analysis or model evaluation for Store {store_nbr}, Family {family}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_all_cases(self, evaluation_cases: List[Dict]) -> Dict:
        """
        Evaluate all cases using adaptive model selection
        """
        
        print("🚀 Starting Adaptive Model Selection Evaluation")
        print(f"📊 Evaluating {len(evaluation_cases)} cases with pattern-based routing")
        print("=" * 60)
        
        all_results = {}
        pattern_classifications = {}
        routing_decisions = {"NEURAL": 0, "TRADITIONAL": 0}
        performance_summary = {
            'cases_evaluated': 0,
            'cases_beat_traditional': 0,
            'cases_beat_neural': 0,
            'cases_beat_both': 0,
            'average_rmsle': 0,
            'total_improvement_traditional': 0,
            'total_improvement_neural': 0
        }
        
        for i, case in enumerate(evaluation_cases, 1):
            store_nbr = case['store_nbr']
            family = case['family']
            case_key = f"store_{store_nbr}_family_{family}"
            
            print(f"\n[{i}/{len(evaluation_cases)}] Case: {case_key}")
            quality_score = case.get('selection_metrics', {}).get('quality_score', 0)
            print(f"    Quality Score: {quality_score:.1f}")
            
            try:
                result = self.evaluate_case_adaptive(store_nbr, family)
                
                if result:
                    all_results[case_key] = result
                    pattern_classifications[case_key] = result.pattern_analysis
                    routing_decisions[result.selected_model_type] += 1
                    
                    # Update summary stats
                    performance_summary['cases_evaluated'] += 1
                    if result.beats_traditional:
                        performance_summary['cases_beat_traditional'] += 1
                    if result.beats_neural:
                        performance_summary['cases_beat_neural'] += 1
                    if result.beats_traditional and result.beats_neural:
                        performance_summary['cases_beat_both'] += 1
                    
                    performance_summary['average_rmsle'] += result.test_rmsle
                    performance_summary['total_improvement_traditional'] += result.improvement_vs_traditional_baseline
                    performance_summary['total_improvement_neural'] += result.improvement_vs_neural_baseline
                    
                    print(f"    ✅ Completed successfully")
                    
                else:
                    print(f"    ❌ Evaluation failed")
                    
            except Exception as e:
                print(f"    ❌ Exception: {str(e)}")
                continue
        
        # Calculate final statistics
        n_cases = performance_summary['cases_evaluated']
        if n_cases > 0:
            performance_summary['average_rmsle'] /= n_cases
            performance_summary['avg_improvement_traditional'] = performance_summary['total_improvement_traditional'] / n_cases
            performance_summary['avg_improvement_neural'] = performance_summary['total_improvement_neural'] / n_cases
        
        print("\n" + "=" * 60)
        print("🎯 ADAPTIVE SELECTION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Cases Evaluated: {n_cases}/{len(evaluation_cases)}")
        print(f"Routing Decisions: {routing_decisions}")
        print(f"Beat Traditional Baseline: {performance_summary['cases_beat_traditional']}/{n_cases}")
        print(f"Beat Neural Baseline: {performance_summary['cases_beat_neural']}/{n_cases}")
        print(f"Beat Both Baselines: {performance_summary['cases_beat_both']}/{n_cases}")
        print(f"Average RMSLE: {performance_summary['average_rmsle']:.4f}")
        
        return {
            'detailed_results': all_results,
            'pattern_classifications': pattern_classifications,
            'routing_decisions': routing_decisions,
            'performance_summary': performance_summary,
            'evaluation_metadata': {
                'pattern_threshold': self.pattern_threshold,
                'neural_available': NEURAL_AVAILABLE,
                'total_cases': len(evaluation_cases),
                'successful_cases': n_cases,
                'traditional_baseline': 0.4755,
                'neural_baseline': 0.5466
            }
        }
    
    def save_results(self, results: Dict, base_filepath: str):
        """
        Save results to structured files following Phase 2/3 pattern
        """
        
        base_path = Path(base_filepath)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save detailed results
        detailed_results = {}
        for case_key, result in results['detailed_results'].items():
            detailed_results[case_key] = {
                'pattern_analysis': {
                    'store_nbr': result.pattern_analysis.store_nbr,
                    'family': result.pattern_analysis.family,
                    'coefficient_variation': result.pattern_analysis.coefficient_variation,
                    'pattern_type': result.pattern_analysis.pattern_type,
                    'confidence_score': result.pattern_analysis.confidence_score,
                    'zero_sales_percentage': result.pattern_analysis.zero_sales_percentage,
                    'seasonal_strength': result.pattern_analysis.seasonal_strength,
                    'trend_strength': result.pattern_analysis.trend_strength,
                    'raw_metrics': result.pattern_analysis.raw_metrics
                },
                'selected_model_type': result.selected_model_type,
                'selected_model_name': result.selected_model_name,
                'test_rmsle': result.test_rmsle,
                'test_mae': result.test_mae,
                'test_mape': result.test_mape,
                'predictions': result.predictions,
                'actuals': result.actuals,
                'model_params': result.model_params,
                'fit_time': result.fit_time,
                'improvement_vs_traditional_baseline': result.improvement_vs_traditional_baseline,
                'improvement_vs_neural_baseline': result.improvement_vs_neural_baseline,
                'beats_traditional': result.beats_traditional,
                'beats_neural': result.beats_neural
            }
        

        with open(base_path / 'adaptive_results.json', 'w') as f:
            json.dump({
                'detailed_results': detailed_results,
                'routing_decisions': results['routing_decisions'],
                'performance_summary': results['performance_summary'],
                'evaluation_metadata': results['evaluation_metadata']
                }, f, indent=2, cls=NumpyEncoder)

        # 2. Save pattern analysis separately
        pattern_data = {}
        for case_key, pattern in results['pattern_classifications'].items():
            pattern_data[case_key] = {
                'store_nbr': pattern.store_nbr,
                'family': pattern.family,
                'coefficient_variation': pattern.coefficient_variation,
                'pattern_type': pattern.pattern_type,
                'confidence_score': pattern.confidence_score,
                'zero_sales_percentage': pattern.zero_sales_percentage,
                'seasonal_strength': pattern.seasonal_strength,
                'trend_strength': pattern.trend_strength,
                'raw_metrics': pattern.raw_metrics
            }
        
        with open(base_path / 'pattern_analysis.json', 'w') as f:
            json.dump({
                'pattern_classifications': pattern_data,
                'threshold_used': self.pattern_threshold,
                'classification_summary': {
                    'regular_count': sum(1 for p in results['pattern_classifications'].values() if p.pattern_type == "REGULAR"),
                    'volatile_count': sum(1 for p in results['pattern_classifications'].values() if p.pattern_type == "VOLATILE")
                }
            }, f, indent=2)
        
        # 3. Save summary report
        summary_report = {
            'phase': 'Phase 3.6: Pattern-Based Model Selection',
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'methodology': {
                'pattern_metric': 'Coefficient of Variation',
                'threshold': self.pattern_threshold,
                'regular_model': 'Neural (LSTM variants)' if NEURAL_AVAILABLE else 'Not Available',
                'volatile_model': 'Traditional (ARIMA, etc.)',
                'baseline_comparisons': {
                    'traditional_baseline': 0.4755,
                    'neural_baseline': 0.5466
                }
            },
            'results_summary': results['performance_summary'],
            'routing_summary': results['routing_decisions'],
            'metadata': results['evaluation_metadata']
        }
        
        with open(base_path / 'summary_report.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print(f"\n💾 Results saved to: {base_path}")
        print(f"   ✅ adaptive_results.json - Detailed evaluation results")
        print(f"   ✅ pattern_analysis.json - Pattern classifications")
        print(f"   ✅ summary_report.json - Academic summary")