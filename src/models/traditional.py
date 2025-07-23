"""
Traditional Baseline Models for Store Sales Forecasting
Integrates with existing EvaluationCaseManager infrastructure
Implements ARIMA, Exponential Smoothing, and Simple Baselines
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import time

# Statistical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

@dataclass
class ModelResults:
    """Container for model results"""
    model_name: str
    store_nbr: int
    family: str
    train_rmsle: float
    test_rmsle: float
    test_mae: float
    test_mape: float
    predictions: List[float]
    actuals: List[float]
    model_params: Dict
    fit_time: float

class TraditionalBaselines:
    """
    Comprehensive traditional baseline models implementation
    Integrates with existing EvaluationCaseManager infrastructure
    """
    
    def __init__(self, evaluation_case_manager=None):
        """
        Initialize with existing EvaluationCaseManager
        
        Args:
            evaluation_case_manager: Existing EvaluationCaseManager instance
        """
        if evaluation_case_manager is None:
            import sys
            sys.path.insert(0, 'src')
            from data import EvaluationCaseManager
            self.case_manager = EvaluationCaseManager()
        else:
            self.case_manager = evaluation_case_manager
            
        self.results = {}
        self.sales_data = None
        self._load_sales_data()
        
    def _load_sales_data(self):
        """Load the sales data once for use with get_case_train_test_data"""
        try:
            print("📂 Loading sales data for traditional baselines...")
            
            # Try multiple possible paths
            possible_paths = [
                'data/raw/train.csv',           # If running from project root
                '../data/raw/train.csv',        # If running from notebooks/
                '../../data/raw/train.csv',     # If running from deeper directory
                './data/raw/train.csv'          # Alternative relative path
            ]
            
            sales_data_loaded = False
            for path in possible_paths:
                try:
                    self.sales_data = pd.read_csv(path)
                    self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
                    print(f"✅ Sales data loaded from: {path}")
                    print(f"   Records: {len(self.sales_data):,}")
                    sales_data_loaded = True
                    break
                except FileNotFoundError:
                    continue
                    
            if not sales_data_loaded:
                print(f"❌ Could not find train.csv in any of these locations:")
                for path in possible_paths:
                    print(f"   - {path}")
                self.sales_data = None
                
        except Exception as e:
            print(f"❌ Could not load sales data: {e}")
            self.sales_data = None
        
    def prepare_case_data(self, store_nbr: int, family: str) -> Tuple[pd.Series, pd.Series]:
        """Use existing get_case_train_test_data function with correct signature"""
        
        import sys
        sys.path.insert(0, 'src')
        from data import get_case_train_test_data
        
        if self.sales_data is None:
            raise ValueError("Sales data not available")
        
        # Use existing train/test split function with CORRECT signature
        train_data, test_data = get_case_train_test_data(self.sales_data, store_nbr, family)
        
        if train_data is None or test_data is None:
            raise ValueError(f"No data available for store {store_nbr}, family {family}")
        
        # Convert to time series if needed
        if isinstance(train_data, pd.DataFrame):
            train_series = train_data.set_index('date')['sales'] if 'date' in train_data.columns else train_data['sales']
        else:
            train_series = train_data
            
        if isinstance(test_data, pd.DataFrame):
            test_series = test_data.set_index('date')['sales'] if 'date' in test_data.columns else test_data['sales']
        else:
            test_series = test_data
            
        return train_series, test_series
    
    def calculate_metrics(self, y_true: np.array, y_pred: np.array) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        # Ensure positive values for RMSLE
        y_true_pos = np.maximum(y_true, 0)
        y_pred_pos = np.maximum(y_pred, 0)
        
        # RMSLE (primary metric)
        rmsle = np.sqrt(np.mean((np.log1p(y_pred_pos) - np.log1p(y_true_pos))**2))
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (handle division by zero)
        mape_values = []
        for true, pred in zip(y_true, y_pred):
            if abs(true) > 1e-6:  # Avoid division by very small numbers
                mape_values.append(abs((true - pred) / true))
        mape = np.mean(mape_values) * 100 if mape_values else float('inf')
        
        return {
            'rmsle': rmsle,
            'mae': mae,
            'mape': mape
        }
    
    def fit_arima(self, train_series: pd.Series, forecast_horizon: int = 15) -> Dict:
        """Fit ARIMA model with optimized parameter selection"""
        
        start_time = time.time()
        
        try:
            print("    🚀 Fast ARIMA fitting...")
            # Optimized Auto ARIMA - much faster parameter space
            model = auto_arima(
                train_series,
                start_p=0, start_q=0, max_p=3, max_q=3,  # Reduced from 5 to 3
                seasonal=True, start_P=0, start_Q=0, max_P=1, max_Q=1, m=7,  # Reduced from 2 to 1
                stepwise=True, suppress_warnings=True, error_action='ignore',
                trace=False, random_state=42,
                maxiter=50,  # Limit iterations
                n_jobs=1     # Single thread to avoid overhead
            )
            
            # Generate predictions
            forecast = model.predict(n_periods=forecast_horizon)
            
            # In-sample predictions for train metrics
            train_pred = model.predict_in_sample()
            
            fit_time = time.time() - start_time
            
            return {
                'model': model,
                'forecast': forecast,
                'train_predictions': train_pred,
                'params': model.get_params(),
                'fit_time': fit_time
            }
            
        except Exception as e:
            print(f"    ⚠️  Auto-ARIMA failed: {str(e)[:50]}...")
            # Progressive fallback strategy
            fallback_orders = [(1,1,1), (0,1,1), (1,0,1), (0,1,0)]
            
            for order in fallback_orders:
                try:
                    print(f"    🔄 Trying simple ARIMA{order}...")
                    model = ARIMA(train_series, order=order)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=forecast_horizon)
                    train_pred = fitted.fittedvalues
                    
                    fit_time = time.time() - start_time
                    print(f"    ✅ ARIMA{order} successful ({fit_time:.1f}s)")
                    
                    return {
                        'model': fitted,
                        'forecast': forecast,
                        'train_predictions': train_pred,
                        'params': {'order': order, 'fallback': True},
                        'fit_time': fit_time
                    }
                except Exception as fallback_error:
                    print(f"    ❌ ARIMA{order} failed")
                    continue
            
            print("    ❌ All ARIMA models failed")
            return None
    
    def fit_exponential_smoothing(self, train_series: pd.Series, forecast_horizon: int = 15) -> Dict:
        """Fit Exponential Smoothing (Holt-Winters) model"""
        
        start_time = time.time()
        
        try:
            # Try different configurations
            configs = [
                {'seasonal': 'add', 'seasonal_periods': 7},
                {'seasonal': 'mul', 'seasonal_periods': 7},
                {'seasonal': None, 'seasonal_periods': None}
            ]
            
            best_model = None
            best_aic = float('inf')
            
            for config in configs:
                try:
                    model = ExponentialSmoothing(
                        train_series,
                        trend='add',
                        seasonal=config['seasonal'],
                        seasonal_periods=config['seasonal_periods']
                    ).fit(optimized=True)
                    
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                        best_config = config
                except:
                    continue
            
            if best_model is None:
                return None
                
            # Generate forecast
            forecast = best_model.forecast(steps=forecast_horizon)
            train_pred = best_model.fittedvalues
            
            fit_time = time.time() - start_time
            
            return {
                'model': best_model,
                'forecast': forecast,
                'train_predictions': train_pred,
                'params': best_config,
                'fit_time': fit_time
            }
            
        except Exception as e:
            print(f"Exponential Smoothing failed: {str(e)}")
            return None
    
    def fit_simple_baselines(self, train_series: pd.Series, forecast_horizon: int = 15) -> Dict:
        """Fit simple baseline models"""
        
        start_time = time.time()
        
        results = {}
        
        # 1. Moving Average (7-day)
        ma_7 = train_series.rolling(window=7).mean().iloc[-1]
        results['moving_average_7'] = {
            'forecast': [ma_7] * forecast_horizon,
            'train_predictions': train_series.rolling(window=7).mean(),
            'params': {'window': 7},
            'fit_time': time.time() - start_time
        }
        
        # 2. Moving Average (14-day)
        ma_14 = train_series.rolling(window=14).mean().iloc[-1]
        results['moving_average_14'] = {
            'forecast': [ma_14] * forecast_horizon,
            'train_predictions': train_series.rolling(window=14).mean(),
            'params': {'window': 14},
            'fit_time': time.time() - start_time
        }
        
        # 3. Seasonal Naive (same day last week)
        if len(train_series) >= 7:
            seasonal_naive = train_series.iloc[-7:].values
            forecast = list(seasonal_naive) * (forecast_horizon // 7 + 1)
            forecast = forecast[:forecast_horizon]
            
            # Generate train predictions
            train_pred = []
            for i in range(len(train_series)):
                if i >= 7:
                    train_pred.append(train_series.iloc[i-7])
                else:
                    train_pred.append(train_series.iloc[0])
            
            results['seasonal_naive'] = {
                'forecast': forecast,
                'train_predictions': pd.Series(train_pred, index=train_series.index),
                'params': {'lag': 7},
                'fit_time': time.time() - start_time
            }
        
        # 4. Linear Trend
        x = np.arange(len(train_series))
        coeffs = np.polyfit(x, train_series.values, 1)
        trend_pred = np.polyval(coeffs, np.arange(len(train_series), len(train_series) + forecast_horizon))
        
        results['linear_trend'] = {
            'forecast': trend_pred.tolist(),
            'train_predictions': pd.Series(np.polyval(coeffs, x), index=train_series.index),
            'params': {'coefficients': coeffs.tolist()},
            'fit_time': time.time() - start_time
        }
        
        return results
    
    def evaluate_case(self, store_nbr: int, family: str, forecast_horizon: int = 15, production_mode: bool = True) -> Dict[str, ModelResults]:
        """
        Evaluate models for a single store-family case
        
        Args:
            production_mode: If True, use only top 2 research-validated models (Phase 6 optimization)
                           If False, use all models for research purposes
        """
        
        print(f"\n📊 Traditional Models Evaluation: Store {store_nbr} - {family}")
        if production_mode:
            print("  ⚡ Production mode: Using top 2 research-validated models")
        
        # Use existing data preparation infrastructure
        train_series, test_series = self.prepare_case_data(store_nbr, family)
        
        if len(test_series) < forecast_horizon:
            print(f"  ⚠️  Warning: Only {len(test_series)} test samples available")
            forecast_horizon = len(test_series)
        
        test_actual = test_series.iloc[:forecast_horizon].values
        case_results = {}
        
        # Import research configuration for production mode
        if production_mode:
            from serving.research_config import get_top_models_for_pattern
            top_models = ["exponential_smoothing", "arima"]  # Research-validated top 2
            print(f"  🎯 Training top models: {top_models}")
        
        # 1. Exponential Smoothing (Research-validated #1 performer)
        if not production_mode or "exponential_smoothing" in top_models:
            print("  📈 Fitting Exponential Smoothing...")
            exp_result = self.fit_exponential_smoothing(train_series, forecast_horizon)
            if exp_result:
                exp_pred = exp_result['forecast'][:forecast_horizon]
                train_metrics = self.calculate_metrics(train_series.values, exp_result['train_predictions'].values)
                test_metrics = self.calculate_metrics(test_actual, exp_pred)
                
                case_results['exponential_smoothing'] = ModelResults(
                    model_name='Exponential_Smoothing',
                    store_nbr=store_nbr,
                    family=family,
                    train_rmsle=train_metrics['rmsle'],
                    test_rmsle=test_metrics['rmsle'],
                    test_mae=test_metrics['mae'],
                    test_mape=test_metrics['mape'],
                    predictions=exp_pred.tolist(),
                    actuals=test_actual.tolist(),
                    model_params=exp_result['params'],
                    fit_time=exp_result['fit_time']
                )
                print(f"    ✅ Exp Smoothing - Test RMSLE: {test_metrics['rmsle']:.4f}")
        
        # 2. ARIMA (Research-validated #2 performer)
        if not production_mode or "arima" in top_models:
            print("  📊 Fitting ARIMA...")
            arima_result = self.fit_arima(train_series, forecast_horizon)
            if arima_result:
                arima_pred = arima_result['forecast'][:forecast_horizon]
                # Handle train predictions length mismatch
                train_pred_vals = arima_result['train_predictions'].values
                if len(train_pred_vals) != len(train_series):
                    train_pred_vals = train_pred_vals[-len(train_series):]
                
                train_metrics = self.calculate_metrics(train_series.values, train_pred_vals)
                test_metrics = self.calculate_metrics(test_actual, arima_pred)
                
                case_results['arima'] = ModelResults(
                    model_name='ARIMA',
                    store_nbr=store_nbr,
                    family=family,
                    train_rmsle=train_metrics['rmsle'],
                    test_rmsle=test_metrics['rmsle'],
                    test_mae=test_metrics['mae'],
                    test_mape=test_metrics['mape'],
                    predictions=arima_pred.tolist(),
                    actuals=test_actual.tolist(),
                    model_params=arima_result['params'],
                    fit_time=arima_result['fit_time']
                )
                print(f"    ✅ ARIMA - Test RMSLE: {test_metrics['rmsle']:.4f}")
        
        # Skip simple baselines in production mode (research showed they underperform)
        if not production_mode:
            print("  📋 Fitting Simple Baselines...")
            simple_results = self.fit_simple_baselines(train_series, forecast_horizon)
            
            for name, result in simple_results.items():
                pred = result['forecast'][:forecast_horizon]
                if isinstance(result['train_predictions'], pd.Series):
                    train_pred = result['train_predictions'].values
                else:
                    train_pred = result['train_predictions']
                
                # Handle train predictions with NaN values
                train_pred_clean = train_pred[-len(train_series):]
                train_pred_clean = train_pred_clean[~np.isnan(train_pred_clean)]
                train_actual_clean = train_series.values[-len(train_pred_clean):]
                
                if len(train_pred_clean) > 0:
                    train_metrics = self.calculate_metrics(train_actual_clean, train_pred_clean)
                else:
                    train_metrics = {'rmsle': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
                    
                test_metrics = self.calculate_metrics(test_actual, pred)
                
                case_results[name] = ModelResults(
                    model_name=name,
                    store_nbr=store_nbr,
                    family=family,
                    train_rmsle=train_metrics['rmsle'],
                    test_rmsle=test_metrics['rmsle'],
                    test_mae=test_metrics['mae'],
                    test_mape=test_metrics['mape'],
                    predictions=pred,
                    actuals=test_actual.tolist(),
                        model_params=result['params'],
                    fit_time=result['fit_time']
                )
                print(f"    ✅ {name} - Test RMSLE: {test_metrics['rmsle']:.4f}")
        
        return case_results
    
    def evaluate_all_cases(self, evaluation_cases: List[Dict]) -> Dict:
        """Evaluate all models on all evaluation cases"""
        
        print("🚀 Starting Traditional Baseline Evaluation")
        print(f"📊 Evaluating {len(evaluation_cases)} cases")
        print("=" * 60)
        
        all_results = {}
        summary_stats = {}
        
        for i, case in enumerate(evaluation_cases, 1):
            store_nbr = case['store_nbr']
            family = case['family']
            case_key = f"store_{store_nbr}_family_{family}"
            
            print(f"\n[{i}/{len(evaluation_cases)}] Case: {case_key}")
            quality_score = case.get('selection_metrics', {}).get('quality_score', 0)
            print(f"    Quality Score: {quality_score:.1f}")
            
            try:
                case_results = self.evaluate_case(store_nbr, family)
                all_results[case_key] = case_results
                print(f"    ✅ Completed successfully")
                
            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
                continue
        
        # Calculate summary statistics
        model_names = set()
        for case_results in all_results.values():
            model_names.update(case_results.keys())
        
        for model_name in model_names:
            rmsle_scores = []
            mae_scores = []
            
            for case_results in all_results.values():
                if model_name in case_results:
                    rmsle_scores.append(case_results[model_name].test_rmsle)
                    mae_scores.append(case_results[model_name].test_mae)
            
            summary_stats[model_name] = {
                'mean_rmsle': np.mean(rmsle_scores),
                'std_rmsle': np.std(rmsle_scores),
                'mean_mae': np.mean(mae_scores),
                'std_mae': np.std(mae_scores),
                'count': len(rmsle_scores)
            }
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY RESULTS")
        print("=" * 60)
        
        for model_name, stats in summary_stats.items():
            print(f"{model_name:20s} | RMSLE: {stats['mean_rmsle']:.4f} ± {stats['std_rmsle']:.4f} | Cases: {stats['count']}")
        
        return {
            'detailed_results': all_results,
            'summary_statistics': summary_stats,
            'evaluation_metadata': {
                'test_split_date': '2017-07-01',
                'forecast_horizon': 15,
                'total_cases': len(evaluation_cases),
                'successful_cases': len(all_results)
            }
        }
    
    def save_results(self, results: Dict, filepath: str):
        """Save results to JSON file"""
        
        # Convert ModelResults to dicts for JSON serialization
        serializable_results = {}
        
        for case_key, case_results in results['detailed_results'].items():
            serializable_results[case_key] = {}
            for model_name, model_result in case_results.items():
                serializable_results[case_key][model_name] = {
                    'model_name': model_result.model_name,
                    'store_nbr': model_result.store_nbr,
                    'family': model_result.family,
                    'train_rmsle': model_result.train_rmsle,
                    'test_rmsle': model_result.test_rmsle,
                    'test_mae': model_result.test_mae,
                    'test_mape': model_result.test_mape,
                    'predictions': model_result.predictions,
                    'actuals': model_result.actuals,
                    'model_params': model_result.model_params,
                    'fit_time': model_result.fit_time
                }
        
        final_results = {
            'detailed_results': serializable_results,
            'summary_statistics': results['summary_statistics'],
            'evaluation_metadata': results['evaluation_metadata']
        }
        
        with open(filepath, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")