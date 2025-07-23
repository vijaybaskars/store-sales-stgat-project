#!/usr/bin/env python3
"""
Debug script for Store 44 - SCHOOL AND OFFICE SUPPLIES timeout issue
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, 'src')

def debug_store44():
    """Debug Store 44 prediction performance"""
    
    print("🔍 Debugging Store 44 - SCHOOL AND OFFICE SUPPLIES")
    print("=" * 60)
    
    # Test imports first
    try:
        from serving.data_service import data_service
        print("✅ Data service imported")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    store_nbr = 44
    family = "SCHOOL AND OFFICE SUPPLIES"
    
    # Step 1: Pattern analysis (should be fast)
    print(f"\n1. Pattern Analysis for Store {store_nbr} - {family}")
    start_time = time.time()
    
    try:
        analysis = data_service.analyze_pattern(store_nbr, family)
        analysis_time = time.time() - start_time
        
        print(f"   ✅ Pattern analysis completed in {analysis_time:.2f}s")
        print(f"   Pattern type: {analysis.pattern_type}")
        print(f"   CV: {analysis.coefficient_variation:.3f}")
        print(f"   Recommended model: {analysis.recommended_model_type}")
    except Exception as e:
        print(f"   ❌ Pattern analysis failed: {e}")
        return
    
    # Step 2: Data preparation (check data quality)
    print(f"\n2. Data Preparation Check")
    start_time = time.time()
    
    try:
        # Access the pattern selector to check data
        selector = data_service.pattern_selector
        train_series, test_series = selector.traditional_evaluator.prepare_case_data(store_nbr, family)
        
        prep_time = time.time() - start_time
        print(f"   ✅ Data preparation completed in {prep_time:.2f}s")
        print(f"   Train data length: {len(train_series)}")
        print(f"   Test data length: {len(test_series)}")
        print(f"   Train data range: {train_series.min():.2f} to {train_series.max():.2f}")
        
    except Exception as e:
        print(f"   ❌ Data preparation failed: {e}")
        return
    
    # Step 3: Quick traditional model test (this is likely the bottleneck)
    print(f"\n3. Traditional Model Test (ARIMA - likely bottleneck)")
    start_time = time.time()
    
    try:
        # Test just the traditional evaluator
        traditional_results = selector.traditional_evaluator.evaluate_case(store_nbr, family, 15)
        
        trad_time = time.time() - start_time
        print(f"   ✅ Traditional models completed in {trad_time:.2f}s")
        print(f"   Models evaluated: {list(traditional_results.keys())}")
        
        # Find best result
        if traditional_results:
            best_model = min(traditional_results.items(), key=lambda x: x[1].test_rmsle)
            print(f"   Best model: {best_model[0]} (RMSLE: {best_model[1].test_rmsle:.4f})")
        
    except Exception as e:
        print(f"   ❌ Traditional model evaluation failed: {e}")
        print(f"   Time taken: {time.time() - start_time:.2f}s")
        
        # This is likely where the timeout occurs
        print("\n💡 DIAGNOSIS: Traditional model fitting (especially ARIMA) is timing out")
        print("   Store 44 has volatile pattern (CV=2.766) which routes to traditional models")
        print("   ARIMA optimization can be very slow for complex time series")
        return
    
    print(f"\n4. Full Adaptive Evaluation (if we get here)")
    start_time = time.time()
    
    try:
        # This should work now if traditional models completed
        result = data_service.generate_prediction(store_nbr, family, 15)
        full_time = time.time() - start_time
        
        print(f"   ✅ Full evaluation completed in {full_time:.2f}s")
        print(f"   Final RMSLE: {result.test_rmsle:.4f}")
        print(f"   Selected model: {result.selected_model_name}")
        
    except Exception as e:
        print(f"   ❌ Full evaluation failed: {e}")
        print(f"   Time taken: {time.time() - start_time:.2f}s")
    
    print("\n=" * 60)
    print("🏁 Debug complete")


if __name__ == "__main__":
    debug_store44()