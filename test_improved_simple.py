#!/usr/bin/env python3
"""
Simple test for improved STGAT system (no complex traditional models)
"""

import sys
sys.path.insert(0, 'src')

import json
import pandas as pd
import numpy as np
from data.evaluation_cases import EvaluationCaseManager, load_evaluation_cases

def test_improved_simple():
    """Test improved STGAT with simple traditional fallback"""
    
    print("🧪 TESTING IMPROVED STGAT (Simple Traditional Methods)")
    print("=" * 55)
    
    # Load infrastructure
    evaluation_cases = load_evaluation_cases()
    case_manager = EvaluationCaseManager()
    
    # Load pattern analysis
    try:
        with open('results/pattern_selection/pattern_analysis.json', 'r') as f:
            pattern_analysis = json.load(f)
        pattern_classifications = pattern_analysis['pattern_classifications']
        print("✅ Pattern analysis loaded")
    except FileNotFoundError:
        print("❌ Pattern analysis not found")
        return False
    
    # Initialize improved STGAT evaluator
    print("\n🔧 Initializing STGAT evaluator with improved fallback...")
    
    # Import after path setup
    from models.stgat import STGATEvaluator
    stgat_evaluator = STGATEvaluator(case_manager, cv_threshold=1.5)
    
    print("✅ STGAT evaluator initialized successfully")
    
    # Find a volatile case
    volatile_case = None
    for case in evaluation_cases:
        case_key = f"store_{case['store_nbr']}_family_{case['family']}"
        if case_key in pattern_classifications:
            cv_value = pattern_classifications[case_key].get('coefficient_variation', 2.0)
            if cv_value >= 1.5:  # Volatile pattern
                volatile_case = case
                break
    
    if volatile_case is None:
        print("❌ No volatile case found")
        return False
    
    store_nbr = volatile_case['store_nbr']
    family = volatile_case['family']
    case_key = f"store_{store_nbr}_family_{family}"
    cv_value = pattern_classifications[case_key].get('coefficient_variation', 2.0)
    
    print(f"\n🎯 Testing volatile case: Store {store_nbr} - {family}")
    print(f"   CV: {cv_value:.3f} (should route to improved traditional fallback)")
    
    # Run evaluation
    result = stgat_evaluator.evaluate_case(
        store_nbr=store_nbr,
        family=family,
        pattern_analysis=pattern_classifications,
        traditional_baseline=0.4755
    )
    
    # Display results
    print(f"\n📊 IMPROVED STGAT RESULT:")
    print(f"   RMSLE: {result['test_rmsle']:.4f}")
    print(f"   Method: {result.get('method_used', 'Unknown')}")
    print(f"   Confidence: {result.get('stgat_confidence', 0.0):.3f}")
    
    # Check if using improved traditional methods
    method_used = result.get('method_used', '')
    if 'Traditional_' in method_used and 'Constant' not in method_used:
        print(f"   ✅ SUCCESS: Using improved traditional method!")
        success = True
    elif 'Traditional_Fallback_Constant' == method_used:
        print(f"   ⚠️  Still using constant fallback")
        success = False
    else:
        print(f"   ❓ Using method: {method_used}")
        success = method_used == 'STGAT'  # STGAT is also success
    
    # Performance comparison
    vs_constant = ((0.4755 - result['test_rmsle']) / 0.4755) * 100
    print(f"   vs Constant baseline (0.4755): {vs_constant:+.1f}%")
    
    if success:
        print(f"\n🎉 IMPROVEMENT SUCCESS!")
        if 'STGAT' in method_used:
            print(f"   Using STGAT model for this case")
        else:
            print(f"   Using improved traditional fallback: {method_used}")
        
        if result['test_rmsle'] < 1.0:
            print(f"   🎯 Reasonable RMSLE achieved: {result['test_rmsle']:.4f}")
        else:
            print(f"   ⚠️  High RMSLE but using proper method: {result['test_rmsle']:.4f}")
    else:
        print(f"\n❌ IMPROVEMENT NEEDED:")
        print(f"   Still using constant fallback instead of improved methods")
    
    return success

if __name__ == "__main__":
    success = test_improved_simple()
    if success:
        print(f"\n✅ Test passed - STGAT improvements working!")
    else:
        print(f"\n❌ Test failed - Further improvements needed")