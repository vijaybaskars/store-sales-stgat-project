#!/usr/bin/env python3
"""
Test improved STGAT system with real traditional model fallback
"""

import sys
sys.path.insert(0, 'src')

import json
import pandas as pd
import numpy as np
from data.evaluation_cases import EvaluationCaseManager, load_evaluation_cases
from models.stgat import STGATEvaluator

def test_improved_stgat_fallback():
    """Test that STGAT now uses actual traditional models for fallback"""
    
    print("🧪 TESTING IMPROVED STGAT FALLBACK SYSTEM")
    print("=" * 50)
    
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
        return
    
    # Initialize improved STGAT evaluator
    print("\n🔧 Initializing STGAT evaluator with traditional models...")
    stgat_evaluator = STGATEvaluator(case_manager, cv_threshold=1.5)
    
    if stgat_evaluator.traditional_baselines is None:
        print("❌ Traditional models not initialized properly")
        return
    
    print("✅ Traditional models initialized successfully")
    
    # Find a volatile case - FIXED: Use correct key format
    volatile_case = None
    for case in evaluation_cases:
        case_key = f"store_{case['store_nbr']}_family_{case['family']}"  # FIXED: Correct key format
        if case_key in pattern_classifications:
            cv_value = pattern_classifications[case_key].get('coefficient_variation', 2.0)
            if cv_value >= 1.5:  # Volatile pattern
                volatile_case = case
                break
    
    if volatile_case is None:
        print("❌ No volatile case found")
        return
    
    store_nbr = volatile_case['store_nbr']
    family = volatile_case['family']
    case_key = f"store_{store_nbr}_family_{family}"  # FIXED: Correct key format
    cv_value = pattern_classifications[case_key].get('coefficient_variation', 2.0)
    
    print(f"\n🎯 Testing volatile case: Store {store_nbr} - {family}")
    print(f"   CV: {cv_value:.3f} (should route to traditional models)")
    
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
    
    # Check if using actual traditional models
    method_used = result.get('method_used', '')
    if 'Traditional_' in method_used:
        if 'Constant' in method_used:
            print("   ⚠️  Still using constant fallback")
            success = False
        else:
            print(f"   ✅ SUCCESS: Using actual traditional model!")
            success = True
    else:
        print(f"   ⚠️  Not using traditional fallback (method: {method_used})")
        success = False
    
    # Performance comparison
    vs_constant = ((0.4755 - result['test_rmsle']) / 0.4755) * 100
    print(f"   vs Constant baseline (0.4755): {vs_constant:+.1f}%")
    
    if success:
        print(f"\n🎉 IMPROVEMENT SUCCESS!")
        print(f"   The STGAT system now uses actual traditional models")
        print(f"   instead of constant fallback values for volatile patterns")
        
        if result['test_rmsle'] < 1.0:
            print(f"   🎯 Reasonable RMSLE achieved: {result['test_rmsle']:.4f}")
        else:
            print(f"   ⚠️  High RMSLE but using proper models: {result['test_rmsle']:.4f}")
    else:
        print(f"\n❌ IMPROVEMENT NEEDED:")
        print(f"   STGAT system still not using actual traditional models")
    
    return success

if __name__ == "__main__":
    success = test_improved_stgat_fallback()
    if success:
        print(f"\n✅ Test passed - STGAT improvements working!")
    else:
        print(f"\n❌ Test failed - Further improvements needed")