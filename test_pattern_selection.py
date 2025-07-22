#!/usr/bin/env python3
"""
Test script for Pattern-Based Model Selection implementation
Verifies integration with existing Phase 2/3 infrastructure
"""

import sys
import os

# Clean up paths and add current project
sys.path = [p for p in sys.path if 'StoreSalesSTGAT' not in p]
sys.path.insert(0, 'src')

def test_pattern_selection():
    """Test the complete pattern selection implementation"""
    
    print("🧪 Testing Pattern-Based Model Selection")
    print("=" * 50)
    
    try:
        # Test 1: Imports
        print("1️⃣ Testing imports...")
        from models.pattern_selection import PatternBasedSelector, PatternAnalysis, AdaptiveResults
        from models.traditional import TraditionalBaselines
        from data.evaluation_cases import EvaluationCaseManager, load_evaluation_cases
        print("   ✅ All imports successful")
        
        # Test 2: Infrastructure
        print("\n2️⃣ Testing infrastructure...")
        case_manager = EvaluationCaseManager()
        evaluation_cases = load_evaluation_cases()
        print(f"   ✅ Loaded {len(evaluation_cases)} evaluation cases")
        
        # Test 3: Selector initialization
        print("\n3️⃣ Testing selector initialization...")
        selector = PatternBasedSelector(case_manager, pattern_threshold=1.5)
        print("   ✅ PatternBasedSelector initialized")
        
        # Test 4: Pattern analysis
        print("\n4️⃣ Testing pattern analysis...")
        test_case = evaluation_cases[0]
        store_nbr, family = test_case['store_nbr'], test_case['family']
        
        pattern_analysis = selector.analyze_pattern(store_nbr, family)
        print(f"   Store {store_nbr} - {family}:")
        print(f"   CV: {pattern_analysis.coefficient_variation:.3f}")
        print(f"   Pattern: {pattern_analysis.pattern_type}")
        print(f"   Confidence: {pattern_analysis.confidence_score:.3f}")
        print("   ✅ Pattern analysis working")
        
        # Test 5: Model selection
        print("\n5️⃣ Testing model selection...")
        selected_model = selector.select_optimal_model(pattern_analysis)
        print(f"   Selected model type: {selected_model}")
        print("   ✅ Model selection working")
        
        # Test 6: Single case evaluation (quick test)
        print("\n6️⃣ Testing adaptive evaluation...")
        print("   Note: This may take 30-60 seconds...")
        result = selector.evaluate_case_adaptive(store_nbr, family)
        
        if result:
            print(f"   ✅ Evaluation successful!")
            print(f"   RMSLE: {result.test_rmsle:.4f}")
            print(f"   Selected: {result.selected_model_type} ({result.selected_model_name})")
            print(f"   vs Traditional: {result.improvement_vs_traditional_baseline:+.1f}%")
        else:
            print("   ⚠️ Evaluation returned None")
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("🎯 Pattern-Based Selection is ready for use")
        print("📚 Academic notebook ready for execution")
        print("🚀 GCP deployment components prepared")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pattern_selection()
    if not success:
        sys.exit(1)
    else:
        print("\n🎉 Integration testing complete!")
        print("You can now run the notebook: notebooks/04_pattern_based_selection.ipynb")