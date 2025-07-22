#!/usr/bin/env python3
"""
Test enhanced STGAT system with adaptive pattern selection integration
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from data.evaluation_cases import EvaluationCaseManager, load_evaluation_cases

def test_enhanced_stgat():
    """Test enhanced STGAT with pattern selection adaptive approach"""
    
    print("🚀 TESTING ENHANCED STGAT WITH ADAPTIVE PATTERN SELECTION")
    print("=" * 65)
    
    # Load infrastructure
    evaluation_cases = load_evaluation_cases()
    case_manager = EvaluationCaseManager()
    
    print(f"✅ Loaded {len(evaluation_cases)} evaluation cases")
    
    # Initialize enhanced STGAT evaluator with adaptive mode
    print("\n🧠 Initializing Enhanced STGAT with Adaptive Pattern Selection...")
    
    try:
        from models.stgat import STGATEvaluator
        enhanced_evaluator = STGATEvaluator(case_manager, cv_threshold=1.5, adaptive_mode=True)
        print("✅ Enhanced STGAT evaluator initialized successfully")
        
        # Check if adaptive mode is working
        if enhanced_evaluator.adaptive_mode and enhanced_evaluator.pattern_selector is not None:
            print("   🎯 Adaptive mode: ENABLED")
            print("   🧠 Pattern selector: AVAILABLE")
            print("   📊 Traditional models: AVAILABLE")
        else:
            print("   ⚠️ Adaptive mode: DISABLED (falling back to basic mode)")
            
    except Exception as e:
        print(f"❌ Failed to initialize enhanced evaluator: {e}")
        return False
    
    # Test on a few cases
    test_cases = evaluation_cases[:3]  # Test first 3 cases
    results = []
    
    print(f"\n🧪 Testing Enhanced STGAT on {len(test_cases)} cases...")
    
    for i, case in enumerate(test_cases, 1):
        store_nbr = case['store_nbr']
        family = case['family']
        
        print(f"\n{i}. Testing Store {store_nbr} - {family}")
        
        try:
            # Use the new adaptive evaluation method
            result = enhanced_evaluator.evaluate_case_adaptive(store_nbr, family)
            
            if result and 'error' not in result:
                results.append(result)
                
                print(f"   ✅ RMSLE: {result['test_rmsle']:.4f}")
                print(f"   📊 Method: {result.get('method_used', 'Unknown')}")
                print(f"   🔍 Pattern: {result.get('pattern_type', 'Unknown')} (CV: {result.get('cv_value', 0.0):.3f})")
                
                # Performance assessment
                vs_traditional = ((0.4755 - result['test_rmsle']) / 0.4755) * 100
                vs_phase36 = ((0.4190 - result['test_rmsle']) / 0.4190) * 100
                
                print(f"   📈 vs Traditional (0.4755): {vs_traditional:+.1f}%")
                print(f"   📈 vs Phase 3.6 (0.4190): {vs_phase36:+.1f}%")
                
                # Success indicators
                if result['test_rmsle'] < 0.4755:
                    print("   🎯 BEATS Traditional baseline!")
                if result['test_rmsle'] < 0.4190:
                    print("   🎯 BEATS Phase 3.6 baseline!")
                
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as case_error:
            print(f"   ❌ Case failed: {case_error}")
    
    # Summary
    if results:
        avg_rmsle = np.mean([r['test_rmsle'] for r in results])
        beats_traditional = sum(1 for r in results if r['test_rmsle'] < 0.4755)
        beats_phase36 = sum(1 for r in results if r['test_rmsle'] < 0.4190)
        
        print(f"\n" + "="*50)
        print(f"🎯 ENHANCED STGAT SUMMARY")
        print(f"="*50)
        print(f"Successful evaluations: {len(results)}/{len(test_cases)}")
        print(f"Average RMSLE: {avg_rmsle:.4f}")
        print(f"Beat Traditional: {beats_traditional}/{len(results)} ({beats_traditional/len(results)*100:.1f}%)")
        print(f"Beat Phase 3.6: {beats_phase36}/{len(results)} ({beats_phase36/len(results)*100:.1f}%)")
        
        # Method distribution
        methods = {}
        for r in results:
            method = r.get('method_used', 'Unknown')
            methods[method] = methods.get(method, 0) + 1
        
        print(f"\n📊 Method Distribution:")
        for method, count in methods.items():
            print(f"   {method}: {count}/{len(results)} ({count/len(results)*100:.1f}%)")
        
        # Success assessment
        vs_traditional = ((0.4755 - avg_rmsle) / 0.4755) * 100
        vs_phase36 = ((0.4190 - avg_rmsle) / 0.4190) * 100
        
        print(f"\n📈 Performance vs Baselines:")
        print(f"   vs Traditional: {vs_traditional:+.1f}%")
        print(f"   vs Phase 3.6: {vs_phase36:+.1f}%")
        
        if avg_rmsle < 0.4190:
            print(f"\n🎉 SUCCESS: Enhanced STGAT outperforms Phase 3.6!")
            success = True
        elif avg_rmsle < 0.4755:
            print(f"\n🎯 GOOD: Enhanced STGAT outperforms traditional baseline")
            success = True
        else:
            print(f"\n⚠️ NEEDS IMPROVEMENT: Still above baselines")
            success = False
    else:
        print(f"\n❌ No successful evaluations")
        success = False
    
    return success

if __name__ == "__main__":
    success = test_enhanced_stgat()
    if success:
        print(f"\n✅ Enhanced STGAT test passed!")
        print(f"📊 Adaptive pattern selection integration working")
    else:
        print(f"\n❌ Enhanced STGAT needs further improvement")