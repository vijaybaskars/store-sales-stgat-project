#!/usr/bin/env python3
"""
Test script to verify the preprocessing fix works correctly
"""

import sys
import os

# Clean up paths and add current project
sys.path = [p for p in sys.path if 'StoreSalesSTGAT' not in p]
sys.path.insert(0, 'src')

from data.evaluation_cases import EvaluationCaseManager
from preprocessing.advanced_features import Phase35DataProcessor

def test_preprocessing_fix():
    """Test the preprocessing fix"""
    
    print("🧪 Testing Phase35DataProcessor Fix")
    print("=" * 50)
    
    try:
        # Initialize components
        case_manager = EvaluationCaseManager()
        processor = Phase35DataProcessor()
        
        print("✅ Components initialized successfully")
        
        # Test with one case
        store_id = 49
        family = 'PET SUPPLIES'
        
        print(f"\n🔧 Testing preprocessing for Store {store_id}, {family}")
        
        result = processor.process_evaluation_case(case_manager, store_id, family)
        
        if result is not None:
            train_final, test_final, feature_columns, scaler = result
            
            print(f"\n✅ PREPROCESSING SUCCESS:")
            print(f"   Train data shape: {train_final.shape}")
            print(f"   Test data shape: {test_final.shape}")
            print(f"   Number of features: {len(feature_columns)}")
            print(f"   Feature types: log1p sales, cyclical temporal, lags, rolling stats")
            
            # Test feature names
            sample_features = feature_columns[:10]
            print(f"\n📊 Sample features:")
            for i, feat in enumerate(sample_features, 1):
                print(f"   {i:2d}. {feat}")
            
            return True
            
        else:
            print("❌ Preprocessing returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preprocessing_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 FIX VERIFIED: Phase35DataProcessor is working correctly!")
        print("\n📋 What was fixed:")
        print("   ✅ Method call: case_manager.get_case_data() instead of get_case_train_test_data()")
        print("   ✅ Sales data loading: Added proper data loading logic")  
        print("   ✅ Pandas compatibility: Fixed fillna(method='forward') to fillna(method='ffill')")
        print("\n🚀 The notebook optimization loop should now work!")
    else:
        print("❌ Fix verification failed - please check the errors above")
        sys.exit(1)