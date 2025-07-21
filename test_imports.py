#!/usr/bin/env python3
"""
Test script to verify all model imports work correctly
Resolves the ModuleNotFoundError issues by testing different import approaches
"""

import sys
import os
import traceback

def test_import_approaches():
    """Test different ways to import the models"""
    
    print("🧪 Testing Model Import Approaches")
    print("=" * 50)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Clean up conflicting paths first
    print("\n🔧 Cleaning up Python path conflicts...")
    paths_to_remove = [p for p in sys.path if 'StoreSalesSTGAT' in p]
    for path in paths_to_remove:
        sys.path.remove(path)
        print(f"   Removed conflicting path: {path}")
    
    # Add current project's src to the beginning
    project_src = os.path.join(current_dir, 'src')
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
        print(f"   Added current project path: {project_src}")
    
    # Test 1: Direct imports (should work from project root)
    print("\n1️⃣ Testing direct imports from project root...")
    try:
        import models
        from models import OptimizedLSTMModel, Phase35LSTMTrainer
        from models import TraditionalBaselines, ModelResults
        print("   ✅ Direct imports successful")
        
        # Test model instantiation
        model = OptimizedLSTMModel(input_size=10)
        print("   ✅ OptimizedLSTMModel instantiation successful")
        
    except Exception as e:
        print(f"   ❌ Direct imports failed: {e}")
    
    # Test 2: sys.path approach (corrected version)
    print("\n2️⃣ Testing sys.path.append approach...")
    try:
        if 'src' not in sys.path:
            sys.path.insert(0, 'src')
        
        # Now import from models (not models.optimized_lstm)
        from models.optimized_lstm import OptimizedLSTMModel as OptLSTM2
        from models.optimized_lstm import Phase35LSTMTrainer as Trainer2
        from models.traditional import TraditionalBaselines as Trad2
        
        print("   ✅ sys.path imports successful")
        
        # Test model instantiation
        model2 = OptLSTM2(input_size=15)
        print("   ✅ OptimizedLSTMModel instantiation successful")
        
    except Exception as e:
        print(f"   ❌ sys.path imports failed: {e}")
        traceback.print_exc()
    
    # Test 3: Import PyTorch and dependencies
    print("\n3️⃣ Testing PyTorch and dependencies...")
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        print("   ✅ All dependencies available")
        
    except Exception as e:
        print(f"   ❌ Dependencies test failed: {e}")
    
    # Test 4: Neural models
    print("\n4️⃣ Testing neural models...")
    try:
        from models.neural.neural_baselines_fixed import NeuralBaselines
        from models.neural.neural_baselines_fixed import VanillaLSTM, BidirectionalLSTM
        
        print("   ✅ Neural models import successful")
        
        # Test instantiation
        vanilla = VanillaLSTM(input_size=1, hidden_size=32, forecast_horizon=15)
        print("   ✅ VanillaLSTM instantiation successful")
        
    except Exception as e:
        print(f"   ❌ Neural models test failed: {e}")
    
    # Test 5: Data modules
    print("\n5️⃣ Testing data modules...")
    try:
        from data.evaluation_cases import EvaluationCaseManager, load_evaluation_cases
        print("   ✅ Data modules import successful")
        
    except Exception as e:
        print(f"   ❌ Data modules test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Import testing complete!")

def create_usage_examples():
    """Create usage examples for corrected imports"""
    
    print("\n📋 CORRECTED USAGE EXAMPLES:")
    print("=" * 50)
    
    print("\n🔧 Method 1: From project root directory")
    print("```python")
    print("# Run from: store-sales-stgat-project/")
    print("import torch")
    print("from models.optimized_lstm import OptimizedLSTMModel, Phase35LSTMTrainer")
    print("from models.traditional import TraditionalBaselines")
    print("from models.neural.neural_baselines_fixed import NeuralBaselines")
    print("")
    print("# Test instantiation")
    print("model = OptimizedLSTMModel(input_size=10)")
    print("print('✅ Model created successfully')")
    print("```")
    
    print("\n🔧 Method 2: Using sys.path (corrected)")
    print("```python")
    print("import sys")
    print("sys.path.insert(0, 'src')  # Add src to path")
    print("")
    print("# Import directly from modules")
    print("from models.optimized_lstm import OptimizedLSTMModel")
    print("from models.traditional import TraditionalBaselines") 
    print("")
    print("# NOT: from models.optimized_lstm import ... (this was the error)")
    print("```")

if __name__ == "__main__":
    test_import_approaches()
    create_usage_examples()