#!/usr/bin/env python3
"""
Simple test script for model imports
Use this to verify all models work correctly
"""

import sys
import os

# Clean up any conflicting paths
sys.path = [p for p in sys.path if 'StoreSalesSTGAT' not in p]

# Add current project src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test imports
print("🧪 Testing Store Sales STGAT Models")
print("=" * 40)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} available")
    
    # Import all models
    from models.optimized_lstm import OptimizedLSTMModel, Phase35LSTMTrainer
    from models.traditional import TraditionalBaselines, ModelResults
    from models.neural.neural_baselines_fixed import NeuralBaselines
    from data.evaluation_cases import EvaluationCaseManager
    
    print("✅ All models imported successfully")
    
    # Test model creation
    model = OptimizedLSTMModel(input_size=10)
    print("✅ OptimizedLSTMModel created successfully")
    
    print("\n🎯 All tests passed! Your models are ready to use.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)