#!/usr/bin/env python3
"""
Debug the numpy.bool deprecation issue
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

def test_numpy_version():
    """Test numpy version and bool handling"""
    print("🔍 Debugging Numpy Issue")
    print("=" * 40)
    
    print(f"NumPy version: {np.__version__}")
    
    # Test what boolean types are available
    print("\nAvailable numpy boolean types:")
    if hasattr(np, 'bool'):
        print(f"  np.bool: {np.bool}")
    else:
        print("  np.bool: NOT AVAILABLE (deprecated)")
        
    if hasattr(np, 'bool_'):
        print(f"  np.bool_: {np.bool_}")
    else:
        print("  np.bool_: NOT AVAILABLE")
    
    # Test the convert function
    print("\nTesting convert_numpy_types function:")
    from serving.flask_api import convert_numpy_types
    
    test_values = [
        True,                    # Python bool
        np.bool_(True),          # Numpy bool
        np.int32(42),           # Numpy int
        np.float64(3.14),       # Numpy float
        np.array([1, 2, 3]),    # Numpy array
        {"a": np.bool_(True), "b": np.int32(5)}  # Dict with numpy values
    ]
    
    for i, val in enumerate(test_values):
        try:
            converted = convert_numpy_types(val)
            print(f"  Test {i+1}: {type(val)} -> {type(converted)} ✅")
        except Exception as e:
            print(f"  Test {i+1}: {type(val)} -> ERROR: {e} ❌")

def test_prediction_data():
    """Test if the prediction generation creates problematic numpy types"""
    print(f"\n🧪 Testing prediction data types")
    
    try:
        from serving.data_service import data_service
        
        # Try a quick pattern analysis first
        analysis = data_service.analyze_pattern(49, "PET SUPPLIES")
        print(f"Pattern analysis type: {type(analysis.beats_traditional) if hasattr(analysis, 'beats_traditional') else 'N/A'}")
        
        # Check the raw metrics
        raw_metrics = analysis.raw_metrics
        print(f"Raw metrics: {type(raw_metrics)}")
        for key, value in raw_metrics.items():
            print(f"  {key}: {type(value)} = {value}")
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    test_numpy_version()
    test_prediction_data()

if __name__ == "__main__":
    main()