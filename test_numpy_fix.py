#!/usr/bin/env python3
"""
Test the comprehensive numpy compatibility fix
"""

import sys
sys.path.insert(0, 'src')

# Import our compatibility fix first
import numpy_compat

# Now test numpy access
import numpy as np

print("Testing numpy compatibility:")
print(f"np.bool: {np.bool}")
print(f"np.bool_: {np.bool_}")
print(f"np.int: {np.int}")  
print(f"np.float: {np.float}")

# Test actual usage
test_bool = np.bool(True)
print(f"np.bool(True) = {test_bool} (type: {type(test_bool)})")

# Test with actual numpy boolean
numpy_bool = np.bool_(True)
print(f"np.bool_(True) = {numpy_bool} (type: {type(numpy_bool)})")

print("✅ Numpy compatibility test passed!")

# Test Flask API import
print("\nTesting Flask API import with compatibility fix...")
try:
    from serving.flask_api import app
    print("✅ Flask API imported successfully!")
    
    # Test the prediction conversion
    from serving.flask_api import convert_numpy_types
    
    test_data = {
        "bool_val": np.bool_(True),
        "int_val": np.int32(42),
        "float_val": np.float64(3.14),
        "array_val": np.array([1, 2, 3])
    }
    
    converted = convert_numpy_types(test_data)
    print("✅ Type conversion test passed!")
    print(f"   Converted types: {[(k, type(v)) for k, v in converted.items()]}")
    
except Exception as e:
    print(f"❌ Import/conversion failed: {e}")
    import traceback
    traceback.print_exc()