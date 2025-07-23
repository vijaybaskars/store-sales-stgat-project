"""
NumPy compatibility module
Apply numpy.bool compatibility fix before any other imports
"""

import numpy as np
import warnings
import sys

# Suppress ALL numpy deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*np.bool.*deprecated.*')
warnings.filterwarnings('ignore', message='.*`np.bool`.*')

# Force compatibility for all deprecated numpy aliases
deprecated_aliases = {
    'bool': bool,
    'int': int, 
    'float': float,
    'complex': complex,
    'object': object,
    'str': str,
    'unicode': str,
}

for alias, replacement in deprecated_aliases.items():
    if not hasattr(np, alias):
        setattr(np, alias, replacement)
        print(f"✅ Added np.{alias} = {replacement.__name__}")

# Double-check numpy.bool specifically
if hasattr(np, 'bool'):
    print("✅ numpy.bool is available")
else:
    print("❌ numpy.bool was missing, added compatibility")

print(f"📍 NumPy version: {np.__version__}")
print(f"📍 Python version: {sys.version_info}")

# Also monkey-patch the __getattr__ to handle dynamic access
original_getattr = getattr(np, '__getattr__', None)

def numpy_getattr_patch(name):
    """Handle access to deprecated numpy attributes gracefully"""
    if name == 'bool':
        warnings.warn("np.bool is deprecated, use builtin bool", FutureWarning, stacklevel=2)
        return bool
    elif name in deprecated_aliases:
        warnings.warn(f"np.{name} is deprecated, use builtin {deprecated_aliases[name].__name__}", 
                     FutureWarning, stacklevel=2)
        return deprecated_aliases[name]
    elif original_getattr:
        return original_getattr(name)
    else:
        raise AttributeError(f"module '{np.__name__}' has no attribute '{name}'")

# Apply the patch
np.__getattr__ = numpy_getattr_patch

print("🔧 Applied comprehensive numpy compatibility patch")