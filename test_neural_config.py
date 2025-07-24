#!/usr/bin/env python3
"""
Quick test to verify neural model configuration is enabled
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_config():
    """Test the neural model configuration"""
    print("🔧 Testing Neural Model Configuration")
    print("="*40)
    
    try:
        from serving.config import config
        
        print(f"✅ Configuration loaded successfully")
        print(f"📊 Neural models enabled: {config.enable_neural_models}")
        print(f"🐛 Debug mode: {config.neural_debug_mode}")
        print(f"⏰ Timeout: {config.neural_timeout_seconds}s")
        print(f"🔄 Fallback on error: {config.neural_fallback_on_error}")
        print(f"🎯 Pattern threshold: {config.pattern_threshold}")
        
        if config.enable_neural_models:
            print(f"\n🎉 SUCCESS: Neural models are ENABLED for debugging!")
            print(f"💡 You can now test CV < 1.5 cases in the dashboard")
            
            # Show the cases that will route to neural
            print(f"\n🧠 Cases that should route to NEURAL models (CV < {config.pattern_threshold}):")
            print(f"   • Store 49 - PET SUPPLIES (CV: 1.018)")
            print(f"   • Store 8 - PET SUPPLIES (CV: 0.999)")  
            print(f"   • Store 53 - PRODUCE (CV: 1.029)")
            
            print(f"\n🚀 Next steps:")
            print(f"   1. Start the dashboard: python run_phase6.py")
            print(f"   2. Select one of the above cases")
            print(f"   3. Click 'Generate Prediction'")
            print(f"   4. Use debug_neural_models.py for detailed error analysis")
            
        else:
            print(f"\n❌ Neural models are still DISABLED")
            print(f"💡 Check if the configuration file was saved properly")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()