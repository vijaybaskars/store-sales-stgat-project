#!/usr/bin/env python3
"""
Neural Model Debugging Script
Debug why neural models crash for CV < 1.5 cases
"""

import sys
import traceback
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neural_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_neural_case(store_nbr, family, cv_value):
    """Test a single neural routing case with detailed error handling"""
    
    print(f"\n{'='*60}")
    print(f"🧠 DEBUGGING: Store {store_nbr} - {family} (CV: {cv_value:.3f})")
    print(f"{'='*60}")
    
    try:
        # Import with error handling
        print("📦 Step 1: Importing modules...")
        from serving.config import config
        from serving.flask_api import app
        import requests
        import time
        
        print(f"✅ Neural models enabled: {config.enable_neural_models}")
        print(f"✅ Debug mode: {config.neural_debug_mode}")
        print(f"✅ Timeout: {config.neural_timeout_seconds}s")
        
        # Test direct API call
        print(f"\n🔄 Step 2: Testing API call...")
        url = f"http://127.0.0.1:8000/predict/{store_nbr}/{family.replace(' ', '%20')}"
        payload = {
            "forecast_horizon": 15,
            "include_analysis": True,
            "fast_mode": True  # Use fast mode for debugging
        }
        
        print(f"📡 Making request to: {url}")
        print(f"📦 Payload: {payload}")
        
        start_time = time.time()
        
        # Make request with timeout
        response = requests.post(
            url, 
            json=payload,
            timeout=config.neural_timeout_seconds
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Request completed in {duration:.1f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"   Model used: {result.get('selected_model_name', 'Unknown')}")
            print(f"   RMSLE: {result.get('test_rmsle', 'N/A')}")
            print(f"   Method: {result.get('method_used', 'Unknown')}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ TIMEOUT: Neural model took longer than {config.neural_timeout_seconds}s")
        return False
        
    except requests.exceptions.ConnectionError:
        print(f"🔌 CONNECTION ERROR: Is the Flask API running on port 8000?")
        print(f"   Start it with: python run_phase6.py")
        return False
        
    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {type(e).__name__}: {e}")
        print(f"\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Main debugging function"""
    
    print("🚀 NEURAL MODEL DEBUGGING SESSION")
    print("="*50)
    
    # Check configuration
    try:
        from serving.config import config
        print(f"✅ Configuration loaded:")
        print(f"   • Neural models: {'ENABLED' if config.enable_neural_models else 'DISABLED'}")
        print(f"   • Debug mode: {'ON' if config.neural_debug_mode else 'OFF'}")
        print(f"   • Timeout: {config.neural_timeout_seconds}s")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Test cases that should route to neural models (CV < 1.5)
    neural_test_cases = [
        {"store": 49, "family": "PET SUPPLIES", "cv": 1.018},
        {"store": 8, "family": "PET SUPPLIES", "cv": 0.999},
        {"store": 53, "family": "PRODUCE", "cv": 1.029}
    ]
    
    print(f"\n🎯 Testing {len(neural_test_cases)} cases that should route to NEURAL models...")
    
    results = []
    for i, case in enumerate(neural_test_cases, 1):
        print(f"\n[{i}/{len(neural_test_cases)}] Testing case...")
        success = test_neural_case(case["store"], case["family"], case["cv"])
        results.append({
            "case": f"Store {case['store']} - {case['family']}",
            "cv": case["cv"],
            "success": success
        })
        
        # Brief pause between tests
        if i < len(neural_test_cases):
            print(f"\n⏸️  Pausing 3 seconds before next test...")
            import time
            time.sleep(3)
    
    # Summary
    print(f"\n🏁 DEBUGGING SUMMARY")
    print("="*50)
    successful = sum(1 for r in results if r["success"])
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {len(results) - successful}/{len(results)}")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"   {status} {result['case']} (CV: {result['cv']:.3f})")
    
    if successful == len(results):
        print(f"\n🎉 ALL TESTS PASSED! Neural models are working correctly.")
    else:
        print(f"\n🔧 DEBUGGING TIPS:")
        print(f"   1. Check neural_debug.log for detailed error messages")
        print(f"   2. Ensure Flask API is running: python run_phase6.py")
        print(f"   3. Check PyTorch installation: python -c 'import torch; print(torch.__version__)'")
        print(f"   4. Monitor system resources during neural training")
        print(f"   5. Try fast_mode=True in dashboard for quicker testing")

if __name__ == "__main__":
    main()