#!/usr/bin/env python3
"""
Quick test to verify the numpy.bool fix works
"""

import sys
import requests
import time
import subprocess
import signal
import os

def test_fixed_api():
    """Test the fixed API"""
    print("🧪 Quick Test: Numpy Boolean Fix")
    print("=" * 40)
    
    # Start Flask API
    print("Starting Flask API...")
    api_process = subprocess.Popen(
        [sys.executable, "src/serving/flask_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for startup
        print("Waiting 20 seconds for API startup...")
        time.sleep(20)
        
        # Test prediction endpoint that was failing
        print("Testing prediction endpoint...")
        base_url = "http://127.0.0.1:8000"
        
        payload = {"forecast_horizon": 15}
        response = requests.post(
            f"{base_url}/predict/26/FROZEN FOODS",  # This was the failing case
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Prediction completed")
            print(f"   RMSLE: {data.get('test_rmsle', 'N/A')}")
            print(f"   Model: {data.get('selected_model_name', 'N/A')}")
            print(f"   Type: {data.get('selected_model_type', 'N/A')}")
            return True
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"   Error: {response.text[:500]}")
            return False
            
    finally:
        # Cleanup
        print("Stopping API...")
        api_process.terminate()
        time.sleep(2)
        try:
            api_process.kill()
        except:
            pass

if __name__ == "__main__":
    success = test_fixed_api()
    if success:
        print("\n🎉 Fix confirmed! You can now restart the dashboard.")
        print("Run: python run_phase6.py")
    else:
        print("\n❌ Fix didn't work. More debugging needed.")