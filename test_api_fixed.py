#!/usr/bin/env python3
"""
Test the Flask API with JSON serialization fixes
Quick test to verify the numpy serialization issue is resolved
"""

import sys
import requests
import time
import subprocess
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_api_endpoints():
    """Test key API endpoints"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing Flask API with JSON serialization fixes")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data.get('status', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Cases endpoint
    print("\n2. Evaluation Cases")
    try:
        response = requests.get(f"{base_url}/cases", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Cases loaded: {data.get('total_cases', 0)}")
        else:
            print(f"   ❌ Cases endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cases error: {e}")
        return False
    
    # Test 3: Pattern analysis (should be fast)
    print("\n3. Pattern Analysis")
    try:
        response = requests.get(f"{base_url}/analysis/49/PET SUPPLIES", timeout=20)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Pattern: {data.get('pattern_type', 'unknown')} (CV: {data.get('coefficient_variation', 0):.3f})")
            print(f"       Recommended: {data.get('recommended_model_type', 'unknown')}")
        else:
            print(f"   ❌ Pattern analysis failed: {response.status_code}")
            print(f"       Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Pattern analysis error: {e}")
        return False
    
    # Test 4: Quick prediction (Store 49 - PET SUPPLIES should be fast)
    print("\n4. Quick Prediction Test (Store 49 - PET SUPPLIES)")
    try:
        payload = {"forecast_horizon": 15, "include_analysis": True}
        response = requests.post(
            f"{base_url}/predict/49/PET SUPPLIES",
            json=payload,
            timeout=60  # Generous timeout
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction successful!")
            print(f"       RMSLE: {data.get('test_rmsle', 0):.4f}")
            print(f"       Model: {data.get('selected_model_name', 'unknown')}")
            print(f"       Type: {data.get('selected_model_type', 'unknown')}")
            print(f"       Beats Traditional: {data.get('beats_traditional', False)}")
            print(f"       Beats Neural: {data.get('beats_neural', False)}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"       Response: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False
    
    print(f"\n✅ All API tests passed! JSON serialization is working correctly.")
    return True

def main():
    print("Starting Flask API for testing...")
    
    # Start Flask API in background
    api_process = subprocess.Popen(
        [sys.executable, "src/serving/flask_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    try:
        # Wait for API to start
        print("Waiting for API to initialize (30 seconds)...")
        time.sleep(30)
        
        # Run tests
        success = test_api_endpoints()
        
        if success:
            print(f"\n🎉 API is working correctly!")
            print(f"You can now restart the dashboard with: python run_phase6.py")
        else:
            print(f"\n❌ API tests failed. Check the error messages above.")
            
    finally:
        # Cleanup
        print(f"\nCleaning up API process...")
        api_process.terminate()
        api_process.wait(timeout=5)

if __name__ == "__main__":
    main()