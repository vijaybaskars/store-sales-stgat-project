#!/usr/bin/env python3
"""
Quick test script for Phase 6 components
Tests individual components without starting the full stack
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing Phase 6 Component Imports...")
    
    try:
        from serving.config import config
        print("✅ Config module")
    except Exception as e:
        print(f"❌ Config module: {e}")
        return False
    
    try:
        from serving.models import HealthResponse, EvaluationCasesResponse
        print("✅ Pydantic models")
    except Exception as e:
        print(f"❌ Pydantic models: {e}")
        return False
    
    try:
        from serving.data_service import data_service
        print("✅ Data service")
    except Exception as e:
        print(f"❌ Data service: {e}")
        return False
    
    try:
        from serving.flask_api import app
        print("✅ Flask API")
    except Exception as e:
        print(f"❌ Flask API: {e}")
        return False
    
    print("✅ All imports successful!")
    return True


def test_config():
    """Test configuration"""
    print("\n🧪 Testing Configuration...")
    
    from serving.config import config
    
    print(f"   Project root: {config.project_root}")
    print(f"   API URL: {config.api_url}")
    print(f"   Streamlit URL: {config.streamlit_url}")
    
    validation = config.validate_paths()
    missing_paths = [name for name, info in validation.items() if not info["exists"]]
    
    if missing_paths:
        print(f"⚠️  Missing paths: {missing_paths}")
        return False
    else:
        print("✅ All required paths exist")
        return True


def test_data_loading():
    """Test data loading capabilities"""
    print("\n🧪 Testing Data Loading...")
    
    from serving.data_service import data_service
    
    try:
        cases = data_service.get_evaluation_cases()
        print(f"✅ Loaded {len(cases)} evaluation cases")
        
        if cases:
            first_case = cases[0]
            print(f"   Sample case: Store {first_case.store_nbr} - {first_case.family}")
            
            # Test pattern analysis (lightweight)
            print("   Testing pattern analysis...")
            analysis = data_service.analyze_pattern(first_case.store_nbr, first_case.family)
            print(f"   ✅ Pattern: {analysis.pattern_type} (CV: {analysis.coefficient_variation:.3f})")
            
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def test_flask_api():
    """Test Flask API endpoints"""
    print("\n🧪 Testing Flask API...")
    
    from serving.flask_api import app
    
    with app.test_client() as client:
        # Test root endpoint
        resp = client.get('/')
        if resp.status_code == 200:
            print("✅ Root endpoint")
        else:
            print(f"❌ Root endpoint: {resp.status_code}")
            return False
        
        # Test health endpoint
        resp = client.get('/health')
        if resp.status_code == 200:
            print("✅ Health endpoint")
        else:
            print(f"❌ Health endpoint: {resp.status_code}")
            return False
        
        # Test cases endpoint
        resp = client.get('/cases')
        if resp.status_code == 200:
            data = resp.get_json()
            print(f"✅ Cases endpoint ({data.get('total_cases', 0)} cases)")
        else:
            print(f"❌ Cases endpoint: {resp.status_code}")
            return False
    
    return True


def main():
    """Run all tests"""
    print("🏪 Phase 6 Component Testing")
    print("=" * 50)
    
    tests = [
        ("Component Imports", test_imports),
        ("Configuration", test_config), 
        ("Data Loading", test_data_loading),
        ("Flask API", test_flask_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All Phase 6 components are ready!")
        print("To start the dashboard: python run_phase6.py")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed - check configuration")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)