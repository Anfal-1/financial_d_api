"""
Comprehensive test script for the Flask API
"""

import requests
import json
import numpy as np
import time

# API configuration
API_URL = "http://localhost:5000"
HEADERS = {'Content-Type': 'application/json'}

def test_health_endpoint():
    """Test the health check endpoint"""
    print("=" * 60)
    print("TESTING HEALTH ENDPOINT")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Health Check Response:")
            print(json.dumps(result, indent=2))
            
            # Check if model and scaler are loaded
            if result.get('model_loaded') and result.get('scaler_loaded'):
                print("✅ Model and scaler are loaded successfully")
                return True
            else:
                print("❌ Model or scaler not loaded")
                return False
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_info_endpoint():
    """Test the API info endpoint"""
    print("\n" + "=" * 60)
    print("TESTING INFO ENDPOINT")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/info", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API Info:")
            print(json.dumps(result, indent=2))
            print("✅ Info endpoint working")
        else:
            print(f"❌ Info endpoint failed with status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Info endpoint failed: {e}")

def test_normal_transaction():
    """Test with normal transaction data"""
    print("\n" + "=" * 60)
    print("TESTING NORMAL TRANSACTION")
    print("=" * 60)
    
    # Normal transaction features (30 features as created by sample model)
    normal_features = [
    1.0,       # step
    200.0,     # amount
    1000.0,    # oldbalanceorg
    800.0,     # newbalanceorg
    0.0,       # oldbalancedest
    0.0,       # newbalancedest
    0          # type_encoded → TRANSFER مثلاً
]
    
    test_data = {"features": normal_features}
    
    try:
        print(f"Sending {len(normal_features)} features...")
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers=HEADERS,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction Result:")
            print(json.dumps(result, indent=2))
            
            if not result.get('is_fraud'):
                print("✅ Correctly identified as normal transaction")
            else:
                print("⚠️  Identified as fraud (might be false positive)")
                
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Normal transaction test failed: {e}")

def test_fraudulent_transaction():
    """Test with potentially fraudulent transaction data"""
    print("\n" + "=" * 60)
    print("TESTING FRAUDULENT TRANSACTION")
    print("=" * 60)
    
    # Fraudulent transaction features (extreme values)
    fraud_features = [
    1.0,        # step
    9000.0,     # amount
    1000.0,     # oldbalanceorg
    0.0,        # newbalanceorg
    0.0,        # oldbalancedest
    0.0,        # newbalancedest
    1           # type_encoded → PAYMENT مثلاً
]
    
    test_data = {"features": fraud_features}
    
    try:
        print(f"Sending {len(fraud_features)} features with extreme values...")
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers=HEADERS,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction Result:")
            print(json.dumps(result, indent=2))
            
            if result.get('is_fraud'):
                print("✅ Correctly identified as fraudulent transaction")
            else:
                print("⚠️  Identified as normal (might be false negative)")
                
        else:
            print(f"❌ Fraud prediction failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Fraudulent transaction test failed: {e}")

def test_error_cases():
    """Test various error scenarios"""
    print("\n" + "=" * 60)
    print("TESTING ERROR CASES")
    print("=" * 60)
    
    error_tests = [
        {
            "name": "Missing features field",
            "data": {"wrong_field": [1, 2, 3]},
            "expected_status": 400
        },
        {
            "name": "Empty features list",
            "data": {"features": []},
            "expected_status": 400
        },
        {
            "name": "Non-numeric features",
            "data": {"features": ["invalid", "data", "types"]},
            "expected_status": 400
        },
        {
            "name": "Mixed valid/invalid features",
            "data": {"features": [1.0, 2.0, "invalid", 4.0]},
            "expected_status": 400
        },
        {
            "name": "None values in features",
            "data": {"features": [1.0, None, 3.0]},
            "expected_status": 400
        },
        {
            "name": "Wrong number of features (too few)",
            "data": {"features": [1.0, 2.0, 3.0]},  # Assuming model expects 30
            "expected_status": 400
        }
    ]
    
    for i, test in enumerate(error_tests, 1):
        print(f"\n{i}. Testing: {test['name']}")
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test['data'],
                headers=HEADERS,
                timeout=10
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == test['expected_status']:
                print(f"   ✅ Correctly returned status {response.status_code}")
            else:
                print(f"   ⚠️  Expected {test['expected_status']}, got {response.status_code}")
            
            if response.status_code >= 400:
                result = response.json()
                print(f"   Error: {result.get('error', 'No error message')}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")

def test_invalid_endpoints():
    """Test invalid endpoints and methods"""
    print("\n" + "=" * 60)
    print("TESTING INVALID ENDPOINTS")
    print("=" * 60)
    
    # Test 404 - invalid endpoint
    print("1. Testing invalid endpoint...")
    try:
        response = requests.get(f"{API_URL}/invalid", timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 404:
            print("   ✅ Correctly returned 404 for invalid endpoint")
        else:
            print(f"   ⚠️  Expected 404, got {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 405 - wrong method
    print("\n2. Testing wrong HTTP method...")
    try:
        response = requests.get(f"{API_URL}/predict", timeout=10)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 405:
            print("   ✅ Correctly returned 405 for wrong method")
        else:
            print(f"   ⚠️  Expected 405, got {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")

def test_performance():
    """Test API performance with multiple requests"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE")
    print("=" * 60)
    
    # Normal transaction for performance testing
    normal_features = [0.1] * 30  # Simple normal transaction
    test_data = {"features": normal_features}
    
    num_requests = 10
    response_times = []
    
    print(f"Sending {num_requests} requests to measure performance...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/predict",
                json=test_data,
                headers=HEADERS,
                timeout=10
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if response.status_code == 200:
                print(f"   Request {i+1}: {response_time:.3f}s ✅")
            else:
                print(f"   Request {i+1}: {response_time:.3f}s ❌ (Status: {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            print(f"   Request {i+1}: Failed - {e}")
    
    if response_times:
        avg_time = np.mean(response_times)
        min_time = np.min(response_times)
        max_time = np.max(response_times)
        
        print(f"\nPerformance Summary:")
        print(f"   Average response time: {avg_time:.3f}s")
        print(f"   Minimum response time: {min_time:.3f}s")
        print(f"   Maximum response time: {max_time:.3f}s")
        
        if avg_time < 1.0:
            print("   ✅ Good performance (< 1s average)")
        else:
            print("   ⚠️  Slow performance (> 1s average)")

def main():
    """Run all tests"""
    print("FRAUD DETECTION API - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("Make sure the Flask app is running on localhost:5000")
    print("Run 'python create_sample_model.py' first to create model files")
    print("=" * 60)
    
    # Test health endpoint first
    health_ok = test_health_endpoint()
    
    if not health_ok:
        print("\n❌ Health check failed. Cannot proceed with other tests.")
        print("Please ensure:")
        print("1. Flask app is running on localhost:5000")
        print("2. Model files exist (run create_sample_model.py)")
        return
    
    # Run all other tests
    test_info_endpoint()
    test_normal_transaction()
    test_fraudulent_transaction()
    test_error_cases()
    test_invalid_endpoints()
    test_performance()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
