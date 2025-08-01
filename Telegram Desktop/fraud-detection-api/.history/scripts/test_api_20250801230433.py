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
            if result.get('model_loaded') and result.get('scaler_loaded'):
                print("✅ Model and scaler are loaded successfully")
                return True
            else:
                print("❌ Model or scaler not loaded")
                return False
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_info_endpoint():
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
    print("\n" + "=" * 60)
    print("TESTING NORMAL TRANSACTION")
    print("=" * 60)
    
    normal_features = [
        1.0, 200.0, 1000.0, 800.0, 0.0, 0.0, 0
    ]
    test_data = {"features": normal_features}
    
    try:
        print(f"Sending {len(normal_features)} features...")
        response = requests.post(f"{API_URL}/predict", json=test_data, headers=HEADERS, timeout=10)
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
    except requests.exceptions.RequestException as e:
        print(f"❌ Normal transaction test failed: {e}")

def test_fraudulent_transaction():
    print("\n" + "=" * 60)
    print("TESTING FRAUDULENT TRANSACTION")
    print("=" * 60)

    fraud_features = [
        1.0, 9000.0, 1000.0, 0.0, 0.0, 0.0, 1
    ]
    test_data = {"features": fraud_features}

    try:
        print(f"Sending {len(fraud_features)} features with extreme values...")
        response = requests.post(f"{API_URL}/predict", json=test_data, headers=HEADERS, timeout=10)
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
    except requests.exceptions.RequestException as e:
        print(f"❌ Fraudulent transaction test failed: {e}")

def test_error_cases():
    print("\n" + "=" * 60)
    print("TESTING ERROR CASES")
    print("=" * 60)
    
    error_tests = [