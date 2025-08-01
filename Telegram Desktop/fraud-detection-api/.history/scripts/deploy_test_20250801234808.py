"""
Test script for deployed API (works with both local and deployed versions)
"""

import requests
import json
import os

def test_deployed_api(base_url):
    """Test the deployed API"""
    print(f"Testing API at: {base_url}")
    print("=" * 60)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        print(f"Health Check - Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API is healthy")
            health_data = response.json()
            print(f"Model loaded: {health_data.get('model_loaded')}")
            print(f"Scaler loaded: {health_data.get('scaler_loaded')}")
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test prediction with normal transaction
    normal_features = [0.1] * 30  # Adjust based on your model's expected input size
    test_data = {"features": normal_features}
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"\nPrediction Test - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"Is fraud: {result.get('is_fraud')}")
            print(f"Reconstruction error: {result.get('reconstruction_error')}")
        else:
            print("❌ Prediction failed")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    return True

if __name__ == "__main__":
    # Test local development server
    local_url = "http://localhost:5000"
    print("Testing local development server...")
    test_deployed_api(local_url)
    
    # Test deployed version (replace with your Vercel URL)
    deployed_url = input("\nEnter your deployed API URL (or press Enter to skip): ").strip()
    if deployed_url:
        print(f"\nTesting deployed API...")
        test_deployed_api(deployed_url)
