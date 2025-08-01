"""
Test script for deployed API (works with both local and deployed versions)
"""

import requests
import json

def test_deployed_api(base_url):
    """Test the deployed API"""
    print(f"\n🔍 Testing API at: {base_url}")
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

    # ✅ Prediction test with correct number of features
   test_data = {
    'amount': 1.0,
    'hour': 200.0,
    'day': 1000.0,
    'location': 800.0,
    'device': 0.0,
    'past_avg_amount': 0.0,
    'past_std_amount': 0.0
}


    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        print(f"\nPrediction Test - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"→ Is fraud: {result.get('is_fraud')}")
            print(f"→ Reconstruction error: {result.get('reconstruction_error')}")
        else:
            print("❌ Prediction failed")
            print(response.text)

    except Exception as e:
        print(f"❌ Prediction error: {e}")

    return True

if __name__ == "__main__":
    # Test local development server
    local_url = "http://localhost:5000"
    print("🚀 Testing local development server...")
    test_deployed_api(local_url)

    # Optional: test deployed version (e.g., on Vercel)
    deployed_url = input("\n🌍 Enter your deployed API URL (or press Enter to skip): ").strip()
    if deployed_url:
        print(f"\n🚀 Testing deployed API...")
        test_deployed_api(deployed_url)
