#!/usr/bin/env python3
"""
Simple test script to verify the Flask app works correctly
"""
import requests
import json
import time
import subprocess
import sys

def test_prediction_api():
    """Test the /predict endpoint"""
    
    # Test data
    test_cases = [
        {
            "name": "Sunny day",
            "data": {
                "precipitation": 0.0,
                "temp_max": 30.0,
                "temp_min": 20.0,
                "wind": 10.0
            }
        },
        {
            "name": "Rainy day",
            "data": {
                "precipitation": 15.5,
                "temp_max": 18.0,
                "temp_min": 12.0,
                "wind": 25.0
            }
        },
        {
            "name": "Snow day",
            "data": {
                "precipitation": 5.0,
                "temp_max": -2.0,
                "temp_min": -8.0,
                "wind": 15.0
            }
        }
    ]
    
    print("Testing Flask Weather Prediction API")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {json.dumps(test_case['data'], indent=2)}")
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=test_case['data'],
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']}%")
                print(f"   Icon: {result['icon']}")
                print(f"   Recommendation: {result['recommendation']}")
            else:
                print(f"❌ FAILED - Status code: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    return True

if __name__ == "__main__":
    # Wait a bit for the server to start
    print("Waiting for Flask server to be ready...")
    time.sleep(2)
    
    # Run tests
    success = test_prediction_api()
    sys.exit(0 if success else 1)
