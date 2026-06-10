
import requests
import json

url = "http://127.0.0.1:5000/api/predict"
payload = {
    "village": "Digha",
    "district": "Patna"
}

try:
    print(f"Requesting {url} with {payload}...")
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Model Used: {data.get('model_used')}")
        print(f"Risk Level: {data.get('risk_level')}")
        print(f"Probability: {data.get('probability')}")
        
        if data.get('model_used') == "BiLSTM+SAR+Stacking":
            print("SUCCESS: ML models are active!")
        else:
            print("FAILURE: Fallback used.")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
