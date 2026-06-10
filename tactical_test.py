import requests
import json
import random
import time

BASE_URL = "http://127.0.0.1:5000"

def test_dashboard():
    print("[TEST] Initiating Bihar Flood Intelligence Stress Test...")
    
    # 1. Get Locations
    try:
        res = requests.get(f"{BASE_URL}/api/locations")
        locs = res.json()
        districts = locs['districts']
        v_by_d = locs['villages_by_district']
        print(f"LOCATIONS LOADED: {len(districts)} Districts Found.")
    except Exception as e:
        print(f"LOCATIONS FAILED: {e}")
        return

    report = []
    
    # 2. Sample Villages across all Districts
    for d in districts:
        villages = v_by_d.get(d, [])
        if not villages: continue
        
        # Pick 2 random villages per district for speed
        sample = random.sample(villages, min(2, len(villages)))
        
        for v in sample:
            print(f"Testing: {d} > {v}...")
            
            # A. Station Test
            try:
                res = requests.get(f"{BASE_URL}/api/station", params={"district": d, "village": v})
                stn_data = res.json()
                if "error" in stn_data:
                    report.append(f"ERR: {d}/{v} STATION: {stn_data['error']}")
                    continue
                obs = stn_data.get("obs", [])
            except Exception as e:
                report.append(f"CRASH: {d}/{v} STATION: {e}")
                continue
                
            # B. Normal Predict Test
            try:
                res = requests.post(f"{BASE_URL}/api/predict", json={
                    "district": d, "village": v, "rain": 10, "wl": 45, "soil": 40, "obs_history": obs
                })
                p_data = res.json()
                if p_data['risk'] != "LOW":
                    report.append(f"WARN: {d}/{v} FALSE POSITIVE: Risk {p_data['risk']} at Low Input")
            except Exception as e:
                report.append(f"CRASH: {d}/{v} PREDICT: {e}")

            # C. Extreme Predict Test (High Sensitivity Check)
            try:
                res = requests.post(f"{BASE_URL}/api/predict", json={
                    "district": d, "village": v, "rain": 140, "wl": 78, "soil": 90, "obs_history": obs
                })
                p_data = res.json()
                if p_data['risk'] != "HIGH":
                    report.append(f"FAIL: {d}/{v} SENSITIVITY: Risk {p_data['risk']} at High Input (Prob: {p_data['prob']})")
                if "phone" not in p_data or "guideline" not in p_data:
                    report.append(f"WARN: {d}/{v} PROTOCOL MISSING: No contact/guideline returned")
            except Exception as e:
                report.append(f"CRASH: {d}/{v} EXTREME PREDICT: {e}")

    # 3. Final Report
    print("\n" + "="*50)
    print("TACTICAL TEST REPORT")
    print("="*50)
    if not report:
        print("SUCCESS: ALL SYSTEMS OPERATIONAL. 100% PARITY ACHIEVED.")
    else:
        for r in report:
            print(r)
    print("="*50)

if __name__ == "__main__":
    test_dashboard()
