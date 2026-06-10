"""
check_models.py  --  Terminal test with CORRECT features matching features.pkl
Features: [wl_3d, rain_7d, soil_moisture, wl_lag1, wl_lag2, rain_lag1, wl_trend, rain_trend]
Run: venv\Scripts\python.exe check_models.py
"""
import os, sys, json, warnings, pickle
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE   = Path(__file__).parent
MODELS = BASE / "models"
DATA   = BASE / "data"

print("=" * 60)
print("  BIHAR FLOOD MODEL CHECK (Corrected Feature Vector)")
print("=" * 60)

# ── Keras ─────────────────────────────────────────────────────────────
try:
    import keras
    print(f"\n[1] Keras {keras.__version__} loaded  OK")
except:
    import tensorflow as tf; keras = tf.keras
    print(f"\n[1] TF {tf.__version__} loaded  OK")

# ── Scaler ────────────────────────────────────────────────────────────
print("\n[2] Loading scaler...")
_fixed = MODELS / "scaler_fixed.pkl"
_orig  = MODELS / "scaler.pkl"
try:
    import joblib
    scaler = joblib.load(_fixed if _fixed.exists() else _orig)
    which  = "scaler_fixed.pkl" if _fixed.exists() else "scaler.pkl (original)"
    print(f"    OK  {which} — {scaler.n_features_in_} features")
except Exception as e:
    print(f"    FAIL: {e}"); sys.exit(1)

# ── BiLSTM ────────────────────────────────────────────────────────────
print("\n[3] Loading BiLSTM...")
try:
    bilstm = keras.models.load_model(MODELS / "final_bilstm_model.keras", compile=False)
    print(f"    OK  Input={bilstm.input_shape}  Output={bilstm.output_shape}")
except Exception as e:
    print(f"    FAIL: {e}"); sys.exit(1)

# ── Threshold ─────────────────────────────────────────────────────────
try:
    THRESHOLD = json.load(open(MODELS / "threshold.json")).get("threshold", 0.3)
    print(f"\n[4] Threshold: {THRESHOLD}")
except:
    THRESHOLD = 0.3
    print(f"\n[4] Threshold: {THRESHOLD} (default)")

# ── Build features from real data ─────────────────────────────────────
print("\n[5] Loading spatial_timeseries to get real lag values...")
df = pd.read_csv(DATA / "spatial_timeseries .csv")
df.columns = [c.strip().lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df.sort_values(["subdistrict","date"], inplace=True)

# Take Patna subdistrict as reference
patna_row = df[df["subdistrict"].str.contains("patna", case=False)].tail(3)
if patna_row.empty:
    patna_row = df.tail(3)

last = patna_row.iloc[-1]
prev = patna_row.iloc[-2] if len(patna_row) >= 2 else last

real_wl_lag1   = float(last["wl_1d"])
real_wl_lag2   = float(prev["wl_1d"])
real_rain_lag1 = float(last["rain"])
print(f"    Real wl_lag1={real_wl_lag1:.2f}  wl_lag2={real_wl_lag2:.2f}  rain_lag1={real_rain_lag1:.2f}")
print(f"    (from subdistrict: {last['subdistrict']}, date: {last['date'].date()})")

# ── Prediction helper ─────────────────────────────────────────────────
def predict_correct(rain, wl, soil_pct):
    """Build correct feature vector and predict. Lags are WL-relative."""
    soil       = soil_pct / 100.0
    # Seed lags proportional to current WL (simulates mild rising trend)
    wl_lag1    = wl * 0.95
    wl_lag2    = wl * 0.90
    rain_lag1  = rain * 0.8
    rain_7d    = rain / 7.0
    wl_trend   = wl - wl_lag1      # positive = rising water
    rain_trend = rain - rain_7d

    X_raw = np.array([[
        wl,          # wl_3d
        rain_7d,     # rain_7d
        soil,        # soil_moisture
        wl_lag1,     # wl_lag1
        wl_lag2,     # wl_lag2
        rain_lag1,   # rain_lag1
        wl_trend,    # wl_trend
        rain_trend,  # rain_trend
    ]], dtype=np.float32)

    X_sc  = scaler.transform(X_raw)
    X_seq = np.tile(X_sc, (1, 7, 1))
    prob  = float(np.clip(bilstm.predict(X_seq, verbose=0)[0][0], 0, 1))
    return prob, X_sc[0]

# ── Test 3 scenarios ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREDICTION TEST (correct features)")
print("=" * 60)

TEST_CASES = [
    {"label": "Monsoon peak  (rain=120, wl=75, soil=80%)", "rain":120, "wl":75, "soil":80},
    {"label": "Pre-monsoon   (rain=35,  wl=50, soil=50%)", "rain": 35, "wl":50, "soil":50},
    {"label": "Dry season    (rain=2,   wl=20, soil=20%)", "rain":  2, "wl":20, "soil":20},
]
print()
get_risk = lambda p: "GREEN" if p < 0.30 else "AMBER" if p <= 0.55 else "RED" if p <= 0.75 else "CRITICAL"

for tc in TEST_CASES:
    prob, scaled = predict_correct(tc["rain"], tc["wl"], tc["soil"])
    risk     = get_risk(prob)
    decision = "FLOOD" if prob >= THRESHOLD else "NO FLOOD"
    print(f"  Scenario : {tc['label']}")
    print(f"  Scaled[0:4]: [{', '.join(f'{v:.3f}' for v in scaled[:4])}]  (should be near -3..+3)")
    print(f"  Probability: {prob:.4f}   Risk: {risk}   Decision: {decision}")
    print()

# ── Sweep to check differentiation ────────────────────────────────────
print("=" * 60)
print("  SENSITIVITY CHECK — Does output vary with input?")
print("=" * 60)
print(f"\n  Varying WATER LEVEL (rain=35, soil=50%):")
print(f"  {'WL (m)':<10} {'Prob':>8}  {'Risk'}")
for wl in [15, 25, 35, 50, 60, 70, 80]:
    p, _ = predict_correct(35, wl, 50)
    risk = get_risk(p)
    bar  = "#" * int(p * 20)
    print(f"  {wl:<10} {p:>8.4f}  {risk:<10}  [{bar:<20}]")

print(f"\n  Varying RAINFALL (wl=50, soil=50%):")
print(f"  {'Rain (mm)':<10} {'Prob':>8}  {'Risk'}")
for rain in [0, 10, 30, 60, 100, 130, 150]:
    p, _ = predict_correct(rain, 50, 50)
    risk = get_risk(p)
    bar  = "#" * int(p * 20)
    print(f"  {rain:<10} {p:>8.4f}  {risk:<10}  [{bar:<20}]")

print("\n" + "=" * 60)
print("  If outputs vary above -> model is differentiating. GOOD.")
print("  If all outputs stay ~1.0 -> model is still saturated.")
print("=" * 60)
