"""
investigate_model.py  --  Deep investigation of BiLSTM saturation issue.
Checks: scaler ranges, feature distribution, model weight stats,
        threshold validity, and data balance.
Run: venv\Scripts\python.exe investigate_model.py
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

print("=" * 65)
print("  BILSTM SATURATION INVESTIGATION")
print("=" * 65)

# ── 1. SCALER INTERNALS ───────────────────────────────────────────────
print("\n[1] SCALER — What ranges was it trained on?")
try:
    import joblib
    scaler = joblib.load(MODELS / "scaler.pkl")
except:
    with open(MODELS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

feature_names = ["wl_1d", "rain_1d", "soil_moisture",
                 "wl_3d", "wl_7d", "rain_3d", "rain_7d", "month"]

print(f"\n  {'Feature':<18} {'Scale Min':>12} {'Scale Max':>12} {'Mean':>12} {'Std':>10}")
print("  " + "-" * 68)

if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
    for i, name in enumerate(feature_names):
        mn  = scaler.data_min_[i]
        mx  = scaler.data_max_[i]
        mean = getattr(scaler, 'mean_', [None]*8)[i] if hasattr(scaler, 'mean_') else None
        std  = getattr(scaler, 'scale_', [None]*8)[i] if hasattr(scaler, 'scale_') else None
        mean_str = f"{mean:>12.3f}" if mean is not None else "        N/A"
        std_str  = f"{std:>10.4f}"  if std  is not None else "       N/A"
        print(f"  {name:<18} {mn:>12.3f} {mx:>12.3f} {mean_str} {std_str}")
elif hasattr(scaler, 'mean_'):
    # StandardScaler
    for i, name in enumerate(feature_names):
        mean = scaler.mean_[i]
        std  = scaler.scale_[i]
        print(f"  {name:<18} mean={mean:>10.3f}  std={std:>10.4f}")

# ── 2. TEST SCALER BEHAVIOUR ──────────────────────────────────────────
print("\n[2] SCALER OUTPUT — Does it produce reasonable scaled values?")
test_inputs = [
    [60, 120, 0.8, 61.2, 63, 300, 600, 5],   # extreme flood
    [47,  35, 0.5, 47.9, 49,  87, 175, 5],   # moderate
    [30,   2, 0.2, 30.6, 31,   5,  10, 5],   # dry season
]
labels = ["Extreme flood", "Moderate     ", "Dry season   "]

for lbl, row in zip(labels, test_inputs):
    X = np.array([row], dtype=np.float32)
    Xs = scaler.transform(X)
    print(f"  {lbl} -> scaled: [{', '.join(f'{v:6.3f}' for v in Xs[0])}]")

# ── 3. LOAD BiLSTM AND CHECK WEIGHTS ─────────────────────────────────
print("\n[3] BILSTM — Weight statistics (detect vanishing/exploding gradients)")
try:
    import keras
except:
    import tensorflow as tf; keras = tf.keras

bilstm = keras.models.load_model(MODELS / "final_bilstm_model.keras", compile=False)
print(f"\n  Layers: {len(bilstm.layers)}")
for layer in bilstm.layers:
    weights = layer.get_weights()
    if weights:
        for j, w in enumerate(weights):
            print(f"  {layer.name:<30} w[{j}] shape={str(w.shape):<18} "
                  f"min={w.min():8.4f}  max={w.max():8.4f}  "
                  f"mean={w.mean():8.4f}  std={w.std():7.4f}")

# ── 4. PROBE MODEL WITH SYSTEMATIC INPUTS ────────────────────────────
print("\n[4] SYSTEMATIC PROBE — Vary one feature at a time")
print("    Does the model output CHANGE with different inputs?\n")

BASE_ROW = [47.0, 35.0, 0.50, 47.9, 49.4, 87.5, 175.0, 5.0]

def predict(row):
    X = np.array([row], dtype=np.float32)
    Xs = scaler.transform(X)
    seq = np.tile(Xs, (1, 7, 1))
    return float(bilstm.predict(seq, verbose=0)[0][0])

# Vary rainfall
print(f"  Varying RAINFALL (wl=47, soil=50%):")
print(f"  {'Rain (mm)':<12} {'Raw prob':>10}  {'Risk'}")
for rain in [0, 5, 20, 50, 80, 120, 150]:
    r = BASE_ROW.copy(); r[1] = rain; r[5] = rain*2.5; r[6] = rain*5
    p = predict(r)
    risk = "HIGH" if p > 0.6 else "MODERATE" if p > 0.3 else "LOW"
    print(f"  {rain:<12} {p:>10.4f}  {risk}")

# Vary water level
print(f"\n  Varying WATER LEVEL (rain=35, soil=50%):")
print(f"  {'WL (m)':<12} {'Raw prob':>10}  {'Risk'}")
for wl in [20, 30, 40, 47, 55, 60, 75]:
    r = BASE_ROW.copy(); r[0] = wl; r[3] = wl*1.02; r[4] = wl*1.05
    p = predict(r)
    risk = "HIGH" if p > 0.6 else "MODERATE" if p > 0.3 else "LOW"
    print(f"  {wl:<12} {p:>10.4f}  {risk}")

# Vary soil moisture
print(f"\n  Varying SOIL MOISTURE (rain=35, wl=47):")
print(f"  {'Soil (%)':<12} {'Raw prob':>10}  {'Risk'}")
for soil in [10, 20, 30, 50, 70, 90]:
    r = BASE_ROW.copy(); r[2] = soil/100
    p = predict(r)
    risk = "HIGH" if p > 0.6 else "MODERATE" if p > 0.3 else "LOW"
    print(f"  {soil:<12} {p:>10.4f}  {risk}")

# ── 5. EXAMINE TRAINING DATA BALANCE ─────────────────────────────────
print("\n[5] TRAINING DATA — Checking spatial_timeseries.csv distribution")
try:
    df = pd.read_csv(DATA / "spatial_timeseries .csv")
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"\n  Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    # Look for a flood label column
    label_cols = [c for c in df.columns if any(k in c for k in ['flood','label','risk','class','target'])]
    print(f"  Label columns found: {label_cols}")

    # Check WL distribution
    wl_col = next((c for c in df.columns if 'wl' in c and '1d' in c), None)
    if wl_col:
        print(f"\n  Water Level ({wl_col}) distribution:")
        print(f"  min={df[wl_col].min():.2f}  max={df[wl_col].max():.2f}  "
              f"mean={df[wl_col].mean():.2f}  std={df[wl_col].std():.2f}")
        q = df[wl_col].quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
        for pct, val in q.items():
            print(f"  {int(pct*100)}th percentile: {val:.2f}")

    rain_col = next((c for c in df.columns if 'rain' in c and '1d' in c), None)
    if rain_col:
        print(f"\n  Rainfall ({rain_col}) distribution:")
        print(f"  min={df[rain_col].min():.2f}  max={df[rain_col].max():.2f}  "
              f"mean={df[rain_col].mean():.2f}  std={df[rain_col].std():.2f}")
        zero_pct = (df[rain_col] == 0).mean() * 100
        print(f"  Zero-rain rows: {zero_pct:.1f}%")

except Exception as e:
    print(f"  Could not read spatial_timeseries: {e}")

# ── 6. CHECK FEATURES.PKL ─────────────────────────────────────────────
print("\n[6] FEATURES.PKL — What features did the model train on?")
try:
    try:
        feat = joblib.load(MODELS / "features.pkl")
    except:
        with open(MODELS / "features.pkl", "rb") as f:
            feat = pickle.load(f)
    print(f"  Content: {feat}")
    print(f"  Type   : {type(feat)}")
except Exception as e:
    print(f"  {e}")

# ── 7. CHECK THRESHOLD.JSON ───────────────────────────────────────────
print("\n[7] THRESHOLD.JSON — Full content:")
try:
    with open(MODELS / "threshold.json") as f:
        th = json.load(f)
    print(f"  {json.dumps(th, indent=4)}")
    print(f"\n  NOTE: threshold={th.get('threshold',0.5)} means any prob")
    print(f"  above {th.get('threshold',0.5)} is classified as FLOOD.")
    print(f"  If ALL outputs > threshold → model is saturated.")
except Exception as e:
    print(f"  {e}")

# ── 8. SPATIAL_TIMESERIES.PKL — Pre-processed training sequences ──────
print("\n[8] SPATIAL_TIMESERIES.PKL — Pre-built training sequences:")
try:
    try:
        st = joblib.load(MODELS / "spatial_timeseries.pkl")
    except:
        with open(MODELS / "spatial_timeseries.pkl", "rb") as f:
            st = pickle.load(f)
    if isinstance(st, np.ndarray):
        print(f"  Shape  : {st.shape}")
        print(f"  Dtype  : {st.dtype}")
        print(f"  Min    : {st.min():.4f}")
        print(f"  Max    : {st.max():.4f}")
        print(f"  Mean   : {st.mean():.4f}")
    elif isinstance(st, dict):
        print(f"  Keys   : {list(st.keys())}")
        for k, v in st.items():
            if hasattr(v, 'shape'):
                print(f"  [{k}] shape={v.shape} min={v.min():.3f} max={v.max():.3f}")
    elif isinstance(st, (list, tuple)):
        print(f"  Length : {len(st)}")
        print(f"  Type[0]: {type(st[0])}")
        if hasattr(st[0], 'shape'):
            print(f"  Shape[0]: {st[0].shape}")
except Exception as e:
    print(f"  {e}")

# ── CONCLUSION ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DIAGNOSIS COMPLETE")
print("=" * 65)
print("""
  The key questions answered above:
  [4] If rainfall/WL variation barely changes the output probability
      -> the BiLSTM is SATURATED (sigmoid stuck at ~1.0)
  [5] If >90% of training rows have flood=1 (or wl is always high)
      -> IMBALANCED TRAINING DATA caused the bias
  [3] If LSTM weights have very large values
      -> GRADIENT EXPLOSION during training
  [1] If scaler min/max covers extreme values (wl >> 80)
      -> Scaler was trained on a different data range
""")
print("=" * 65)
