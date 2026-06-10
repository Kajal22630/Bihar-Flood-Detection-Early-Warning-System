import os, joblib, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

print("--- Asset Creation Protocol ---")

# 1. Create Pickles
try:
    path = DATA / "spatial_timeseries .csv"
    if not path.exists(): path = DATA / "spatial_timeseries.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    FEAT_COLS = ['wl_3d', 'rain_7d', 'soil_moisture', 'wl_1d', 'rain']
    X = df[FEAT_COLS].values
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    joblib.dump(scaler, MODELS / "scaler.pkl")
    with open(MODELS / "features.pkl", "wb") as f:
        pickle.dump(FEAT_COLS, f)
    
    print("[SUCCESS] scaler.pkl and features.pkl created.")
except Exception as e:
    print(f"[ERROR] Pickle generation failed: {e}")

# 2. Create Keras Model
print("Attempting Keras model generation...")
try:
    import tensorflow as tf
    # Robust import for tf.keras
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7, len(FEAT_COLS))),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model_path = MODELS / "final_bilstm_model.keras"
    model.save(str(model_path))
    print(f"[SUCCESS] {model_path.name} created.")
except Exception as e:
    print(f"[ERROR] Keras generation failed: {e}")

print("--- Protocol Complete ---")
