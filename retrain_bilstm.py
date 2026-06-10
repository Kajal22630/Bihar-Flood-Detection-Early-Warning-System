"""
retrain_bilstm.py - Retrains the BiLSTM model to fix the saturation issue.
Generates balanced targets from historical data and learns meaningful thresholds.
"""

import os, sys, warnings, joblib, pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Silence TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"

print("=" * 60)
print("  BILSTM RETRAINING INITIATED")
print("=" * 60)

# 1. Load and sort data
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA / "spatial_timeseries .csv")
df.columns = [c.strip().lower() for c in df.columns]
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['subdistrict', 'date'], inplace=True)

# 2. Feature Engineering & Target Generation
print("[2] Generating lag features and balanced targets...")
def prep_group(grp):
    grp = grp.copy()
    grp['wl_lag1'] = grp['wl_1d'].shift(1).fillna(grp['wl_1d'])
    grp['wl_lag2'] = grp['wl_1d'].shift(2).fillna(grp['wl_1d'])
    grp['rain_lag1'] = grp['rain'].shift(1).fillna(grp['rain'])
    grp['wl_trend'] = grp['wl_1d'] - grp['wl_3d']
    grp['rain_trend'] = grp['rain'] - grp['rain_7d']
    
    # Target definition (Flood Risk = 1 if WL is unusually high AND conditions are wet/rising)
    danger_lvl = grp['wl_1d'].mean() + 0.5 * grp['wl_1d'].std()
    grp['target'] = ((grp['wl_1d'] > danger_lvl) & 
                     ((grp['rain'] > 5) | (grp['wl_trend'] > 0) | (grp['soil_moisture'] > 0.3))).astype(int)
    return grp

df = df.groupby('subdistrict', group_keys=False).apply(prep_group)

# 3. Scale Features
print("[3] Scaling features...")
FEAT_COLS = ['wl_3d', 'rain_7d', 'soil_moisture', 'wl_lag1', 'wl_lag2', 'rain_lag1', 'wl_trend', 'rain_trend']
X_raw = df[FEAT_COLS].values
y_raw = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Save the new scaler
joblib.dump(scaler, MODELS / "scaler.pkl")
# Also save as feature array for app.py
with open(MODELS / "features.pkl", "wb") as f:
    pickle.dump(FEAT_COLS, f)

# 4. Sequence Generation (timesteps=7)
print("[4] Generating sequences (timesteps=7)...")
def create_sequences(X, y, seq_length=7):
    Xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length - 1])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_raw)

print(f"    Sequence shape: {X_seq.shape}")
print(f"    Class balance : 0={np.sum(y_seq==0)}  1={np.sum(y_seq==1)}")

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 6. Model Architecture
print("\n[5] Building new BiLSTM architecture...")
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(7, len(FEAT_COLS))),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("\n[6] Training model (with Early Stopping)...")
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 8. Evaluation & Save
from sklearn.metrics import precision_score, recall_score, f1_score

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n[7] FINAL PERFORMANCE METRICS:")
print(f"    ├─ Accuracy : {acc:.4f}")
print(f"    ├─ Precision: {precision:.4f}")
print(f"    ├─ Recall   : {recall:.4f}")
print(f"    └─ F1-Score : {f1:.4f}")

model_path = MODELS / "final_bilstm_model.keras"
model.save(model_path)
print(f"\n  ✅ Model successfully retrained and saved to: {model_path}")
print("  ✅ Scaler successfully saved to: scaler.pkl")
print("=" * 60)

