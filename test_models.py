
import os
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import keras
    print(f"Keras version: {keras.__version__}")
except ImportError as e:
    print(f"Failed to import Keras: {e}")
    sys.exit(1)

MODELS = Path("models")

model_files = [
    "final_bilstm_model.h5",
    "model_sar.keras",
    "final_model.keras"
]

for fname in model_files:
    p = MODELS / fname
    if p.exists():
        print(f"Attempting to load {fname}...")
        try:
            model = keras.models.load_model(str(p), compile=False)
            print(f"Successfully loaded {fname}")
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    else:
        print(f"File {fname} does not exist in models/")
