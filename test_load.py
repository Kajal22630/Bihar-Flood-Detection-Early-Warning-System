import os
import tensorflow as tf
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE = Path(__file__).parent.absolute()
MODEL_PATH = BASE / "models" / "final_bilstm_model.h5"

try:
    print(f"Loading {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("SUCCESS: Model Loaded.")
    print(f"Input Shape: {model.input_shape}")
except Exception as e:
    print(f"FAILURE: {e}")
