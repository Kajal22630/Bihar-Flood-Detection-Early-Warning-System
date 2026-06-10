import os, sys, warnings
from pathlib import Path

# Fix for potential binary compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    import keras
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {keras.__version__}")
except ImportError:
    print("TF/Keras not found")
    sys.exit(1)

BASE = Path(__file__).parent.absolute()
MODELS = BASE / "models"

def convert_to_keras(h5_name):
    h5_path = MODELS / h5_name
    keras_path = MODELS / h5_name.replace(".h5", ".keras")
    
    if not h5_path.exists():
        print(f"File not found: {h5_path}")
        return

    print(f"Loading {h5_name}...")
    try:
        # Load with compile=False to avoid layer initialization issues
        model = keras.models.load_model(str(h5_path), compile=False)
        print(f"Saving to {keras_path.name}...")
        model.save(str(keras_path))
        print("Success!")
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    convert_to_keras("final_bilstm_model.h5")
