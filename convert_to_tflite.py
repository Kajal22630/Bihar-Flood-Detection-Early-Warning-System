import tensorflow as tf
import os
from pathlib import Path

MODELS = Path("models")
OUTPUT = Path("models/tflite")
OUTPUT.mkdir(exist_ok=True)

def convert(name):
    keras_path = MODELS / name
    if not keras_path.exists():
        print(f"Skipping {name} (not found)")
        return
    
    print(f"--- Converting {name} to TFLite ---")
    try:
        model = tf.keras.models.load_model(keras_path, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Try to avoid Flex ops first
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        out_name = name.replace(".keras", ".tflite").replace(".h5", ".tflite")
        with open(OUTPUT / out_name, "wb") as f:
            f.write(tflite_model)
        print(f"DONE: Saved to {OUTPUT / out_name}")
    except Exception as e:
        print(f"Retrying with Flex ops for {name}...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            out_name = name.replace(".keras", ".tflite").replace(".h5", ".tflite")
            with open(OUTPUT / out_name, "wb") as f:
                f.write(tflite_model)
            print(f"DONE (with Flex): Saved to {OUTPUT / out_name}")
        except Exception as e2:
            print(f"FAILED {name}: {e2}")


if __name__ == "__main__":
    # Already done one, but re-run for safety
    convert("final_bilstm_model.keras")
    convert("model_sar.keras")
    convert("final_model.keras")
