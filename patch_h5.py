
import h5py
import json
import os
import zipfile
import shutil

def clean_config_dict(obj):
    if isinstance(obj, dict):
        if 'quantization_config' in obj:
            del obj['quantization_config']
        if obj.get('class_name') == 'InputLayer' and 'optional' in obj.get('config', {}):
            del obj['config']['optional']
        for k, v in obj.items():
            clean_config_dict(v)
    elif isinstance(obj, list):
        for item in obj:
            clean_config_dict(item)

def patch_h5(path):
    print(f"Patching H5: {path}...")
    try:
        with h5py.File(path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config = json.loads(config_str)
                clean_config_dict(config)
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                print("  Done.")
    except Exception as e:
        print(f"  Failed: {e}")

def patch_keras(path):
    print(f"Patching Keras: {path}...")
    try:
        temp_dir = path + "_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        config_path = os.path.join(temp_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            clean_config_dict(config)
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Re-zip
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_ref.write(file_path, arcname)
            print("  Done.")
        else:
            print("  config.json not found in archive.")
        
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"  Failed: {e}")

models_dir = 'models'
for f in os.listdir(models_dir):
    p = os.path.join(models_dir, f)
    if f.endswith('.h5'):
        patch_h5(p)
    elif f.endswith('.keras'):
        patch_keras(p)
