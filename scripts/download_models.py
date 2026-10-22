import os
import sys
import zipfile
import urllib.request

RELEASE_URL = "https://github.com/hashan-7/hvtm-care-pharma-forecasting/releases/download/v4.1.0/models.zip"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TEMP_ZIP_PATH = os.path.join(PROJECT_ROOT, "models_temp.zip")

REQUIRED_MODELS = [
    "scaler_X.pkl", "scaler_y.pkl", "linear_regression_model.pkl",
    "random_forest_model.pkl", "xgboost_model.pkl", "BiLSTM_model.h5",
    "GRU_model.h5", "NeuroStack_MetaLearner.pkl"
]

def check_models_exist():
    if not os.path.exists(MODELS_DIR):
        return False
    return all(os.path.exists(os.path.join(MODELS_DIR, m)) for m in REQUIRED_MODELS)

def download_models(force=False):
    if not force and check_models_exist():
        print("[INFO] All required model artifacts already exist in models/.")
        return True

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[DOWNLOAD] Fetching pre-trained model weights from GitHub Release v4.1.0...")
    print(f"URL: {RELEASE_URL}")

    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rDownloading: {percent}% [{count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB]")
                sys.stdout.flush()

        urllib.request.urlretrieve(RELEASE_URL, TEMP_ZIP_PATH, reporthook)
        print("\n[EXTRACT] Unpacking models into models/ directory...")
        
        with zipfile.ZipFile(TEMP_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)

        if os.path.exists(TEMP_ZIP_PATH):
            os.remove(TEMP_ZIP_PATH)

        print("[SUCCESS] All model artifacts downloaded and unpacked successfully!")
        return True
    except Exception as e:
        if os.path.exists(TEMP_ZIP_PATH):
            os.remove(TEMP_ZIP_PATH)
        print(f"\n[ERROR] Failed to download model weights: {e}")
        return False

if __name__ == "__main__":
    download_models(force=True)
