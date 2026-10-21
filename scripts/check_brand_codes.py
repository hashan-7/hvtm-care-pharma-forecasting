import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Resolve project data path dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "drugs.csv")
if not os.path.exists(DATA_FILE_PATH):
    DATA_FILE_PATH = os.path.join(PROJECT_ROOT, "drugs.csv")

df = pd.read_csv(DATA_FILE_PATH)

le = LabelEncoder()
le.fit(df['Brand_Name'])

print("\nBRAND CODE REFERENCE LIST")
print("=" * 40)
print(f"{'CODE':<5} | {'BRAND NAME'}")
print("-" * 40)

brand_mapping = dict(zip(le.transform(le.classes_), le.classes_))

for code, name in brand_mapping.items():
    print(f"{code:<5} | {name}")

print("=" * 40)
