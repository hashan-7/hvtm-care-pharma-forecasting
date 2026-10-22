"""
HVTM CARE - Data Preprocessing & Date Normalization Utility
Author: Thiloka Indhuwari (@ThilokaIndhuwari)
Module: Temporal Alignment & Dataset Chronological Formatting
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "drugs.csv")
if not os.path.exists(DATA_FILE_PATH):
    DATA_FILE_PATH = os.path.join(PROJECT_ROOT, "drugs.csv")

df = pd.read_csv(DATA_FILE_PATH)

correct_dates_list = pd.date_range(start='2020-01-01', periods=73, freq='MS').strftime('%Y-%m-%d').tolist()
full_dates = correct_dates_list * 30

df['Date'] = full_dates
df.to_csv(DATA_FILE_PATH, index=False)

print("[SUCCESS] File Updated Successfully.\n")
print("[VERIFICATION] First Brand - 2020/2021 Pattern:")
print(df[['Date', 'Brand_Name']].iloc[0:15]) 

print("\n[VERIFICATION] Middle Brand - Transition Check:")
print(df[['Date', 'Brand_Name']].iloc[70:80])
