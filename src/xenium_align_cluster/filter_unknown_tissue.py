import pandas as pd
from pathlib import Path
import shutil

parquet_dir = Path(r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\parquets")

def filter_parquet(filename, condition_func, desc):
    filepath = parquet_dir / filename
    backup_path = parquet_dir / (filename + ".bak")
    
    if not filepath.exists():
        print(f"File not found: {filename}")
        return
        
    if not backup_path.exists():
        shutil.copy(filepath, backup_path)
        print(f"Created backup: {backup_path.name}")
        
    df = pd.read_parquet(filepath)
    original_len = len(df)
    
    df_filtered = condition_func(df)
    filtered_len = len(df_filtered)
    
    print(f"[{desc}] {filename}:")
    print(f"  Original cells: {original_len}")
    print(f"  Filtered cells: {filtered_len}")
    print(f"  Removed cells : {original_len - filtered_len}")
    
    df_filtered.to_parquet(filepath)
    print(f"  Overwrote {filename} with filtered data.\n")

# SNT227: Delete top (y < 26000), keep bottom (y >= 26000)
def filter_227(df):
    return df[df['global_y'] >= 26000].copy()

# SNT675: Delete bottom (y >= 29000), keep top (y < 29000)
def filter_675(df):
    return df[df['global_y'] < 29000].copy()

filter_parquet("features_dual_SNT227_age69.parquet", filter_227, "Keep Bottom")
filter_parquet("features_dual_SNT675_age69.parquet", filter_675, "Keep Top")

print("Filtering complete! Unknown tissues purged.")
