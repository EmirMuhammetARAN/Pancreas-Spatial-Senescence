import os
import subprocess
import shutil
import sys

python_exe = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\venv\Scripts\python.exe"

donors = [
    ("SNT393", 37, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\PC24043_Scan1.qptiff"),
    ("SNT348", 37, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\PC24056_Scan1.qptiff"),
    ("SNT354", 35, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\SNT354_PC24058_Scan1.qptiff"),
    ("SNT869", 35, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\SNT869_PC24058_Scan1.qptiff"),
    ("SNT899", 35, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\SNT899_PC24046_Scan1.qptiff"),
    ("SNT675", 69, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\SNT675_PC24049_Scan1.qptiff"),
    ("SNT227", 69, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\SNT227_PC24038_Scan1.qptiff"),
    ("SNT484", 69, r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images\SNT484_PC24045_Scan1.qptiff"),
]

parquet_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\parquets"
tiles_base = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\tiles"

def run_command(cmd):
    print(f"\n---> RUNNING: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"!!! ERROR: {cmd} !!!")
        sys.exit(1)

for snt, age, qptiff_path in donors:
    parquet_path = os.path.join(parquet_dir, f"features_{snt}_age{age}.parquet")
    
    if os.path.exists(parquet_path):
        print(f"[{snt}] Parquet exists, skipping.")
        continue
        
    print(f"\n{'='*50}\nSTARTING: {snt} (AGE {age}) - PIPELINE\n{'='*50}")
    
    tile_dir = os.path.join(tiles_base, f"age_{age}_{snt}")
    
    run_command(f'"{python_exe}" src/cutting/extract_all_tiles.py')
    run_command(f'"{python_exe}" src/cutting/run_cellpose.py')
    run_command(f'"{python_exe}" src/cutting/extract_features.py')
    
    if os.path.exists(parquet_path):
        print(f"[{snt}] Parquet successfully generated. Cleaning up tiles and masks...")
        try:
            shutil.rmtree(tile_dir)
            print(f"[{snt}] Cleanup successful: {tile_dir} removed.")
        except Exception as e:
            print(f"[{snt}] Error during cleanup: {e}")
    else:
        print(f"[{snt}] WARNING: Parquet not generated! Skipping cleanup.")
        sys.exit(1)

print("\nPIPELINE COMPLETED SUCCESSFULLY!")
