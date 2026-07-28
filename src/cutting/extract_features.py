import os
import glob
import time
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

base_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\tiles"
age_folders = sorted(glob.glob(os.path.join(base_dir, "age_*")))

properties = ['label', 'area', 'centroid', 'mean_intensity']
parquet_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\parquets"
os.makedirs(parquet_dir, exist_ok=True)

for folder in age_folders:
    folder_name = os.path.basename(folder)
    parts = folder_name.split('_')
    age_val = int(parts[1])
    patient_id = parts[2]
    
    out_parquet = os.path.join(parquet_dir, f"features_{patient_id}_age{age_val}.parquet")
    
    if os.path.exists(out_parquet):
        print(f"SKIPPED: {out_parquet} already exists.")
        continue
        
    print(f"\n{'='*40}\nEXTRACTING FEATURES: {patient_id} (Age {age_val})\n{'='*40}")
    
    mask_files = sorted(glob.glob(os.path.join(folder, "*_mask.tiff")))
    patient_df_list = []
    start_patient = time.time()
    
    for i, mask_path in enumerate(mask_files):
        raw_path = mask_path.replace('_mask.tiff', '.tiff')
        if not os.path.exists(raw_path):
            continue
            
        basename = os.path.basename(raw_path).replace('.tiff', '')
        _, y_str, x_str = basename.split('_')
        tile_y, tile_x = int(y_str.replace('y', '')), int(x_str.replace('x', ''))
        
        masks = tifffile.imread(mask_path)
        raw_img = tifffile.imread(raw_path) 
        
        raw_img_hwc = np.moveaxis(raw_img, 0, -1)
        
        props = regionprops_table(masks, intensity_image=raw_img_hwc, properties=properties)
        df_tile = pd.DataFrame(props)
        
        if df_tile.empty:
            continue
            
        df_tile['patient_id'] = patient_id
        df_tile['age'] = age_val
        df_tile['tile_y'] = tile_y
        df_tile['tile_x'] = tile_x
        df_tile['global_y'] = df_tile['centroid-0'] + tile_y
        df_tile['global_x'] = df_tile['centroid-1'] + tile_x
        
        df_tile.drop(columns=['centroid-0', 'centroid-1'], inplace=True)
        patient_df_list.append(df_tile)
        
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(mask_files)}] tiles processed...")
            
    if patient_df_list:
        final_patient_df = pd.concat(patient_df_list, ignore_index=True)
        rename_dict = {f"mean_intensity-{ch}": f"CH_{ch}" for ch in range(38)}
        final_patient_df.rename(columns=rename_dict, inplace=True)
        
        final_patient_df.to_parquet(out_parquet, engine='pyarrow', index=False)
        
        elapsed = time.time() - start_patient
        print(f"-> SUCCESS! {len(final_patient_df)} cells saved. ({elapsed:.1f} s)")
        print(f"-> Saved to: {out_parquet}")

print("\n\nFeature extraction completed for all patients!")
