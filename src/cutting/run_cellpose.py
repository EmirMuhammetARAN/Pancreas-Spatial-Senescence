import os
import glob
import sys
import numpy as np
import tifffile
from cellpose.models import CellposeModel
import time
from skimage.segmentation import clear_border

sys.stdout.reconfigure(encoding='utf-8')

CH_DAPI = 0
CH_ECAD = 28

model = CellposeModel(gpu=True, pretrained_model='cpsam_v2')
base_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\tiles"

def norm(ch):
    p1, p99 = np.percentile(ch, 1), np.percentile(ch, 99)
    return np.clip((ch - p1) / (p99 - p1 + 1e-8), 0, 1)

age_folders = sorted(glob.glob(os.path.join(base_dir, "age_*")))

for folder in age_folders:
    age_name = os.path.basename(folder)
    tiff_files = sorted([f for f in glob.glob(os.path.join(folder, "*.tiff")) if '_mask' not in f])
    
    print(f"\n{'='*40}\n{age_name}: Processing {len(tiff_files)} tiles...\n{'='*40}")

    for i, tiff_path in enumerate(tiff_files):
        out_path = tiff_path.replace('.tiff', '_mask.tiff')
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(tiff_files)}] Skipped (already exists): {os.path.basename(tiff_path)}")
            continue

        try:
            img = tifffile.imread(tiff_path)
            
            if img is None or img.ndim < 3 or img.shape[0] == 0 or img.shape[0] <= max(CH_ECAD, CH_DAPI):
                print(f"  [{i+1}/{len(tiff_files)}] SKIPPED (empty/corrupted): {os.path.basename(tiff_path)}")
                continue

            two_channel = np.stack([norm(img[CH_ECAD]), norm(img[CH_DAPI])], axis=0)

            start = time.time()
            masks, flows, styles = model.eval(two_channel, diameter=None, channels=[1, 2])
            elapsed = time.time() - start

            masks_clean = clear_border(masks)
            n_cells = len(np.unique(masks_clean)) - 1
            
            tifffile.imwrite(out_path, masks_clean.astype(np.uint32))
            print(f"  [{i+1}/{len(tiff_files)}] {os.path.basename(tiff_path)}: {n_cells} cells ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(tiff_files)}] ERROR ({os.path.basename(tiff_path)}): {e}")

print("\n\nAll processing completed!")
