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

# Pass 1: Cell boundary model (E-Cadherin + DAPI)
model_cell = CellposeModel(gpu=True, pretrained_model='cpsam_v2')
# Pass 2: Dedicated nuclear model on DAPI
try:
    model_nuc = CellposeModel(gpu=True, pretrained_model='nuclei')
except Exception:
    # Fallback to cpsam_v2 if separate nuclei weights not downloaded
    model_nuc = model_cell

base_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\tiles"

def norm(ch):
    p1, p99 = np.percentile(ch, 1), np.percentile(ch, 99)
    return np.clip((ch - p1) / (p99 - p1 + 1e-8), 0, 1)

age_folders = sorted(glob.glob(os.path.join(base_dir, "age_*")))

for folder in age_folders:
    age_name = os.path.basename(folder)
    tiff_files = sorted([f for f in glob.glob(os.path.join(folder, "*.tiff")) if '_mask' not in f])
    
    print(f"\n{'='*40}\n{age_name}: Processing {len(tiff_files)} tiles (Dual-Mask Mode)...\n{'='*40}")

    for i, tiff_path in enumerate(tiff_files):
        out_cell_path = tiff_path.replace('.tiff', '_mask.tiff')
        out_core_path = tiff_path.replace('.tiff', '_core_mask.tiff')

        if os.path.exists(out_cell_path) and os.path.exists(out_core_path):
            print(f"  [{i+1}/{len(tiff_files)}] Skipped (already exists): {os.path.basename(tiff_path)}")
            continue

        try:
            img = tifffile.imread(tiff_path)
            
            if img is None or img.ndim < 3 or img.shape[0] == 0 or img.shape[0] <= max(CH_ECAD, CH_DAPI):
                print(f"  [{i+1}/{len(tiff_files)}] SKIPPED (empty/corrupted): {os.path.basename(tiff_path)}")
                continue

            ecad_norm = norm(img[CH_ECAD])
            dapi_norm = norm(img[CH_DAPI])

            start = time.time()

            # 1. Whole-Cell Segmentation Pass (E-Cadherin + DAPI)
            two_channel = np.stack([ecad_norm, dapi_norm], axis=0)
            masks_cell, flows_c, styles_c = model_cell.eval(two_channel, diameter=None, channels=[1, 2])
            masks_cell_clean = clear_border(masks_cell)

            # 2. True Nuclear Segmentation Pass (DAPI standalone)
            masks_nuc, flows_n, styles_n = model_nuc.eval(dapi_norm, diameter=None, channels=[0, 0])
            masks_nuc_clean = clear_border(masks_nuc)

            # 3. 1-to-1 Label Harmonization:
            # Nuclear pixels inside cell k strictly inherit label k.
            # Cells with no nucleus in this 2D section naturally receive 0 pixels.
            core_masks_clean = np.where((masks_cell_clean > 0) & (masks_nuc_clean > 0), masks_cell_clean, 0)

            elapsed = time.time() - start

            n_cells = len(np.unique(masks_cell_clean)) - 1
            n_nuclei = len(np.unique(core_masks_clean)) - 1
            
            tifffile.imwrite(out_cell_path, masks_cell_clean.astype(np.uint32))
            tifffile.imwrite(out_core_path, core_masks_clean.astype(np.uint32))

            print(f"  [{i+1}/{len(tiff_files)}] {os.path.basename(tiff_path)}: {n_cells} cells, {n_nuclei} matched nuclei ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(tiff_files)}] ERROR ({os.path.basename(tiff_path)}): {e}")

print("\n\nAll dual-mask processing completed!")
