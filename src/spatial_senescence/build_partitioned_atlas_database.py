# -*- coding: utf-8 -*-
"""
build_partitioned_atlas_database.py
===================================
Constructs the partitioned, single-cell spatial atlas database across all 9.4 Million Cells:
- Directory Partition Schema:
    dataset/derived/cells/donor_id=.../region=.../tissue_piece_id=.../cells.parquet
- Model:
    LineageMLP trained on 1,041,443 ground-truth matched cells (32 lineage proteins only;
    p16, p21, Lamin B1, HMGB1, Ki67, DAPI strictly EXCLUDED to avoid senescence leakage).
- Output Fields per Cell:
    cell_id, x_um, y_um, cell_area, tissue_piece_id, donor_id, region, age, sex
    CH_0_full ... CH_37_full (38 full-cell channels)
    CH_0_core ... CH_37_core (38 raw nuclear channels, preserved with NaNs)
    is_core_imputed (1 if nucleus missing, 0 if segmented)
    predicted_cell_type (Acinar, Beta, Alpha, Delta, Ductal, Endothelial, Gamma, Immune, Stroma, Unknown)
    cell_type_confidence (softmax probability; confidence < 0.50 flagged as 'Unknown')
- Outputs Manifest Summary:
    dataset/derived/cells_manifest_summary.csv
"""

import sys, os, time, gc
from pathlib import Path
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print("  BUILDING 9.4M CELL PARTITIONED SPATIAL SENESCENCE ATLAS DATABASE")
print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 80)

BASE_DIR = Path(r'D:\GitHub\Pancreas-Spatial-Senescence')
PARQUET_DIR = BASE_DIR / 'dataset/parquets'
DERIVED_DIR = BASE_DIR / 'dataset/derived/cells'
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = [
    'Acinar', 'Alpha (GCG)', 'Beta (INS)', 'Delta (SST)',
    'Ductal', 'Endothelial', 'Gamma (PPY)', 'Immune', 'Stroma'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(TARGET_CLASSES)}

# 32 Lineage channels (excluding DAPI:0, p21:9, p16:16, Lamin B1:20, HMGB1:23, Ki67:31)
EXCLUDED_CHANNELS = [0, 9, 16, 20, 23, 31]
LINEAGE_CHANNELS = [i for i in range(38) if i not in EXCLUDED_CHANNELS]
LINEAGE_COLS = [f'CH_{i}_full' for i in LINEAGE_CHANNELS]

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAIN LINEAGE MLP ON 1M GROUND-TRUTH CELLS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Training Senescence-Free Lineage MLP on 1M Ground-Truth Cells...")
t_train_start = time.time()

p393 = pd.read_parquet(BASE_DIR / 'src/xenium_align_cluster/snt393_xenium_registration/SNT393_matches_high_confidence_5um.parquet')
p227 = pd.read_parquet(BASE_DIR / 'src/xenium_align_cluster/snt227_xenium_registration/SNT227_matches_high_confidence_5um.parquet')

ad393 = sc.read_h5ad(BASE_DIR / 'dataset/spatial-transkriptomics/annotated_secondary_analysis_393.h5ad')
ad227 = sc.read_h5ad(BASE_DIR / 'dataset/spatial-transkriptomics/annotated_secondary_analysis_227.h5ad')

cid393 = dict(zip(ad393.obs['cell_id'], ad393.obs['final_cell_type']))
cid227 = dict(zip(ad227.obs['cell_id'], ad227.obs['final_cell_type']))

p393['cell_type'] = p393['xenium_cell_id'].map(cid393)
p227['cell_type'] = p227['xenium_cell_id'].map(cid227)

p393_clean = p393[p393['cell_type'].isin(TARGET_CLASSES)].copy()
p227_clean = p227[p227['cell_type'].isin(TARGET_CLASSES)].copy()

# Standardize slide-level for training
X393 = p393_clean[LINEAGE_COLS].values.astype(np.float32)
X227 = p227_clean[LINEAGE_COLS].values.astype(np.float32)

X393 = (X393 - X393.mean(axis=0, keepdims=True)) / (X393.std(axis=0, keepdims=True) + 1e-6)
X227 = (X227 - X227.mean(axis=0, keepdims=True)) / (X227.std(axis=0, keepdims=True) + 1e-6)

X_train = np.vstack([X393, X227])
y_train = np.concatenate([
    p393_clean['cell_type'].map(CLASS_TO_IDX).values.astype(np.int64),
    p227_clean['cell_type'].map(CLASS_TO_IDX).values.astype(np.int64)
])

del p393, p227, ad393, ad227, p393_clean, p227_clean, X393, X227
gc.collect()

print(f"  * Combined Training Set: {len(X_train):,} cells across SNT393 and SNT227")

# Class weights
counts = np.bincount(y_train, minlength=9)
class_weights = torch.tensor(len(y_train) / (9.0 * np.maximum(counts, 1.0)), dtype=torch.float32).to(device)

class LineageMLP(nn.Module):
    def __init__(self, in_dim=32, num_classes=9, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

mlp = LineageMLP(in_dim=32, num_classes=9).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=4096, shuffle=True)
mlp.train()
for ep in range(1, 16):
    ep_loss = 0.0
    for bx, by in tr_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(mlp(bx), by)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    scheduler.step()
    if ep % 5 == 0 or ep == 15:
        print(f"    Epoch {ep:02d}/15 - Loss: {ep_loss / len(tr_loader):.4f}")

mlp.eval()
print(f"  * Lineage MLP training finished in {time.time() - t_train_start:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINE THE 12 SECTIONS SPECIFICATION
# ─────────────────────────────────────────────────────────────────────────────
sections_def = [
    # Donor 1: 35y Male
    {
        'piece_id': 'SNT354', 'donor_id': 'Donor_35y_Male', 'region': 'Head_Inferior',
        'parquet_file': 'features_dual_SNT354_age35.parquet', 'y_filter': lambda y: y >= 21217,
        'age': 35, 'sex': 'Male'
    },
    {
        'piece_id': 'SNT869', 'donor_id': 'Donor_35y_Male', 'region': 'Body_Superior',
        'parquet_file': 'features_dual_SNT354_age35.parquet', 'y_filter': lambda y: y < 21217,
        'age': 35, 'sex': 'Male'
    },
    {
        'piece_id': 'SNT899', 'donor_id': 'Donor_35y_Male', 'region': 'Head_Superior',
        'parquet_file': 'features_dual_SNT899_age35.parquet', 'y_filter': lambda y: y >= 17203,
        'age': 35, 'sex': 'Male'
    },
    {
        'piece_id': 'SNT648', 'donor_id': 'Donor_35y_Male', 'region': 'Tail_Superior',
        'parquet_file': 'features_dual_SNT899_age35.parquet', 'y_filter': lambda y: y < 17203,
        'age': 35, 'sex': 'Male'
    },
    # Donor 2: 37y Female
    {
        'piece_id': 'SNT348', 'donor_id': 'Donor_37y_Female', 'region': 'Head_Inferior',
        'parquet_file': 'features_dual_SNT348_age37.parquet', 'y_filter': lambda y: y < 25559,
        'age': 37, 'sex': 'Female'
    },
    {
        'piece_id': 'SNT854', 'donor_id': 'Donor_37y_Female', 'region': 'Body_Superior',
        'parquet_file': 'features_dual_SNT348_age37.parquet', 'y_filter': lambda y: y >= 25559,
        'age': 37, 'sex': 'Female'
    },
    {
        'piece_id': 'SNT393', 'donor_id': 'Donor_37y_Female', 'region': 'Tail',
        'parquet_file': 'features_dual_SNT393_age37.parquet', 'y_filter': lambda y: y < 17203,
        'age': 37, 'sex': 'Female'
    },
    {
        'piece_id': 'SNT875', 'donor_id': 'Donor_37y_Female', 'region': 'Head_Superior',
        'parquet_file': 'features_dual_SNT393_age37.parquet', 'y_filter': lambda y: y >= 17203,
        'age': 37, 'sex': 'Female'
    },
    # Donor 3: 69y Female
    {
        'piece_id': 'SNT484', 'donor_id': 'Donor_69y_Female', 'region': 'Head_Superior',
        'parquet_file': 'features_dual_SNT484_age69.parquet', 'y_filter': lambda y: y < 17203,
        'age': 69, 'sex': 'Female'
    },
    {
        'piece_id': 'SNT282', 'donor_id': 'Donor_69y_Female', 'region': 'Tail_Superior',
        'parquet_file': 'features_dual_SNT484_age69.parquet', 'y_filter': lambda y: y >= 17203,
        'age': 69, 'sex': 'Female'
    },
    {
        'piece_id': 'SNT227', 'donor_id': 'Donor_69y_Female', 'region': 'Head_Inferior',
        'parquet_file': 'features_dual_SNT227_age69.parquet', 'y_filter': lambda y: np.ones_like(y, dtype=bool),
        'age': 69, 'sex': 'Female'
    },
    {
        'piece_id': 'SNT675', 'donor_id': 'Donor_69y_Female', 'region': 'Body_Superior',
        'parquet_file': 'features_dual_SNT675_age69.parquet', 'y_filter': lambda y: np.ones_like(y, dtype=bool),
        'age': 69, 'sex': 'Female'
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROCESS EACH SECTION & BUILD PARTITIONED DATABASE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Processing 12 Anatomical Sections into Partitioned Parquets...")
manifest_rows = []
total_processed_cells = 0

# Cache loaded raw parquets to avoid re-reading the same file twice
loaded_parquets = {}

for s_idx, sec in enumerate(sections_def, 1):
    t_sec_start = time.time()
    fname = sec['parquet_file']
    piece_id = sec['piece_id']
    donor_id = sec['donor_id']
    region = sec['region']
    
    print(f"\n  >> [{s_idx}/12] Processing Piece: {piece_id} | Donor: {donor_id} | Region: {region}...")
    
    if fname not in loaded_parquets:
        print(f"     Loading raw parquet: {fname}...")
        df_raw = pd.read_parquet(PARQUET_DIR / fname)
        loaded_parquets[fname] = df_raw
    else:
        df_raw = loaded_parquets[fname]
        
    # Apply Y-filter to isolate the exact tissue island
    y_vals = df_raw['tile_y'].values
    mask = sec['y_filter'](y_vals)
    df_island = df_raw[mask].copy()
    n_cells = len(df_island)
    total_processed_cells += n_cells
    print(f"     Isolate Island: {n_cells:,} cells")
    
    # 1. Feature normalization for classifier
    feat_mat = df_island[LINEAGE_COLS].values.astype(np.float32)
    feat_std = (feat_mat - feat_mat.mean(axis=0, keepdims=True)) / (feat_mat.std(axis=0, keepdims=True) + 1e-6)
    
    # 2. Predict cell types and confidence with Lineage MLP on GPU
    preds_list = []
    confs_list = []
    with torch.no_grad():
        for i in range(0, len(feat_std), 8192):
            bx = torch.from_numpy(feat_std[i:i+8192]).to(device)
            probs = F.softmax(mlp(bx), dim=-1)
            max_p, pred_c = torch.max(probs, dim=-1)
            preds_list.append(pred_c.cpu().numpy())
            confs_list.append(max_p.cpu().numpy())
            
    pred_indices = np.concatenate(preds_list)
    pred_confidences = np.concatenate(confs_list).astype(np.float32)
    
    # Assign names and threshold at confidence < 0.50 -> 'Unknown'
    pred_names = np.array([IDX_TO_CLASS[idx] for idx in pred_indices], dtype=object)
    low_conf_mask = (pred_confidences < 0.50)
    pred_names[low_conf_mask] = 'Unknown'
    n_unknown = low_conf_mask.sum()
    
    # 3. Detect core missingness flag
    core_missing_mask = df_island['CH_0_core'].isna().values
    core_missing_frac = float(core_missing_mask.mean())
    is_core_imp = core_missing_mask.astype(np.uint8)
    
    # 4. Construct clean dataframe for this piece
    df_out = pd.DataFrame()
    df_out['cell_id'] = df_island['label'].values.astype(np.int64) if 'label' in df_island else np.arange(n_cells, dtype=np.int64)
    df_out['x_um'] = df_island['global_x'].values.astype(np.float32) if 'global_x' in df_island else df_island['tile_x'].values.astype(np.float32)
    df_out['y_um'] = df_island['global_y'].values.astype(np.float32) if 'global_y' in df_island else df_island['tile_y'].values.astype(np.float32)
    df_out['cell_area'] = df_island['area'].values.astype(np.float32) if 'area' in df_island else np.zeros(n_cells, dtype=np.float32)
    df_out['tissue_piece_id'] = piece_id
    df_out['donor_id'] = donor_id
    df_out['region'] = region
    df_out['age'] = sec['age']
    df_out['sex'] = sec['sex']
    
    # Copy all 38 raw full channels
    for ch in range(38):
        df_out[f'CH_{ch}_full'] = df_island[f'CH_{ch}_full'].values.astype(np.float32)
        
    # Copy all 38 raw core channels (preserving original NaNs)
    for ch in range(38):
        df_out[f'CH_{ch}_core'] = df_island[f'CH_{ch}_core'].values.astype(np.float32)
        
    df_out['is_core_imputed'] = is_core_imp
    if 'area_core' in df_island:
        df_out['area_core'] = df_island['area_core'].values.astype(np.float32)
    if 'n_c_ratio' in df_island:
        df_out['n_c_ratio'] = df_island['n_c_ratio'].values.astype(np.float32)
    df_out['predicted_cell_type'] = pred_names
    df_out['cell_type_confidence'] = pred_confidences
    
    # 5. Save to partitioned directory
    # Path: dataset/derived/cells/donor_id=.../region=.../tissue_piece_id=.../cells.parquet
    target_dir = DERIVED_DIR / f"donor_id={donor_id}" / f"region={region}" / f"tissue_piece_id={piece_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "cells.parquet"
    df_out.to_parquet(out_file, index=False, compression='zstd')
    
    sec_time = time.time() - t_sec_start
    print(f"     Saved Partition -> {out_file.relative_to(BASE_DIR)} ({os.path.getsize(out_file) / 1e6:.1f} MB, {sec_time:.1f}s)")
    print(f"     Core Missing: {core_missing_frac:.1%} | Unknown (conf<0.50): {n_unknown:,} ({n_unknown/n_cells:.1%})")
    
    # Track statistics for manifest
    ct_counts = df_out['predicted_cell_type'].value_counts()
    manifest_rows.append({
        'tissue_piece_id': piece_id,
        'donor_id': donor_id,
        'region': region,
        'age': sec['age'],
        'sex': sec['sex'],
        'cell_count': n_cells,
        'core_missing_pct': round(core_missing_frac * 100, 2),
        'unknown_pct': round(n_unknown / n_cells * 100, 2),
        'pct_Acinar': round(ct_counts.get('Acinar', 0) / n_cells * 100, 2),
        'pct_Beta': round(ct_counts.get('Beta (INS)', 0) / n_cells * 100, 2),
        'pct_Alpha': round(ct_counts.get('Alpha (GCG)', 0) / n_cells * 100, 2),
        'pct_Delta': round(ct_counts.get('Delta (SST)', 0) / n_cells * 100, 2),
        'pct_Ductal': round(ct_counts.get('Ductal', 0) / n_cells * 100, 2),
        'pct_Immune': round(ct_counts.get('Immune', 0) / n_cells * 100, 2),
        'pct_Stroma': round(ct_counts.get('Stroma', 0) / n_cells * 100, 2),
        'pct_Endothelial': round(ct_counts.get('Endothelial', 0) / n_cells * 100, 2),
        'pct_Gamma': round(ct_counts.get('Gamma (PPY)', 0) / n_cells * 100, 2),
        'parquet_path': str(out_file.relative_to(BASE_DIR))
    })

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE MANIFEST SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Compiling Cohort Manifest Summary...")
df_manifest = pd.DataFrame(manifest_rows)
manifest_path = BASE_DIR / 'dataset/derived/cells_manifest_summary.csv'
df_manifest.to_csv(manifest_path, index=False)
print("Manifest Summary saved to:", manifest_path)
print("\n" + "=" * 80)
print(df_manifest[['tissue_piece_id', 'donor_id', 'region', 'cell_count', 'core_missing_pct', 'pct_Beta', 'pct_Alpha', 'pct_Immune', 'unknown_pct']].to_string(index=False))
print("=" * 80)

print(f"\n[4/4] COMPLETE! Successfully partitioned {total_processed_cells:,} cells into 12 clean Parquets.")
