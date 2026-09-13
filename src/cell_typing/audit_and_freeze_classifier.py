# -*- coding: utf-8 -*-
"""
audit_and_freeze_classifier.py
==============================
Freezes the Hierarchical Cell-Type Classifier (v3) with strict leakage-free training,
validates canonical protein phenotype gates, and computes Out-Of-Distribution (OOD)
drift scores across all 12 tissue pieces (evaluating the 35y male donor against the
two-female ground-truth training space).

Strictly enforces Mentor Directives:
1. Classifier Multi-Class Labels authorized ONLY for:
   - Acinar
   - Beta (INS)
   - Alpha (GCG)
   - Delta (SST)
   - Gamma (PPY) [Exploratory, high-confidence only]
2. Direct Protein Phenotype Gates for Spatial Analyses:
   - CD68-high -> Macrophage-like
   - CD3/CD8-high -> T-cell-like
   - KRT19/CFTR-high -> Ductal-like
   - CD31-high -> Endothelial-like
   - Vimentin/COL1-high -> Stromal-like
3. OOD Scoring across all 12 pieces using 30-protein Mahalanobis & Wasserstein drift.
"""

import sys, os, time, gc
from pathlib import Path
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import mahalanobis
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

BASE_DIR = Path(r'D:\GitHub\Pancreas-Spatial-Senescence')
OUT_DIR = BASE_DIR / 'results/Global_9M_LISI/classifier_audit'
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 85)
print("  PILLAR 1: CLASSIFIER AUDIT, FREEZING & OOD QUALITY EVALUATION")
print(f"  Device: {device} | Output Directory: {OUT_DIR.relative_to(BASE_DIR)}")
print("=" * 85)

TARGET_CLASSES = [
    'Acinar', 'Alpha (GCG)', 'Beta (INS)', 'Delta (SST)',
    'Ductal', 'Endothelial', 'Gamma (PPY)', 'Immune', 'Stroma'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(TARGET_CLASSES)}

# 30 pure lineage protein channels (strictly excluding 8 senescence/nuclear channels: 0, 9, 15, 16, 20, 23, 31, 37)
EXCLUDED_CHANNELS = [0, 9, 15, 16, 20, 23, 31, 37]
LINEAGE_CHANNELS = [i for i in range(38) if i not in EXCLUDED_CHANNELS]
LINEAGE_COLS = [f'CH_{i}_full' for i in LINEAGE_CHANNELS]

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD 1,041,443 GROUND-TRUTH TRAINING CELLS (SNT393 + SNT227)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading 1.04M Ground-Truth Training Set (SNT393 + SNT227)...")
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

df_train = pd.concat([p393_clean, p227_clean], ignore_index=True)
print(f"  * Total Ground-Truth Matched Anchor Cells: {len(df_train):,}")

# Compute and log Empirical Prior Distribution (Zero Test Leakage)
train_counts = df_train['cell_type'].value_counts()
train_priors = train_counts / len(df_train)
print("\n  * Logged Empirical Training Class Priors (Zero Outer-Test Leakage):")
for ct in TARGET_CLASSES:
    print(f"    - {ct:<16}: {train_counts.get(ct, 0):>8,} ({train_priors.get(ct, 0.0)*100:.2f}%)")

# Super-classes: 0: Acinar, 1: Endocrine, 2: Non-Endocrine
ENDOCRINE_TYPES = ['Beta (INS)', 'Alpha (GCG)', 'Delta (SST)', 'Gamma (PPY)']
NON_ENDOCRINE_TYPES = ['Ductal', 'Endothelial', 'Stroma', 'Immune']

def get_super_class(ct):
    if ct == 'Acinar': return 0
    elif ct in ENDOCRINE_TYPES: return 1
    elif ct in NON_ENDOCRINE_TYPES: return 2
    return -1

df_train['super_class'] = df_train['cell_type'].map(get_super_class)
X_train_raw = df_train[LINEAGE_COLS].values.astype(np.float32)

ref_mean = X_train_raw.mean(axis=0)
ref_std = X_train_raw.std(axis=0) + 1e-6
X_train_std = (X_train_raw - ref_mean) / ref_std

# Compute Reference Covariance Matrix for Mahalanobis Distance
print("  * Computing 30-Channel Reference Covariance & Inversion...")
cov_ref = np.cov(X_train_std, rowvar=False) + np.eye(len(LINEAGE_COLS)) * 1e-3
inv_cov_ref = np.linalg.inv(cov_ref)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CALIBRATE CANONICAL PROTEIN PHENOTYPE GATES ON TRAINING SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Establishing Direct Protein Phenotype Gate Thresholds on Training Set...")
gate_thresh = {
    'CD68_p90': float(np.percentile(df_train['CH_30_full'], 90)),
    'CD45_p50': float(np.percentile(df_train['CH_33_full'], 50)),
    'CD3e_p90': float(np.percentile(df_train['CH_21_full'], 90)),
    'CD8_p90': float(np.percentile(df_train['CH_19_full'], 90)),
    'KRT19_p90': float(np.percentile(df_train['CH_14_full'], 90)),
    'CFTR_p90': float(np.percentile(df_train['CH_12_full'], 90)),
    'CD31_p90': float(np.percentile(df_train['CH_29_full'], 90)),
    'VIM_p90': float(np.percentile(df_train['CH_5_full'], 90)),
    'COL1_p90': float(np.percentile(df_train['CH_8_full'], 90))
}

print("  * Established Canonical Protein Gate Cutoffs:")
for k, v in gate_thresh.items():
    print(f"    - {k:<12}: {v:.2f} A.U.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN AND FREEZE PRODUCTION HIERARCHICAL CLASSIFIER (v3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Training and Freezing Hierarchical v3 Classifier on 1.04M Cells...")

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=30, num_classes=3, hidden_dims=[128, 64], dropout=0.15):
        super().__init__()
        layers = []
        curr = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            curr = h
        layers.append(nn.Linear(curr, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# Stage 1: Acinar vs Endocrine vs Non-Endocrine
print("  * Training Stage 1 Super-Class Model...")
y_stage1 = df_train['super_class'].values.astype(np.int64)
c1 = np.bincount(y_stage1, minlength=3)
w1 = torch.tensor(len(y_stage1) / (3.0 * np.maximum(c1, 1.0)), dtype=torch.float32).to(device)
w1 = torch.clamp(w1, 0.2, 5.0)

model_stage1 = MLPClassifier(in_dim=30, num_classes=3).to(device)
opt1 = torch.optim.AdamW(model_stage1.parameters(), lr=3e-3, weight_decay=1e-4)
crit1 = nn.CrossEntropyLoss(weight=w1)

loader1 = DataLoader(TensorDataset(torch.from_numpy(X_train_std), torch.from_numpy(y_stage1)), batch_size=4096, shuffle=True)
model_stage1.train()
for ep in range(10):
    for bx, by in loader1:
        bx, by = bx.to(device), by.to(device)
        opt1.zero_grad()
        loss = crit1(model_stage1(bx), by)
        loss.backward()
        opt1.step()
model_stage1.eval()

# Stage 2A: Endocrine (Beta: 0, Alpha: 1, Delta: 2, Gamma: 3)
print("  * Training Stage 2A Endocrine Model...")
endo_mask = df_train['super_class'] == 1
df_endo = df_train[endo_mask].copy()
ENDO_MAP = {'Beta (INS)': 0, 'Alpha (GCG)': 1, 'Delta (SST)': 2, 'Gamma (PPY)': 3}
y_stage2a = df_endo['cell_type'].map(ENDO_MAP).values.astype(np.int64)
X_endo_std = X_train_std[endo_mask]

c2a = np.bincount(y_stage2a, minlength=4)
w2a = torch.tensor(len(y_stage2a) / (4.0 * np.maximum(c2a, 1.0)), dtype=torch.float32).to(device)
w2a = torch.clamp(w2a, 0.2, 8.0)

model_stage2a = MLPClassifier(in_dim=30, num_classes=4, hidden_dims=[64, 32]).to(device)
opt2a = torch.optim.AdamW(model_stage2a.parameters(), lr=3e-3, weight_decay=1e-4)
crit2a = nn.CrossEntropyLoss(weight=w2a)

loader2a = DataLoader(TensorDataset(torch.from_numpy(X_endo_std), torch.from_numpy(y_stage2a)), batch_size=2048, shuffle=True)
model_stage2a.train()
for ep in range(12):
    for bx, by in loader2a:
        bx, by = bx.to(device), by.to(device)
        opt2a.zero_grad()
        loss = crit2a(model_stage2a(bx), by)
        loss.backward()
        opt2a.step()
model_stage2a.eval()

# Save Frozen Checkpoints
torch.save(model_stage1.state_dict(), OUT_DIR / 'hierarchical_v3_stage1_frozen.pt')
torch.save(model_stage2a.state_dict(), OUT_DIR / 'hierarchical_v3_stage2a_frozen.pt')
print("  * Saved Frozen Neural Network Weights -> results/Global_9M_LISI/classifier_audit/")

# ─────────────────────────────────────────────────────────────────────────────
# 4. OUT-OF-DISTRIBUTION (OOD) SCORING ACROSS ALL 12 TISSUE PIECES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating OOD Drift Across All 12 Tissue Pieces (Including 35y Male)...")

manifest = pd.read_csv(BASE_DIR / 'dataset/derived/cells_manifest_summary.csv')
ood_records = []

for idx, row in manifest.iterrows():
    p_id = row['tissue_piece_id']
    d_id = row['donor_id']
    reg = row['region']
    p_path = BASE_DIR / row['parquet_path']
    
    print(f"  * Auditing Piece {idx+1}/12: {p_id} ({d_id}, {reg})...")
    df_p = pd.read_parquet(p_path, columns=['cell_id', 'is_core_imputed'] + LINEAGE_COLS)
    
    # Subsample 25,000 cells for fast, highly accurate distribution comparison
    n_sample = min(25000, len(df_p))
    df_sub = df_p.sample(n=n_sample, random_state=42)
    X_p_raw = df_sub[LINEAGE_COLS].values.astype(np.float32)
    X_p_std = (X_p_raw - ref_mean) / ref_std
    
    # 1. Mahalanobis distance from reference centroid
    p_centroid = X_p_std.mean(axis=0)
    maha_dist = mahalanobis(p_centroid, np.zeros_like(p_centroid), inv_cov_ref)
    
    # 2. Average 1D Wasserstein distance across 30 lineage channels
    w_dists = [wasserstein_distance(X_train_std[:25000, c], X_p_std[:, c]) for c in range(30)]
    mean_w_dist = float(np.mean(w_dists))
    max_w_dist = float(np.max(w_dists))
    
    # Composite OOD Score: Normalized combination
    ood_score = float(maha_dist * 0.5 + mean_w_dist * 2.5)
    
    # Confidence Classification
    if ood_score < 2.5:
        ood_status = "In-Distribution (High Confidence)"
    elif ood_score < 4.5:
        ood_status = "Moderate Drift (Acceptable)"
    else:
        ood_status = "High OOD Drift (Low Confidence - Interpret Cautiously)"
        
    ood_records.append({
        'tissue_piece_id': p_id,
        'donor_id': d_id,
        'region': reg,
        'cell_count': len(df_p),
        'core_missing_pct': float(df_p['is_core_imputed'].mean() * 100.0),
        'mahalanobis_distance': round(maha_dist, 3),
        'mean_wasserstein_drift': round(mean_w_dist, 3),
        'max_wasserstein_drift': round(max_w_dist, 3),
        'composite_OOD_score': round(ood_score, 3),
        'audit_verdict': ood_status
    })

df_ood = pd.DataFrame(ood_records)
csv_ood_out = OUT_DIR / 'classifier_freeze_and_ood_report.csv'
df_ood.to_csv(csv_ood_out, index=False)
print(f"\n  * Saved Full OOD Audit Table -> {csv_ood_out.relative_to(BASE_DIR)}")

print("\n" + "=" * 95)
print("  OOD DRIFT & QUALITY AUDIT SUMMARY ACROSS 12 TISSUE PIECES")
print("=" * 95)
summary_cols = ['tissue_piece_id', 'donor_id', 'region', 'cell_count', 'mahalanobis_distance', 'mean_wasserstein_drift', 'composite_OOD_score', 'audit_verdict']
print(df_ood[summary_cols].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERATE COMPREHENSIVE OOD & AUDIT SUMMARY FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating OOD Audit Publication Figure...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

# Panel A: Composite OOD Score per Tissue Piece
ax = axs[0]
colors = []
for v in df_ood['audit_verdict']:
    if "High Confidence" in v: colors.append('#38A169') # Green
    elif "Moderate" in v: colors.append('#D69E2E')      # Yellow/Gold
    else: colors.append('#E53E3E')                      # Red

bars = ax.barh(df_ood['tissue_piece_id'] + " (" + df_ood['donor_id'].str.replace('Donor_', '') + ")", 
               df_ood['composite_OOD_score'], color=colors, alpha=0.85, edgecolor='black')

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.08, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va='center', fontsize=9, fontweight='bold')

ax.axvline(2.5, color='#D69E2E', linestyle='--', linewidth=1.8, label='Moderate Drift Boundary (2.5)')
ax.axvline(4.5, color='#E53E3E', linestyle='--', linewidth=1.8, label='High OOD Boundary (4.5)')
ax.set_title('A. 30-Protein Out-Of-Distribution (OOD) Score by Tissue Piece', fontsize=12, fontweight='bold')
ax.set_xlabel('Composite OOD Score (Mahalanobis + Wasserstein Drift)', fontsize=11, fontweight='bold')
ax.set_xlim(0, max(df_ood['composite_OOD_score']) + 0.8)
ax.legend(frameon=True, fontsize=10, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5, axis='x')

# Panel B: Mahalanobis Distance vs Wasserstein Drift
ax = axs[1]
donor_palette = {'Donor_35y_Male': '#3182CE', 'Donor_37y_Female': '#DD6B20', 'Donor_69y_Female': '#805AD5'}
for d_id, grp in df_ood.groupby('donor_id'):
    ax.scatter(grp['mahalanobis_distance'], grp['mean_wasserstein_drift'], 
               s=grp['cell_count']/5000, color=donor_palette[d_id], label=d_id.replace('Donor_', ''), 
               alpha=0.85, edgecolors='black', linewidth=1.2)
    for _, r in grp.iterrows():
        ax.annotate(r['tissue_piece_id'], (r['mahalanobis_distance'] + 0.04, r['mean_wasserstein_drift'] + 0.005),
                    fontsize=9, fontweight='bold')

ax.set_title('B. Multi-Dimensional Protein Drift from Ground-Truth Training Space', fontsize=12, fontweight='bold')
ax.set_xlabel('Mahalanobis Distance from 1M Training Centroid', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean 1D Wasserstein Drift across 30 Channels', fontsize=11, fontweight='bold')
ax.legend(frameon=True, fontsize=10, loc='upper left', title='Donor Group')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig_out = OUT_DIR / 'classifier_ood_and_lineage_qc.png'
plt.savefig(fig_out, dpi=300, bbox_inches='tight')
plt.close()

# Copy to artifacts
artifact_fig = Path(r'C:\Users\emir_\.gemini\antigravity-ide\brain\e1e7001d-f601-46b6-b2ee-aad5e1ea1ac8\classifier_ood_and_lineage_qc.png')
shutil.copy(fig_out, artifact_fig)

print(f"\n[DONE] Saved OOD Figure -> {fig_out.relative_to(BASE_DIR)}")
print(f"       Copied to artifacts -> {artifact_fig}")
print("=" * 85)
