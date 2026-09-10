# -*- coding: utf-8 -*-
"""
analyze_spatial_senescence_atlas.py
===================================
Executes comprehensive biological analysis on the 9.4 Million Cell Partitioned Atlas:
1. Multi-marker Senescence-Associated Phenotype (SAP) quantification:
   - Stratified per cell type & per section:
     * SAP_p16: p16-high (top 10%) & Ki67-low (bottom 50%)
     * SAP_p21: p21-high (top 10%) & Ki67-low (bottom 50%)
     * SAP_Dual_LMNB1: (p16-high OR p21-high) & Lamin B1 loss (bottom 20%) & Ki67-low
     * SAP_DDR: DNA damage checkpoint (gH2AX or 53BP1 top 10%) & CDK inhibitor high & Ki67-low
     * Continuous SAP Score: Composite continuous index [0.0, 1.0]
2. Islet Spatial Architecture:
   - Endocrine cell clustering into discrete Islets (Beta, Alpha, Delta, Gamma; eps=25um, min=10 cells)
   - Distance to nearest exocrine boundary:
     * Islet_Border (dist <= 15 um)
     * Islet_Mantle (15 < dist <= 30 um)
     * Islet_Core (dist > 30 um)
   - Testing Border vs Core senescence enrichment.
3. Microenvironmental Niche Analysis:
   - Spatial proximity of Beta/Alpha cells to CD68 Macrophages / Immune cells
   - Local Collagen I / Stroma intensity
4. Outputs:
   - results/Global_9M_LISI/senescence_atlas/sap_summary_by_donor_and_lineage.csv
   - results/Global_9M_LISI/senescence_atlas/islet_border_core_enrichment.csv
   - results/Global_9M_LISI/senescence_atlas/microenvironment_proximity_summary.csv
   - Publication-quality multi-panel figures.
"""

import sys, os, time, gc
from pathlib import Path
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(r'D:\GitHub\Pancreas-Spatial-Senescence')
DERIVED_DIR = BASE_DIR / 'dataset/derived/cells'
OUT_DIR = BASE_DIR / 'results/Global_9M_LISI/senescence_atlas'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = BASE_DIR / 'dataset/derived/cells_manifest_summary.csv'
manifest = pd.read_csv(MANIFEST_PATH)

print("=" * 80)
print("  SPATIAL SENESCENCE ATLAS: 9.4M CELL BIOLOGICAL PHENOTYPING & ISLET NICHES")
print(f"  Sections: {len(manifest)} anatomical pieces across 3 donors")
print(f"  Output Directory: {OUT_DIR.relative_to(BASE_DIR)}")
print("=" * 80)

# Channel Indices
CH_P16 = 'CH_16_full'
CH_P21 = 'CH_9_full'
CH_KI67 = 'CH_31_full'
CH_LMNB1 = 'CH_20_full'
CH_HMGB1 = 'CH_23_full'
CH_GH2AX = 'CH_37_full'
CH_53BP1 = 'CH_15_full'
CH_COL1 = 'CH_8_full'

ENDOCRINE_TYPES = ['Beta (INS)', 'Alpha (GCG)', 'Delta (SST)', 'Gamma (PPY)']
LINEAGES_TO_EVALUATE = [
    'Acinar', 'Ductal', 'Beta (INS)', 'Alpha (GCG)',
    'Delta (SST)', 'Gamma (PPY)', 'Immune', 'Stroma', 'Endothelial'
]

# Trackers for aggregate data
sap_lineage_records = []
islet_records = []
microenv_records = []

t_all_start = time.time()

for idx, row in manifest.iterrows():
    piece_id = row['tissue_piece_id']
    donor_id = row['donor_id']
    region = row['region']
    age = row['age']
    sex = row['sex']
    p_path = BASE_DIR / row['parquet_path']
    
    print(f"\n[{idx+1}/{len(manifest)}] Processing Piece: {piece_id} | {donor_id} | {region} ({row['cell_count']:,} cells)...")
    t_piece = time.time()
    
    df = pd.read_parquet(p_path)
    n_cells = len(df)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. COMPUTE SAP PHENOTYPES PER CELL TYPE
    # ─────────────────────────────────────────────────────────────────────────
    df['is_sap_p16'] = False
    df['is_sap_p21'] = False
    df['is_sap_dual'] = False
    df['is_sap_ddr'] = False
    df['sap_continuous_score'] = 0.0
    
    for ctype in LINEAGES_TO_EVALUATE:
        c_mask = (df['predicted_cell_type'] == ctype).values
        n_c = c_mask.sum()
        if n_c < 50:
            continue
            
        p16_vals = df.loc[c_mask, CH_P16].values
        p21_vals = df.loc[c_mask, CH_P21].values
        ki67_vals = df.loc[c_mask, CH_KI67].values
        lmnb1_vals = df.loc[c_mask, CH_LMNB1].values
        gh2ax_vals = df.loc[c_mask, CH_GH2AX].values
        bp1_vals = df.loc[c_mask, CH_53BP1].values
        
        # Percentile ranks within this cell type [0.0, 1.0]
        r_p16 = rankdata(p16_vals) / n_c
        r_p21 = rankdata(p21_vals) / n_c
        r_ki67 = rankdata(ki67_vals) / n_c
        r_lmnb1 = rankdata(lmnb1_vals) / n_c
        r_gh2ax = rankdata(gh2ax_vals) / n_c
        r_bp1 = rankdata(bp1_vals) / n_c
        
        # Binary phenotypes
        p16_hi = (r_p16 >= 0.90)
        p21_hi = (r_p21 >= 0.90)
        ki67_lo = (r_ki67 <= 0.50)
        lmnb1_lo = (r_lmnb1 <= 0.20)
        ddr_hi = (r_gh2ax >= 0.90) | (r_bp1 >= 0.90)
        
        sap_p16 = p16_hi & ki67_lo
        sap_p21 = p21_hi & ki67_lo
        sap_dual = (p16_hi | p21_hi) & lmnb1_lo & ki67_lo
        sap_ddr = ddr_hi & (p16_hi | p21_hi) & ki67_lo
        
        # Continuous composite score
        score = (r_p16 + r_p21 + 0.5 * r_gh2ax + 0.5 * (1.0 - r_lmnb1)) / 3.0 * (1.0 - 0.5 * r_ki67)
        score = np.clip(score, 0.0, 1.0)
        
        df.loc[c_mask, 'is_sap_p16'] = sap_p16
        df.loc[c_mask, 'is_sap_p21'] = sap_p21
        df.loc[c_mask, 'is_sap_dual'] = sap_dual
        df.loc[c_mask, 'is_sap_ddr'] = sap_ddr
        df.loc[c_mask, 'sap_continuous_score'] = score
        
        # Record lineage-level rates
        sap_lineage_records.append({
            'tissue_piece_id': piece_id,
            'donor_id': donor_id,
            'region': region,
            'age': age,
            'sex': sex,
            'cell_type': ctype,
            'total_cells': n_c,
            'pct_sap_p16': round(float(sap_p16.mean() * 100), 2),
            'pct_sap_p21': round(float(sap_p21.mean() * 100), 2),
            'pct_sap_dual': round(float(sap_dual.mean() * 100), 2),
            'pct_sap_ddr': round(float(sap_ddr.mean() * 100), 2),
            'mean_sap_score': round(float(score.mean()), 4),
            'median_sap_score': round(float(np.median(score)), 4)
        })

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ISLET ARCHITECTURE & BORDER VS CORE ENRICHMENT
    # ─────────────────────────────────────────────────────────────────────────
    endo_mask = df['predicted_cell_type'].isin(ENDOCRINE_TYPES).values
    exo_mask = (~endo_mask) & (df['predicted_cell_type'] != 'Unknown').values
    n_endo = endo_mask.sum()
    
    if n_endo >= 50 and exo_mask.sum() >= 50:
        endo_coords = df.loc[endo_mask, ['x_um', 'y_um']].values
        exo_coords = df.loc[exo_mask, ['x_um', 'y_um']].values
        
        # Spatial clustering of endocrine cells into islets (eps=25 um, min_samples=10)
        db = DBSCAN(eps=25.0, min_samples=10, algorithm='kd_tree', n_jobs=-1).fit(endo_coords)
        islet_labels = db.labels_
        n_islets = len(set(islet_labels)) - (1 if -1 in islet_labels else 0)
        
        # Distance to nearest non-endocrine (exocrine/stromal) cell
        exo_tree = cKDTree(exo_coords)
        dists_to_exo, _ = exo_tree.query(endo_coords, k=1, workers=-1)
        
        df_endo = df.loc[endo_mask].copy()
        df_endo['islet_id'] = islet_labels
        df_endo['dist_to_exocrine_um'] = dists_to_exo
        
        # Filter cells assigned to real islets (not noise)
        in_islet_mask = (islet_labels >= 0)
        df_islet_cells = df_endo[in_islet_mask].copy()
        
        # Define anatomical zones within islets
        # Border: <= 15 um to exocrine boundary
        # Mantle: 15 um - 30 um
        # Core: > 30 um deep inside islet
        zones = np.full(len(df_islet_cells), 'Mantle', dtype=object)
        zones[df_islet_cells['dist_to_exocrine_um'] <= 15.0] = 'Border'
        zones[df_islet_cells['dist_to_exocrine_um'] > 30.0] = 'Core'
        df_islet_cells['islet_zone'] = zones
        
        print(f"     Identified {n_islets:,} Islets ({len(df_islet_cells):,} cells: "
              f"Border={(zones=='Border').sum():,}, Mantle={(zones=='Mantle').sum():,}, Core={(zones=='Core').sum():,})")
        
        for ctype in ['Beta (INS)', 'Alpha (GCG)']:
            df_c = df_islet_cells[df_islet_cells['predicted_cell_type'] == ctype]
            if len(df_c) < 30:
                continue
            for zone in ['Border', 'Mantle', 'Core']:
                df_z = df_c[df_c['islet_zone'] == zone]
                if len(df_z) < 10:
                    continue
                islet_records.append({
                    'tissue_piece_id': piece_id,
                    'donor_id': donor_id,
                    'region': region,
                    'age': age,
                    'sex': sex,
                    'cell_type': ctype,
                    'islet_zone': zone,
                    'cell_count': len(df_z),
                    'pct_sap_p16': round(float(df_z['is_sap_p16'].mean() * 100), 2),
                    'pct_sap_p21': round(float(df_z['is_sap_p21'].mean() * 100), 2),
                    'pct_sap_dual': round(float(df_z['is_sap_dual'].mean() * 100), 2),
                    'mean_sap_score': round(float(df_z['sap_continuous_score'].mean()), 4)
                })

    # ─────────────────────────────────────────────────────────────────────────
    # 3. IMMUNE & STROMA MICROENVIRONMENT
    # ─────────────────────────────────────────────────────────────────────────
    immune_mask = (df['predicted_cell_type'] == 'Immune').values
    beta_mask = (df['predicted_cell_type'] == 'Beta (INS)').values
    
    if immune_mask.sum() >= 20 and beta_mask.sum() >= 50:
        immune_coords = df.loc[immune_mask, ['x_um', 'y_um']].values
        beta_coords = df.loc[beta_mask, ['x_um', 'y_um']].values
        
        immune_tree = cKDTree(immune_coords)
        beta_to_immune_dist, _ = immune_tree.query(beta_coords, k=1, workers=-1)
        
        # Count immune cells within 25 um radius
        immune_in_25um = [len(idx_list) for idx_list in immune_tree.query_ball_point(beta_coords, r=25.0, workers=-1)]
        
        df_beta = df.loc[beta_mask].copy()
        df_beta['dist_to_immune_um'] = beta_to_immune_dist
        df_beta['immune_count_25um'] = immune_in_25um
        
        # Compare SAP-high vs SAP-low Beta cells
        # SAP-high: top 15% continuous score; SAP-low: bottom 50%
        thresh_hi = df_beta['sap_continuous_score'].quantile(0.85)
        thresh_lo = df_beta['sap_continuous_score'].quantile(0.50)
        
        beta_sap_hi = df_beta[df_beta['sap_continuous_score'] >= thresh_hi]
        beta_sap_lo = df_beta[df_beta['sap_continuous_score'] <= thresh_lo]
        
        if len(beta_sap_hi) >= 10 and len(beta_sap_lo) >= 10:
            microenv_records.append({
                'tissue_piece_id': piece_id,
                'donor_id': donor_id,
                'region': region,
                'age': age,
                'sex': sex,
                'sap_high_mean_dist_to_immune_um': round(float(beta_sap_hi['dist_to_immune_um'].mean()), 2),
                'sap_low_mean_dist_to_immune_um': round(float(beta_sap_lo['dist_to_immune_um'].mean()), 2),
                'sap_high_pct_with_immune_25um': round(float((beta_sap_hi['immune_count_25um'] > 0).mean() * 100), 2),
                'sap_low_pct_with_immune_25um': round(float((beta_sap_lo['immune_count_25um'] > 0).mean() * 100), 2),
                'sap_high_mean_collagen1': round(float(beta_sap_hi[CH_COL1].mean()), 2),
                'sap_low_mean_collagen1': round(float(beta_sap_lo[CH_COL1].mean()), 2),
            })
            
    print(f"     Completed piece in {time.time() - t_piece:.1f}s")
    del df
    gc.collect()

print(f"\n[ALL 12 SECTIONS PROCESSED in {time.time() - t_all_start:.1f}s]")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE SUMMARY CSV TABLES
# ─────────────────────────────────────────────────────────────────────────────
df_sap_lineage = pd.DataFrame(sap_lineage_records)
df_islet = pd.DataFrame(islet_records)
df_microenv = pd.DataFrame(microenv_records)

df_sap_lineage.to_csv(OUT_DIR / 'sap_summary_by_donor_and_lineage.csv', index=False)
df_islet.to_csv(OUT_DIR / 'islet_border_core_enrichment.csv', index=False)
df_microenv.to_csv(OUT_DIR / 'microenvironment_proximity_summary.csv', index=False)

print("\nSaved CSV tables:")
print(f"  * {OUT_DIR / 'sap_summary_by_donor_and_lineage.csv'}")
print(f"  * {OUT_DIR / 'islet_border_core_enrichment.csv'}")
print(f"  * {OUT_DIR / 'microenvironment_proximity_summary.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERATE PUBLICATION-QUALITY FIGURES
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# FIGURE 1: Senescence-Associated Phenotype (SAP) Landscape across Donors and Lineages
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# 1A: Mean SAP Score across Lineages by Donor
ax = axes[0, 0]
donor_palette = {'Donor_35y_Male': '#2b5c8f', 'Donor_37y_Female': '#388e3c', 'Donor_69y_Female': '#c2185b'}
sns.barplot(
    data=df_sap_lineage, x='cell_type', y='mean_sap_score', hue='donor_id',
    palette=donor_palette, ax=ax, errorbar='se', capsize=0.1
)
ax.set_title('A: Continuous SAP Score across Cell Lineages (by Donor)', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Cell Type', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Continuous SAP Score [0, 1]', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=35)
ax.legend(title='Donor Cohort', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

# 1B: Dual Senescence Rate (p16/p21 + Lamin B1 loss) in Endocrine vs Non-Endocrine
ax = axes[0, 1]
endo_sub = df_sap_lineage[df_sap_lineage['cell_type'].isin(['Beta (INS)', 'Alpha (GCG)', 'Acinar', 'Ductal', 'Immune'])]
sns.boxplot(
    data=endo_sub, x='cell_type', y='pct_sap_dual', hue='donor_id',
    palette=donor_palette, ax=ax, boxprops=dict(alpha=0.8)
)
ax.set_title('B: Canonical Dual SAP Rate (p16/p21-high + LMNB1-loss)', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Cell Type', fontsize=11, fontweight='bold')
ax.set_ylabel('% Dual SAP Cells', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.legend(title='Donor Cohort', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

# 1C: Regional Distribution of Beta Cell Senescence
ax = axes[1, 0]
beta_sub = df_sap_lineage[df_sap_lineage['cell_type'] == 'Beta (INS)']
sns.barplot(
    data=beta_sub, x='region', y='pct_sap_p16', hue='donor_id',
    palette=donor_palette, ax=ax, errorbar=None
)
ax.set_title('C: Beta Cell p16-high Rate across Pancreatic Anatomical Regions', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Anatomical Region', fontsize=11, fontweight='bold')
ax.set_ylabel('% Beta Cells p16-high (Ki67-low)', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=20)
ax.legend(title='Donor Cohort', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

# 1D: DDR (gH2AX / 53BP1) Rate across Donors
ax = axes[1, 1]
sns.barplot(
    data=df_sap_lineage[df_sap_lineage['cell_type'].isin(['Beta (INS)', 'Alpha (GCG)', 'Ductal', 'Acinar'])],
    x='cell_type', y='pct_sap_ddr', hue='donor_id',
    palette=donor_palette, ax=ax, errorbar='se', capsize=0.1
)
ax.set_title('D: DNA Damage Response (DDR) SAP Rate (gH2AX/53BP1 + CDK inhibitor)', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Cell Type', fontsize=11, fontweight='bold')
ax.set_ylabel('% DDR-Positive Senescent Cells', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=20)
ax.legend(title='Donor Cohort', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
fig1_path = OUT_DIR / 'fig1_sap_cohort_distribution.png'
plt.savefig(fig1_path)
plt.close()
print(f"Generated Figure 1 -> {fig1_path.relative_to(BASE_DIR)}")

# FIGURE 2: Islet Spatial Architecture (Border vs Mantle vs Core Senescence Enrichment)
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

zone_order = ['Border', 'Mantle', 'Core']
zone_colors = ['#d32f2f', '#f57c00', '#1976d2']

# 2A: Beta Cells SAP Score in Border vs Core
ax = axes[0]
beta_islet = df_islet[df_islet['cell_type'] == 'Beta (INS)']
sns.barplot(
    data=beta_islet, x='islet_zone', y='mean_sap_score', order=zone_order,
    palette=zone_colors, ax=ax, errorbar='se', capsize=0.1
)
ax.set_title('A: Beta Cell SAP Score: Islet Border vs Mantle vs Core', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Islet Anatomical Zone (Distance to Exocrine Pancreas)', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Continuous SAP Score', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

# Annotate zone definition
ax.text(0, ax.get_ylim()[0] + 0.01, '<=15 um\n(Border)', ha='center', va='bottom', fontsize=9, color='#333333')
ax.text(1, ax.get_ylim()[0] + 0.01, '15-30 um\n(Mantle)', ha='center', va='bottom', fontsize=9, color='#333333')
ax.text(2, ax.get_ylim()[0] + 0.01, '>30 um\n(Core)', ha='center', va='bottom', fontsize=9, color='#333333')

# 2B: Alpha Cells SAP Score in Border vs Core
ax = axes[1]
alpha_islet = df_islet[df_islet['cell_type'] == 'Alpha (GCG)']
sns.barplot(
    data=alpha_islet, x='islet_zone', y='mean_sap_score', order=zone_order,
    palette=zone_colors, ax=ax, errorbar='se', capsize=0.1
)
ax.set_title('B: Alpha Cell SAP Score: Islet Border vs Mantle vs Core', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Islet Anatomical Zone (Distance to Exocrine Pancreas)', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Continuous SAP Score', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

ax.text(0, ax.get_ylim()[0] + 0.01, '<=15 um\n(Border)', ha='center', va='bottom', fontsize=9, color='#333333')
ax.text(1, ax.get_ylim()[0] + 0.01, '15-30 um\n(Mantle)', ha='center', va='bottom', fontsize=9, color='#333333')
ax.text(2, ax.get_ylim()[0] + 0.01, '>30 um\n(Core)', ha='center', va='bottom', fontsize=9, color='#333333')

plt.tight_layout()
fig2_path = OUT_DIR / 'fig2_islet_core_vs_border_sap.png'
plt.savefig(fig2_path)
plt.close()
print(f"Generated Figure 2 -> {fig2_path.relative_to(BASE_DIR)}")

# FIGURE 3: Microenvironment Niche (Immune Proximity & Fibrosis around SAP-high Beta Cells)
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# 3A: Distance to Nearest Immune Cell (SAP-High vs SAP-Low Beta Cells)
ax = axes[0]
dist_comp = pd.DataFrame([
    {'State': 'SAP-High Beta', 'Distance_um': r['sap_high_mean_dist_to_immune_um'], 'Piece': r['tissue_piece_id']}
    for _, r in df_microenv.iterrows()
] + [
    {'State': 'SAP-Low Beta', 'Distance_um': r['sap_low_mean_dist_to_immune_um'], 'Piece': r['tissue_piece_id']}
    for _, r in df_microenv.iterrows()
])

sns.boxplot(data=dist_comp, x='State', y='Distance_um', palette=['#d32f2f', '#1976d2'], ax=ax, width=0.4)
sns.stripplot(data=dist_comp, x='State', y='Distance_um', color='black', alpha=0.6, jitter=0.1, size=7, ax=ax)
ax.set_title('A: Distance to Nearest Immune Cell from Beta Cells', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Beta Cell State', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Distance to Nearest Immune Cell (um)', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

# 3B: Collagen I (Fibrosis) around SAP-High vs SAP-Low Beta Cells
ax = axes[1]
col_comp = pd.DataFrame([
    {'State': 'SAP-High Beta', 'Collagen_I': r['sap_high_mean_collagen1'], 'Piece': r['tissue_piece_id']}
    for _, r in df_microenv.iterrows()
] + [
    {'State': 'SAP-Low Beta', 'Collagen_I': r['sap_low_mean_collagen1'], 'Piece': r['tissue_piece_id']}
    for _, r in df_microenv.iterrows()
])

sns.boxplot(data=col_comp, x='State', y='Collagen_I', palette=['#d32f2f', '#1976d2'], ax=ax, width=0.4)
sns.stripplot(data=col_comp, x='State', y='Collagen_I', color='black', alpha=0.6, jitter=0.1, size=7, ax=ax)
ax.set_title('B: Local Collagen I (Fibrotic Stroma) around Beta Cells', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Beta Cell State', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Collagen I Intensity (CH_8)', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
fig3_path = OUT_DIR / 'fig3_microenvironment_immune_fibrosis_niche.png'
plt.savefig(fig3_path)
plt.close()
print(f"Generated Figure 3 -> {fig3_path.relative_to(BASE_DIR)}")

print("\n" + "=" * 80)
print("  ALL ANALYSES AND FIGURES GENERATED SUCCESSFULLY!")
print("=" * 80)
