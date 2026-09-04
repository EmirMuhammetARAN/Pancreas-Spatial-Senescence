# -*- coding: utf-8 -*-
import os, sys
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
"""
lisi_03_wnn_joint.py
====================
WNN (RNA + Protein Birlikte) için LISI Hesaplama
-------------------------------------------------
Strateji:
  - cLISI: Ground Truth etiketine sahip hücrelerden cell_type'a göre
            TABAKALı (stratified) örnekleme ile ~150.000 hücre seçilir.
  - iLISI: 9.4M hücrelerin TAMAMINDAN patient_id'ye göre
            TABAKALı (stratified) örnekleme ile ~200.000 hücre seçilir.
  - Embedding: global_9M_embeddings.npy
      * Sütun 0:30  → RNA (Harmony, 30 boyut)
      * Sütun 30:94 → Protein (Novae, 64 boyut)
  - WNN: muon.pp.neighbors() ile MuData üzerinde joint ağ
  - KNN: 90 komşu ile scib_metrics.ilisi_knn / clisi_knn
"""

import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import scib_metrics
import gc
import time
from pathlib import Path

# ─────────────────────────────────────────────
# YOLLAR
# ─────────────────────────────────────────────
DATA_DIR   = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset")
MATCH_393  = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt393_xenium_registration")
MATCH_227  = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt227_xenium_registration")
OUT_DIR    = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\results\Global_9M_LISI")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLISI = 150_000   # cLISI orneklem buyuklugu
N_ILISI = 200_000   # iLISI orneklem buyuklugu
N_NEIGHBORS = 90
RANDOM_STATE = 42

print("=" * 60)
print("  WNN (RNA + PROTEIN JOINT) LISI - STRATIFIED SAMPLING")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. GROUND TRUTH ETİKETLERİ
# ─────────────────────────────────────────────
print("\n[1/7] Ground Truth etiketleri yukleniyor...")

adata_ref_393 = sc.read_h5ad(
    DATA_DIR / "spatial-transkriptomics" / "annotated_secondary_analysis_393.h5ad",
    backed='r'
)
df_type_393 = adata_ref_393.obs[['cell_id', 'final_cell_type']].copy().rename(
    columns={'cell_id': 'xenium_cell_id'}
)
match_393 = pd.read_parquet(MATCH_393 / "SNT393_matches_high_confidence_5um.parquet")
match_393 = match_393.merge(df_type_393, on='xenium_cell_id', how='inner')
match_393['uid'] = (match_393['tile_x'].astype(str) + "_" +
                    match_393['tile_y'].astype(str) + "_" +
                    match_393['label'].astype(str))
gt_dict_393 = match_393.set_index('uid')['final_cell_type'].to_dict()

adata_ref_227 = sc.read_h5ad(
    DATA_DIR / "spatial-transkriptomics" / "annotated_secondary_analysis_227.h5ad",
    backed='r'
)
df_type_227 = adata_ref_227.obs[['cell_id', 'final_cell_type']].copy().rename(
    columns={'cell_id': 'xenium_cell_id'}
)
match_227 = pd.read_parquet(MATCH_227 / "SNT227_matches_high_confidence_5um.parquet")
match_227 = match_227.merge(df_type_227, on='xenium_cell_id', how='inner')
match_227['uid'] = (match_227['tile_x'].astype(str) + "_" +
                    match_227['tile_y'].astype(str) + "_" +
                    match_227['label'].astype(str))
gt_dict_227 = match_227.set_index('uid')['final_cell_type'].to_dict()

gt_dict = {**gt_dict_393, **gt_dict_227}
del adata_ref_393, df_type_393, match_393, adata_ref_227, df_type_227, match_227
gc.collect()

print(f"  Toplam GT uid sayisi: {len(gt_dict):,}")

# ─────────────────────────────────────────────
# 2. META VE EMBEDDİNG (MEMMAP)
# ─────────────────────────────────────────────
print("\n[2/7] Metadata yukleniyor (sadece uid + patient_id)...")
# mmap: 7GB dosyayi RAM'e YUKLEMEZ, sadece adresler
embeddings_full = np.load(DATA_DIR / "global_9M_embeddings.npy", mmap_mode='r')

# Sadece ihtiyac olan 2 kolon -> RAM'de kucuk
meta_full = pd.read_parquet(
    DATA_DIR / "global_9M_embeddings_metadata.parquet",
    columns=['uid', 'patient_id']
)
meta_full = meta_full.reset_index(drop=True)

print(f"  Toplam hucre: {len(meta_full):,}")

# cLISI filtresi: once isin() ile GT satirlari bul (hizli hash)
# sonra sadece o ~1M satira .map() uygula
gt_set = set(gt_dict.keys())
mask_gt = meta_full['uid'].isin(gt_set)
meta_ref = meta_full[mask_gt].copy()
meta_ref = meta_ref.reset_index(drop=False)  # orijinal satir no -> 'index' kolonu
meta_ref['ground_truth_type'] = meta_ref['uid'].map(gt_dict)  # sadece ~1M satir

print(f"  GT etiketli hucre: {len(meta_ref):,}")
del gt_set, mask_gt
gc.collect()

# ─────────────────────────────────────────────
# 3. YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────
def stratified_sample_with_min(df, group_col, n_total, min_per_group=50,
                                random_state=42):
    """
    group_col'a gore orantili stratified orneklem.
    Hicbir grup min_per_group'tan az temsil edilemez (nadir hucre tipi korumasi).
    """
    frac = n_total / len(df)
    samples = []
    for name, grp in df.groupby(group_col):
        n_sample = max(min_per_group, int(round(len(grp) * frac)))
        n_sample = min(n_sample, len(grp))
        samples.append(grp.sample(n=n_sample, random_state=random_state))
    result = pd.concat(samples)
    if len(result) > n_total:
        result = result.sample(n=n_total, random_state=random_state)
    return result


def stratified_sample_proportional(df, group_col, n_total, random_state=42):
    """
    group_col'a gore tam orantili stratified orneklem.
    Her hastanin agirliği toplam veriyle orantili kalir.
    """
    samples = []
    for name, grp in df.groupby(group_col):
        n_sample = int(round(len(grp) / len(df) * n_total))
        n_sample = min(n_sample, len(grp))
        if n_sample > 0:
            samples.append(grp.sample(n=n_sample, random_state=random_state))
    result = pd.concat(samples)
    if len(result) > n_total:
        result = result.sample(n=n_total, random_state=random_state)
    return result


def sparse_to_nn_results(dist_mat, n_neighbors):
    """Scipy sparse KNN matrisini scib_metrics NeighborsResults'a donusturur."""
    N = dist_mat.shape[0]
    k = n_neighbors
    indices_arr   = np.zeros((N, k), dtype=np.int32)
    distances_arr = np.zeros((N, k), dtype=np.float32)
    for i in range(N):
        s = dist_mat.indptr[i]
        e = dist_mat.indptr[i + 1]
        cols = dist_mat.indices[s:e]
        data = dist_mat.data[s:e]
        order = np.argsort(data)
        cols, data = cols[order], data[order]
        kk = min(k, len(cols))
        indices_arr[i, :kk]   = cols[:kk]
        distances_arr[i, :kk] = data[:kk]
    return scib_metrics.nearest_neighbors.NeighborsResults(
        indices=indices_arr, distances=distances_arr
    )

print("\n[3/7] cLISI icin Stratified Orneklem (cell_type'a gore)...")
# meta_ref zaten yukarida isin()+map() ile olusturuldu (~1M GT hucre)

meta_clisi = stratified_sample_with_min(
    meta_ref, 'ground_truth_type', N_CLISI, min_per_group=50,
    random_state=RANDOM_STATE
)
print(f"  cLISI orneklem boyutu: {len(meta_clisi):,}")
print("  Cell type dagilimi:")
print(meta_clisi['ground_truth_type'].value_counts().to_string())

# ─────────────────────────────────────────────
# 5. iLISI — STRATİFİED BY PATIENT ID (FULL 9.4M)
# ─────────────────────────────────────────────
print("\n[4/7] iLISI icin Stratified Orneklem (patient_id'ye gore, tum 9.4M)...")

meta_ilisi = stratified_sample_proportional(
    meta_full, 'patient_id', N_ILISI, random_state=RANDOM_STATE
)
print(f"  iLISI orneklem boyutu: {len(meta_ilisi):,}")
print("  Patient ID dagilimi:")
print(meta_ilisi['patient_id'].value_counts().to_string())

# ─────────────────────────────────────────────
# 6. EMBEDDİNG YÜKLEMESİ
#    RNA (0:30) ve Protein (30:94)
# ─────────────────────────────────────────────
print("\n[5/7] RNA ve Protein embedding'leri RAM'e aliniyor...")

clisi_idx = meta_clisi['index'].to_numpy()
ilisi_idx = meta_ilisi.index.to_numpy()

# cLISI için RNA + Protein
rna_clisi  = np.array(embeddings_full[clisi_idx, :30],  dtype=np.float32)
prot_clisi = np.array(embeddings_full[clisi_idx, 30:],  dtype=np.float32)

# iLISI için RNA + Protein
rna_ilisi  = np.array(embeddings_full[ilisi_idx, :30],  dtype=np.float32)
prot_ilisi = np.array(embeddings_full[ilisi_idx, 30:],  dtype=np.float32)

del embeddings_full
gc.collect()

# ─────────────────────────────────────────────
# 7. WNN GRAPH + LISI HESAPLAMA
# ─────────────────────────────────────────────
print("\n[6/7] WNN Grafi + LISI hesaplaniyor...")

# ────── 7a. cLISI (WNN) ──────
print("\n  --- cLISI icin WNN ---")
t0 = time.time()

adata_rna_c  = sc.AnnData(X=rna_clisi)
adata_prot_c = sc.AnnData(X=prot_clisi)

adata_rna_c.obs['batch']      = meta_clisi['patient_id'].astype('category').values
adata_rna_c.obs['cell_type']  = meta_clisi['ground_truth_type'].astype('category').values
adata_prot_c.obs['batch']     = meta_clisi['patient_id'].astype('category').values
adata_prot_c.obs['cell_type'] = meta_clisi['ground_truth_type'].astype('category').values

mdata_c = mu.MuData({'rna': adata_rna_c, 'prot': adata_prot_c})
mdata_c.obs['batch']     = meta_clisi['patient_id'].astype('category').values
mdata_c.obs['cell_type'] = meta_clisi['ground_truth_type'].astype('category').values

# Modalite bazli KNN (WNN on-kosulu)
sc.pp.neighbors(mdata_c['rna'],  n_neighbors=N_NEIGHBORS, use_rep='X')
sc.pp.neighbors(mdata_c['prot'], n_neighbors=N_NEIGHBORS, use_rep='X')

# WNN
mu.pp.neighbors(mdata_c, n_neighbors=N_NEIGHBORS)

nn_c = sparse_to_nn_results(mdata_c.obsp['distances'], N_NEIGHBORS)
clisi_scores = scib_metrics.clisi_knn(
    X=nn_c, labels=mdata_c.obs['cell_type'].values, scale=False
)
wnn_clisi_mean = float(np.mean(clisi_scores))
print(f"  cLISI (WNN) tamamlandi: {wnn_clisi_mean:.4f}  ({time.time()-t0:.1f} sn)")

del adata_rna_c, adata_prot_c, mdata_c, nn_c, rna_clisi, prot_clisi
gc.collect()

# ────── 7b. iLISI (WNN) ──────
print("\n  --- iLISI icin WNN ---")
t0 = time.time()

adata_rna_i  = sc.AnnData(X=rna_ilisi)
adata_prot_i = sc.AnnData(X=prot_ilisi)

adata_rna_i.obs['batch']  = meta_ilisi['patient_id'].astype('category').values
adata_prot_i.obs['batch'] = meta_ilisi['patient_id'].astype('category').values

mdata_i = mu.MuData({'rna': adata_rna_i, 'prot': adata_prot_i})
mdata_i.obs['batch'] = meta_ilisi['patient_id'].astype('category').values

# Modalite bazli KNN
sc.pp.neighbors(mdata_i['rna'],  n_neighbors=N_NEIGHBORS, use_rep='X')
sc.pp.neighbors(mdata_i['prot'], n_neighbors=N_NEIGHBORS, use_rep='X')

# WNN
mu.pp.neighbors(mdata_i, n_neighbors=N_NEIGHBORS)

nn_i = sparse_to_nn_results(mdata_i.obsp['distances'], N_NEIGHBORS)
ilisi_scores = scib_metrics.ilisi_knn(
    X=nn_i, batches=mdata_i.obs['batch'].values, scale=False
)
wnn_ilisi_mean = float(np.mean(ilisi_scores))
print(f"  iLISI (WNN) tamamlandi: {wnn_ilisi_mean:.4f}  ({time.time()-t0:.1f} sn)")

del adata_rna_i, adata_prot_i, mdata_i, nn_i, rna_ilisi, prot_ilisi
gc.collect()

# ─────────────────────────────────────────────
# SONUÇLAR
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  WNN (JOINT)  |  iLISI: {wnn_ilisi_mean:.4f}  |  cLISI: {wnn_clisi_mean:.4f}")
print("=" * 60)

md_path = OUT_DIR / "LISI_WNN_Joint.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# WNN (RNA + Protein Joint) LISI Sonuclari\n\n")
    f.write("## Orneklem Stratejisi\n")
    f.write(f"- **iLISI**: Tum 9.4M hucreden `patient_id`'ye gore tabakali, **{len(meta_ilisi):,}** hucre\n")
    f.write(f"- **cLISI**: GT etiketli hucrelerden `cell_type`'a gore tabakali (min 50/tip), **{len(meta_clisi):,}** hucre\n")
    f.write(f"- KNN komsu sayisi: `{N_NEIGHBORS}`\n")
    f.write("- WNN: muon.pp.neighbors() ile RNA + Protein joint agirlikli komsu grafi\n\n")
    f.write("## Sonuclar\n\n")
    f.write("| Metrik | Deger | Yorum |\n")
    f.write("|--------|-------|-------|\n")
    f.write(f"| iLISI (Batch Mixing) | **{wnn_ilisi_mean:.4f}** | Yuksek = iyi karisim |\n")
    f.write(f"| cLISI (Cell Type Purity) | **{wnn_clisi_mean:.4f}** | Dusuk = iyi saflik |\n")
    f.write("\n## Cell Type Dagilimi (cLISI orneklem)\n\n```\n")
    f.write(meta_clisi['ground_truth_type'].value_counts().to_string())
    f.write("\n```\n\n## Patient ID Dagilimi (iLISI orneklem)\n\n```\n")
    f.write(meta_ilisi['patient_id'].value_counts().to_string())
    f.write("\n```\n")

# ─────────────────────────────────────────────
# 8. OZET TABLOSU (3 modality)
# ─────────────────────────────────────────────
print("\n[7/7] Ozet sonuc tablosu yaziliyor...")
summary_path = OUT_DIR / "LISI_Summary_Stratified.md"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("# 9.4M Global Stratified LISI Ozet Tablosu\n\n")
    f.write("> Ornekleme Yontemi: Tabakali (Stratified) — cLISI icin cell_type'a gore, iLISI icin patient_id'ye gore\n\n")
    f.write("| Modality | iLISI (Batch Mixing) | cLISI (Cell Type Purity) | cLISI Orneklem | iLISI Orneklem |\n")
    f.write("|----------|---------------------|--------------------------|----------------|----------------|\n")
    f.write(f"| Protein (Novae) | — | — | ~{N_CLISI//1000}K | ~{N_ILISI//1000}K |\n")
    f.write(f"| RNA (Harmony)   | — | — | ~{N_CLISI//1000}K | ~{N_ILISI//1000}K |\n")
    f.write(f"| Joint (WNN)     | **{wnn_ilisi_mean:.4f}** | **{wnn_clisi_mean:.4f}** | ~{N_CLISI//1000}K | ~{N_ILISI//1000}K |\n")
    f.write("\n> Not: Protein ve RNA sonuclari ilgili scriptlerden (lisi_01, lisi_02) alınarak doldurunuz.\n")

print(f"\nSonuclar kaydedildi:")
print(f"  - {md_path}")
print(f"  - {summary_path}")
print("\nTUM ISLEMLER BASARIYLA TAMAMLANDI!")
