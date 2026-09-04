# -*- coding: utf-8 -*-
import os, sys
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
"""
lisi_02_rna_harmony.py
======================
RNA (Harmony) Modality için LISI Hesaplama
-------------------------------------------
Strateji:
  - cLISI: Ground Truth etiketine sahip hücrelerden cell_type'a göre
            TABAKALı (stratified) örnekleme ile ~150.000 hücre seçilir.
  - iLISI: 9.4M hücrelerin TAMAMINDAN patient_id'ye göre
            TABAKALı (stratified) örnekleme ile ~200.000 hücre seçilir.
  - Embedding: global_9M_embeddings.npy içinde sütunlar 0:30 → RNA (Harmony, 30 boyut)
  - KNN: 90 komşu ile scib_metrics.ilisi_knn / clisi_knn
"""

import numpy as np
import pandas as pd
import scanpy as sc
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
print("  RNA (HARMONY) LISI - STRATIFIED SAMPLING")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. GROUND TRUTH ETİKETLERİ
# ─────────────────────────────────────────────
print("\n[1/6] Ground Truth etiketleri yukleniyor...")

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
print("\n[2/6] Metadata yukleniyor (sadece uid + patient_id)...")
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

print("\n[3/6] cLISI icin Stratified Orneklem (cell_type'a gore)...")


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

meta_clisi = stratified_sample_with_min(
    meta_ref, 'ground_truth_type', N_CLISI, min_per_group=50,
    random_state=RANDOM_STATE
)
print(f"  cLISI orneklem boyutu: {len(meta_clisi):,}")
print("  Cell type dagilimi:")
print(meta_clisi['ground_truth_type'].value_counts().to_string())

# ─────────────────────────────────────────────
# 4. iLISI — STRATİFİED BY PATIENT ID (FULL 9.4M)
# ─────────────────────────────────────────────
print("\n[4/6] iLISI icin Stratified Orneklem (patient_id'ye gore, tum 9.4M)...")

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

meta_ilisi = stratified_sample_proportional(
    meta_full, 'patient_id', N_ILISI, random_state=RANDOM_STATE
)
print(f"  iLISI orneklem boyutu: {len(meta_ilisi):,}")
print("  Patient ID dagilimi:")
print(meta_ilisi['patient_id'].value_counts().to_string())

# ─────────────────────────────────────────────
# 5. EMBEDDİNG SÜTUNLARI: RNA = 0:30
# ─────────────────────────────────────────────
print("\n[5/6] RNA (Harmony) embedding'leri RAM'e aliniyor...")

clisi_idx = meta_clisi['index'].to_numpy()   # orijinal meta_full satir no
ilisi_idx = meta_ilisi.index.to_numpy()       # meta_full'dan direkt

rna_clisi = np.array(embeddings_full[clisi_idx, :30])
rna_ilisi = np.array(embeddings_full[ilisi_idx, :30])

del embeddings_full
gc.collect()

# ─────────────────────────────────────────────
# 6. LISI HESAPLAMA
# ─────────────────────────────────────────────
def get_nn_results(X, n_neighbors):
    """scanpy KNN -> scib_metrics NeighborsResults"""
    adata_tmp = sc.AnnData(X=X.astype(np.float32))
    sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, use_rep='X')
    dist_mat = adata_tmp.obsp['distances']

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

    del adata_tmp; gc.collect()
    return scib_metrics.nearest_neighbors.NeighborsResults(
        indices=indices_arr, distances=distances_arr
    )

print("\n[6/6] LISI hesaplaniyor...")

# --- iLISI (batch mixing) ---
print("  iLISI (batch mixing) hesaplaniyor...")
t0 = time.time()
nn_ilisi = get_nn_results(rna_ilisi, N_NEIGHBORS)
batch_labels = meta_ilisi['patient_id'].astype(str).values
ilisi_scores = scib_metrics.ilisi_knn(X=nn_ilisi, batches=batch_labels, scale=False)
rna_ilisi_mean = float(np.mean(ilisi_scores))
print(f"  iLISI tamamlandi: {rna_ilisi_mean:.4f}  ({time.time()-t0:.1f} sn)")
del rna_ilisi, nn_ilisi; gc.collect()

# --- cLISI (cell type purity) ---
print("  cLISI (cell type purity) hesaplaniyor...")
t0 = time.time()
nn_clisi = get_nn_results(rna_clisi, N_NEIGHBORS)
celltype_labels = meta_clisi['ground_truth_type'].values
clisi_scores = scib_metrics.clisi_knn(X=nn_clisi, labels=celltype_labels, scale=False)
rna_clisi_mean = float(np.mean(clisi_scores))
print(f"  cLISI tamamlandi: {rna_clisi_mean:.4f}  ({time.time()-t0:.1f} sn)")
del rna_clisi, nn_clisi; gc.collect()

# ─────────────────────────────────────────────
# SONUÇLAR
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RNA (HARMONY)  |  iLISI: {rna_ilisi_mean:.4f}  |  cLISI: {rna_clisi_mean:.4f}")
print("=" * 60)

md_path = OUT_DIR / "LISI_RNA_Harmony.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# RNA (Harmony) LISI Sonuclari\n\n")
    f.write("## Orneklem Stratejisi\n")
    f.write(f"- **iLISI**: Tum 9.4M hucreden `patient_id`'ye gore tabakali, **{len(meta_ilisi):,}** hucre\n")
    f.write(f"- **cLISI**: GT etiketli hucrelerden `cell_type`'a gore tabakali (min 50/tip), **{len(meta_clisi):,}** hucre\n")
    f.write(f"- KNN komsu sayisi: `{N_NEIGHBORS}`\n\n")
    f.write("## Sonuclar\n\n")
    f.write("| Metrik | Deger | Yorum |\n")
    f.write("|--------|-------|-------|\n")
    f.write(f"| iLISI (Batch Mixing) | **{rna_ilisi_mean:.4f}** | Yuksek = iyi karisim |\n")
    f.write(f"| cLISI (Cell Type Purity) | **{rna_clisi_mean:.4f}** | Dusuk = iyi saflik |\n")
    f.write("\n## Cell Type Dagilimi (cLISI orneklem)\n\n```\n")
    f.write(meta_clisi['ground_truth_type'].value_counts().to_string())
    f.write("\n```\n\n## Patient ID Dagilimi (iLISI orneklem)\n\n```\n")
    f.write(meta_ilisi['patient_id'].value_counts().to_string())
    f.write("\n```\n")

print(f"\nSonuclar kaydedildi: {md_path}")
print("TUM ISLEMLER BASARIYLA TAMAMLANDI!")
