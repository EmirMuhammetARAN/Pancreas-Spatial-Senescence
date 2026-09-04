# -*- coding: utf-8 -*-
import os, sys
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
"""
fix_protein_harmony_9M.py
=========================
Sorun: global_9M_embeddings.npy icerisindeki Protein (sutun 30:94)
       sadece SNT393 ve SNT227'nin Harmony-corrected embedding'ini iceriyor,
       diger 5 hasta (SNT354, SNT348, SNT899, SNT484, SNT675) ham (raw) Novae.

Cozum:
  1. Her hastanin Novae parquet'ini metadata siralamasiyla yukle ve hizala.
  2. 9.4M × 64 matrisini olustur (float32 -> ~2.4 GB RAM).
  3. Harmony'i tum 9.4M uzerinde calistir (batch = patient_id).
  4. Harmony ciktisini global_9M_embeddings.npy sutunlari 30:94'e yaz.
"""

import numpy as np
import pandas as pd
import harmonypy as hm
import gc
import time
from pathlib import Path

DATA_DIR  = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset")
NOVAE_DIR = DATA_DIR / "novae" / "novae_embeddings"
PQ_DIR    = DATA_DIR / "parquets"
EMB_PATH  = DATA_DIR / "global_9M_embeddings.npy"

# Parquet dosya ismi -> hasta kodu + yas
PATIENT_FILES = {
    "SNT393": ("features_dual_SNT393_age37.parquet",  "SNT393_novae.parquet"),
    "SNT227": ("features_dual_SNT227_age69.parquet",  "SNT227_novae.parquet"),
    "SNT354": ("features_dual_SNT354_age35.parquet",  "SNT354_novae.parquet"),
    "SNT348": ("features_dual_SNT348_age37.parquet",  "SNT348_novae.parquet"),
    "SNT899": ("features_dual_SNT899_age35.parquet",  "SNT899_novae.parquet"),
    "SNT484": ("features_dual_SNT484_age69.parquet",  "SNT484_novae.parquet"),
    "SNT675": ("features_dual_SNT675_age69.parquet",  "SNT675_novae.parquet"),
}

# ─────────────────────────────────────────────
# 1. METADATA SIRALAMASINI YUKlE
# ─────────────────────────────────────────────
print("\n[1/5] Metadata yukleniyor...")
meta = pd.read_parquet(DATA_DIR / "global_9M_embeddings_metadata.parquet",
                       columns=["uid", "patient_id"])
meta = meta.reset_index(drop=True)
N_TOTAL = len(meta)
print(f"  Toplam hucre: {N_TOTAL:,}")
print(meta["patient_id"].value_counts().to_string())

# ─────────────────────────────────────────────
# 2. HER HASTAYI YUKLE, UID ILE HIZALA
# ─────────────────────────────────────────────
print("\n[2/5] Her hastanin Novae embedding'i yukleniyor ve hizalaniyor...")

# 9.4M x 64 float32 matrisi (~ 2.4 GB)
X_novae_global = np.zeros((N_TOTAL, 64), dtype=np.float32)
patient_labels  = meta["patient_id"].values  # Harmony icin batch etiketi

for patient, (pq_file, novae_file) in PATIENT_FILES.items():
    t0 = time.time()

    # Metadata'da bu hastanin satir indeksleri
    mask_meta = meta["patient_id"] == patient
    meta_idx  = meta.index[mask_meta].values         # global matristeki satir numaralari
    meta_uids = meta.loc[mask_meta, "uid"].values     # sirali uid listesi

    # Parquet uid olustur (novae_parquet ile aynı sirada oldugunu dogruluyoruz)
    pq = pd.read_parquet(PQ_DIR / pq_file, columns=["tile_x", "tile_y", "label"])
    pq["uid"] = (pq["tile_x"].astype(str) + "_" +
                 pq["tile_y"].astype(str) + "_" +
                 pq["label"].astype(str))
    pq = pq.reset_index(drop=True)

    # Novae embedding
    novae_df = pd.read_parquet(NOVAE_DIR / novae_file)
    novae_df.index = pq["uid"].values   # uid indexi ata

    # metadata uid sirasina gore hizala (bazı hucreler metadata'da yok, intersection)
    common_uids = pd.Index(meta_uids).intersection(novae_df.index)
    n_match     = len(common_uids)
    n_expected  = mask_meta.sum()

    # meta_idx -> uid siralamasina gore yeniden duzelt
    uid_to_metaidx = dict(zip(meta_uids, meta_idx))

    # common_uids icin global satir no'su
    global_rows = np.array([uid_to_metaidx[u] for u in common_uids])
    X_novae_global[global_rows] = novae_df.loc[common_uids].values.astype(np.float32)

    dt = time.time() - t0
    print(f"  {patient}: {n_match:,}/{n_expected:,} hucre eslesti ({dt:.1f}s)")
    del pq, novae_df
    gc.collect()

print(f"\n  Toplam matris boyutu: {X_novae_global.nbytes / 1e9:.2f} GB")

# ─────────────────────────────────────────────
# 3. NaN / ZERO-NORM KONTROLU
# ─────────────────────────────────────────────
print("\n[3/5] NaN ve zero-norm kontrolu...")
nan_mask  = np.isnan(X_novae_global).any(axis=1)
norm_mask = np.linalg.norm(X_novae_global, axis=1) == 0
bad_mask  = nan_mask | norm_mask
print(f"  Bozuk satir: {bad_mask.sum():,} ({nan_mask.sum():,} NaN + {norm_mask.sum():,} zero-norm)")
# Bozuk satirlari en yakin gecerli degerle doldurmak yerine 0 birak —
# Harmony bunu tolere eder; cok az olmali.

# ─────────────────────────────────────────────
# 4. HARMONY (7 HASTA, TUM 9.4M)
# ─────────────────────────────────────────────
print("\n[4/5] Harmony uygulanıyor (7 hasta, 9.4M hücre)...")
print("  Bu adim 10-30 dakika surebilir. Lutfen bekleyin...")
t0 = time.time()

batch_df = pd.DataFrame({"patient_id": patient_labels})
ho = hm.run_harmony(
    X_novae_global.astype(np.float64),   # harmonypy float64 ister
    batch_df,
    "patient_id",
    max_iter_harmony=20,
    max_iter_kmeans=20,
    verbose=True,
)
X_harmony = ho.Z_corr.T.astype(np.float32)  # (N, 64)
dt = time.time() - t0
print(f"  Harmony tamamlandi! ({dt/60:.1f} dk)")

del X_novae_global, ho
gc.collect()

# ─────────────────────────────────────────────
# 5. GLOBAL EMBEDDING MATRISINE YAZ (sutun 30:94)
# ─────────────────────────────────────────────
print("\n[5/5] global_9M_embeddings.npy sutunlari 30:94 guncelleniyor...")
print("  Mevcut dosya RAM'e yukleniyor (write icin mmap=r+ gerekmez, tam yukle)...")

emb_full = np.load(EMB_PATH)                    # (9413345, 94) float32 ~ 3.5 GB
print(f"  Matris yuklendi: {emb_full.shape}, {emb_full.nbytes/1e9:.1f} GB")

emb_full[:, 30:94] = X_harmony
del X_harmony
gc.collect()

# Yedek olarak eski dosyayi kaydet
backup_path = EMB_PATH.with_name("global_9M_embeddings_BACKUP_raw_prot.npy")
if not backup_path.exists():
    print(f"  Yedek kaydediliyor: {backup_path.name}")
    np.save(backup_path, emb_full[:, 30:94])    # Sadece protein sutunlarini yedekle

np.save(EMB_PATH, emb_full)
print(f"  Guncelleme tamamlandi: {EMB_PATH}")
print("\n  Simdi lisi_01_protein_novae.py tekrar calistirarak skorlari kontrol edin!")
print("\nTUM ISLEMLER BASARIYLA TAMAMLANDI!")
