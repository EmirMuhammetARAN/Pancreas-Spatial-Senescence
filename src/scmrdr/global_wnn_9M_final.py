import pandas as pd
import numpy as np
import scanpy as sc
import muon as mu
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import gc
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset")
MATCH_DIR_393 = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt393_xenium_registration")
MATCH_DIR_227 = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt227_xenium_registration")

print("1. Ground Truth (SNT227 ve SNT393) Hucreleri Icin Multi-Omics Referans Hazirlaniyor...")

# RNA Yükleme
adata_rna_393 = sc.read_h5ad(DATA_DIR / "imputed_h5ad" / "imputed_xenium_SNT393_age37.h5ad")
adata_rna_227 = sc.read_h5ad(DATA_DIR / "imputed_h5ad" / "imputed_xenium_SNT227_age69.h5ad")
adata_rna_393.obs['uid'] = adata_rna_393.obs['tile_x'].astype(str) + "_" + adata_rna_393.obs['tile_y'].astype(str) + "_" + adata_rna_393.obs['label'].astype(str)
adata_rna_227.obs['uid'] = adata_rna_227.obs['tile_x'].astype(str) + "_" + adata_rna_227.obs['tile_y'].astype(str) + "_" + adata_rna_227.obs['label'].astype(str)

adata_rna_393.obs.set_index('uid', inplace=True)
adata_rna_227.obs.set_index('uid', inplace=True)
adata_rna_393.obs_names = adata_rna_393.obs.index
adata_rna_227.obs_names = adata_rna_227.obs.index

# Ground Truth etiketlerini alma
adata_ref_393 = sc.read_h5ad(DATA_DIR / "spatial-transkriptomics" / "annotated_secondary_analysis_393.h5ad", backed='r')
df_type_393 = adata_ref_393.obs[['cell_id', 'final_cell_type']].copy().rename(columns={'cell_id': 'xenium_cell_id'})
match_393 = pd.read_parquet(MATCH_DIR_393 / "SNT393_matches_high_confidence_5um.parquet")
match_393 = match_393.merge(df_type_393, on='xenium_cell_id', how='inner')
match_393['uid'] = match_393['tile_x'].astype(str) + "_" + match_393['tile_y'].astype(str) + "_" + match_393['label'].astype(str)
gt_dict_393 = match_393.set_index('uid')['final_cell_type'].to_dict()

adata_ref_227 = sc.read_h5ad(DATA_DIR / "spatial-transkriptomics" / "annotated_secondary_analysis_227.h5ad", backed='r')
df_type_227 = adata_ref_227.obs[['cell_id', 'final_cell_type']].copy().rename(columns={'cell_id': 'xenium_cell_id'})
match_227 = pd.read_parquet(MATCH_DIR_227 / "SNT227_matches_high_confidence_5um.parquet")
match_227 = match_227.merge(df_type_227, on='xenium_cell_id', how='inner')
match_227['uid'] = match_227['tile_x'].astype(str) + "_" + match_227['tile_y'].astype(str) + "_" + match_227['label'].astype(str)
gt_dict_227 = match_227.set_index('uid')['final_cell_type'].to_dict()

uid_gt_393 = list(gt_dict_393.keys())
uid_gt_227 = list(gt_dict_227.keys())

# Sadece Referans olan (GT) hucreleri filtrele
adata_rna_393_ref = adata_rna_393[adata_rna_393.obs_names.intersection(uid_gt_393)].copy()
adata_rna_227_ref = adata_rna_227[adata_rna_227.obs_names.intersection(uid_gt_227)].copy()
del adata_rna_393, adata_rna_227
gc.collect()

print("2. Protein Verileri (Parquet) Okunuyor...")
pq_393 = pd.read_parquet(DATA_DIR / "parquets" / "features_dual_SNT393_age37.parquet")
pq_227 = pd.read_parquet(DATA_DIR / "parquets" / "features_dual_SNT227_age69.parquet")
pq_393['uid'] = pq_393['tile_x'].astype(str) + "_" + pq_393['tile_y'].astype(str) + "_" + pq_393['label'].astype(str)
pq_227['uid'] = pq_227['tile_x'].astype(str) + "_" + pq_227['tile_y'].astype(str) + "_" + pq_227['label'].astype(str)
pq_393.set_index('uid', inplace=True)
pq_227.set_index('uid', inplace=True)

prot_cols = [c for c in pq_393.columns if c not in ['label', 'patient_id', 'age', 'tile_y', 'tile_x', 'global_y', 'global_x', 'uid']]

pq_393_ref = pq_393.loc[pq_393.index.intersection(uid_gt_393)].copy()
pq_227_ref = pq_227.loc[pq_227.index.intersection(uid_gt_227)].copy()
del pq_393, pq_227

adata_prot_393 = sc.AnnData(X=pq_393_ref[prot_cols].fillna(0).values.astype(np.float32), obs=pq_393_ref[['patient_id']])
adata_prot_393.obs_names = pq_393_ref.index
adata_prot_227 = sc.AnnData(X=pq_227_ref[prot_cols].fillna(0).values.astype(np.float32), obs=pq_227_ref[['patient_id']])
adata_prot_227.obs_names = pq_227_ref.index

# Hizalama (Alignment)
common_393 = adata_rna_393_ref.obs_names.intersection(adata_prot_393.obs_names)
adata_rna_393_ref = adata_rna_393_ref[common_393]
adata_prot_393 = adata_prot_393[common_393]

common_227 = adata_rna_227_ref.obs_names.intersection(adata_prot_227.obs_names)
adata_rna_227_ref = adata_rna_227_ref[common_227]
adata_prot_227 = adata_prot_227[common_227]

# Birlestirme
adata_rna_ref = sc.concat([adata_rna_393_ref, adata_rna_227_ref], join='inner')
adata_rna_ref.obs['ground_truth_type'] = adata_rna_ref.obs_names.map({**gt_dict_393, **gt_dict_227})
adata_rna_ref.obs['batch'] = adata_rna_ref.obs['patient_id'].astype(str)

adata_prot_ref = sc.concat([adata_prot_393, adata_prot_227], join='inner')
adata_prot_ref.obs['batch'] = adata_prot_ref.obs['patient_id'].astype(str)
adata_prot_ref.obs['ground_truth_type'] = adata_prot_ref.obs_names.map({**gt_dict_393, **gt_dict_227})

mdata_ref = mu.MuData({'rna': adata_rna_ref, 'prot': adata_prot_ref})
mdata_ref.obs['ground_truth_type'] = mdata_ref.obs_names.map({**gt_dict_393, **gt_dict_227})
del adata_rna_393_ref, adata_rna_227_ref, pq_393_ref, pq_227_ref, adata_prot_393, adata_prot_227
gc.collect()

print("3. Preprocessing: Protein CLR Normalizasyonu ve RNA PCA (Dinamik)...")
# Protein icin CLR Normalization
mu.prot.pp.clr(mdata_ref['prot'])
# Proteinde PCA yapmiyoruz, 38 raw/clr kanalin tamamini kullaniyoruz
mdata_ref['prot'].obsm['X_pca'] = mdata_ref['prot'].X.copy()

# RNA icin PCA (30 Bilesen - Varyans dirsegine gore)
sc.pp.pca(mdata_ref['rna'], n_comps=30)

print("4. Harmony Batch Correction (RNA ve Protein Ayri Ayri)...")
sc.external.pp.harmony_integrate(mdata_ref['rna'], 'batch', basis='X_pca', adjusted_basis='X_pca_harmony')
sc.external.pp.harmony_integrate(mdata_ref['prot'], 'batch', basis='X_pca', adjusted_basis='X_pca_harmony')

# Muon varsayılan olarak X_pca'i kullanır, bu yüzden batch-corrected veriyi oraya kopyalıyoruz
mdata_ref['rna'].obsm['X_pca'] = mdata_ref['rna'].obsm['X_pca_harmony'].copy()
mdata_ref['prot'].obsm['X_pca'] = mdata_ref['prot'].obsm['X_pca_harmony'].copy()

print("5. WNN (Weighted Nearest Neighbor) Grafi Kurulumu (Referans Verisi Uzerinde)...")
sc.pp.neighbors(mdata_ref['rna'], n_neighbors=30)
sc.pp.neighbors(mdata_ref['prot'], n_neighbors=30)
mu.pp.neighbors(mdata_ref, n_neighbors=30)
mu.tl.umap(mdata_ref)

print("6. WNN UMAP Uzayinda Etiket Yansitici (Direct KNN) Egitiliyor...")
# UMAP projeksiyonu scanpy ingest tarafindan desteklenmedigi icin, 
# Birlestirilmis (Joint) Harmony-PCA uzayinda agirlikli KNN (manuel mesafe matrisi) egitiyoruz.
# Q1 standartlarına uygun olarak RNA ve Proteinin birbirini ezmemesi icin L2 norm uyguluyoruz:
rna_l2 = normalize(mdata_ref['rna'].obsm['X_pca_harmony'], norm='l2')
prot_l2 = normalize(mdata_ref['prot'].obsm['X_pca_harmony'], norm='l2')
X_ref_joint = np.hstack([rna_l2, prot_l2])
y_ref = mdata_ref.obs['ground_truth_type'].values
knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
knn.fit(X_ref_joint, y_ref)

print("7. Bilinmeyen 8.3 Milyon Hucre (Chunklar Halinde) Referansa Yansitiliyor...")
imputed_files = list((DATA_DIR / "imputed_h5ad").glob("imputed_xenium_*.h5ad"))
res_list = []

for f in imputed_files:
    patient = f.stem.split('_')[2] 
    print(f"  -> {patient} isleniyor...")
    
    ad_rna_q = sc.read_h5ad(f)
    ad_rna_q.obs['uid'] = ad_rna_q.obs['tile_x'].astype(str) + "_" + ad_rna_q.obs['tile_y'].astype(str) + "_" + ad_rna_q.obs['label'].astype(str)
    ad_rna_q.obs.set_index('uid', inplace=True)
    
    # Referans hastaysa, yalnizca bilinmeyenlerini al (GT hucrelerini tekrar hesaplama)
    if patient in ["SNT227", "SNT393"]:
        gt_dict = gt_dict_393 if patient == "SNT393" else gt_dict_227
        unknown_uids = ad_rna_q.obs.index.difference(list(gt_dict.keys()))
        ad_rna_q = ad_rna_q[unknown_uids].copy()
    
    if len(ad_rna_q) == 0:
        continue
        
    pq_q = pd.read_parquet(DATA_DIR / "parquets" / f"features_dual_{patient}_age{f.stem.split('age')[1].split('.')[0]}.parquet")
    pq_q['uid'] = pq_q['tile_x'].astype(str) + "_" + pq_q['tile_y'].astype(str) + "_" + pq_q['label'].astype(str)
    pq_q.set_index('uid', inplace=True)
    
    common_q = ad_rna_q.obs_names.intersection(pq_q.index)
    ad_rna_q = ad_rna_q[common_q]
    pq_q = pq_q.loc[common_q]
    
    ad_prot_q = sc.AnnData(X=pq_q[prot_cols].fillna(0).values.astype(np.float32), obs=pq_q[['patient_id']])
    ad_prot_q.obs_names = pq_q.index
    
    mdata_q = mu.MuData({'rna': ad_rna_q, 'prot': ad_prot_q})
    del ad_rna_q, pq_q, ad_prot_q; gc.collect()
    
    # CLR Normalizasyonu
    mu.prot.pp.clr(mdata_q['prot'])
    mdata_q['prot'].obsm['X_pca'] = mdata_q['prot'].X.copy()
    
    # Ingest PCA Koordinatları (Sadece RNA icin, cunku Protein dogrudan 38 ozellik)
    sc.tl.ingest(mdata_q['rna'], mdata_ref['rna'], obs='ground_truth_type', embedding_method='pca')
    # Protein icin ingest'e gerek yok, X_pca zaten X.copy() olarak atandi.
    
    # Query hucrelerinin Joint PCA uzayini olustur
    # Referansta oldugu gibi burada da oncelikle L2 norm uyguluyoruz!
    rna_q_l2 = normalize(mdata_q['rna'].obsm['X_pca'], norm='l2')
    prot_q_l2 = normalize(mdata_q['prot'].obsm['X_pca'], norm='l2')
    X_q_joint = np.hstack([rna_q_l2, prot_q_l2])
    
    # Direct KNN ile tahmin et
    preds = knn.predict(X_q_joint)
    
    res_df = pd.DataFrame({
        'uid': mdata_q.obs_names,
        'predicted_cell_type': preds,
        'confidence': 1.0, # KNN classification
        'patient_id': mdata_q['rna'].obs['patient_id'],
        'is_reference': False
    })
    res_list.append(res_df)
    
    # Gecici (Ara) Kayit - Olası bir çökmeye karsi guvenlik
    res_df.to_parquet(DATA_DIR / f"temp_pred_{patient}.parquet")
    
    del mdata_q
    gc.collect()

print("8. Sonuclar Birlestirilip Kaydediliyor...")
final_df = pd.concat(res_list, ignore_index=True)

gt_df = pd.DataFrame({
    'uid': mdata_ref.obs_names,
    'predicted_cell_type': mdata_ref.obs['ground_truth_type'],
    'confidence': 1.0,
    'patient_id': mdata_ref['rna'].obs['batch'],
    'is_reference': True
})
final_df = pd.concat([final_df, gt_df], ignore_index=True)

out_path = DATA_DIR / "global_9M_wnn_predictions.parquet"
final_df.to_parquet(out_path)
print(f"PROCESS COMPLETED SUCCESSFULLY! Sonuc dosyasi: {out_path}")
