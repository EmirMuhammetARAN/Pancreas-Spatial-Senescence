import pandas as pd
import numpy as np
import scanpy as sc
from scipy.spatial import cKDTree
from pathlib import Path
import harmonypy as hm
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings("ignore")

def compute_knn_lisi(embeddings, labels, k=90):
    tree = cKDTree(embeddings)
    _, indices = tree.query(embeddings, k=k)
    unique_labels = np.unique(labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_to_idx[l] for l in labels])
    lisi_scores = []
    n_labels = len(unique_labels)
    for neighbors in indices:
        counts = np.bincount(int_labels[neighbors], minlength=n_labels)
        probs = counts / k
        simpson = np.sum(probs**2)
        lisi_scores.append(1.0 / simpson if simpson > 0 else 1.0)
    return np.mean(lisi_scores)

DATA_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset")
MATCH_DIR_393 = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt393_xenium_registration")
MATCH_DIR_227 = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt227_xenium_registration")

print("1. Referans (Ground Truth) Eslesmeleri Hazirlaniyor...")
adata_ref_393 = sc.read_h5ad(DATA_DIR / r"spatial-transkriptomics\annotated_secondary_analysis(37 393 üst).h5ad", backed='r')
df_type_393 = adata_ref_393.obs[['cell_id', 'final_cell_type']].copy().rename(columns={'cell_id': 'xenium_cell_id'})
match_393 = pd.read_parquet(MATCH_DIR_393 / "SNT393_matches_high_confidence_5um.parquet")
match_393 = match_393.merge(df_type_393, on='xenium_cell_id', how='inner')
match_393['uid'] = match_393['tile_x'].astype(str) + "_" + match_393['tile_y'].astype(str) + "_" + match_393['label'].astype(str)
gt_dict_393 = match_393.set_index('uid')['final_cell_type'].to_dict()
del adata_ref_393, df_type_393, match_393

adata_ref_227 = sc.read_h5ad(DATA_DIR / r"spatial-transkriptomics\annotated_secondary_analysis(69yaş 227alt).h5ad", backed='r')
df_type_227 = adata_ref_227.obs[['cell_id', 'final_cell_type']].copy().rename(columns={'cell_id': 'xenium_cell_id'})
match_227 = pd.read_parquet(MATCH_DIR_227 / "SNT227_matches_high_confidence_5um.parquet")
match_227 = match_227.merge(df_type_227, on='xenium_cell_id', how='inner')
match_227['uid'] = match_227['tile_x'].astype(str) + "_" + match_227['tile_y'].astype(str) + "_" + match_227['label'].astype(str)
gt_dict_227 = match_227.set_index('uid')['final_cell_type'].to_dict()
del adata_ref_227, df_type_227, match_227

gc.collect()

print("2. Tum Hastalarin Imputed Dosyalari Birlestiriliyor (9.4 Milyon Hucre)...")
adatas = []
imputed_files = list((DATA_DIR / "imputed_h5ad").glob("imputed_xenium_*.h5ad"))

for f in imputed_files:
    print(f"  -> Okunuyor: {f.name}")
    ad = sc.read_h5ad(f)
    ad.obs['uid'] = ad.obs['tile_x'].astype(str) + "_" + ad.obs['tile_y'].astype(str) + "_" + ad.obs['label'].astype(str)
    
    if "SNT393" in f.name:
        ad.obs['ground_truth_type'] = ad.obs['uid'].map(gt_dict_393).fillna("Unknown")
    elif "SNT227" in f.name:
        ad.obs['ground_truth_type'] = ad.obs['uid'].map(gt_dict_227).fillna("Unknown")
    else:
        ad.obs['ground_truth_type'] = "Unknown"
        
    ad.obs['batch'] = ad.obs['patient_id'].astype(str)
    adatas.append(ad)

adata_all = sc.concat(adatas, join='inner')
del adatas
gc.collect()

print(f"\nToplam Hucre Sayisi: {adata_all.n_obs:,}")
mask_gt = adata_all.obs['ground_truth_type'] != "Unknown"
print(f"Ground Truth Olan Hucre: {mask_gt.sum():,}")
print(f"Tahmin Bekleyen Hucre: {(~mask_gt).sum():,}")

print("\n3. PCA Hesaplanıyor (Chunked Mode ile RAM Tasarrufu)...")
sc.pp.pca(adata_all, n_comps=30, chunked=True, chunk_size=50000)

print("\n4. Harmony Çalıştırılıyor (Tum Hastalar İcin)...")
sc.external.pp.harmony_integrate(adata_all, 'batch')

print("\n5. LISI Skorlari Hesaplanıyor...")
idx_sub_ilisi = np.random.choice(adata_all.n_obs, 100000, replace=False)
ilisi = compute_knn_lisi(adata_all.obsm['X_pca_harmony'][idx_sub_ilisi], adata_all.obs['batch'].values[idx_sub_ilisi])
idx_gt = np.where(mask_gt)[0]
idx_sub_clisi = np.random.choice(idx_gt, 100000, replace=False) if len(idx_gt) > 100000 else idx_gt
clisi = compute_knn_lisi(adata_all.obsm['X_pca_harmony'][idx_sub_clisi], adata_all.obs['ground_truth_type'].values[idx_sub_clisi])
print(f"iLISI (Batch Mixing): {ilisi:.3f}")
print(f"cLISI (Cluster Purity): {clisi:.3f}")

print("\n6. MiniBatch KMeans ile Kumeleme (12 Kume - Otomatik Etiketleme)...")
kmeans = MiniBatchKMeans(n_clusters=12, random_state=42, batch_size=50000)
adata_all.obs['kmeans_12'] = kmeans.fit_predict(adata_all.obsm['X_pca_harmony'])

# Kumeleri biyolojik isimlere cevirme
cluster_to_type = {}
df_gt = adata_all.obs[mask_gt]
for cluster_id in range(12):
    subset = df_gt[df_gt['kmeans_12'] == cluster_id]
    if len(subset) > 0:
        most_common = subset['ground_truth_type'].mode()[0]
        cluster_to_type[cluster_id] = most_common
    else:
        cluster_to_type[cluster_id] = "Unknown"
        
print("Otomatik Etiket Eslesmeleri:")
for k, v in cluster_to_type.items():
    print(f"  Kume {k} -> {v}")

adata_all.obs['predicted_cell_type'] = adata_all.obs['kmeans_12'].map(cluster_to_type)

print("\n7. UMAP İcin 100k Hucrelik Alt Orneklem Cekiliyor ve Kaydediliyor...")
adata_sub = sc.pp.subsample(adata_all, n_obs=100000, random_state=42, copy=True)
sc.pp.neighbors(adata_sub, n_neighbors=30, use_rep='X_pca_harmony')
sc.tl.umap(adata_sub)

sc.pl.umap(adata_sub, color=['batch'], show=False)
plt.savefig("D:/GitHub/Pancreas-Spatial-Senescence/sil/Global_9M_UMAP_Batch.png", bbox_inches='tight', dpi=300)
plt.close()

sc.pl.umap(adata_sub, color=['predicted_cell_type'], show=False)
plt.savefig("D:/GitHub/Pancreas-Spatial-Senescence/sil/Global_9M_UMAP_PredictedTypes.png", bbox_inches='tight', dpi=300)
plt.close()

print("\n8. Saving all predictions...")
adata_all.obs[['uid', 'patient_id', 'batch', 'ground_truth_type', 'kmeans_12', 'predicted_cell_type']].to_csv("D:/GitHub/Pancreas-Spatial-Senescence/dataset/global_9M_predictions.csv")
print("PROCESS COMPLETED SUCCESSFULLY!")
