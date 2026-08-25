import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

DATA_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset")
OBS_FILE = DATA_DIR / "global_9M_obs.parquet"
HARMONY_CACHE_FILE = DATA_DIR / "global_9M_harmony.npy"

print("1. Harmony Matrix ve Metadata Loading...")
X_harmony = np.load(HARMONY_CACHE_FILE)
obs_df = pd.read_parquet(OBS_FILE)

print("2. Dogrudan KNN Label Transfer Hazirligi...")
# Sadece "Unknown" olmayan Ground Truth hucrelerini egitim seti olarak aliyoruz
mask_gt = obs_df['ground_truth_type'] != "Unknown"

X_train = X_harmony[mask_gt]
y_train = obs_df.loc[mask_gt, 'ground_truth_type'].values

print(f" -> Egitim seti (Referans) hucre sayisi: {len(X_train)}")
print(" -> Referans Hucre Tipleri:")
counts = pd.Series(y_train).value_counts()
print(counts)

print("\n3. KNN Modeli Egitiliyor (k=30)...")
knn = KNeighborsClassifier(n_neighbors=30, weights='distance', n_jobs=-1)
knn.fit(X_train, y_train)

print("\n4. Kalan 8.3 Milyon Hucre Icin Tahmin Yapiliyor (Chunked)...")
X_test = X_harmony[~mask_gt]
chunk_size = 500000
predictions = []

for i in range(0, len(X_test), chunk_size):
    pred_chunk = knn.predict(X_test[i:i+chunk_size])
    predictions.append(pred_chunk)

all_predictions = np.concatenate(predictions)

# Guncel tahminleri ana DataFrame'e yaz
obs_df['predicted_cell_type_knn'] = obs_df['ground_truth_type'].copy()
obs_df.loc[~mask_gt, 'predicted_cell_type_knn'] = all_predictions

print("\n5. Yeni Genel Dagilim (Tum 9.4 Milyon):")
print(obs_df['predicted_cell_type_knn'].value_counts())

# UMAP icin kucuk bir alt orneklem olustur (Gorsellestirme amaciyla)
print("\n6. Yeni UMAP Cizimi Icin Hazirlik...")
np.random.seed(42)
idx_sub = np.random.choice(len(obs_df), 100000, replace=False)
adata_sub = sc.AnnData(X=np.empty((100000, 1), dtype=np.float32))
adata_sub.obs = obs_df.iloc[idx_sub].copy()
adata_sub.obsm['X_pca_harmony'] = X_harmony[idx_sub]

sc.pp.neighbors(adata_sub, n_neighbors=30, use_rep='X_pca_harmony')
sc.tl.umap(adata_sub)

sc.pl.umap(adata_sub, color=['predicted_cell_type_knn'], show=False)
plt.savefig("D:/GitHub/Pancreas-Spatial-Senescence/sil/Global_9M_UMAP_DirectKNN.png", bbox_inches='tight', dpi=300)
plt.close()

print("\n7. Sonuclar Kaydediliyor...")
obs_df.to_parquet(DATA_DIR / "global_9M_obs_knn_final.parquet")
obs_df.to_csv(DATA_DIR / "global_9M_predictions_direct_knn.csv")
print("PROCESS COMPLETED SUCCESSFULLY!")
