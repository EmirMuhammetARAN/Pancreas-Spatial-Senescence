import os
import pandas as pd
import numpy as np
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\plots")
emb_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\novae_embeddings"

def compute_knn_lisi(embeddings, labels, k=90):
    print("  Computing LISI...")
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

print("--- VERI YUKLEME (7 KESIT, 20k/kesit) ---")
adatas = []
files = [f for f in os.listdir(emb_dir) if f.endswith(".parquet")]
for f in files:
    batch_name = f.split("_")[0]
    df_emb = pd.read_parquet(os.path.join(emb_dir, f))
    X_emb = df_emb.values.astype(np.float64)
    
    # NaN ve zero-norm satirlari temizle
    nan_mask = np.isnan(X_emb).any(axis=1)
    zero_mask = np.linalg.norm(X_emb, axis=1) == 0
    bad_mask = nan_mask | zero_mask
    n_bad = bad_mask.sum()
    if n_bad > 0:
        print(f"  {batch_name}: {n_bad} bozuk satir silindi (NaN={nan_mask.sum()}, ZeroNorm={zero_mask.sum()})")
        X_emb = X_emb[~bad_mask]
    
    np.random.seed(42)
    idx = np.random.choice(len(X_emb), size=min(20000, len(X_emb)), replace=False)
    X_sub = X_emb[idx]
    
    adata = sc.AnnData(X=np.zeros((len(X_sub), 1)))
    adata.obsm["X_novae"] = X_sub
    adata.obs["batch"] = batch_name
    adatas.append(adata)

global_adata = sc.concat(adatas)
print(f"Total cells: {len(global_adata)}")

# Son kontrol
assert not np.isnan(global_adata.obsm["X_novae"]).any(), "Hala NaN var!"
assert (np.linalg.norm(global_adata.obsm["X_novae"], axis=1) > 0).all(), "Hala zero-norm var!"
print("Veri kontrolu gecti!")

print("\n--- BEFORE HARMONY ---")
sc.pp.neighbors(global_adata, use_rep="X_novae", n_neighbors=30)
sc.tl.leiden(global_adata, resolution=0.1, key_added="leiden_before", flavor="igraph", n_iterations=2, directed=False)
sc.tl.umap(global_adata, min_dist=0.3)
global_adata.obsm["X_umap_before"] = global_adata.obsm["X_umap"].copy()

np.random.seed(42)
lisi_idx = np.random.choice(len(global_adata), size=20000, replace=False)
lisi_emb_before = global_adata.obsm["X_novae"][lisi_idx]
ilisi_before = compute_knn_lisi(lisi_emb_before, global_adata.obs["batch"].values[lisi_idx], k=90)
clisi_before = compute_knn_lisi(lisi_emb_before, global_adata.obs["leiden_before"].values[lisi_idx], k=90)
print(f"  BEFORE -> iLISI = {ilisi_before:.3f} | cLISI = {clisi_before:.3f}")

print("\n--- HARMONY UYGULANIYOR ---")
batch_df = global_adata.obs[["batch"]].copy()
ho = hm.run_harmony(global_adata.obsm["X_novae"], batch_df, "batch",
                    max_iter_harmony=30, max_iter_kmeans=50, verbose=True)
global_adata.obsm["X_harmony"] = ho.Z_corr.T
print("  Harmony tamamlandi!")

print("\n--- AFTER HARMONY ---")
sc.pp.neighbors(global_adata, use_rep="X_harmony", n_neighbors=30)
sc.tl.leiden(global_adata, resolution=0.1, key_added="leiden_after", flavor="igraph", n_iterations=2, directed=False)
sc.tl.umap(global_adata, min_dist=0.3)
global_adata.obsm["X_umap_after"] = global_adata.obsm["X_umap"].copy()

lisi_emb_after = global_adata.obsm["X_harmony"][lisi_idx]
ilisi_after = compute_knn_lisi(lisi_emb_after, global_adata.obs["batch"].values[lisi_idx], k=90)
clisi_after = compute_knn_lisi(lisi_emb_after, global_adata.obs["leiden_after"].values[lisi_idx], k=90)
print(f"  AFTER  -> iLISI = {ilisi_after:.3f} | cLISI = {clisi_after:.3f}")

print("\n--- DORT PANEL GRAFIK ---")
fig, axes = plt.subplots(2, 2, figsize=(22, 16), facecolor="#0f0f1a")

global_adata.obsm["X_umap"] = global_adata.obsm["X_umap_before"]
sc.pl.umap(global_adata, color="batch", ax=axes[0,0], show=False, s=2, alpha=0.6)
axes[0,0].set_title(f"BEFORE Harmony - Batch (iLISI={ilisi_before:.2f}/7.0)", color="white", fontweight="bold")
sc.pl.umap(global_adata, color="leiden_before", ax=axes[0,1], show=False, s=2, alpha=0.6)
axes[0,1].set_title(f"BEFORE Harmony - Biyoloji (cLISI={clisi_before:.2f})", color="white", fontweight="bold")

global_adata.obsm["X_umap"] = global_adata.obsm["X_umap_after"]
sc.pl.umap(global_adata, color="batch", ax=axes[1,0], show=False, s=2, alpha=0.6)
axes[1,0].set_title(f"AFTER Harmony - Batch (iLISI={ilisi_after:.2f}/7.0)", color="white", fontweight="bold")
sc.pl.umap(global_adata, color="leiden_after", ax=axes[1,1], show=False, s=2, alpha=0.6)
axes[1,1].set_title(f"AFTER Harmony - Biyoloji (cLISI={clisi_after:.2f})", color="white", fontweight="bold")

for ax in axes.flat:
    ax.set_facecolor("#0f0f1a")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")

plt.tight_layout()
out_path = OUT_DIR / "Novae_Harmony_Comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Grafik kaydedildi!")

print("\n--- OZET ---")
print(f"  iLISI: {ilisi_before:.3f} -> {ilisi_after:.3f}  (max=7.0)")
print(f"  cLISI: {clisi_before:.3f} -> {clisi_after:.3f}")
print("BITTI!")
