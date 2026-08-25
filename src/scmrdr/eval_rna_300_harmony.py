import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCMRD_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\scmrd-datas")
OUT_DIR = Path(r"C:\Users\emir_\.gemini\antigravity-ide\brain\48c92ca2-503c-4c91-87e8-1c561cbd2e6b")

print("Yukleniyor...")
df_393 = pd.read_parquet(SCMRD_DIR / "rna_harmony_SNT393.parquet")
df_393["batch"] = "SNT393"
df_227 = pd.read_parquet(SCMRD_DIR / "rna_harmony_SNT227.parquet")
df_227["batch"] = "SNT227"

df_all = pd.concat([df_393, df_227], ignore_index=True)
print(f"Toplam hucre: {len(df_all)}")

# Alt orneklem (LISI ve UMAP icin 20k)
df_sub = df_all.sample(n=20000, random_state=42)
harmony_cols = [c for c in df_sub.columns if c.endswith("_harmony")]

X_harm = df_sub[harmony_cols].values
batches = df_sub["batch"].values
cell_types = df_sub["final_cell_type"].values

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

print("LISI hesaplaniyor...")
ilisi = compute_knn_lisi(X_harm, batches)
clisi = compute_knn_lisi(X_harm, cell_types)
print(f"300-Gen Harmony -> iLISI: {ilisi:.3f} | cLISI: {clisi:.3f}")

print("UMAP calistiriliyor...")
adata = ad.AnnData(X=X_harm)
adata.obs["batch"] = pd.Categorical(batches)
adata.obs["cell_type"] = pd.Categorical(cell_types)

sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
sc.tl.umap(adata)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc.pl.umap(adata, color="batch", ax=axes[0], title=f"RNA Harmony (300 Gen) - Batch\niLISI: {ilisi:.3f}", show=False)
sc.pl.umap(adata, color="cell_type", ax=axes[1], title=f"RNA Harmony (300 Gen) - Cell Type\ncLISI: {clisi:.3f}", show=False)
plt.tight_layout()
fig.savefig(OUT_DIR / "RNA_300Gen_Harmony_UMAP.png", dpi=150, bbox_inches="tight")
print("UMAP kaydedildi.")
