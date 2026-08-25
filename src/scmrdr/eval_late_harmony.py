import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import harmonypy as hm
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\imputed_h5ad")
OUT_DIR = Path(r"C:\Users\emir_\.gemini\antigravity-ide\brain\48c92ca2-503c-4c91-87e8-1c561cbd2e6b")

print("1. Imputed RNA Dosyalari Yukleniyor (SNT393 ve SNT227)...")
adata_393 = sc.read_h5ad(DATA_DIR / "imputed_xenium_SNT393_age37.h5ad")
adata_393.obs['batch'] = 'SNT393'

adata_227 = sc.read_h5ad(DATA_DIR / "imputed_xenium_SNT227_age69.h5ad")
adata_227.obs['batch'] = 'SNT227'

adata_imputed = adata_393.concatenate(adata_227, join='inner')
print(f"Toplam Hucre: {adata_imputed.n_obs}, Gen Sayisi: {adata_imputed.n_vars}")

print("Alt orneklem aliniyor (UMAP icin hizli sonuc - 60k hucre)...")
sc.pp.subsample(adata_imputed, n_obs=60000, random_state=42)

print("Veri on isleme (PCA)...")
sc.pp.pca(adata_imputed, n_comps=30)

print("\n[ADIM 1] Imputed RNA icin Gec Harmony (Late Batch Correction)...")
batch_df = adata_imputed.obs[['batch']].copy()
ho = hm.run_harmony(adata_imputed.obsm['X_pca'], batch_df, "batch", max_iter_harmony=20)
adata_imputed.obsm['X_harmony'] = ho.Z_corr.T

print("\n[ADIM 2] Yeniden Kumeleme ve UMAP (Harmony Uzayinda)...")
sc.pp.neighbors(adata_imputed, use_rep='X_harmony', n_neighbors=30)
sc.tl.leiden(adata_imputed, resolution=0.2) 
sc.tl.umap(adata_imputed, min_dist=0.3)

print("\n[ADIM 3] Temizlenmis UMAP'ler Ciziliyor...")
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor="#0f0f1a")

sc.pl.umap(adata_imputed, color="batch", ax=axes[0], show=False)
axes[0].set_title("Harmony Sonrasi - Batch (Kesitler)", color='white', fontsize=16, fontweight='bold')

sc.pl.umap(adata_imputed, color="leiden", ax=axes[1], show=False, palette='tab20')
axes[1].set_title(f"Harmony Sonrasi - Kumeler (n={adata_imputed.obs['leiden'].nunique()})", color='white', fontsize=16, fontweight='bold')

if 'AMY2A' in adata_imputed.var_names:
    sc.pl.umap(adata_imputed, color="AMY2A", cmap='magma', ax=axes[2], show=False)
    axes[2].set_title("Gen Ifadesi: AMY2A", color='white', fontsize=16, fontweight='bold')
else:
    # try to find AMY2A
    found = [g for g in adata_imputed.var_names if 'AMY2A' in g]
    if found:
        sc.pl.umap(adata_imputed, color=found[0], cmap='magma', ax=axes[2], show=False)
        axes[2].set_title(f"Gen Ifadesi: {found[0]}", color='white', fontsize=16, fontweight='bold')

for ax in axes:
    ax.set_facecolor('#0f0f1a')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')

plt.tight_layout()
out_file = OUT_DIR / "scMRDR_Late_Harmony.png"
fig.savefig(out_file, facecolor="#0f0f1a", bbox_inches='tight', dpi=150)
print(f"Gec Harmony Tamamlandi! UMAP Kaydedildi: {out_file}")
