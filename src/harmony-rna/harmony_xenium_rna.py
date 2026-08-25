import scanpy as sc
import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATASET_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\spatial-transkriptomics")
SCMRD_DIR = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\scmrd-datas")
SCMRD_DIR.mkdir(parents=True, exist_ok=True)

H5AD_FILES = list(DATASET_DIR.glob("annotated_*.h5ad"))

print("=" * 60)
print("  Xenium RNA - 300 GEN HARMONY (FULL DATASET)")
print("=" * 60)

print("\n[1/4] H5AD Dosyalari Yukleniyor...")
dfs = []
for h5ad_path in sorted(H5AD_FILES):
    adata = sc.read_h5ad(h5ad_path)
    batch_name = "SNT393_37yas" if "393" in h5ad_path.name else "SNT227_69yas"
    adata.obs["batch"] = batch_name
    print(f"  {batch_name}: {len(adata):,} hucre")
    dfs.append(adata)

adata = ad.concat(dfs, join="inner")
adata.obs_names_make_unique()
print(f"  Birlesik: {len(adata):,} hucre x {adata.n_vars} gen")

print("\n[2/4] Normalizasyon (TUM GENLER)...")
# Xenium'un ham (raw) datasi, bu yuzden normalize edip log-transform uyguluyoruz
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10) # Harmony icin scale etmek iyi sonuclar verir

# PCA YAPMIYORUZ! Direkt 300 gen kullanacagiz.
# adata.X scipy sparse olabilecegi icin dense matrix'e ceviriyoruz
if hasattr(adata.X, "toarray"):
    data_mat = adata.X.toarray()
else:
    data_mat = adata.X

print(f"\n[3/4] Harmony Uygulanıyor (PCA'siz, {adata.n_vars} gen uzerinden, {len(adata):,} hucre)...")
# Bu islem 1.2M hucre icin biraz surebilir (tahmini 5-10 dk)
batch_df = adata.obs[["batch"]].copy()
ho = hm.run_harmony(data_mat, batch_df, "batch",
                    max_iter_harmony=20, max_iter_kmeans=20, verbose=True)
harmony_out = ho.Z_corr.T
print("  Harmony tamamlandi!")

print("\n[4/4] RNA X_HARMONY KAYIT (scMRDR icin)...")
gene_names = adata.var_names.tolist()
# Harmony ciktisindaki gen isimlerinin sonuna "_" ekleyelim veya ayni birakalim
rna_harmony_cols = [f"{g}_harmony" for g in gene_names]

batch_series = adata.obs["batch"].values
ct_series = adata.obs["final_cell_type"].values
cell_ids = adata.obs_names.values

for bn in np.unique(batch_series):
    mask = batch_series == bn
    snt = "SNT393" if "393" in bn else "SNT227"
    
    df_rna = pd.DataFrame(
        harmony_out[mask],
        columns=rna_harmony_cols
    )
    df_rna["cell_id"] = cell_ids[mask]
    df_rna["final_cell_type"] = ct_series[mask]
    
    out_path = SCMRD_DIR / f"rna_harmony_{snt}.parquet"
    df_rna.to_parquet(out_path, index=False)
    print(f"  {snt}: {len(df_rna):,} hucre basariyla kaydedildi -> {out_path.name}")

print("\nISLEM TAMAMLANDI! scMRDR icin veriler hazir.")
