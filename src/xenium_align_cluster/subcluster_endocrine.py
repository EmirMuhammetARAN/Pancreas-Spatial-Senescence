"""Sub-cluster the Endocrine cells to distinguish Alpha, Beta, and Delta cells.
"""

import argparse
from pathlib import Path
import warnings

import pandas as pd
import scanpy as sc
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def subcluster_donor(h5ad: Path, donor: str, target_cluster: str, output_dir: Path) -> None:
    print(f"Loading {donor} Xenium dataset...")
    adata = sc.read_h5ad(h5ad)
    
    if "leiden" not in adata.obs:
        print(f"Skipping donor {donor}, 'leiden' not found in obs.")
        return
        
    adata.obs["leiden"] = adata.obs["leiden"].astype(str)
    
    # Filter to the target endocrine cluster
    valid_mask = adata.obs["leiden"] == target_cluster
    if not valid_mask.any():
        print(f"Target cluster {target_cluster} not found for donor {donor}.")
        return
        
    adata_endo = adata[valid_mask].copy()
    print(f"Donor {donor}: Extracted {len(adata_endo)} endocrine cells (Cluster {target_cluster}).")
    
    # Preprocessing (Normalize & Log)
    print("Normalizing subset...")
    sc.pp.normalize_total(adata_endo, target_sum=1e4)
    sc.pp.log1p(adata_endo)
    
    # Run PCA, Neighbors, and Leiden (resolution=0.5 for stable large clusters)
    print("Running PCA, Neighbors, and Leiden (resolution=0.5)...")
    sc.tl.pca(adata_endo, svd_solver='arpack')
    sc.pp.neighbors(adata_endo, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(adata_endo, resolution=0.5, key_added='endo_leiden')
    
    # Differential Expression on the new sub-clusters
    print("Running Wilcoxon rank-sum test on sub-clusters...")
    sc.tl.rank_genes_groups(adata_endo, 'endo_leiden', method='wilcoxon')
    
    n_cells_per_cluster = adata_endo.obs['endo_leiden'].value_counts().to_dict()
    
    rows = []
    result_names = adata_endo.uns['rank_genes_groups']['names']
    
    for cluster in adata_endo.obs['endo_leiden'].cat.categories:
        cluster_genes = result_names[str(cluster)]
        top = [gene for gene in cluster_genes if not str(gene).startswith(("NEGATIVE", "BLANK", "Unassigned"))][:20]
        rows.append({
            "endo_leiden": cluster,
            "n_cells": n_cells_per_cluster.get(cluster, 0),
            "top_20_wilcoxon_genes": "; ".join(top)
        })
        
    out_csv = output_dir / f"donor_{donor}_endocrine_subclusters_top_genes.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved sub-cluster results to {out_csv}\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    spatial = root / "dataset" / "spatial-transkriptomics"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-37", type=Path, default=spatial / "secondary_analysis(37 393 üst).h5ad")
    parser.add_argument("--h5ad-69", type=Path, default=spatial / "secondary_analysis(69yaş 227alt).h5ad")
    parser.add_argument("--output-dir", type=Path, default=root / "deneme" / "xenium_leiden_reports")
    args = parser.parse_args()
    
    for input_path in (args.h5ad_37, args.h5ad_69):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Donor 37: Endocrine cluster is '10'
    subcluster_donor(args.h5ad_37, "37", "10", args.output_dir)
    
    # Donor 69: Endocrine cluster is '8'
    subcluster_donor(args.h5ad_69, "69", "8", args.output_dir)


if __name__ == "__main__":
    main()
