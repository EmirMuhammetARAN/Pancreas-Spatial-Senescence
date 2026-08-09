"""Create marker-gene reports for Leiden-to-cell-type annotation.

The output is deliberately a report, not automatic annotation. It lets us
map each donor's arbitrary Leiden labels to one shared cell-type taxonomy
before any PhenoCycler classifier is trained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd
import scanpy as sc
import numpy as np

# Suppress scanpy warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# The 300 genes in the Xenium panel
PANCREAS_MARKERS = [
    "ABCC8", "ACKR1", "ACKR3", "ACSL1", "ACTA2", "ADCYAP1", "ADIPOQ", "ALDH1A1", "AMY2A", "ANKRD33B",
    "AQP1", "AQP8", "ARX", "ASCL1", "AURKA", "BARX2", "BAX", "BCL2", "BCL2L1", "BCL2L2",
    "BUB1", "C11orf96", "C1QA", "C3", "CALCA", "CALCB", "CALD1", "CARTPT", "CASR", "CAVIN1",
    "CCL2", "CCL5", "CCNB2", "CCND1", "CCNE2", "CCNL2", "CCR2", "CD2", "CD274", "CD3D",
    "CD3E", "CD3G", "CD52", "CD55", "CD63", "CD68", "CD7", "CD74", "CD84", "CD8A",
    "CD9", "CD96", "CDC20", "CDH19", "CDK1", "CDK5RAP3", "CDKN1A", "CDKN1C", "CDKN2A", "CDX2",
    "CEACAM6", "CFTR", "CHGA", "CHGB", "CHRM3", "CLDN18", "CLU", "CMTM8", "COL1A1", "COL1A2",
    "COL6A2", "COMP", "CORT", "CPA3", "CREB5", "CRHR2", "CSF1R", "CTNNB1", "CTRB1", "CTSB",
    "CTSK", "CXCL1", "CXCL12", "CXCL2", "CXCL6", "CXCL8", "CXCR4", "DDX60L", "DLK1", "DPT",
    "E2F1", "ENO1", "ERO1B", "ERP27", "ESAM", "ETV1", "FABP4", "FAP", "FDXR", "FFAR3",
    "FFAR4", "FOLR1", "FOSB", "FOXB1", "FOXB2", "FOXO1", "FOXP3", "FRZB", "G6PC2", "GABRA6",
    "GAL", "GATM", "GC", "GCG", "GCGR", "GCNT3", "GDF15", "GEM", "GHRL", "GHSR",
    "GLIS3", "GLP1R", "GLS", "GPM6B", "GPNMB", "GPX3", "GSDMB", "GSDMD", "GSDME", "GSN",
    "GZMK", "HADH", "HES1", "HES4", "HHEX", "HLA-E", "HNF1B", "HPGDS", "HSF4", "HSPB1",
    "IAPP", "IFNG", "IGFBP7", "IGHD", "IGHG1", "IGHM", "IL1RL1", "IL2RB", "IL32", "INS",
    "IRX2", "ISL1", "JUN", "JUND", "KIT", "KLRC1", "KLRG1", "KRAS", "KRT19", "KRT7",
    "KRT8", "L1CAM", "LCN2", "LENG8", "LEPR", "LGALS1", "LGI4", "LGMN", "LOXL4", "LPL",
    "LRFN5", "LYZ", "MAFA", "MALL", "MAP1B", "MCM6", "MDM2", "MEIS2", "MLXIPL", "MMP7",
    "MRC1", "MS4A8", "MT1A", "MUC13", "MUC5AC", "MUC5B", "MYC", "NAP1L4", "NEUROG1", "NEUROG3",
    "NKX2-2", "NKX6-1", "NPTX2", "NPY", "NPY1R", "NR5A2", "NUSAP1", "ONECUT1", "OPN4", "PCNA",
    "PDGFRA", "PDGFRB", "PDPN", "PDX1", "PECAM1", "PGC", "PGF", "PHGR1", "PHIP", "PHLDA3",
    "PLIN2", "PLK3", "PLXNB2", "POLD3", "PPP1R1B", "PPY", "PRDX1", "PRG4", "PROX1", "PRSS1",
    "PTF1A", "PYY", "RB1", "RBP4", "RBPJL", "RGS5", "S100A11", "S100A6", "SAMSN1", "SCNN1G",
    "SCTR", "SDC1", "SDS", "SERPINA1", "SERPINE1", "SERPINH1", "SERPINI2", "SIGLEC6", "SIX2", "SIX3",
    "SLC22A6", "SLC30A8", "SLC6A4", "SLC6A6", "SLC7A2", "SLC7A8", "SNRNP70", "SORL1", "SOX10", "SOX4",
    "SOX9", "SPARC", "SPATA2L", "SPHK1", "SPIC", "SPINK1", "SPINT2", "SQSTM1", "SSR1", "SST",
    "SSTR1", "SSTR2", "STMN2", "STX1A", "SULT1C2", "SYCN", "SYT13", "TAC1", "TAGLN", "TEN1",
    "TFF1", "TFF2", "TGFB1I1", "TGFBR2", "TIMP1", "TM4SF1", "TM4SF4", "TMSB10", "TNFRSF10B", "TNFRSF12A",
    "TNFRSF1A", "TP53", "TRIAP1", "TSPAN1", "TSPAN8", "TTR", "TUBGCP2", "TXNIP", "UBALD2", "UCHL1",
    "UCP1", "VEGFA", "VGF", "VIM", "VIP", "VWA5A", "XCL1", "YBX3", "ZBTB7A", "ZMAT3"
]


def report_one(h5ad: Path, donor: str, output_dir: Path) -> None:
    print(f"Loading {donor} Xenium dataset...")
    adata = sc.read_h5ad(h5ad)
    
    # Scanpy reads Xenium h5ad natively. Ensure leiden is categorical string.
    if "leiden" not in adata.obs:
        print(f"Skipping donor {donor}, 'leiden' not found in obs.")
        return
        
    adata.obs["leiden"] = adata.obs["leiden"].astype(str)
    
    # Filter out unassigned cells
    valid_mask = ~adata.obs["leiden"].isin(["-1", "Unknown", "Unassigned", "nan", "NaN"])
    if not valid_mask.all():
        adata = adata[valid_mask].copy()
    
    # Ensure leiden remains a categorical variable for rank_genes_groups
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    
    n_cells_per_cluster = adata.obs["leiden"].value_counts().to_dict()
    
    print(f"Normalizing {donor}...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 1. Normalized Means Report
    print(f"Calculating normalized means for {donor}...")
    # Find which markers are actually present in the adata var names (case-insensitive fallback)
    gene_to_index = {str(gene).upper(): str(gene) for gene in adata.var_names}
    present_markers = [gene_to_index[marker.upper()] for marker in PANCREAS_MARKERS if marker.upper() in gene_to_index]
    
    # Calculate means grouping by leiden cluster
    means_df = sc.get.obs_df(adata, keys=present_markers + ["leiden"]).groupby("leiden").mean()
    means_df.reset_index(inplace=True)
    means_df.insert(1, "n_cells", means_df["leiden"].map(n_cells_per_cluster))
    means_df.to_csv(output_dir / f"donor_{donor}_leiden_marker_means.csv", index=False)
    
    # 2. Differential Expression (Wilcoxon Rank-Sum Test)
    print(f"Running Wilcoxon rank-sum tests for {donor}...")
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
    
    rows = []
    # rank_genes_groups stores results as a record array
    result_names = adata.uns["rank_genes_groups"]["names"]
    
    for cluster in means_df["leiden"].unique():
        # Get top 20 genes for this cluster, filtering out non-biological negatives
        cluster_genes = result_names[str(cluster)]
        top = [gene for gene in cluster_genes if not str(gene).startswith(("NEGATIVE", "BLANK", "Unassigned"))][:20]
        rows.append({
            "leiden": cluster,
            "n_cells": n_cells_per_cluster.get(cluster, 0),
            "top_20_wilcoxon_genes": "; ".join(top)
        })
        
    pd.DataFrame(rows).to_csv(output_dir / f"donor_{donor}_leiden_top_genes.csv", index=False)
    print(f"Donor {donor}: {len(means_df)} Leiden clusters, {len(present_markers)} panel markers analyzed.\n")


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
    report_one(args.h5ad_37, "37", args.output_dir)
    report_one(args.h5ad_69, "69", args.output_dir)


if __name__ == "__main__":
    main()
