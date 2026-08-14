import scanpy as sc
import pandas as pd
import numpy as np
import ot
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph, KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from pathlib import Path
import gc
import torch
import warnings
warnings.filterwarnings('ignore')

print("Starting Stage 3: USHER (FGW) Topological Alignment & Label Transfer")

# Paths
dataset_dir = Path(r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset")
imputed_dir = dataset_dir / "imputed_h5ad"
annotated_dir = dataset_dir / "annotated_h5ad"
annotated_dir.mkdir(exist_ok=True)

ref_h5ad_393_path = dataset_dir / r"spatial-transkriptomics\annotated_secondary_analysis(37 393 üst).h5ad"
ref_h5ad_227_path = dataset_dir / r"spatial-transkriptomics\annotated_secondary_analysis(69yaş 227alt).h5ad"
matches_393_path = Path(r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt393_xenium_registration\SNT393_matches_high_confidence_5um.parquet")
matches_227_path = Path(r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt227_xenium_registration\SNT227_matches_high_confidence_5um.parquet")

def load_true_ref(h5ad_path, matches_path):
    adata = sc.read_h5ad(h5ad_path)
    matches = pd.read_parquet(matches_path)
    adata = adata[adata.obs['cell_id'].isin(matches['xenium_cell_id'])].copy()
    
    cell_type_col = None
    for col in adata.obs.columns:
        if 'cell_type' in col.lower() or 'cluster' in col.lower() or 'annotation' in col.lower() or 'leiden' in col.lower():
            cell_type_col = col
            break
            
    if cell_type_col:
        adata.obs['CellType'] = adata.obs[cell_type_col].astype(str)
    else:
        adata.obs['CellType'] = 'Unknown'
        
    if hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()
    
    return adata

print("Loading Reference Datasets...")
ref_393 = load_true_ref(ref_h5ad_393_path, matches_393_path)
ref_227 = load_true_ref(ref_h5ad_227_path, matches_227_path)

# Concatenate reference datasets
common_genes = ref_393.var_names.intersection(ref_227.var_names)
ref_393 = ref_393[:, common_genes].copy()
ref_227 = ref_227[:, common_genes].copy()

adata_ref = ref_393.concatenate(ref_227, batch_key='ref_batch')
del ref_393, ref_227
gc.collect()

X_ref_full = adata_ref.X.astype(np.float32)
Y_ref_full = adata_ref.obs['CellType'].values

# Stratified sampling for Reference Anchors (5000 cells)
anchor_size = 5000
ref_indices = []
cell_counts = adata_ref.obs['CellType'].value_counts()
for ct, count in cell_counts.items():
    n_sample = int(np.round((count / len(adata_ref)) * anchor_size))
    if n_sample > 0:
        idx = np.where(Y_ref_full == ct)[0]
        ref_indices.extend(np.random.choice(idx, size=min(n_sample, len(idx)), replace=False))

# Fallback if rounding causes fewer than 5000
if len(ref_indices) < anchor_size:
    remaining = anchor_size - len(ref_indices)
    avail = list(set(range(len(adata_ref))) - set(ref_indices))
    ref_indices.extend(np.random.choice(avail, size=min(remaining, len(avail)), replace=False))

ref_indices = np.array(ref_indices[:anchor_size])
X_ref_anchor = X_ref_full[ref_indices]

print("Computing Geodesic Distances for Reference Anchors...")
# kNN graph (k=30)
ref_knn = kneighbors_graph(X_ref_anchor, n_neighbors=30, mode='distance', include_self=False)
C1 = shortest_path(ref_knn, directed=False)
# Handle unreachable nodes (inf) by replacing with max finite distance
C1[np.isinf(C1)] = np.nanmax(C1[C1 != np.inf]) * 1.5 
C1 = C1 / C1.max()  # Normalize

imputed_files = [
    "imputed_xenium_SNT348_age37.h5ad",
    "imputed_xenium_SNT354_age35.h5ad",
    "imputed_xenium_SNT484_age69.h5ad",
    "imputed_xenium_SNT675_age69.h5ad",
    "imputed_xenium_SNT899_age35.h5ad",
    "features_dual_SNT393_age37.h5ad"
]

print("Training kNN Classifier (k=30) on Full Reference...")
knn_classifier = KNeighborsClassifier(n_neighbors=30, weights='distance')
knn_classifier.fit(X_ref_full, Y_ref_full)

for f in imputed_files:
    f_path = imputed_dir / f
    if not f_path.exists():
        continue
    
    print(f"\n========================================")
    print(f"Processing Target: {f}")
    adata_tgt = sc.read_h5ad(f_path)
    
    if "SNT393" in f and 'global_y' in adata_tgt.obs.columns:
        adata_tgt = adata_tgt[adata_tgt.obs['global_y'] >= 25250].copy()
        
    if hasattr(adata_tgt.X, 'toarray'):
        adata_tgt.X = adata_tgt.X.toarray()
    
    adata_tgt = adata_tgt[:, common_genes].copy()
    X_tgt_full = adata_tgt.X.astype(np.float32)
    
    # Target anchors
    actual_anchor_size = min(anchor_size, len(adata_tgt))
    tgt_indices = np.random.choice(len(adata_tgt), size=actual_anchor_size, replace=False)
    X_tgt_anchor = X_tgt_full[tgt_indices]
    
    print("Computing Geodesic Distances for Target Anchors...")
    tgt_knn = kneighbors_graph(X_tgt_anchor, n_neighbors=30, mode='distance', include_self=False)
    C2 = shortest_path(tgt_knn, directed=False)
    C2[np.isinf(C2)] = np.nanmax(C2[C2 != np.inf]) * 1.5
    C2 = C2 / C2.max()
    
    print("Computing Cross-Domain Feature Distance...")
    M = pairwise_distances(X_tgt_anchor, X_ref_anchor[:actual_anchor_size], metric='euclidean')
    M = M / M.max()
    
    print("Running Fused Gromov-Wasserstein (FGW)...")
    p = ot.unif(actual_anchor_size)
    q = ot.unif(actual_anchor_size)
    
    # alpha controls tradeoff between feature distance (M) and structural distance (C1, C2)
    # alpha=0.5 gives equal weight
    T = ot.gromov.fused_gromov_wasserstein(M, C2, C1[:actual_anchor_size, :actual_anchor_size], p, q, loss_fun='square_loss', alpha=0.5)
    
    # Barycentric mapping of Target anchors to Reference space
    # T is of shape (Target, Reference)
    T_norm = T / T.sum(axis=1, keepdims=True)
    X_tgt_anchor_mapped = T_norm.dot(X_ref_anchor[:actual_anchor_size])
    
    print("Extrapolating alignment to all target cells via Barycentric Projection...")
    # Train regressor to map all target cells based on anchor mapping
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')
    regressor.fit(X_tgt_anchor, X_tgt_anchor_mapped)
    
    # Process in chunks to save memory
    chunk_size = 50000
    X_tgt_aligned = np.zeros_like(X_tgt_full)
    for i in range(0, len(X_tgt_full), chunk_size):
        X_tgt_aligned[i:i+chunk_size] = regressor.predict(X_tgt_full[i:i+chunk_size])
    
    print("Transferring Xenium Labels (kNN k=30)...")
    pred_labels = []
    for i in range(0, len(X_tgt_aligned), chunk_size):
        pred_labels.extend(knn_classifier.predict(X_tgt_aligned[i:i+chunk_size]))
        
    adata_tgt.X = X_tgt_aligned
    adata_tgt.obs['CellType_USHER'] = pred_labels
    
    out_path = annotated_dir / f.replace("imputed_xenium_", "usher_annotated_")
    if "features_dual_SNT393" in f:
        out_path = annotated_dir / "usher_annotated_SNT393_age37.h5ad"
        
    adata_tgt.write_h5ad(out_path)
    print(f"Saved aligned and annotated target to: {out_path.name}")
    
    del adata_tgt, X_tgt_full, X_tgt_aligned, T, M, C2
    gc.collect()

print("\nStage 3: Topological Alignment and Label Transfer Completed Successfully!")
