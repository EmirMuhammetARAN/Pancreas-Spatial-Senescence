import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dataset_dir = Path(r'C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\dataset\spatial-transkriptomics')
h5ad_37 = dataset_dir / 'secondary_analysis(37 393 üst).h5ad'
h5ad_69 = dataset_dir / 'secondary_analysis(69yaş 227alt).h5ad'
reports_dir = Path(r'C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\xenium_leiden_reports')

markers = {
    'Acinar': ['PRSS1', 'SPINK1', 'CTRB1', 'AMY2A'],
    'Ductal': ['KRT19', 'CFTR', 'SOX9', 'MUC5B'],
    'Stroma': ['COL1A1', 'ACTA2', 'PDGFRB', 'SPARC'],
    'Endothelial': ['PECAM1', 'ESAM', 'RGS5'],
    'Immune': ['CD68', 'LYZ', 'CD74'],
    'Endocrine_Beta': ['INS', 'IAPP'],
    'Endocrine_Alpha': ['GCG', 'ARX'],
    'Endocrine_Delta': ['SST'],
    'Endocrine_Gamma': ['PPY']
}

def annotate_dataset(h5ad_path, donor):
    print(f"\n{'='*50}\nAnnotating Donor {donor}\n{'='*50}")
    adata = sc.read_h5ad(h5ad_path)
    adata.obs['leiden'] = adata.obs['leiden'].astype(str)
    
    # 1. Base Cluster Identification
    print("Mapping Base Clusters via Wilcoxon...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    result_names = adata.uns['rank_genes_groups']['names']
    
    cluster_mapping = {}
    for cluster in adata.obs['leiden'].cat.categories:
        top_genes = [str(g) for g in result_names[str(cluster)] if not str(g).startswith(('NEGATIVE', 'BLANK', 'Unassigned'))][:10]
        
        best_match = 'Unknown'
        best_score = 0
        for cell_type, genes in markers.items():
            if cell_type.startswith('Endocrine'):
                continue
            
            overlap = len(set(top_genes).intersection(set(genes)))
            if overlap > best_score:
                best_score = overlap
                best_match = cell_type
                
        endo_genes = markers['Endocrine_Beta'] + markers['Endocrine_Alpha'] + markers['Endocrine_Delta'] + markers['Endocrine_Gamma']
        endo_overlap = len(set(top_genes).intersection(set(endo_genes)))
        if endo_overlap > best_score:
            best_match = 'Endocrine'
            
        cluster_mapping[cluster] = best_match

    adata.obs['base_cell_type'] = adata.obs['leiden'].map(cluster_mapping)
    
    # 2. Per-Cell Thresholding / Re-assignment
    print("Applying global thresholding to resolve spillovers...")
    all_markers = [g for sublist in markers.values() for g in sublist]
    available_markers = [g for g in all_markers if g in adata.var_names]
    
    expr = adata[:, available_markers].X
    if hasattr(expr, "todense"):
        expr = expr.todense()
    expr = np.asarray(expr)
    
    df_expr = pd.DataFrame(expr, columns=available_markers, index=adata.obs.index)
    
    score_df = pd.DataFrame(index=adata.obs.index)
    for cell_type, genes in markers.items():
        valid_genes = [g for g in genes if g in available_markers]
        if valid_genes:
            score_df[cell_type] = df_expr[valid_genes].mean(axis=1)
        else:
            score_df[cell_type] = 0
            
    score_df['Endocrine_General'] = score_df[['Endocrine_Beta', 'Endocrine_Alpha', 'Endocrine_Delta', 'Endocrine_Gamma']].max(axis=1)
    competition_cols = ['Acinar', 'Ductal', 'Stroma', 'Endothelial', 'Immune', 'Endocrine_General']
    
    final_annotations = []
    
    for i, row in score_df.iterrows():
        base_type = adata.obs.at[i, 'base_cell_type']
        comp_scores = row[competition_cols]
        max_score = comp_scores.max()
        
        if max_score > 0:
            winning_category = comp_scores.idxmax()
            if winning_category == 'Endocrine_General':
                endo_scores = row[['Endocrine_Beta', 'Endocrine_Alpha', 'Endocrine_Delta', 'Endocrine_Gamma']]
                final_annotations.append(endo_scores.idxmax())
            else:
                final_annotations.append(winning_category)
        else:
            if base_type == 'Endocrine':
                final_annotations.append('Endocrine_Unknown')
            else:
                final_annotations.append(base_type)
                
    adata.obs['cell_type_annotation'] = final_annotations
    
    name_map = {
        'Endocrine_Beta': 'Beta (INS)',
        'Endocrine_Alpha': 'Alpha (GCG)',
        'Endocrine_Delta': 'Delta (SST)',
        'Endocrine_Gamma': 'Gamma (PPY)',
        'Endocrine_Unknown': 'Endocrine_Unknown'
    }
    adata.obs['final_cell_type'] = adata.obs['cell_type_annotation'].map(lambda x: name_map.get(x, x))
    
    counts = adata.obs['final_cell_type'].value_counts().reset_index()
    counts.columns = ['Cell Type', 'Count']
    counts['Percentage'] = (counts['Count'] / len(adata) * 100).round(1).astype(str) + '%'
    
    print("\nFinal Global Cell Type Proportions:")
    print(counts.to_string(index=False))
    
    out_csv = reports_dir / f'donor_{donor}_global_annotation_proportions.csv'
    counts.to_csv(out_csv, index=False)
    print(f"Saved CSV report to {out_csv.name}")
    
    out_h5ad = dataset_dir / f'annotated_{h5ad_path.name}'
    adata.write_h5ad(out_h5ad)
    print("Saved annotated H5AD successfully.")

annotate_dataset(h5ad_37, '37')
annotate_dataset(h5ad_69, '69')
print("\nGlobal Annotation Pipeline Completed!")
