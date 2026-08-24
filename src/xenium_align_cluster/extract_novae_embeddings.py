import os
import pandas as pd
import numpy as np
import scanpy as sc
import novae
import torch
import gc

def main():
    print("Sopa-Inspired Novae Inference Basliyor...")
    torch.set_float32_matmul_precision('medium')
    
    input_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\parquets"
    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    PROTEIN_COLS = [f"CH_{i}_core" for i in range(38)]
    
    model_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\novae_model_weights"
    out_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\novae_embeddings"
    
    print("Egitilmis Novae Modeli Yukleniyor...")
    model = novae.Novae.from_pretrained(model_dir)
    
    for f in files:
        batch_name = f.split('_')[2]
        
        out_file = os.path.join(out_dir, f"{batch_name}_novae.parquet")
        if os.path.exists(out_file):
            print(f"Zaten mevcut: {batch_name}")
            continue
            
        print(f"\n---> {batch_name} yukleniyor ve graf oruluyor...")
        df_path = os.path.join(input_dir, f)
        df = pd.read_parquet(df_path)
        
        X = df[PROTEIN_COLS].fillna(0).values.astype(np.float32)
        spatial_coords = df[['global_x', 'global_y']].values
        
        adata = sc.AnnData(X=X)
        adata.obsm['spatial'] = spatial_coords
        adata.obs['batch'] = pd.Categorical([batch_name] * len(adata))
        sc.pp.log1p(adata)
        
        novae.spatial_neighbors(adata, n_neighs=15, delaunay=False, radius=None)
        
        print(f"---> {batch_name} icin Latent uzay hesaplaniyor...")
        model.compute_representations(adata, accelerator='gpu')
        
        emb = adata.obsm['novae_latent']
        emb_df = pd.DataFrame(emb, columns=[f"Novae_{d}" for d in range(emb.shape[1])])
        emb_df.to_parquet(out_file, index=False)
        
        del adata
        del df
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{batch_name} embeddings basariyla diske yazildi!")

if __name__ == "__main__":
    main()
