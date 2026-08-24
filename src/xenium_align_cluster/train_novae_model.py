import os
import pandas as pd
import numpy as np
import scanpy as sc
import novae
import torch
import gc
from lightning.pytorch.callbacks import Callback

class MemoryCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
    def on_train_epoch_end(self, trainer, pl_module):
        gc.collect()
        torch.cuda.empty_cache()

def main():
    print("Sopa-Inspired (Chunked) Novae Pipeline Basliyor...")
    torch.set_float32_matmul_precision('medium')
    
    input_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\parquets"
    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    PROTEIN_COLS = [f"CH_{i}_core" for i in range(38)]
    
    adatas = []
    for f in files:
        df_path = os.path.join(input_dir, f)
        df = pd.read_parquet(df_path)
        batch_name = f.split('_')[2]
        
        X = df[PROTEIN_COLS].fillna(0).values.astype(np.float32)
        spatial_coords = df[['global_x', 'global_y']].values
        
        adata = sc.AnnData(X=X)
        adata.obsm['spatial'] = spatial_coords
        adata.obs['batch'] = pd.Categorical([batch_name] * len(adata))
        sc.pp.log1p(adata)
        adatas.append(adata)
    
    print("\n[ADIM 1.5] Tum kesitler icin Novae Spatial Graph örülüyor...")
    novae.spatial_neighbors(adatas, n_neighs=15, delaunay=False, radius=None)
    
    print("\n[ADIM 2] Novae GNN Kuruluyor...")
    torch.cuda.empty_cache()
    model = novae.Novae(adatas, embedding_size=32, batch_size=128)
    
    print("\n[ADIM 3] Novae Egitimi Basliyor (Tensör çekirdekleri devrede)...")
    mem_cb = MemoryCallback()
    model.fit(max_epochs=100, accelerator='gpu', patience=3, num_workers=0, callbacks=[mem_cb])
    
    print("\n[ADIM 3.5] Model agirliklari diske kaydediliyor (Guvenlik noktasi)...")
    model_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\novae_model_weights"
    os.makedirs(model_dir, exist_ok=True)
    try:
        model.save_pretrained(model_dir)
        print("Model basariyla kaydedildi.")
    except Exception as e:
        print("Model kaydedilirken hata (onemsiz, devam ediliyor):", str(e))
    
    print("\n[ADIM 4] GNN Latent uzayi (X_novae) TEK TEK cikartiliyor (RAM dostu)...")
    out_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\novae_embeddings"
    os.makedirs(out_dir, exist_ok=True)
    
    for i, adata in enumerate(adatas):
        batch_name = adata.obs['batch'].iloc[0]
        print(f"\n---> {batch_name} icin Latent uzay hesaplaniyor...")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        model.compute_representations(adata, accelerator='gpu')
        
        emb = adata.obsm['novae_latent']
        emb_df = pd.DataFrame(emb, columns=[f"Novae_{d}" for d in range(emb.shape[1])])
        emb_df.to_parquet(os.path.join(out_dir, f"{batch_name}_novae.parquet"), index=False)
        
        # RAM sismemesi icin gereksiz devasa matrisleri aninda sil
        if 'novae_latent' in adata.obsm:
            del adata.obsm['novae_latent']
        if 'novae_latent_corrected' in adata.obsm:
            del adata.obsm['novae_latent_corrected']
            
        print(f"{batch_name} embeddings basariyla diske yazildi!")

if __name__ == "__main__":
    main()
