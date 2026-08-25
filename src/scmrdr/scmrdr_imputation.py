import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import pandas as pd
import numpy as np
import scanpy as sc
import copy
from pathlib import Path
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def zinb_loss(x, mean, disp, pi, eps=1e-8):
    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_case = t1 + t2 - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.lt(x, 1e-8), zero_case, nb_case)
    return torch.mean(result)

class scMRDR(nn.Module):
    def __init__(self, prot_dim=38, rna_dim=300, latent_dim=32):
        super(scMRDR, self).__init__()
        
        self.encoder_p = nn.Sequential(
            nn.Linear(prot_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.encoder_r = nn.Sequential(
            nn.Linear(rna_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU()
        )
        
        self.fc_mu = nn.Linear(64 + 128, latent_dim)
        self.fc_logvar = nn.Linear(64 + 128, latent_dim)
        
        self.decoder_p = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, prot_dim)
        )
        
        self.decoder_r_base = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ELU()
        )
        self.dec_r_mean = nn.Sequential(nn.Linear(256, rna_dim), nn.Softplus())
        self.dec_r_disp = nn.Sequential(nn.Linear(256, rna_dim), nn.Softplus())
        self.dec_r_pi = nn.Sequential(nn.Linear(256, rna_dim), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, p, r, mask):
        h_p = self.encoder_p(p)
        h_r = self.encoder_r(r)
        
        h_r = h_r * mask.unsqueeze(1)
        
        h_concat = torch.cat([h_p, h_r], dim=1)
        
        mu_u = self.fc_mu(h_concat)
        logvar_u = self.fc_logvar(h_concat)
        
        z_u = self.reparameterize(mu_u, logvar_u)
        
        p_recon = self.decoder_p(z_u)
        r_base = self.decoder_r_base(z_u)
        r_mean = self.dec_r_mean(r_base)
        r_disp = self.dec_r_disp(r_base)
        r_pi = self.dec_r_pi(r_base)
        
        return p_recon, (r_mean, r_disp, r_pi), mu_u, logvar_u

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0

dataset_dir = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset")
parquet_dir = dataset_dir / "parquets"

print("Merging Reference Data using True Spatial Registration Matches...")
adata_393 = sc.read_h5ad(dataset_dir / r"spatial-transkriptomics\annotated_secondary_analysis(37 393 üst).h5ad")
pq_393 = pd.read_parquet(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt393_xenium_registration\SNT393_matches_high_confidence_5um.parquet")

df_ref_393 = adata_393.obs[['cell_id']].copy()
rna_df_393 = pd.DataFrame(adata_393.X.toarray() if hasattr(adata_393.X, 'toarray') else adata_393.X, index=adata_393.obs.index)
df_ref_393 = pd.concat([df_ref_393, rna_df_393], axis=1)
merged_393 = df_ref_393.merge(pq_393, left_on='cell_id', right_on='xenium_cell_id', how='inner')
print(f"Matched {len(merged_393)} true cells for SNT393.")

adata_227 = sc.read_h5ad(dataset_dir / r"spatial-transkriptomics\annotated_secondary_analysis(69yaş 227alt).h5ad")
pq_227 = pd.read_parquet(r"D:\GitHub\Pancreas-Spatial-Senescence\src\xenium_align_cluster\snt227_xenium_registration\SNT227_matches_high_confidence_5um.parquet")

df_ref_227 = adata_227.obs[['cell_id']].copy()
rna_df_227 = pd.DataFrame(adata_227.X.toarray() if hasattr(adata_227.X, 'toarray') else adata_227.X, index=adata_227.obs.index)
df_ref_227 = pd.concat([df_ref_227, rna_df_227], axis=1)
merged_227 = df_ref_227.merge(pq_227, left_on='cell_id', right_on='xenium_cell_id', how='inner')
print(f"Matched {len(merged_227)} true cells for SNT227.")

ref_data = pd.concat([merged_393, merged_227], ignore_index=True)

prot_cols = [f"CH_{i}_full" for i in range(38)]
rna_cols = list(range(300))

X_prot_ref = torch.FloatTensor(np.nan_to_num(ref_data[prot_cols].values))
X_rna_ref = torch.FloatTensor(np.nan_to_num(ref_data[rna_cols].values)) 
Mask_ref = torch.ones(X_prot_ref.size(0))

idx = torch.randperm(X_prot_ref.size(0))
ref_dataset = TensorDataset(X_prot_ref[idx], X_rna_ref[idx], Mask_ref[idx])

total_size = len(ref_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(ref_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

model = scMRDR(prot_dim=38, rna_dim=300, latent_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
mse_loss = nn.MSELoss()
beta = 1.0 
early_stopping = EarlyStopping(patience=10)


log_file = open(dataset_dir / "scmrdr_training.log", "w", encoding="utf-8")
csv_file = open(dataset_dir / "scmrdr_loss.csv", "w", encoding="utf-8")
csv_file.write("Epoch,Train_Prot,Train_RNA,Val_Prot,Val_RNA\n")
train_losses, val_losses = [], []

print("\nTraining scMRDR Model with Early Stopping (Max 200 Epochs)...")
epochs = 200

for epoch in range(epochs):
    model.train()
    epoch_p_loss = 0
    epoch_r_loss = 0
    epoch_kld = 0
    
    for p, r, m in train_loader:
        p, r, m = p.to(device), r.to(device), m.to(device)
        
        optimizer.zero_grad()
        p_recon, (r_mean, r_disp, r_pi), mu, logvar = model(p, r, m)
        
        loss_p = mse_loss(p_recon, p)
        loss_r = zinb_loss(r, r_mean, r_disp, r_pi)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / p.size(0)
        
        loss = loss_p + loss_r + beta * kld
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_p_loss += loss_p.item()
        epoch_r_loss += loss_r.item()
        epoch_kld += kld.item()
    
    train_p = epoch_p_loss/len(train_loader)
    train_r = epoch_r_loss/len(train_loader)
        
    model.eval()
    val_p_loss = 0
    val_r_loss = 0
    with torch.no_grad():
        for p, r, m in val_loader:
            p, r, m = p.to(device), r.to(device), m.to(device)
            p_recon, (r_mean, r_disp, r_pi), mu, logvar = model(p, r, m)
            
            loss_p = mse_loss(p_recon, p)
            loss_r = zinb_loss(r, r_mean, r_disp, r_pi)
            
            val_p_loss += loss_p.item()
            val_r_loss += loss_r.item()
            
    val_p = val_p_loss/len(val_loader)
    val_r = val_r_loss/len(val_loader)
    val_total = val_p + val_r
    
    log_line = f"Epoch {epoch+1:03d}/{epochs} | Train [Prot:{train_p:.2f} RNA:{train_r:.4f}] | Val [Prot:{val_p:.2f} RNA:{val_r:.4f}]"
    print(log_line)
    log_file.write(log_line + "\n")
    csv_file.write(f"{epoch+1},{train_p},{train_r},{val_p},{val_r}\n")
    train_losses.append(train_p + train_r)
    val_losses.append(val_p + val_r)
    
    early_stopping(val_total, model)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}! Restoring best weights.")
        model.load_state_dict(early_stopping.best_state)
        break
else:
    print("Reached max epochs! Restoring best weights anyway.")
    model.load_state_dict(early_stopping.best_state)


log_file.close()
csv_file.close()
torch.save(early_stopping.best_state, dataset_dir / "scmrdr_best_weights.pth")
plt.figure()
plt.plot(train_losses, label="Train Total Loss")
plt.plot(val_losses, label="Val Total Loss")
plt.legend()
plt.title("scMRDR Autoencoder Training Loss")
plt.savefig(dataset_dir / "scmrdr_loss_curve.png", dpi=300)

print("\nImputing Target Datasets...")
target_files = [
    "features_dual_SNT227_age69.parquet",
    "features_dual_SNT348_age37.parquet",
    "features_dual_SNT354_age35.parquet",
    "features_dual_SNT393_age37.parquet",
    "features_dual_SNT484_age69.parquet",
    "features_dual_SNT675_age69.parquet",
    "features_dual_SNT899_age35.parquet"
]

out_dir = dataset_dir / "imputed_h5ad"
out_dir.mkdir(exist_ok=True)
model.eval()
gene_names = adata_393.var_names 

for tgt_file in target_files:
    file_path = parquet_dir / tgt_file
    
    if not file_path.exists():
        print(f"Warning: File not found {file_path}")
        continue
    print(f"Processing {tgt_file}...")
    pq_tgt = pd.read_parquet(file_path)
    X_prot_tgt = torch.FloatTensor(np.nan_to_num(pq_tgt[prot_cols].values))
    X_rna_dummy = torch.zeros((X_prot_tgt.size(0), 300))
    Mask_tgt = torch.zeros(X_prot_tgt.size(0))
    
    tgt_loader = DataLoader(TensorDataset(X_prot_tgt, X_rna_dummy, Mask_tgt), batch_size=4096, shuffle=False)
    
    imputed_rna = np.empty((X_prot_tgt.size(0), 300), dtype=np.float32)
    idx_counter = 0
    with torch.no_grad():
        for p, r, m in tgt_loader:
            p, r, m = p.to(device), r.to(device), m.to(device)
            _, (r_mean, _, _), _, _ = model(p, r, m)
            batch_size = p.size(0)
            imputed_rna[idx_counter:idx_counter+batch_size] = r_mean.cpu().numpy()
            idx_counter += batch_size
    
    new_adata = sc.AnnData(X=imputed_rna)
    new_adata.var_names = gene_names
    new_adata.obs = pq_tgt[['label', 'patient_id', 'age', 'tile_y', 'tile_x', 'global_y', 'global_x']].copy()
    new_adata.obs.index = new_adata.obs.index.astype(str)
    
    out_name = tgt_file.replace('features_dual_', 'imputed_xenium_').replace('.parquet', '.h5ad')
    new_adata.write_h5ad(out_dir / out_name)
    print(f"Saved imputed H5AD to {out_name}")

print("\nStage 2.5: Optimized scMRDR Imputation Completed Successfully!")
