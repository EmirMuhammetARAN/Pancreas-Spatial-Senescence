import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_mmd(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz
    bandwidth_range = [10, 15, 20, 50]
    loss = 0
    for a in bandwidth_range:
        loss += (torch.exp(-0.5 * dxx / a).mean() + 
                 torch.exp(-0.5 * dyy / a).mean() - 
                 2 * torch.exp(-0.5 * dxy / a).mean())
    return loss

class CytoOneModel(nn.Module):
    def __init__(self, input_dim):
        super(CytoOneModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim)
        )
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.softplus(self.decoder(z))
        return x_recon, z

parquet_dir = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\parquets")
ref_files = ["features_dual_SNT227_age69.parquet", "features_dual_SNT393_age37.parquet"]
target_files = [f.name for f in parquet_dir.glob("*.parquet") if f.name not in ref_files]

feature_cols = [f"CH_{i}_full" for i in range(38)]

def prepare_tensor(df):
    data = df[feature_cols].values
    data = np.nan_to_num(data)
    data = np.log1p(data) 
    return torch.FloatTensor(data)

print("Loading Reference Data (SNT227 & SNT393)...")
ref_dfs = [pd.read_parquet(parquet_dir / f) for f in ref_files]
ref_df = pd.concat(ref_dfs, ignore_index=True)
ref_tensor = prepare_tensor(ref_df)

idx = torch.randperm(ref_tensor.size(0))[:100000]
ref_tensor_sub = ref_tensor[idx].to(device)
ref_loader = DataLoader(TensorDataset(ref_tensor_sub), batch_size=2048, shuffle=True)

out_dir = Path(r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\corrected_parquets")
out_dir.mkdir(exist_ok=True)

for tgt_file in target_files:
    print(f"\n{'='*50}\nProcessing Target: {tgt_file}\n{'='*50}")
    tgt_df = pd.read_parquet(parquet_dir / tgt_file)
    tgt_tensor = prepare_tensor(tgt_df)
    
    tgt_loader = DataLoader(TensorDataset(tgt_tensor), batch_size=2048, shuffle=True)
    
    model = CytoOneModel(input_dim=len(feature_cols)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 10
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        ref_iter = iter(ref_loader)
        
        for batch in tgt_loader:
            x_tgt = batch[0].to(device)
            
            try:
                x_ref = next(ref_iter)[0].to(device)
            except StopIteration:
                ref_iter = iter(ref_loader)
                x_ref = next(ref_iter)[0].to(device)
                
            min_size = min(x_tgt.size(0), x_ref.size(0))
            x_tgt_sub, x_ref_sub = x_tgt[:min_size], x_ref[:min_size]
            
            optimizer.zero_grad()
            
            tgt_recon, tgt_z = model(x_tgt_sub)
            _, ref_z = model(x_ref_sub)
            
            recon_loss = criterion(tgt_recon, x_tgt_sub)
            mmd_loss = compute_mmd(tgt_z, ref_z)
            
            loss = recon_loss + 10.0 * mmd_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(tgt_loader):.4f}")
    
    print("Applying CytoOne correction to target dataset...")
    model.eval()
    corrected_data = []
    eval_loader = DataLoader(TensorDataset(tgt_tensor), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for batch in eval_loader:
            x = batch[0].to(device)
            x_recon, _ = model(x)
            corrected_data.append(x_recon.cpu().numpy())
            
    corrected_data = np.concatenate(corrected_data, axis=0)
    
    tgt_df[feature_cols] = np.expm1(corrected_data) 
    
    out_path = out_dir / f"cytoone_corrected_{tgt_file}"
    tgt_df.to_parquet(out_path)
    print(f"Saved corrected dataset to {out_path}")

print("\nStage 1: CytoOne Batch Effect Correction Completed Successfully!")
