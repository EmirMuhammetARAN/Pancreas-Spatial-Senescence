import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

import os

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print("\nVeriler yukleniyor...")
df_list = []
for snt, age, region, label in ACTIVE_COHORT:
    d_clean = load_and_filter_tissue(snt, age)
    if d_clean is not None:
        d_clean['Group_Label'] = label
        d_clean['patient_id'] = label  # Geriye donuk uyumluluk
        df_list.append(d_clean)

df_all = pd.concat(df_list, ignore_index=True)

global_p16 = df_all['young_p16_95th'].iloc[0]
global_lamin = df_all['young_lamin_50th'].iloc[0]
print(f"Young Ref Esikleri: p16>{global_p16:.2f}, Lamin_low<{global_lamin:.2f}")

results = []

for patient_id, group in df_all.groupby('patient_id'):
    age = group['age'].iloc[0]
    
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95)).values
    n_beta = is_beta.sum()
    
    if n_beta < 10:
        print(f"  {age} yaş: Yeterli beta yok (n={n_beta}), atlanıyor")
        continue
    
    beta_cells = group[is_beta]
    
    p16_only = (beta_cells['CH_16_robust'] > global_p16).sum()
    oran_p16_only = (p16_only / n_beta) * 100
    
    p16_lamin = ((beta_cells['CH_16_robust'] > global_p16) & (beta_cells['CH_20_robust'] < global_lamin)).sum()
    oran_p16_lamin = (p16_lamin / n_beta) * 100
    
    print(f"  {age} yaş: Beta={n_beta:,} | p16-only: {p16_only} ({oran_p16_only:.1f}%) | "
          f"p16+Lamin: {p16_lamin} ({oran_p16_lamin:.1f}%)")
    
    results.append({
        'Yaş': age,
        'Hasta': patient_id,
        'Toplam Beta': n_beta,
        'p16-only (n)': p16_only,
        'p16-only (%)': oran_p16_only,
        'p16+Lamin (n)': p16_lamin,
        'p16+Lamin (%)': oran_p16_lamin
    })

df_res = pd.DataFrame(results).sort_values('Yaş')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(df_res))
w = 0.35

bars1 = ax1.bar(x - w/2, df_res['p16-only (%)'], w, label='p16-only (Patra Tanımı)', 
                color='#E8A838', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + w/2, df_res['p16+Lamin (%)'], w, label='p16↑ + LaminB1↓ (Bizim Tanım)', 
                color='#C44E52', alpha=0.85, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars1, df_res['p16-only (%)']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, df_res['p16+Lamin (%)']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(a)} yaş" for a in df_res['Yaş']], fontsize=11)
ax1.set_ylabel('p16+ Beta Hücre Ratioı (%)', fontsize=12)
ax1.set_title('Patra et al. (2024) Doğrulaması:\np16+ Beta Hücre Ratioı Yaşla Artıyor mu?', 
              fontweight='bold', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

ax1.axhline(y=6.3, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Patra <28y (%6.3)')
ax1.axhline(y=25.1, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Patra >38y (%25.1)')
ax1.legend(fontsize=9)

ax2.plot(df_res['Yaş'], df_res['p16-only (%)'], marker='o', linewidth=2.5, markersize=10,
         color='#E8A838', label='p16-only (Patra Tanımı)')
ax2.plot(df_res['Yaş'], df_res['p16+Lamin (%)'], marker='s', linewidth=2.5, markersize=10,
         color='#C44E52', label='p16↑ + LaminB1↓ (Bizim Tanım)')

ax2.axhspan(0, 6.3, alpha=0.1, color='green', label='Patra Genç Bandı (<%6.3)')
ax2.axhspan(25.1, 35, alpha=0.1, color='red', label='Patra Yaşlı Bandı (>%25.1)')

ax2.set_xlabel('Donör Yaşı', fontsize=12)
ax2.set_ylabel('p16+ Beta Hücre Ratioı (%)', fontsize=12)
ax2.set_title('Yaşa Bağlı p16+ Ratio Trendi\n(Patra Referans Bantları ile)', fontweight='bold', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\saglikli grafikler 35-69"
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "patra_p16_oran_trendi.png")
plt.savefig(save_path, dpi=300)
print(f"\nGrafik '{save_path}' olarak kaydedildi.")
