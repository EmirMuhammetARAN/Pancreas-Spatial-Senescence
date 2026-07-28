import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import os

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print("\nVeriler yükleniyor...")
df_list = []
for snt, age, region, label in ACTIVE_COHORT:
    d_clean = load_and_filter_tissue(snt, age)
    if d_clean is not None:
        d_clean['Group_Label'] = label
        d_clean['patient_id'] = label  # Geriye donuk uyumluluk
        df_list.append(d_clean)

df = pd.concat(df_list, ignore_index=True)

global_p16 = df['young_p16_95th'].iloc[0]
global_lamin = df['young_lamin_50th'].iloc[0]
print(f"Young Ref Eşikleri: p16>{global_p16:.2f}, Lamin_low<{global_lamin:.2f}")

results_insulin = []
results_cpeptide = []

for patient_id, group in df.groupby('patient_id'):
    age = group['age'].iloc[0]
    
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95)).values
    
    is_sen = ((group['CH_16_robust'] > global_p16) & (group['CH_20_robust'] < global_lamin)).values
    
    sen_beta = is_sen & is_beta
    non_sen_beta = (~is_sen) & is_beta
    
    n_sen = sen_beta.sum()
    n_non = non_sen_beta.sum()
    
    if n_sen < 3:
        print(f"  {age} yaş: Yeterli senesans beta yok (n={n_sen}), atlanıyor")
        continue
    
    ins_sen = group.loc[sen_beta, 'CH_3'].values
    ins_non = group.loc[non_sen_beta, 'CH_3'].values
    
    stat_ins, p_ins = mannwhitneyu(ins_sen, ins_non, alternative='greater')
    
    cpep_sen = group.loc[sen_beta, 'CH_22'].values
    cpep_non = group.loc[non_sen_beta, 'CH_22'].values
    
    stat_cpep, p_cpep = mannwhitneyu(cpep_sen, cpep_non, alternative='greater')
    
    sig_ins = '***' if p_ins < 0.001 else ('**' if p_ins < 0.01 else ('*' if p_ins < 0.05 else 'ns'))
    sig_cpep = '***' if p_cpep < 0.001 else ('**' if p_cpep < 0.01 else ('*' if p_cpep < 0.05 else 'ns'))
    
    print(f"  {age} yaş: İnsülin Sen={ins_sen.mean():.1f} vs Non={ins_non.mean():.1f} (p={p_ins:.2e}) {sig_ins} | "
          f"C-Peptide Sen={cpep_sen.mean():.1f} vs Non={cpep_non.mean():.1f} (p={p_cpep:.2e}) {sig_cpep}")
    
    results_insulin.append({
        'Yaş': age, 'Hasta': patient_id,
        'Sen_mean': ins_sen.mean(), 'Non_mean': ins_non.mean(),
        'Sen_std': ins_sen.std(), 'Non_std': ins_non.std(),
        'p-value': p_ins, 'n_sen': n_sen, 'n_non': n_non
    })
    results_cpeptide.append({
        'Yaş': age, 'Hasta': patient_id,
        'Sen_mean': cpep_sen.mean(), 'Non_mean': cpep_non.mean(),
        'Sen_std': cpep_sen.std(), 'Non_std': cpep_non.std(),
        'p-value': p_cpep, 'n_sen': n_sen, 'n_non': n_non
    })

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

if results_insulin:
    res1 = pd.DataFrame(results_insulin).sort_values('Yaş')
    x = np.arange(len(res1))
    w = 0.35
    ax1.bar(x - w/2, res1['Non_mean'], w, label='Normal Beta', color='#4C72B0', alpha=0.8,
            yerr=res1['Non_std'], capsize=3)
    ax1.bar(x + w/2, res1['Sen_mean'], w, label='Senescence Beta', color='#C44E52', alpha=0.8,
            yerr=res1['Sen_std'], capsize=3)
    for i, (_, row) in enumerate(res1.iterrows()):
        sig = '***' if row['p-value'] < 0.001 else ('**' if row['p-value'] < 0.01 else ('*' if row['p-value'] < 0.05 else 'ns'))
        max_val = max(row['Non_mean'] + row['Non_std'], row['Sen_mean'] + row['Sen_std'])
        ax1.text(i, max_val + 2, sig, ha='center', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(a)} yaş" for a in res1['Yaş']], fontsize=11)
    ax1.set_ylabel('İnsülin (CH_3) İntensitesi', fontsize=12)
    ax1.set_title('İnsülin İntensitesi\n(Senescence vs Normal Beta)', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

if results_cpeptide:
    res2 = pd.DataFrame(results_cpeptide).sort_values('Yaş')
    x = np.arange(len(res2))
    w = 0.35
    ax2.bar(x - w/2, res2['Non_mean'], w, label='Normal Beta', color='#4C72B0', alpha=0.8,
            yerr=res2['Non_std'], capsize=3)
    ax2.bar(x + w/2, res2['Sen_mean'], w, label='Senescence Beta', color='#C44E52', alpha=0.8,
            yerr=res2['Sen_std'], capsize=3)
    for i, (_, row) in enumerate(res2.iterrows()):
        sig = '***' if row['p-value'] < 0.001 else ('**' if row['p-value'] < 0.01 else ('*' if row['p-value'] < 0.05 else 'ns'))
        max_val = max(row['Non_mean'] + row['Non_std'], row['Sen_mean'] + row['Sen_std'])
        ax2.text(i, max_val + 2, sig, ha='center', fontsize=12, fontweight='bold', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(a)} yaş" for a in res2['Yaş']], fontsize=11)
    ax2.set_ylabel('C-Peptide (CH_22) İntensitesi', fontsize=12)
    ax2.set_title('C-Peptide İntensitesi\n(Senescence vs Normal Beta)', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle("Patra et al. (2024) Doğrulaması: Senescence Beta Hücreleri Daha Aktif mi?\n"
             "İnsülin + C-Peptide Çift Kanıt (p16↑ + LaminB1↓ Tanımı)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
out_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\saglikli grafikler 35-69"
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, "patra_insulin_test.png")
plt.savefig(output_path, dpi=300)
print(f"\nGrafik '{output_path}' olarak kaydedildi.")
