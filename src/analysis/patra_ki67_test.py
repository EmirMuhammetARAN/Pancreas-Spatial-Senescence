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
print(f"Ki67 (CH_31) global stats: mean={df['CH_31'].mean():.2f}, "
      f"median={df['CH_31'].median():.2f}, std={df['CH_31'].std():.2f}")

results = []

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
    
    ki67_sen = group.loc[sen_beta, 'CH_31'].values
    ki67_non = group.loc[non_sen_beta, 'CH_31'].values
    
    stat, p_val = mannwhitneyu(ki67_sen, ki67_non, alternative='less')
    
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    
    ki67_threshold = group['CH_31'].quantile(0.95)
    ki67_pos_sen = (ki67_sen > ki67_threshold).mean() * 100
    ki67_pos_non = (ki67_non > ki67_threshold).mean() * 100
    
    print(f"  {age} yaş: Ki67 Sen={ki67_sen.mean():.2f} vs Non={ki67_non.mean():.2f} "
          f"(p={p_val:.2e}) {sig} | Ki67+ oran: Sen={ki67_pos_sen:.1f}% vs Non={ki67_pos_non:.1f}%")
    
    results.append({
        'Yaş': age, 'Hasta': patient_id,
        'Ki67_Sen_mean': ki67_sen.mean(), 'Ki67_Non_mean': ki67_non.mean(),
        'Ki67_Sen_std': ki67_sen.std(), 'Ki67_Non_std': ki67_non.std(),
        'Ki67+_Sen(%)': ki67_pos_sen, 'Ki67+_Non(%)': ki67_pos_non,
        'p-value': p_val, 'n_sen': n_sen, 'n_non': n_non
    })

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

if results:
    res = pd.DataFrame(results).sort_values('Yaş')
    
    x = np.arange(len(res))
    w = 0.35
    ax1.bar(x - w/2, res['Ki67_Non_mean'], w, label='Normal Beta', color='#4C72B0', alpha=0.8,
            yerr=res['Ki67_Non_std'], capsize=3)
    ax1.bar(x + w/2, res['Ki67_Sen_mean'], w, label='Senescence Beta', color='#C44E52', alpha=0.8,
            yerr=res['Ki67_Sen_std'], capsize=3)
    for i, (_, row) in enumerate(res.iterrows()):
        sig = '***' if row['p-value'] < 0.001 else ('**' if row['p-value'] < 0.01 else ('*' if row['p-value'] < 0.05 else 'ns'))
        max_val = max(row['Ki67_Non_mean'] + row['Ki67_Non_std'], row['Ki67_Sen_mean'] + row['Ki67_Sen_std'])
        ax1.text(i, max_val + 0.5, sig, ha='center', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(a)} yaş" for a in res['Yaş']], fontsize=11)
    ax1.set_ylabel('Ki67 (CH_31) İntensitesi', fontsize=12)
    ax1.set_title('Ki67 İntensitesi: Senescence = Bölünme Durması\n(Düşük Ki67 = Hücre Bölünmüyor)', 
                  fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax2.bar(x - w/2, res['Ki67+_Non(%)'], w, label='Normal Beta', color='#4C72B0', alpha=0.8)
    ax2.bar(x + w/2, res['Ki67+_Sen(%)'], w, label='Senescence Beta', color='#C44E52', alpha=0.8)
    for i, (_, row) in enumerate(res.iterrows()):
        max_val = max(row['Ki67+_Non(%)'], row['Ki67+_Sen(%)'])
        ax2.text(i - w/2, row['Ki67+_Non(%)'] + 0.3, f"{row['Ki67+_Non(%)']:.1f}%", 
                ha='center', fontsize=9, fontweight='bold', color='#4C72B0')
        ax2.text(i + w/2, row['Ki67+_Sen(%)'] + 0.3, f"{row['Ki67+_Sen(%)']:.1f}%", 
                ha='center', fontsize=9, fontweight='bold', color='#C44E52')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(a)} yaş" for a in res['Yaş']], fontsize=11)
    ax2.set_ylabel('Ki67+ (Bölünen) Hücre Ratioı (%)', fontsize=12)
    ax2.set_title('Bölünen Hücre Ratioı: Senescenceta Sıfıra Yakın mı?\n(Ki67+ = Aktif Bölünme)', 
                  fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle("Patra et al. (2024) Doğrulaması: Senescence = Hücre Bölünmesinin Durması\n"
             "Ki67 (Proliferasyon Markeri) Senescence Beta Hücrelerinde Çökmeli",
             fontsize=14, fontweight='bold')
plt.tight_layout()
out_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\saglikli grafikler 35-69"
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, "patra_ki67_test.png")
plt.savefig(output_path, dpi=300)
print(f"\nGrafik '{output_path}' olarak kaydedildi.")
