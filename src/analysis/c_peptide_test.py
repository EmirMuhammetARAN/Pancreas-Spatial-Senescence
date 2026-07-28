import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print(f"{COHORT_NAME} - Veriler yukleniyor ve DBSCAN filtresi uygulaniyor...")

df_list = []
for snt, age, region, label in ACTIVE_COHORT:
    print(f"{snt} ({region}) isleniyor...")
    d_clean = load_and_filter_tissue(snt, age)
    if d_clean is not None:
        d_clean['Group_Label'] = label
        df_list.append(d_clean)

if not df_list:
    print("HATA: Hcbir veri yuklenemedi.")
    sys.exit()

df = pd.concat(df_list, ignore_index=True)
print(f"Toplam hucre sayisi (DBSCAN Sonrasi): {len(df):,}")

results = []
plot_data = []

print("\n--- Patra et al. (2024) Hipotezi: C-Peptid (CH_22) Yogunlugu Testi ---")
print("Hipotez: Senescence Beta hucrelerinde (p16 high) C-Peptid sinyali, normal Beta hucrelerine gore DAHA YUKSEKTIR.\n")

group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df['Group_Label'] = pd.Categorical(df['Group_Label'], categories=group_order, ordered=True)

p16_sinir = df['young_p16_95th'].iloc[0]
lamin_sinir = df['young_lamin_50th'].iloc[0]
df['is_senescent'] = ((df['CH_16_robust'] > p16_sinir) & (df['CH_20_robust'] < lamin_sinir)).astype(bool)

for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95)).values.astype(bool)
    
    is_sen = group['is_senescent'].values
    
    beta_mask = is_beta
    senescent_mask = is_sen[beta_mask]
    
    c_pep_values = group.loc[beta_mask, 'CH_22'].values
    
    sen_beta_cpep = c_pep_values[senescent_mask]
    non_sen_beta_cpep = c_pep_values[~senescent_mask]
    
    if len(sen_beta_cpep) > 0 and len(non_sen_beta_cpep) > 0:
        mean_sen = sen_beta_cpep.mean()
        mean_non = non_sen_beta_cpep.mean()
        
        stat, p_val = mannwhitneyu(sen_beta_cpep, non_sen_beta_cpep, alternative='greater')
        
        results.append({
            'Group_Label': region,
            'Normal Beta C-Peptid': mean_non,
            'Yasli Beta C-Peptid': mean_sen,
            'p-value': p_val,
        })
        
        np.random.seed(42)
        n_sample_non = min(5000, len(non_sen_beta_cpep))
        n_sample_sen = min(5000, len(sen_beta_cpep))
        
        for val in np.random.choice(non_sen_beta_cpep, n_sample_non, replace=False):
            plot_data.append({'Group_Label': region, 'Durum': 'Normal Beta', 'C-Peptid (CH_22)': val})
        for val in np.random.choice(sen_beta_cpep, n_sample_sen, replace=False):
            plot_data.append({'Group_Label': region, 'Durum': 'Yasli Beta', 'C-Peptid (CH_22)': val})

        print(f"[{region}]")
        print(f"  Normal Beta (n={len(non_sen_beta_cpep):,}) Ort. C-Peptid: {mean_non:.2f}")
        print(f"  Yasli Beta  (n={len(sen_beta_cpep):,}) Ort. C-Peptid: {mean_sen:.2f}")
        print(f"  P-degeri (Mann-Whitney U): {p_val:.2e} -> {'*** DESTEKLIYOR ***' if p_val < 0.05 else 'Desteklemiyor'}\n")
    else:
        print(f"[{region}] icin yeterli yasli beta hucresi yok.\n")

if len(plot_data) > 0:
    plot_df = pd.DataFrame(plot_data)
    plot_df['Group_Label'] = pd.Categorical(plot_df['Group_Label'], categories=group_order, ordered=True)
    plot_df = plot_df.sort_values('Group_Label')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Group_Label', y='C-Peptid (CH_22)', hue='Durum', data=plot_df, 
                palette={'Normal Beta': '#a1c9f4', 'Yasli Beta': '#ffb482'}, showfliers=False)

    plt.title(f"{COHORT_NAME} - Patra et al. (2024) Hipotez Dogrulamasi:\nYasli (Senescence) Beta Hucrelerinde Artmis C-Peptid Birikimi", fontsize=14, fontweight='bold')
    plt.xlabel("Grup", fontsize=12, fontweight='bold')
    plt.ylabel("C-Peptid (CH_22) Sinyal Siddeti", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Hucre Durumu")
    plt.xticks(rotation=15)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_c_peptide_hypothesis_test.png")
    plt.savefig(save_path, dpi=300)
    print(f"Grafik '{save_path}' olarak kaydedildi.")
