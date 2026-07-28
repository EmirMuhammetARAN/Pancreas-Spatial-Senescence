import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
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

print("\nRobust Scaling (Zaten filtering.py icinde yapildi)...")

p16_sinir = df['young_p16_95th'].iloc[0]
lamin_sinir = df['young_lamin_50th'].iloc[0]
df['is_senescent'] = ((df['CH_16_robust'] > p16_sinir) & (df['CH_20_robust'] < lamin_sinir)).astype(bool)

print("\n--- Bolgelere Gore Senescence Alfa ve Beta Hucre Dagilimi ---")
results = []
total_sen_beta, total_non_sen_beta = 0, 0
total_sen_alpha, total_non_sen_alpha = 0, 0

group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df['Group_Label'] = pd.Categorical(df['Group_Label'], categories=group_order, ordered=True)

for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95)).values.astype(bool)
    is_alpha = (group['CH_6'] > group['CH_6'].quantile(0.95)).values.astype(bool)
    
    is_sen_beta = group['is_senescent'].values
    is_sen_alpha = group['is_senescent'].values
    
    n_beta = is_beta.sum()
    n_alpha = is_alpha.sum()
    
    sen_beta = (is_sen_beta & is_beta).sum()
    sen_alpha = (is_sen_alpha & is_alpha).sum()
    
    total_sen_beta += sen_beta
    total_non_sen_beta += (n_beta - sen_beta)
    total_sen_alpha += sen_alpha
    total_non_sen_alpha += (n_alpha - sen_alpha)
    
    pct_beta_sen = (sen_beta / n_beta) * 100 if n_beta > 0 else 0
    pct_alpha_sen = (sen_alpha / n_alpha) * 100 if n_alpha > 0 else 0
    
    results.append({'Group_Label': region, 'Beta (%)': pct_beta_sen, 'Alfa (%)': pct_alpha_sen})
    
    print(f"\n{region}:")
    print(f"  Toplam Beta: {n_beta:,} -> Senescence: {sen_beta:,} (%{pct_beta_sen:.2f})")
    print(f"  Toplam Alfa: {n_alpha:,} -> Senescence: {sen_alpha:,} (%{pct_alpha_sen:.2f})")

res_df = pd.DataFrame(results)

contingency_table = [
    [total_sen_beta, total_non_sen_beta],
    [total_sen_alpha, total_non_sen_alpha]
]
if (total_sen_beta + total_sen_alpha) == 0:
    chi2, p_val = 0, 1.0
else:
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

if (total_non_sen_beta * total_sen_alpha) > 0:
    odds_ratio = (total_sen_beta * total_non_sen_alpha) / (total_non_sen_beta * total_sen_alpha)
else:
    odds_ratio = np.nan

n_total = total_sen_beta + total_non_sen_beta + total_sen_alpha + total_non_sen_alpha
cramers_v = np.sqrt(chi2 / n_total) if n_total > 0 else 0

print(f"\n--- Istatistiksel Testler ve Etki Buyuklugu (Effect Size) ---")
print(f"Chi-Square p-degeri : {p_val:.2e}")
print(f"Odds Ratio (OR)     : {odds_ratio:.2f} (Beta hucreleri, Alfa'ya gore {odds_ratio:.2f} kat daha fazla senesans riski tasiyor)")
print(f"Cramer's V          : {cramers_v:.4f} (Iliski Gucu: 0 zayif, 1 guclu)")

if p_val < 0.001:
    print("Sonuc: Beta hucrelerinin senesansa girme orani Alfa hucrelerinden ISTATISTIKSEL OLARAK ANLAMLI derecede farklidir (p < 0.001) ***")
else:
    print("Sonuc: Istatistiksel olarak anlamli bir fark bulunamadi.")

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(res_df))

plt.bar(index, res_df['Beta (%)'], bar_width, label='Beta Hucreleri', color='#4C72B0', edgecolor='black')
plt.bar(index + bar_width, res_df['Alfa (%)'], bar_width, label='Alfa Hucreleri', color='#C44E52', edgecolor='black')

plt.xlabel('Grup', fontsize=12, fontweight='bold')
plt.ylabel('Senescence Ratioi (%)', fontsize=12, fontweight='bold')
plt.title(f'{COHORT_NAME} - Beta vs Alfa Hucrelerinde Senescence Ratiolari\n(Chi-Square p-value: {p_val:.2e})', fontsize=14, fontweight='bold')
plt.xticks(index + bar_width / 2, res_df['Group_Label'], fontsize=11, rotation=15)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_beta_vs_alpha_senescence.png")
plt.savefig(save_path, dpi=300)
print(f"\nGrafik '{save_path}' olarak kaydedildi.")
