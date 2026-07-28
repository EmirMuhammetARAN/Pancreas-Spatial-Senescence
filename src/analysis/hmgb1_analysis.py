import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, pearsonr
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print(f"{COHORT_NAME} - Veriler yukleniyor ve DBSCAN filtresi uygulaniyor...")

columns_to_load = ['patient_id', 'age', 'CH_0', 'CH_3', 'CH_16', 'CH_20', 'CH_23', 'CH_30', 'global_x', 'global_y']

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

print("Robust Normalizasyon hesaplanıyor...")
df['CH_16_robust'] = 0.0
df['CH_20_robust'] = 0.0
for reg in df['Group_Label'].unique():
    mask = df['Group_Label'] == reg
    for ch in ['CH_16', 'CH_20']:
        data = df.loc[mask, ch].dropna()
        if len(data) == 0: continue
        med = data.median()
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25 if (q75 - q25) > 0 else 1
        df.loc[mask, f'{ch}_robust'] = (df.loc[mask, ch] - med) / iqr

global_p16 = df['young_p16_95th'].iloc[0]
global_lamin = df['CH_20_robust'].quantile(0.50)

CHUNK_SIZE = 50000
RADII = [25, 50, 100]

print(f"Global eşikler: p16 > {global_p16:.2f}, Lamin < {global_lamin:.2f}")
print(f"HMGB1 (CH_23) global stats: mean={df['CH_23'].mean():.2f}, "
      f"median={df['CH_23'].median():.2f}, std={df['CH_23'].std():.2f}")

group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df['Group_Label'] = pd.Categorical(df['Group_Label'], categories=group_order, ordered=True)

print("\n" + "="*60)
print("  ANALİZ 1: HMGB1 Nükleer Kaçış (Senescence vs Normal)")
print("="*60)

is_sen_global = (df['CH_16_robust'] > global_p16) & (df['CH_20_robust'] < global_lamin)

results_hmgb1 = []

for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    
    is_sen = ((group['CH_16_robust'] > global_p16) & (group['CH_20_robust'] < global_lamin)).values
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95)).values
    
    sen_beta = is_sen & is_beta
    non_sen_beta = (~is_sen) & is_beta
    
    n_sen = sen_beta.sum()
    n_non = non_sen_beta.sum()
    
    if n_sen < 3:
        print(f"  {region}: Yeterli senesans beta yok (n={n_sen}), atlanıyor")
        continue
    
    hmgb1_sen = group.loc[sen_beta, 'CH_23'].values
    hmgb1_non = group.loc[non_sen_beta, 'CH_23'].values
    
    stat, p_val = mannwhitneyu(hmgb1_sen, hmgb1_non, alternative='less')
    
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    
    print(f"  {region}: HMGB1 Sen={hmgb1_sen.mean():.2f} vs Non={hmgb1_non.mean():.2f} "
          f"(p={p_val:.2e}) {sig} [n_sen={n_sen}, n_non={n_non}]")
    
    results_hmgb1.append({
        'Group_Label': region,
        'HMGB1_Senescence': hmgb1_sen.mean(),
        'HMGB1_Normal': hmgb1_non.mean(),
        'p-value': p_val,
        'n_sen': n_sen,
        'n_non': n_non,
        'HMGB1_sen_std': hmgb1_sen.std(),
        'HMGB1_non_std': hmgb1_non.std()
    })

print("\n" + "="*60)
print("  ANALİZ 2: Tüm Hücrelerde HMGB1 (Senescence vs Normal)")
print("="*60)

results_all = []
for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    is_sen = ((group['CH_16_robust'] > global_p16) & (group['CH_20_robust'] < global_lamin)).values
    
    n_sen = is_sen.sum()
    if n_sen < 3:
        print(f"  {region}: Yeterli senesans yok (n={n_sen}), atlanıyor")
        continue
    
    hmgb1_sen = group.loc[is_sen, 'CH_23'].values
    hmgb1_non = group.loc[~is_sen, 'CH_23'].values
    
    stat, p_val = mannwhitneyu(hmgb1_sen, hmgb1_non, alternative='less')
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    
    print(f"  {region}: HMGB1 Sen={hmgb1_sen.mean():.2f} vs Non={hmgb1_non.mean():.2f} "
          f"(p={p_val:.2e}) {sig} [n_sen={n_sen}]")
    
    results_all.append({
        'Group_Label': region,
        'HMGB1_Senescence': hmgb1_sen.mean(),
        'HMGB1_Normal': hmgb1_non.mean(),
        'p-value': p_val,
        'n_sen': n_sen
    })

print("\n" + "="*60)
print("  ANALİZ 3: HMGB1^low Senescence Hücre Etrafında CD68+ Makrofaj")
print("="*60)

results_macro = []

for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    
    is_sen = ((group['CH_16_robust'] > global_p16) & (group['CH_20_robust'] < global_lamin)).values
    
    hmgb1_median = group['CH_23'].median()
    is_sen_hmgb1_low = is_sen & (group['CH_23'].values < hmgb1_median)
    is_sen_hmgb1_high = is_sen & (group['CH_23'].values >= hmgb1_median)
    
    is_macro = (group['CH_30'] > group['CH_30'].quantile(0.95)).values
    
    n_sen_low = is_sen_hmgb1_low.sum()
    n_sen_high = is_sen_hmgb1_high.sum()
    
    print(f"\n  {region}:")
    print(f"    Senescence+HMGB1_low: {n_sen_low} | Senescence+HMGB1_high: {n_sen_high}")
    print(f"    CD68+ makrofaj: {is_macro.sum()}")
    
    if n_sen_low < 3 or n_sen_high < 3:
        print(f"    UYARI: Yeterli hücre yok, atlanıyor")
        continue
    
    coords = group[['global_x', 'global_y']].values
    tree = cKDTree(coords)
    
    for radius in RADII:
        macro_near_sen_low = np.zeros(len(coords))
        macro_near_sen_high = np.zeros(len(coords))
        total_near = np.zeros(len(coords))
        
        for start_idx in range(0, len(coords), CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, len(coords))
            chunk_coords = coords[start_idx:end_idx]
            neighbors_list = tree.query_ball_point(chunk_coords, r=radius)
            
            for i_chunk, neighbors in enumerate(neighbors_list):
                i = start_idx + i_chunk
                n_list = [n for n in neighbors if n != i]
                if not n_list:
                    continue
                total_near[i] = len(n_list)
                macro_near_sen_low[i] = sum(1 for n in n_list if is_macro[n])
                macro_near_sen_high[i] = sum(1 for n in n_list if is_macro[n])
        
        density_low = macro_near_sen_low[is_sen_hmgb1_low] / np.maximum(total_near[is_sen_hmgb1_low], 1)
        density_high = macro_near_sen_high[is_sen_hmgb1_high] / np.maximum(total_near[is_sen_hmgb1_high], 1)
        density_non = macro_near_sen_low[~is_sen] / np.maximum(total_near[~is_sen], 1)
        
        mean_low = density_low.mean() * 100
        mean_high = density_high.mean() * 100
        mean_non = density_non.mean() * 100
        
        if len(density_low) > 1 and len(density_high) > 1:
            _, p_lh = mannwhitneyu(density_low, density_high, alternative='greater')
        else:
            p_lh = np.nan
        
        if len(density_low) > 1 and len(density_non) > 1:
            _, p_ln = mannwhitneyu(density_low, density_non, alternative='greater')
        else:
            p_ln = np.nan
        
        sig_lh = '***' if p_lh < 0.001 else ('**' if p_lh < 0.01 else ('*' if p_lh < 0.05 else 'ns'))
        sig_ln = '***' if p_ln < 0.001 else ('**' if p_ln < 0.01 else ('*' if p_ln < 0.05 else 'ns'))
        
        print(f"    r={radius}px: HMGB1_low={mean_low:.2f}% vs HMGB1_high={mean_high:.2f}% "
              f"vs Normal={mean_non:.2f}% | low>high: p={p_lh:.2e} {sig_lh} | low>non: p={p_ln:.2e} {sig_ln}")
        
        results_macro.append({
            'Group_Label': region,
            'Yarıçap': radius,
            'Makrofaj_HMGB1_low(%)': mean_low,
            'Makrofaj_HMGB1_high(%)': mean_high,
            'Makrofaj_Normal(%)': mean_non,
            'p_low_vs_high': p_lh,
            'p_low_vs_normal': p_ln
        })

fig = plt.figure(figsize=(22, 14))

def safe_categorical(df, col='Group_Label'):
    df[col] = pd.Categorical(df[col], categories=group_order, ordered=True)
    return df.sort_values(col)

ax1 = fig.add_subplot(2, 3, 1)
if results_hmgb1:
    res1 = safe_categorical(pd.DataFrame(results_hmgb1))
    x = np.arange(len(res1))
    w = 0.35
    ax1.bar(x - w/2, res1['HMGB1_Normal'], w, label='Normal Beta', color='#4C72B0', alpha=0.8,
            yerr=res1['HMGB1_non_std'], capsize=3)
    ax1.bar(x + w/2, res1['HMGB1_Senescence'], w, label='Senescence Beta', color='#C44E52', alpha=0.8,
            yerr=res1['HMGB1_sen_std'], capsize=3)
    for i, (_, row) in enumerate(res1.iterrows()):
        sig = '***' if row['p-value'] < 0.001 else ('**' if row['p-value'] < 0.01 else ('*' if row['p-value'] < 0.05 else 'ns'))
        max_val = max(row['HMGB1_Normal'] + row['HMGB1_non_std'], row['HMGB1_Senescence'] + row['HMGB1_sen_std'])
        ax1.text(i, max_val + 0.3, sig, ha='center', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(res1['Group_Label'], rotation=15)
    ax1.set_ylabel('HMGB1 İntensitesi')
    ax1.set_title('HMGB1 Nükleer Kaçış\n(Beta Hücreleri)', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

ax2 = fig.add_subplot(2, 3, 2)
if results_all:
    res2 = safe_categorical(pd.DataFrame(results_all))
    x = np.arange(len(res2))
    w = 0.35
    ax2.bar(x - w/2, res2['HMGB1_Normal'], w, label='Normal', color='#4C72B0', alpha=0.8)
    ax2.bar(x + w/2, res2['HMGB1_Senescence'], w, label='Senescence', color='#C44E52', alpha=0.8)
    for i, (_, row) in enumerate(res2.iterrows()):
        sig = '***' if row['p-value'] < 0.001 else ('**' if row['p-value'] < 0.01 else ('*' if row['p-value'] < 0.05 else 'ns'))
        max_val = max(row['HMGB1_Normal'], row['HMGB1_Senescence'])
        ax2.text(i, max_val + 0.1, sig, ha='center', fontsize=12, fontweight='bold', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(res2['Group_Label'], rotation=15)
    ax2.set_ylabel('HMGB1 İntensitesi')
    ax2.set_title('HMGB1 Nükleer Kaçış\n(Tüm Hücreler)', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

ax3 = fig.add_subplot(2, 3, 3)
best_region = df['Group_Label'].value_counts().idxmax()
sample_df = df[df['Group_Label'] == best_region]
sample_size = min(50000, len(sample_df))
if sample_size > 0:
    sample = sample_df.sample(n=sample_size, random_state=42)
    ax3.scatter(sample['CH_16'], sample['CH_23'], alpha=0.1, s=1, c='gray')
    sen_sample = sample_df[(sample_df['CH_16_robust'] > global_p16) & (sample_df['CH_20_robust'] < global_lamin)]
    ax3.scatter(sen_sample['CH_16'], sen_sample['CH_23'], alpha=0.5, s=5, c='red', label='Senescence')
    if len(sample_df) > 1:
        r, p = pearsonr(sample_df['CH_16'], sample_df['CH_23'])
    else:
        r, p = 0, 1
    ax3.set_xlabel('p16 (CH_16)')
    ax3.set_ylabel('HMGB1 (CH_23)')
    ax3.set_title(f'{best_region} - p16 vs HMGB1 Korelasyonu\n(r={r:.3f}, p={p:.2e})', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

if results_macro:
    res_m = safe_categorical(pd.DataFrame(results_macro))
    
    for idx, radius in enumerate(RADII):
        ax = fig.add_subplot(2, 3, 4 + idx)
        subset = res_m[res_m['Yarıçap'] == radius]
        
        if len(subset) == 0:
            continue
        
        x = np.arange(len(subset))
        w = 0.25
        
        ax.bar(x - w, subset['Makrofaj_Normal(%)'], w, label='Normal', color='#4C72B0', alpha=0.8)
        ax.bar(x, subset['Makrofaj_HMGB1_high(%)'], w, label='Sen+HMGB1↑', color='#DD8452', alpha=0.8)
        ax.bar(x + w, subset['Makrofaj_HMGB1_low(%)'], w, label='Sen+HMGB1↓', color='#C44E52', alpha=0.8)
        
        for i, (_, row) in enumerate(subset.iterrows()):
            sig = '***' if row['p_low_vs_normal'] < 0.001 else ('**' if row['p_low_vs_normal'] < 0.01 else ('*' if row['p_low_vs_normal'] < 0.05 else 'ns'))
            max_val = max(row['Makrofaj_Normal(%)'], row['Makrofaj_HMGB1_high(%)'], row['Makrofaj_HMGB1_low(%)'])
            ax.text(i, max_val + 0.2, sig, ha='center', fontsize=10, fontweight='bold', color='red')
        
        ax.set_xticks(x)
        ax.set_xticklabels(subset['Group_Label'], rotation=15)
        ax.set_ylabel('CD68+ Makrofaj (%)')
        ax.set_title(f'Makrofaj Kemotaksisi (r={radius}px)\nHMGB1↓ = DAMP salgılanmış', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle(f"{COHORT_NAME} - HMGB1 Nükleer Kaçış ve Makrofaj Kemotaksisi\n"
             "Senescence → HMGB1 sızması → Ekstraselüler DAMP → Makrofaj Birikimi",
             fontsize=15, fontweight='bold')
plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_hmgb1_analysis.png")
plt.savefig(output_path, dpi=300)
print(f"\n\nGrafik '{output_path}' olarak kaydedildi.")
