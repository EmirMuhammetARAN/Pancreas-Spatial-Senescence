import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print(f"{COHORT_NAME} - Veriler yükleniyor ve DBSCAN filtresi uygulaniyor...")
columns_to_load = ['patient_id', 'age', 'CH_0', 'CH_3', 'CH_16', 'CH_19', 'CH_20', 'CH_27', 'CH_30', 'CH_34', 'CH_36', 'global_x', 'global_y']

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

RADII = [25, 50, 100]
CHUNK_SIZE = 50000

IMMUNE_CHANNELS = {
    'CD8 (Killer T)': 'CH_19',
    'CD4 (Helper T)': 'CH_27',
    'CD68 (Makrofaj)': 'CH_30',
    'FOXP3 (Treg)': 'CH_36',
    'CD56 (NK)': 'CH_34',
}

global_p16 = df['young_p16_95th'].iloc[0]
global_lamin = df['CH_20_robust'].quantile(0.50)

print(f"Global eşikler: p16 > {global_p16:.2f}, Lamin < {global_lamin:.2f}")
print(f"Yarıçaplar: {RADII} px\n")

all_results = []

group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df['Group_Label'] = pd.Categorical(df['Group_Label'], categories=group_order, ordered=True)

for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    print(f"\n{'='*60}")
    print(f"  {region} - {len(group):,} hücre işleniyor...")
    print(f"{'='*60}")
    
    is_sen = ((group['CH_16_robust'] > global_p16) & (group['CH_20_robust'] < global_lamin)).values.astype(bool)
    
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95)).values.astype(bool)
    
    immune_masks = {}
    for name, ch in IMMUNE_CHANNELS.items():
        immune_masks[name] = (group[ch] > group[ch].quantile(0.95)).values.astype(bool)
        print(f"  {name}: {immune_masks[name].sum():,} hücre")
    
    sen_beta = is_sen & is_beta
    non_sen_beta = (~is_sen) & is_beta
    
    n_sen_beta = sen_beta.sum()
    n_non_sen_beta = non_sen_beta.sum()
    print(f"\n  Senescence Beta: {n_sen_beta:,} | Normal Beta: {n_non_sen_beta:,}")
    
    if n_sen_beta < 3:
        print(f"  UYARI: Yeterli senesans beta hücresi yok, atlanıyor...")
        continue
    
    coords = group[['global_x', 'global_y']].values
    tree = cKDTree(coords)
    
    for radius in RADII:
        print(f"\n  --- Yarıçap: {radius} px ---")
        
        immune_counts = {name: np.zeros(len(coords)) for name in IMMUNE_CHANNELS}
        total_neighbors = np.zeros(len(coords))
        
        for start_idx in range(0, len(coords), CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, len(coords))
            chunk_coords = coords[start_idx:end_idx]
            neighbors_list = tree.query_ball_point(chunk_coords, r=radius)
            
            for i_chunk, neighbors in enumerate(neighbors_list):
                i = start_idx + i_chunk
                n_list = [n for n in neighbors if n != i]
                if not n_list:
                    continue
                total_neighbors[i] = len(n_list)
                
                for name, mask in immune_masks.items():
                    immune_counts[name][i] = sum(1 for n in n_list if mask[n])
        
        for name in IMMUNE_CHANNELS:
            sen_densities = immune_counts[name][sen_beta] / np.maximum(total_neighbors[sen_beta], 1)
            non_densities = immune_counts[name][non_sen_beta] / np.maximum(total_neighbors[non_sen_beta], 1)
            
            mean_sen = sen_densities.mean() * 100
            mean_non = non_densities.mean() * 100
            
            if len(sen_densities) > 1 and len(non_densities) > 1:
                stat, p_val = mannwhitneyu(sen_densities, non_densities, alternative='greater')
            else:
                p_val = np.nan
            
            fold_change = mean_sen / mean_non if mean_non > 0 else np.nan
            
            sig = ''
            if not np.isnan(p_val):
                if p_val < 0.001: sig = '***'
                elif p_val < 0.01: sig = '**'
                elif p_val < 0.05: sig = '*'
            
            print(f"    {name}: Sen={mean_sen:.2f}% vs Non={mean_non:.2f}% "
                  f"(FC={fold_change:.2f}x, p={p_val:.2e}) {sig}")
            
            all_results.append({
                'Group_Label': region,
                'Yarıçap': radius,
                'Bağışıklık Hücresi': name,
                'Senescence Beta (%)': mean_sen,
                'Normal Beta (%)': mean_non,
                'Fold Change': fold_change,
                'p-value': p_val
            })

res_df = pd.DataFrame(all_results)

if len(res_df) == 0:
    print("\nYeterli veri yok, grafik oluşturulamadı.")
    exit()

def safe_categorical(df, col='Group_Label'):
    df[col] = pd.Categorical(df[col], categories=group_order, ordered=True)
    return df.sort_values(col)

res_df = safe_categorical(res_df)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

r100 = res_df[res_df['Yarıçap'] == 100]

for idx, immune_name in enumerate(IMMUNE_CHANNELS.keys()):
    if idx >= len(axes):
        break
    ax = axes[idx]
    subset = r100[r100['Bağışıklık Hücresi'] == immune_name]
    
    x = np.arange(len(subset))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, subset['Normal Beta (%)'], w, label='Normal Beta', color='#4C72B0', alpha=0.8)
    bars2 = ax.bar(x + w/2, subset['Senescence Beta (%)'], w, label='Senescence Beta', color='#C44E52', alpha=0.8)
    
    for i, (_, row) in enumerate(subset.iterrows()):
        if row['p-value'] < 0.001:
            sig_text = '***'
        elif row['p-value'] < 0.01:
            sig_text = '**'
        elif row['p-value'] < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        max_val = max(row['Normal Beta (%)'], row['Senescence Beta (%)'])
        ax.text(i, max_val + 0.2, sig_text, ha='center', fontsize=10, fontweight='bold', color='red')
    
    ax.set_title(immune_name, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subset['Group_Label'], fontsize=10, rotation=15)
    ax.set_ylabel('Komşu Yoğunluğu (%)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

ax_heat = axes[5]
pivot = r100.pivot_table(values='Fold Change', index='Bağışıklık Hücresi', columns='Group_Label')
pivot = pivot.reindex(columns=[c for c in group_order if c in pivot.columns])

im = ax_heat.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
ax_heat.set_xticks(range(len(pivot.columns)))
ax_heat.set_xticklabels(pivot.columns, fontsize=10, rotation=15)
ax_heat.set_yticks(range(len(pivot.index)))
ax_heat.set_yticklabels(pivot.index, fontsize=9)
ax_heat.set_title("Fold Change Heatmap\n(Senescence / Normal)", fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax_heat, shrink=0.8)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax_heat.text(j, i, f'{val:.2f}x', ha='center', va='center', fontsize=9, fontweight='bold',
                        color='white' if val > 1.5 or val < 0.7 else 'black')

plt.suptitle(f"{COHORT_NAME} - Senescence Beta Hücreleri Etrafında Bağışıklık İnfiltrasyonu (r=100px)\n"
             "Makale 1 Doğrulaması: HLA-I artışı → CD8+ birikimi beklenir",
             fontsize=15, fontweight='bold')
plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path1 = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_immune_infiltration_analysis.png")
plt.savefig(output_path1, dpi=300)
print(f"\n\nGrafik '{output_path1}' olarak kaydedildi.")

fig2, axes2 = plt.subplots(1, len(IMMUNE_CHANNELS), figsize=(25, 5))

for idx, immune_name in enumerate(IMMUNE_CHANNELS.keys()):
    ax = axes2[idx]
    subset = res_df[res_df['Bağışıklık Hücresi'] == immune_name]
    
    for region in subset['Group_Label'].unique():
        region_data = subset[subset['Group_Label'] == region].sort_values('Yarıçap')
        if len(region_data) > 0:
            fc_values = region_data['Fold Change'].values
            ax.plot(RADII[:len(fc_values)], fc_values, marker='o', linewidth=2, markersize=8, label=region)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_title(immune_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Yarıçap (px)', fontsize=10)
    ax.set_ylabel('Fold Change', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle(f"{COHORT_NAME} - Mesafeye Bağlı Bağışıklık İnfiltrasyonu (Distance Decay)\n"
             "FC > 1 = Senescence Beta etrafında artış", fontsize=14, fontweight='bold')
plt.tight_layout()
output_path2 = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_immune_distance_decay.png")
plt.savefig(output_path2, dpi=300)
print(f"Grafik '{output_path2}' olarak kaydedildi.")
