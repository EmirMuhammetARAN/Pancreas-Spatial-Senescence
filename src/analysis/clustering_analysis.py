import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
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

p16_sinir = df['young_p16_95th'].iloc[0]
lamin_sinir = df['young_lamin_50th'].iloc[0]
df['is_senescent'] = ((df['CH_16_robust'] > p16_sinir) & (df['CH_20_robust'] < lamin_sinir)).astype(int)

results = []

print("\n--- Kumelenme (Clustering) Analizi Basliyor ---")
group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df['Group_Label'] = pd.Categorical(df['Group_Label'], categories=group_order, ordered=True)

for region, df_region in df.groupby('Group_Label'):
    if len(df_region) == 0: continue
    
    sen_cells = df_region[df_region['is_senescent'] == 1].copy()
    
    n_total = len(df_region)
    n_sen = len(sen_cells)
    
    if n_sen < 10:
        print(f"{region}: Yeterli senesans hucresi yok ({n_sen}). Atlaniyor.")
        continue
        
    print(f"Isleniyor: {region} - Toplam: {n_total:,}, Senescence: {n_sen:,}")
    
    sen_coords = sen_cells[['global_x','global_y']].values
    tree = cKDTree(sen_coords)
    distances, _ = tree.query(sen_coords, k=6)
    sen_mean_dist = distances[:, 1:].mean()
    
    rand_dists = []
    for seed in [42, 43, 44]:
        np.random.seed(seed)
        random_idx = np.random.choice(n_total, size=n_sen, replace=False)
        random_coords = df_region.iloc[random_idx][['global_x','global_y']].values
        tree_r = cKDTree(random_coords)
        d_r, _ = tree_r.query(random_coords, k=6)
        rand_dists.append(d_r[:, 1:].mean())
        
    rand_mean_dist = np.mean(rand_dists)
    clustering_factor = rand_mean_dist / sen_mean_dist
    
    results.append({
        'Group_Label': region,
        'senescent_distance': sen_mean_dist,
        'random_distance': rand_mean_dist,
        'clustering_factor': clustering_factor
    })

results_df = pd.DataFrame(results)
if results_df.empty:
    print("Yeterli senesans hucresi bulunmadigi icin grafikler cizilemiyor.")
    sys.exit(0)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df))
width = 0.35

rects1 = ax1.bar(x - width/2, results_df['random_distance'], width, label='Beklenen Rastgele Mesafe', color='#A0AEC0', alpha=0.8, edgecolor='black')
rects2 = ax1.bar(x + width/2, results_df['senescent_distance'], width, label='Gercek Senescence Mesafesi', color='#E53E3E', alpha=0.9, edgecolor='black')

ax1.set_ylabel('En Yakin 5 Komsuya Ortalama Mesafe (Piksel)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Group_Label'], fontweight='bold', rotation=15)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(x, results_df['clustering_factor'], color='#2B6CB0', marker='o', markersize=10, linewidth=3, label='Kumelenme Faktoru (K-Kat)')
ax2.set_ylabel('Kumelenme Faktoru (Rastgele / Gercek)', fontweight='bold', color='#2B6CB0')
ax2.tick_params(axis='y', labelcolor='#2B6CB0')

for i, val in enumerate(results_df['clustering_factor']):
    ax2.annotate(f"{val:.1f}x", (x[i], val), xytext=(0, 10), textcoords="offset points", ha='center', color='#2B6CB0', fontweight='bold')

plt.title(f"{COHORT_NAME} - Senescence Hucrelerinin Mekansal Kumelenmesi (Spatial Clustering)", fontweight='bold', pad=20)
fig.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_spatial_clustering_analysis.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nGrafik kaydedildi: {save_path}")
print("\nVeri Ozeti:")
print(results_df.to_string(index=False))
