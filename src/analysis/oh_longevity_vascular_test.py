import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import mannwhitneyu, pearsonr
import gc

print("=" * 60)
print("Oh et al. (2025) Pankreas Longevity & Vascular Aging Testi")
print("=" * 60)

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

global_p16_r = df['young_p16_95th'].iloc[0]
global_p21_r = df['young_p21_95th'].iloc[0]
global_lamin_low_r = df['young_lamin_50th'].iloc[0]
print(f"Young Ref Esikleri: p16>{global_p16_r:.2f}, p21>{global_p21_r:.2f}, lamin_low<{global_lamin_low_r:.2f}")

RADIUS = 100

results_longevity = []
results_vascular = []

for patient_id, group in df.groupby('patient_id'):
        age = int(group['age'].iloc[0])
        print(f"\n--- {age} Yas Isleniyor ---")
        
        is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95))
        
        is_sen_beta = is_beta & (group['CH_16_robust'] > global_p16_r) & (group['CH_20_robust'] < global_lamin_low_r)
        
        is_young_beta = is_beta & (group['CH_16_robust'] <= global_p16_r) & (group['CH_9_robust'] <= global_p21_r) & (group['CH_20_robust'] > global_lamin_high_r)
        
        sen_beta = group[is_sen_beta]
        young_beta = group[is_young_beta]
        
        n_sen = len(sen_beta)
        n_young = len(young_beta)
        
        print(f"  Senescence Beta: {n_sen}")
        print(f"  Genc Beta (p16-/p21-/LaminB1 high): {n_young}")
        
        if n_sen >= 5 and n_young >= 5:
            is_cd8 = (group['CH_19'] > group['CH_19'].quantile(0.95))
            is_cd68 = (group['CH_30'] > group['CH_30'].quantile(0.95))
            is_foxp3 = (group['CH_36'] > group['CH_36'].quantile(0.95))
            
            cd8_coords = group[is_cd8][['global_x', 'global_y']].values
            cd68_coords = group[is_cd68][['global_x', 'global_y']].values
            foxp3_coords = group[is_foxp3][['global_x', 'global_y']].values
            
            def count_neighbors(source_coords, target_coords, radius):
                if len(source_coords) == 0 or len(target_coords) == 0: return 0
                tree = KDTree(target_coords)
                counts = tree.query_ball_point(source_coords, r=radius)
                return np.mean([len(c) for c in counts])
            
            coords_sen = sen_beta[['global_x', 'global_y']].values
            coords_young = young_beta[['global_x', 'global_y']].values
            
            cd8_sen = count_neighbors(coords_sen, cd8_coords, RADIUS)
            cd8_young = count_neighbors(coords_young, cd8_coords, RADIUS)
            
            cd68_sen = count_neighbors(coords_sen, cd68_coords, RADIUS)
            cd68_young = count_neighbors(coords_young, cd68_coords, RADIUS)
            
            foxp3_sen = count_neighbors(coords_sen, foxp3_coords, RADIUS)
            foxp3_young = count_neighbors(coords_young, foxp3_coords, RADIUS)
            
            results_longevity.append({
                'Yas': age,
                'CD8_Sen': cd8_sen, 'CD8_Young': cd8_young,
                'CD68_Sen': cd68_sen, 'CD68_Young': cd68_young,
                'FOXP3_Sen': foxp3_sen, 'FOXP3_Young': foxp3_young,
                'n_sen': n_sen, 'n_young': n_young
            })
            
            print(f"  [Longevity] CD8:  Sen={cd8_sen:.2f} vs Young={cd8_young:.2f}")
            print(f"  [Longevity] CD68: Sen={cd68_sen:.2f} vs Young={cd68_young:.2f}")
            print(f"  [Longevity] FOXP3: Sen={foxp3_sen:.2f} vs Young={foxp3_young:.2f}")
        else:
            print("  Longevity testi icin yeterli hucre yok.")
        
        is_vessel = (group['CH_29'] > group['CH_29'].quantile(0.95))
        vessel_coords = group[is_vessel][['global_x', 'global_y']].values
        
        if len(vessel_coords) < 10:
            print("  Yeterli damar hucresi yok, atlaniyoor...")
            continue
        
        beta_cells = group[is_beta].copy()
        beta_coords = beta_cells[['global_x', 'global_y']].values
        
        vessel_tree = KDTree(vessel_coords)
        distances, _ = vessel_tree.query(beta_coords, k=1)
        beta_cells = beta_cells.assign(dist_to_vessel=distances)
        
        close_beta = beta_cells[beta_cells['dist_to_vessel'] < 50]
        far_beta = beta_cells[beta_cells['dist_to_vessel'] > 150]
        
        n_close = len(close_beta)
        n_far = len(far_beta)
        
        if n_close < 10 or n_far < 10:
            print(f"  Damar testi icin yeterli hucre yok (close={n_close}, far={n_far})")
            continue
        
        close_sen_rate = ((close_beta['CH_16_robust'] > global_p16_r) & (close_beta['CH_20_robust'] < global_lamin_low_r)).mean() * 100
        far_sen_rate = ((far_beta['CH_16_robust'] > global_p16_r) & (far_beta['CH_20_robust'] < global_lamin_low_r)).mean() * 100
        
        close_p16 = close_beta['CH_16'].mean()
        far_p16 = far_beta['CH_16'].mean()
        
        close_ca9 = close_beta['CH_4'].mean()
        far_ca9 = far_beta['CH_4'].mean()
        
        p16_dist_corr, p16_dist_p = pearsonr(beta_cells['dist_to_vessel'], beta_cells['CH_16'])
        
        ca9_dist_corr, ca9_dist_p = pearsonr(beta_cells['dist_to_vessel'], beta_cells['CH_4'])
        
        results_vascular.append({
            'Yas': age,
            'Close_Sen%': close_sen_rate,
            'Far_Sen%': far_sen_rate,
            'Close_p16': close_p16,
            'Far_p16': far_p16,
            'Close_CA9': close_ca9,
            'Far_CA9': far_ca9,
            'p16_dist_corr': p16_dist_corr,
            'p16_dist_p': p16_dist_p,
            'ca9_dist_corr': ca9_dist_corr,
            'ca9_dist_p': ca9_dist_p,
            'n_close': n_close,
            'n_far': n_far
        })
        
        print(f"  [Vascular] Yakin(<50px): Sen%={close_sen_rate:.2f}, p16={close_p16:.2f}, CA9={close_ca9:.2f}")
        print(f"  [Vascular] Uzak(>150px): Sen%={far_sen_rate:.2f}, p16={far_p16:.2f}, CA9={far_ca9:.2f}")
        print(f"  [Vascular] p16~Distance r={p16_dist_corr:.3f}, CA9~Distance r={ca9_dist_corr:.3f}")
    
if not results_longevity and not results_vascular:
    print("Yeterli veri bulunamadi.")
    exit()

fig, axs = plt.subplots(2, 3, figsize=(22, 13))

c_sen = '#E74C3C'   # Kirmizi (senesans/yasli)
c_young = '#27AE60'  # Yesil (genc)
c_close = '#E67E22'  # Turuncu (damara yakin)
c_far = '#8E44AD'    # Mor (damara uzak)

if results_longevity:
    df_long = pd.DataFrame(results_longevity).sort_values('Yas')
    ages = [f"{int(a)} yas" for a in df_long['Yas']]
    x = np.arange(len(ages))
    w = 0.35
    
    axs[0,0].bar(x - w/2, df_long['CD8_Sen'], w, label='Senescence Beta', color=c_sen, edgecolor='black')
    axs[0,0].bar(x + w/2, df_long['CD8_Young'], w, label='Genc Beta', color=c_young, edgecolor='black')
    axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(ages)
    axs[0,0].set_title('CD8 (Killer T) Komsulugu\nYasli Beta Etrafinda Artmali', fontweight='bold')
    axs[0,0].set_ylabel('Ort. Komsu Sayisi (R=100px)')
    axs[0,0].legend(); axs[0,0].grid(axis='y', alpha=0.3)
    
    axs[0,1].bar(x - w/2, df_long['CD68_Sen'], w, label='Senescence Beta', color=c_sen, edgecolor='black')
    axs[0,1].bar(x + w/2, df_long['CD68_Young'], w, label='Genc Beta', color=c_young, edgecolor='black')
    axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(ages)
    axs[0,1].set_title('CD68 (Makrofaj) Komsulugu\nYasli Beta Etrafinda Artmali', fontweight='bold')
    axs[0,1].set_ylabel('Ort. Komsu Sayisi (R=100px)')
    axs[0,1].legend(); axs[0,1].grid(axis='y', alpha=0.3)
    
    axs[0,2].bar(x - w/2, df_long['FOXP3_Sen'], w, label='Senescence Beta', color=c_sen, edgecolor='black')
    axs[0,2].bar(x + w/2, df_long['FOXP3_Young'], w, label='Genc Beta', color=c_young, edgecolor='black')
    axs[0,2].set_xticks(x); axs[0,2].set_xticklabels(ages)
    axs[0,2].set_title('FOXP3 (Treg) Komsulugu\nGenc Beta Etrafinda Artmali (Koruma)', fontweight='bold')
    axs[0,2].set_ylabel('Ort. Komsu Sayisi (R=100px)')
    axs[0,2].legend(); axs[0,2].grid(axis='y', alpha=0.3)

if results_vascular:
    df_vasc = pd.DataFrame(results_vascular).sort_values('Yas')
    ages_v = [f"{int(a)} yas" for a in df_vasc['Yas']]
    xv = np.arange(len(ages_v))
    
    axs[1,0].bar(xv - w/2, df_vasc['Close_Sen%'], w, label='Damara Yakin (<50px)', color=c_close, edgecolor='black')
    axs[1,0].bar(xv + w/2, df_vasc['Far_Sen%'], w, label='Damara Uzak (>150px)', color=c_far, edgecolor='black')
    axs[1,0].set_xticks(xv); axs[1,0].set_xticklabels(ages_v)
    axs[1,0].set_title('Damar Yakinligi vs Senescence Ratioi\nYakin = Daha Cok Yaslanma?', fontweight='bold')
    axs[1,0].set_ylabel('Senescence Ratioi (%)')
    axs[1,0].legend(); axs[1,0].grid(axis='y', alpha=0.3)
    
    axs[1,1].bar(xv - w/2, df_vasc['Close_p16'], w, label='Damara Yakin (<50px)', color=c_close, edgecolor='black')
    axs[1,1].bar(xv + w/2, df_vasc['Far_p16'], w, label='Damara Uzak (>150px)', color=c_far, edgecolor='black')
    axs[1,1].set_xticks(xv); axs[1,1].set_xticklabels(ages_v)
    axs[1,1].set_title('Damar Yakinligi vs p16 Seviyesi\nYakin = Daha Yuksek p16?', fontweight='bold')
    axs[1,1].set_ylabel('Ort. p16 Intensitesi')
    axs[1,1].legend(); axs[1,1].grid(axis='y', alpha=0.3)
    
    axs[1,2].bar(xv - w/2, df_vasc['Close_CA9'], w, label='Damara Yakin (<50px)', color=c_close, edgecolor='black')
    axs[1,2].bar(xv + w/2, df_vasc['Far_CA9'], w, label='Damara Uzak (>150px)', color=c_far, edgecolor='black')
    for k, row in enumerate(df_vasc.itertuples()):
        axs[1,2].text(k, max(row.Close_CA9, row.Far_CA9) * 1.05, 
                      f"p16~dist r={row.p16_dist_corr:.3f}\nCA9~dist r={row.ca9_dist_corr:.3f}",
                      ha='center', fontsize=7, color='navy')
    axs[1,2].set_xticks(xv); axs[1,2].set_xticklabels(ages_v)
    axs[1,2].set_title('Damar Yakinligi vs CA9 (Hipoksi)\nUzak = Daha Fazla Hipoksi?', fontweight='bold')
    axs[1,2].set_ylabel('Ort. CA9 Intensitesi')
    axs[1,2].legend(); axs[1,2].grid(axis='y', alpha=0.3)

plt.suptitle("Oh et al. (Wyss-Coray, 2025) Dogrulamasi: Pankreas Longevity & Vascular Aging\n"
             "Ust: Genc vs Yasli Beta Hucrelerin Immun Cevresi | Alt: Damar Yakinligi vs Yaslanma",
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

out_dir = r'C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\saglikli grafikler 35-69'
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, 'oh_longevity_vascular_test.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGrafik kaydedildi: {output_path}")

print("\n" + "=" * 80)
print("OZET: LONGEVITY IMZASI (Genc vs Yasli Beta Hucre Immun Cevresi)")
print("=" * 80)
if results_longevity:
    for r in sorted(results_longevity, key=lambda x: x['Yas']):
        age = r['Yas']
        print(f"  {age} yas (n_sen={r['n_sen']}, n_young={r['n_young']}):")
        print(f"    CD8:   Sen={r['CD8_Sen']:.2f}  Young={r['CD8_Young']:.2f}  Ratio={r['CD8_Sen']/(r['CD8_Young']+0.001):.2f}")
        print(f"    CD68:  Sen={r['CD68_Sen']:.2f}  Young={r['CD68_Young']:.2f}  Ratio={r['CD68_Sen']/(r['CD68_Young']+0.001):.2f}")
        print(f"    FOXP3: Sen={r['FOXP3_Sen']:.2f}  Young={r['FOXP3_Young']:.2f}  Ratio={r['FOXP3_Sen']/(r['FOXP3_Young']+0.001):.2f}")

print("\n" + "=" * 80)
print("OZET: VASCULAR AGING (Damara Yakin vs Uzak Beta Hucreleri)")
print("=" * 80)
if results_vascular:
    for r in sorted(results_vascular, key=lambda x: x['Yas']):
        age = r['Yas']
        print(f"  {age} yas (close={r['n_close']}, far={r['n_far']}):")
        print(f"    Sen%:  Close={r['Close_Sen%']:.3f}%  Far={r['Far_Sen%']:.3f}%")
        print(f"    p16:   Close={r['Close_p16']:.2f}  Far={r['Far_p16']:.2f}")
        print(f"    CA9:   Close={r['Close_CA9']:.2f}  Far={r['Far_CA9']:.2f}")
        print(f"    Corr:  p16~dist r={r['p16_dist_corr']:.3f} (p={r['p16_dist_p']:.4f})")
        print(f"           CA9~dist r={r['ca9_dist_corr']:.3f} (p={r['ca9_dist_p']:.4f})")

print("\nTamamlandi!")
