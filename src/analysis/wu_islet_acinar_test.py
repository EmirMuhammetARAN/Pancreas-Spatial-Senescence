import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, ConvexHull
from scipy.stats import mannwhitneyu
import gc

print("=" * 60)
print("Wu/Iwasaki Testi: Adacik vs Asiner & Morfoloji")
print("=" * 60)

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

df = pd.concat(df_list, ignore_index=True)

global_p16_r = df['young_p16_95th'].iloc[0]
global_p21_r = df['young_p21_95th'].iloc[0]
global_lamin_r = df['young_lamin_50th'].iloc[0]
print(f"Young Ref Esikleri: p16>{global_p16_r:.2f}, p21>{global_p21_r:.2f}, Lamin_low<{global_lamin_r:.2f}")

results_distribution = []
results_cell_ratios = []
results_islet_morph = []

for patient_id, group in df.groupby('patient_id'):
        age = int(group['age'].iloc[0])
        print(f"\n--- {age} Yas ---")
        n_total = len(group)
        
        q95 = {ch: group[ch].quantile(0.95) for ch in ['CH_2','CH_3','CH_6','CH_18','CH_26','CH_15']}
        
        is_islet = ((group['CH_3'] > q95['CH_3']) | 
                    (group['CH_6'] > q95['CH_6']) | 
                    (group['CH_18'] > q95['CH_18']) | 
                    (group['CH_26'] > q95['CH_26']))
        
        is_acinar = (group['CH_2'] > q95['CH_2']) & (~is_islet)  # PRSS2+ ama adacik degil
        
        islet_cells = group[is_islet]
        acinar_cells = group[is_acinar]
        
        n_islet = len(islet_cells)
        n_acinar = len(acinar_cells)
        
        islet_p16_rate = (islet_cells['CH_16_robust'] > global_p16_r).mean() * 100 if n_islet > 0 else 0
        acinar_p16_rate = (acinar_cells['CH_16_robust'] > global_p16_r).mean() * 100 if n_acinar > 0 else 0
        islet_p21_rate = (islet_cells['CH_9_robust'] > global_p21_r).mean() * 100 if n_islet > 0 else 0
        acinar_p21_rate = (acinar_cells['CH_9_robust'] > global_p21_r).mean() * 100 if n_acinar > 0 else 0
        
        islet_p16_mean = islet_cells['CH_16'].mean() if n_islet > 0 else 0
        acinar_p16_mean = acinar_cells['CH_16'].mean() if n_acinar > 0 else 0
        islet_p21_mean = islet_cells['CH_9'].mean() if n_islet > 0 else 0
        acinar_p21_mean = acinar_cells['CH_9'].mean() if n_acinar > 0 else 0
        
        results_distribution.append({
            'Yas': age,
            'Islet_p16%': islet_p16_rate, 'Acinar_p16%': acinar_p16_rate,
            'Islet_p21%': islet_p21_rate, 'Acinar_p21%': acinar_p21_rate,
            'Islet_p16_mean': islet_p16_mean, 'Acinar_p16_mean': acinar_p16_mean,
            'Islet_p21_mean': islet_p21_mean, 'Acinar_p21_mean': acinar_p21_mean,
            'n_islet': n_islet, 'n_acinar': n_acinar
        })
        
        print(f"  [Dagilim] Islet p16={islet_p16_rate:.2f}% p21={islet_p21_rate:.2f}% (n={n_islet})")
        print(f"  [Dagilim] Acinar p16={acinar_p16_rate:.2f}% p21={acinar_p21_rate:.2f}% (n={n_acinar})")
        
        is_beta = (group['CH_3'] > q95['CH_3'])
        is_alfa = (group['CH_6'] > q95['CH_6'])
        is_delta = (group['CH_18'] > q95['CH_18'])
        is_pp = (group['CH_26'] > q95['CH_26'])
        is_ghrelin = (group['CH_15'] > q95['CH_15'])
        
        cell_types_map = {
            'Beta': is_beta, 'Alfa': is_alfa, 'Delta': is_delta,
            'PP': is_pp, 'Ghrelin': is_ghrelin
        }
        
        ratios = {'Yas': age}
        for name, mask in cell_types_map.items():
            cells = group[mask]
            n = len(cells)
            if n > 10:
                sen_rate = ((cells['CH_16_robust'] > global_p16_r) & (cells['CH_20_robust'] < global_lamin_r)).mean() * 100
                p16_rate = (cells['CH_16_robust'] > global_p16_r).mean() * 100
                p21_rate = (cells['CH_9_robust'] > global_p21_r).mean() * 100
            else:
                sen_rate = p16_rate = p21_rate = np.nan
            ratios[f'{name}_n'] = n
            ratios[f'{name}_sen%'] = sen_rate
            ratios[f'{name}_p16%'] = p16_rate
            ratios[f'{name}_p21%'] = p21_rate
        
        results_cell_ratios.append(ratios)
        
        for name in cell_types_map:
            if not np.isnan(ratios.get(f'{name}_sen%', np.nan)):
                print(f"  [{name}] n={ratios[f'{name}_n']}, Sen={ratios[f'{name}_sen%']:.2f}%, p16={ratios[f'{name}_p16%']:.2f}%, p21={ratios[f'{name}_p21%']:.2f}%")
        
        beta_cells = group[is_beta]
        if len(beta_cells) > 20:
            beta_coords = beta_cells[['global_x', 'global_y']].values
            tree = KDTree(beta_coords)
            
            neighbor_counts = tree.query_ball_point(beta_coords, r=50)
            dense_mask = np.array([len(n) >= 3 for n in neighbor_counts])
            
            if dense_mask.sum() > 10:
                dense_coords = beta_coords[dense_mask]
                
                x_range = dense_coords[:, 0].max() - dense_coords[:, 0].min()
                y_range = dense_coords[:, 1].max() - dense_coords[:, 1].min()
                
                aspect_ratio = max(x_range, y_range) / (min(x_range, y_range) + 1)
                spread_area = x_range * y_range
                
                alfa_count = is_alfa.sum()
                beta_count = is_beta.sum()
                delta_count = is_delta.sum()
                alfa_beta_ratio = alfa_count / (beta_count + 1)
                
                results_islet_morph.append({
                    'Yas': age,
                    'Aspect_Ratio': aspect_ratio,
                    'Spread_Area': spread_area,
                    'Alfa_Beta_Ratio': alfa_beta_ratio,
                    'Beta_Count': beta_count,
                    'Alfa_Count': alfa_count,
                    'Delta_Count': delta_count,
                    'Dense_Beta': dense_mask.sum()
                })
                print(f"  [Morfoloji] AR={aspect_ratio:.2f}, Area={spread_area:.0f}, A/B={alfa_beta_ratio:.2f}")
    
fig, axs = plt.subplots(2, 4, figsize=(28, 13))

if results_distribution:
    df_d = pd.DataFrame(results_distribution).sort_values('Yas')
    ages = [f"{int(a)}y" for a in df_d['Yas']]
    x = np.arange(len(ages))
    w = 0.35
    
    axs[0,0].bar(x - w/2, df_d['Islet_p16%'], w, label='Adacik (Islet)', color='#3498DB', edgecolor='black')
    axs[0,0].bar(x + w/2, df_d['Acinar_p16%'], w, label='Asiner (PRSS2+)', color='#E67E22', edgecolor='black')
    axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(ages)
    axs[0,0].set_title('p16+ Ratioi: Adacik vs Asiner\n(Wu: p16 adacikta yogun)', fontweight='bold')
    axs[0,0].set_ylabel('p16+ Hucre Ratioi (%)')
    axs[0,0].legend(); axs[0,0].grid(axis='y', alpha=0.3)
    
    axs[0,1].bar(x - w/2, df_d['Islet_p21%'], w, label='Adacik (Islet)', color='#3498DB', edgecolor='black')
    axs[0,1].bar(x + w/2, df_d['Acinar_p21%'], w, label='Asiner (PRSS2+)', color='#E67E22', edgecolor='black')
    axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(ages)
    axs[0,1].set_title('p21+ Ratioi: Adacik vs Asiner\n(Wu: p21 asinerde yogun)', fontweight='bold')
    axs[0,1].set_ylabel('p21+ Hucre Ratioi (%)')
    axs[0,1].legend(); axs[0,1].grid(axis='y', alpha=0.3)

if results_cell_ratios:
    df_r = pd.DataFrame(results_cell_ratios).sort_values('Yas')
    ages_r = [f"{int(a)}y" for a in df_r['Yas']]
    xr = np.arange(len(ages_r))
    
    cell_names = ['Beta', 'Alfa', 'Delta', 'PP', 'Ghrelin']
    colors = ['#3498DB', '#E74C3C', '#F1C40F', '#2ECC71', '#9B59B6']
    bw = 0.15
    
    for j, (name, color) in enumerate(zip(cell_names, colors)):
        col = f'{name}_sen%'
        vals = df_r[col].fillna(0).values
        axs[0,2].bar(xr + (j-2)*bw, vals, bw, label=name, color=color, edgecolor='black', linewidth=0.5)
    
    axs[0,2].set_xticks(xr); axs[0,2].set_xticklabels(ages_r)
    axs[0,2].set_title('Senescence Ratioi: 5 Adacik Hucre Tipi\n(Iwasaki: Beta en cok yaslanir)', fontweight='bold')
    axs[0,2].set_ylabel('Senescence Ratioi (%)')
    axs[0,2].legend(fontsize=7); axs[0,2].grid(axis='y', alpha=0.3)
    
    for j, (name, color) in enumerate(zip(cell_names, colors)):
        p16_col = f'{name}_p16%'
        p21_col = f'{name}_p21%'
        p16_vals = df_r[p16_col].fillna(0).values
        p21_vals = df_r[p21_col].fillna(0).values
        avg_p16 = np.nanmean(p16_vals)
        avg_p21 = np.nanmean(p21_vals)
        axs[0,3].bar(j - 0.15, avg_p16, 0.3, color=color, edgecolor='black', alpha=0.8)
        axs[0,3].bar(j + 0.15, avg_p21, 0.3, color=color, edgecolor='black', alpha=0.4, hatch='//')
        if j == 0:
            axs[0,3].bar(j - 0.15, 0, 0.3, color='gray', edgecolor='black', alpha=0.8, label='p16+')
            axs[0,3].bar(j + 0.15, 0, 0.3, color='gray', edgecolor='black', alpha=0.4, hatch='//', label='p21+')
    
    axs[0,3].set_xticks(range(len(cell_names))); axs[0,3].set_xticklabels(cell_names)
    axs[0,3].set_title('p16 vs p21 Dominansi\n(Her Hucre Tipinde Ortalama)', fontweight='bold')
    axs[0,3].set_ylabel('Ortalama Ratio (%)')
    axs[0,3].legend(); axs[0,3].grid(axis='y', alpha=0.3)

if results_islet_morph:
    df_m = pd.DataFrame(results_islet_morph).sort_values('Yas')
    ages_m = [f"{int(a)}y" for a in df_m['Yas']]
    xm = np.arange(len(ages_m))
    
    axs[1,0].bar(xm, df_m['Alfa_Beta_Ratio'], color='#E74C3C', edgecolor='black')
    axs[1,0].set_xticks(xm); axs[1,0].set_xticklabels(ages_m)
    axs[1,0].set_title('Alfa/Beta Hucre Ratioi\n(Iwasaki: Yasla alfa artar)', fontweight='bold')
    axs[1,0].set_ylabel('Alfa / Beta Ratioi')
    axs[1,0].grid(axis='y', alpha=0.3)
    
    axs[1,1].bar(xm, df_m['Aspect_Ratio'], color='#8E44AD', edgecolor='black')
    axs[1,1].set_xticks(xm); axs[1,1].set_xticklabels(ages_m)
    axs[1,1].set_title('Adacik Aspect Ratio\n(Iwasaki: Yasla uzama)', fontweight='bold')
    axs[1,1].set_ylabel('Aspect Ratio (max/min)')
    axs[1,1].grid(axis='y', alpha=0.3)
    
    axs[1,2].bar(xm, df_m['Beta_Count'], color='#3498DB', edgecolor='black', label='Beta')
    axs[1,2].bar(xm, df_m['Alfa_Count'], bottom=df_m['Beta_Count'], color='#E74C3C', edgecolor='black', label='Alfa')
    axs[1,2].bar(xm, df_m['Delta_Count'], bottom=df_m['Beta_Count'].values + df_m['Alfa_Count'].values, color='#F1C40F', edgecolor='black', label='Delta')
    axs[1,2].set_xticks(xm); axs[1,2].set_xticklabels(ages_m)
    axs[1,2].set_title('Adacik Hucre Sayilari\n(Iwasaki: Beta azalir)', fontweight='bold')
    axs[1,2].set_ylabel('Hucre Sayisi')
    axs[1,2].legend(); axs[1,2].grid(axis='y', alpha=0.3)
    
    if results_distribution:
        axs[1,3].plot([int(a) for a in df_d['Yas']], df_d['Islet_p16_mean'], 'o-', color='#3498DB', label='Adacik p16', linewidth=2)
        axs[1,3].plot([int(a) for a in df_d['Yas']], df_d['Acinar_p16_mean'], 's-', color='#E67E22', label='Asiner p16', linewidth=2)
        axs[1,3].plot([int(a) for a in df_d['Yas']], df_d['Islet_p21_mean'], 'o--', color='#3498DB', label='Adacik p21', linewidth=2, alpha=0.5)
        axs[1,3].plot([int(a) for a in df_d['Yas']], df_d['Acinar_p21_mean'], 's--', color='#E67E22', label='Asiner p21', linewidth=2, alpha=0.5)
        axs[1,3].set_xlabel('Yas')
        axs[1,3].set_ylabel('Ortalama Intensite')
        axs[1,3].set_title('p16/p21 Intensite Trendi\nAdacik vs Asiner', fontweight='bold')
        axs[1,3].legend(fontsize=7); axs[1,3].grid(alpha=0.3)

plt.suptitle("Wu et al. (2026) & Iwasaki (2026) Dogrulamasi: Adacik vs Asiner & Morfoloji\n"
             "Ust: p16/p21 Dagilimi & Hucre Tipi Yaslanmasi | Alt: Adacik Morfoloji & Ratiolar",
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

out_dir = r'C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\saglikli grafikler 35-69'
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, 'wu_islet_acinar_test.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGrafik kaydedildi: {output_path}")

print("\n" + "=" * 80)
print("OZET")
print("=" * 80)
if results_distribution:
    print("\n--- p16/p21 Adacik vs Asiner ---")
    for r in sorted(results_distribution, key=lambda x: x['Yas']):
        print(f"  {r['Yas']}y: Islet p16={r['Islet_p16%']:.2f}% Acinar p16={r['Acinar_p16%']:.2f}% | Islet p21={r['Islet_p21%']:.2f}% Acinar p21={r['Acinar_p21%']:.2f}%")

if results_cell_ratios:
    print("\n--- Hucre Tipi Senescence Ratiolari ---")
    for r in sorted(results_cell_ratios, key=lambda x: x['Yas']):
        line = f"  {r['Yas']}y:"
        for name in ['Beta', 'Alfa', 'Delta', 'PP', 'Ghrelin']:
            s = r.get(f'{name}_sen%', np.nan)
            if not np.isnan(s):
                line += f" {name}={s:.1f}%"
        print(line)

print("\nTamamlandi!")
