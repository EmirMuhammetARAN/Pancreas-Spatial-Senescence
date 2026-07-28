import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import entropy as shannon_entropy
import gc

print("=" * 60)
print("Ma et al. (2024) B Hucre Birikimi & Entropi Testi")
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

df_all = pd.concat(df_list, ignore_index=True)

global_p16 = df_all['young_p16_95th'].iloc[0]
global_lamin = df_all['young_lamin_50th'].iloc[0]
print(f"Young Ref Esikleri: p16_robust > {global_p16:.2f}, Lamin_robust < {global_lamin:.2f}")

RADIUS = 100

results_bcell = []
results_entropy = []

for patient_id, group in df_all.groupby('patient_id'):
    age = int(group['age'].iloc[0])
    print(f"\n--- {age} Yas ---")
    
    is_sen = (group['CH_16_robust'] > global_p16) & (group['CH_20_robust'] < global_lamin)
    is_normal = ~is_sen
        
    sen_cells = group[is_sen]
    normal_cells = group[is_normal]
    n_sen = len(sen_cells)
    
    is_cd20 = (group['CH_35'] > group['CH_35'].quantile(0.95))
    cd20_coords = group[is_cd20][['global_x', 'global_y']].values
    
    if n_sen >= 5 and len(cd20_coords) >= 3:
        sen_coords = sen_cells[['global_x', 'global_y']].values
        normal_sample = normal_cells.sample(n=min(len(normal_cells), n_sen * 3), random_state=42)
        normal_coords = normal_sample[['global_x', 'global_y']].values
        
        cd20_tree = KDTree(cd20_coords)
        
        sen_cd20 = np.mean([len(c) for c in cd20_tree.query_ball_point(sen_coords, r=RADIUS)])
        normal_cd20 = np.mean([len(c) for c in cd20_tree.query_ball_point(normal_coords, r=RADIUS)])
        fc_cd20 = sen_cd20 / (normal_cd20 + 0.001)
        
        is_cd8 = (group['CH_19'] > group['CH_19'].quantile(0.95))
        is_cd68 = (group['CH_30'] > group['CH_30'].quantile(0.95))
        is_foxp3 = (group['CH_36'] > group['CH_36'].quantile(0.95))
        
        def count_neighbors(src, tgt_coords, r):
            if len(tgt_coords) < 2: return 0
            tree = KDTree(tgt_coords)
            return np.mean([len(c) for c in tree.query_ball_point(src, r=r)])
        
        cd8_coords = group[is_cd8][['global_x', 'global_y']].values
        cd68_coords = group[is_cd68][['global_x', 'global_y']].values
        foxp3_coords = group[is_foxp3][['global_x', 'global_y']].values
        
        results_bcell.append({
            'Yas': age,
            'CD20_Sen': sen_cd20, 'CD20_Normal': normal_cd20, 'CD20_FC': fc_cd20,
            'CD8_Sen': count_neighbors(sen_coords, cd8_coords, RADIUS),
            'CD8_Normal': count_neighbors(normal_coords, cd8_coords, RADIUS),
            'CD68_Sen': count_neighbors(sen_coords, cd68_coords, RADIUS),
            'CD68_Normal': count_neighbors(normal_coords, cd68_coords, RADIUS),
            'FOXP3_Sen': count_neighbors(sen_coords, foxp3_coords, RADIUS),
            'FOXP3_Normal': count_neighbors(normal_coords, foxp3_coords, RADIUS),
            'n_sen': n_sen, 'n_cd20': len(cd20_coords)
        })
        print(f"  [B Cell] CD20: Sen={sen_cd20:.2f} vs Normal={normal_cd20:.2f} FC={fc_cd20:.2f}")
    
    q95 = {ch: group[ch].quantile(0.95) for ch in ['CH_3','CH_6','CH_18','CH_26','CH_29','CH_30']}
    
    type_counts = {
        'Beta': (group['CH_3'] > q95['CH_3']).sum(),
        'Alfa': (group['CH_6'] > q95['CH_6']).sum(),
        'Delta': (group['CH_18'] > q95['CH_18']).sum(),
        'PP': (group['CH_26'] > q95['CH_26']).sum(),
        'Endotel': (group['CH_29'] > q95['CH_29']).sum(),
        'Immune': (group['CH_30'] > q95['CH_30']).sum(),
    }
    
    if n_sen >= 5:
        sen_coords = sen_cells[['global_x', 'global_y']].values
        all_coords = group[['global_x', 'global_y']].values
        all_tree = KDTree(all_coords)
        
        sen_type_counts = np.zeros(6)
        normal_type_counts = np.zeros(6)
        
        channels = ['CH_3', 'CH_6', 'CH_18', 'CH_26', 'CH_29', 'CH_30']
        
        for idx_list in all_tree.query_ball_point(sen_coords[:min(200, n_sen)], r=200):
            neighbors = group.iloc[idx_list]
            for i, ch in enumerate(channels):
                sen_type_counts[i] += (neighbors[ch] > q95[ch]).sum()
        
        normal_sample_coords = normal_cells.sample(n=min(200, len(normal_cells)), random_state=42)[['global_x', 'global_y']].values
        for idx_list in all_tree.query_ball_point(normal_sample_coords, r=200):
            neighbors = group.iloc[idx_list]
            for i, ch in enumerate(channels):
                normal_type_counts[i] += (neighbors[ch] > q95[ch]).sum()
        
        sen_probs = sen_type_counts / (sen_type_counts.sum() + 1e-10)
        normal_probs = normal_type_counts / (normal_type_counts.sum() + 1e-10)
        
        sen_entropy = shannon_entropy(sen_probs + 1e-10)
        normal_entropy = shannon_entropy(normal_probs + 1e-10)
        
        results_entropy.append({
            'Yas': age,
            'Sen_Entropy': sen_entropy,
            'Normal_Entropy': normal_entropy,
            'Global_Type_Counts': type_counts
        })
        print(f"  [Entropy] Senescence bolge={sen_entropy:.3f} vs Normal bolge={normal_entropy:.3f}")
    
    pass
    gc.collect()

fig, axs = plt.subplots(2, 3, figsize=(22, 13))

c_sen = '#E74C3C'
c_normal = '#3498DB'

if results_bcell:
    df_b = pd.DataFrame(results_bcell).sort_values('Yas')
    ages = [f"{int(a)}y" for a in df_b['Yas']]
    x = np.arange(len(ages))
    w = 0.35
    
    axs[0,0].bar(x - w/2, df_b['CD20_Sen'], w, label='Senescence Etrafinda', color=c_sen, edgecolor='black')
    axs[0,0].bar(x + w/2, df_b['CD20_Normal'], w, label='Normal Etrafinda', color=c_normal, edgecolor='black')
    for i, fc in enumerate(df_b['CD20_FC']):
        axs[0,0].text(i, max(df_b['CD20_Sen'].iloc[i], df_b['CD20_Normal'].iloc[i]) * 1.05,
                      f'FC={fc:.2f}', ha='center', fontsize=8, fontweight='bold')
    axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(ages)
    axs[0,0].set_title('CD20+ B Hucre Birikimi\n(Ma et al. IgG Hipotezi)', fontweight='bold')
    axs[0,0].set_ylabel('Ort. Komsu Sayisi (R=100px)')
    axs[0,0].legend(); axs[0,0].grid(axis='y', alpha=0.3)
    
    cd8_fc = df_b['CD8_Sen'] / (df_b['CD8_Normal'] + 0.001)
    cd68_fc = df_b['CD68_Sen'] / (df_b['CD68_Normal'] + 0.001)
    foxp3_fc = df_b['FOXP3_Sen'] / (df_b['FOXP3_Normal'] + 0.001)
    
    axs[0,1].bar(x - 0.2, cd8_fc, 0.2, label='CD8 FC', color='#E74C3C', edgecolor='black')
    axs[0,1].bar(x, cd68_fc, 0.2, label='CD68 FC', color='#F39C12', edgecolor='black')
    axs[0,1].bar(x + 0.2, foxp3_fc, 0.2, label='FOXP3 FC', color='#27AE60', edgecolor='black')
    axs[0,1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(ages)
    axs[0,1].set_title('Immun Hucre FC (Senescence/Normal)\nKontrol Karsilastirmasi', fontweight='bold')
    axs[0,1].set_ylabel('Fold Change')
    axs[0,1].legend(); axs[0,1].grid(axis='y', alpha=0.3)
    
    cd20_fc_vals = df_b['CD20_FC'].values
    axs[0,2].bar(x - 0.3, cd20_fc_vals, 0.15, label='CD20 (B)', color='#9B59B6', edgecolor='black')
    axs[0,2].bar(x - 0.15, cd8_fc, 0.15, label='CD8 (KillerT)', color='#E74C3C', edgecolor='black')
    axs[0,2].bar(x, cd68_fc, 0.15, label='CD68 (Makrofaj)', color='#F39C12', edgecolor='black')
    axs[0,2].bar(x + 0.15, foxp3_fc, 0.15, label='FOXP3 (Treg)', color='#27AE60', edgecolor='black')
    axs[0,2].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='FC=1 (fark yok)')
    axs[0,2].set_xticks(x); axs[0,2].set_xticklabels(ages)
    axs[0,2].set_title('Tum Immun Hucre FC Karsilastirmasi\nCD20 vs CD8 vs CD68 vs FOXP3', fontweight='bold')
    axs[0,2].set_ylabel('Fold Change (Senescence/Normal)')
    axs[0,2].legend(fontsize=7); axs[0,2].grid(axis='y', alpha=0.3)

if results_entropy:
    df_e = pd.DataFrame(results_entropy).sort_values('Yas')
    ages_e = [f"{int(a)}y" for a in df_e['Yas']]
    xe = np.arange(len(ages_e))
    
    axs[1,0].bar(xe - w/2, df_e['Sen_Entropy'], w, label='Senescence Bolgesi', color=c_sen, edgecolor='black')
    axs[1,0].bar(xe + w/2, df_e['Normal_Entropy'], w, label='Normal Bolge', color=c_normal, edgecolor='black')
    axs[1,0].set_xticks(xe); axs[1,0].set_xticklabels(ages_e)
    axs[1,0].set_title('Shannon Entropisi\n(Hucre Tipi Karismasi)', fontweight='bold')
    axs[1,0].set_ylabel('Shannon Entropy (H)')
    axs[1,0].legend(); axs[1,0].grid(axis='y', alpha=0.3)
    
    axs[1,1].plot([int(a) for a in df_e['Yas']], df_e['Sen_Entropy'], 'o-', color=c_sen, label='Senescence Bolgesi', linewidth=2, markersize=8)
    axs[1,1].plot([int(a) for a in df_e['Yas']], df_e['Normal_Entropy'], 's-', color=c_normal, label='Normal Bolge', linewidth=2, markersize=8)
    axs[1,1].set_xlabel('Yas')
    axs[1,1].set_ylabel('Shannon Entropy (H)')
    axs[1,1].set_title('Entropi Trendi (Yasa Gore)\nMa et al.: Organizasyonel Karmasiklik Artar', fontweight='bold')
    axs[1,1].legend(); axs[1,1].grid(alpha=0.3)
    
    cell_types = ['Beta', 'Alfa', 'Delta', 'PP', 'Endotel', 'Immune']
    colors_ct = ['#3498DB', '#E74C3C', '#F1C40F', '#2ECC71', '#9B59B6', '#E67E22']
    
    for i, row in enumerate(df_e.itertuples()):
        counts = [row.Global_Type_Counts[ct] for ct in cell_types]
        total = sum(counts)
        fracs = [c/total for c in counts]
        bottom = 0
        for j, (ct, frac) in enumerate(zip(cell_types, fracs)):
            axs[1,2].bar(i, frac, bottom=bottom, color=colors_ct[j], label=ct if i==0 else '', edgecolor='black', linewidth=0.3)
            bottom += frac
    
    axs[1,2].set_xticks(range(len(ages_e))); axs[1,2].set_xticklabels(ages_e)
    axs[1,2].set_title('Hucre Tipi Ratiolari (Yasa Gore)\nYasla Degisim Var mi?', fontweight='bold')
    axs[1,2].set_ylabel('Ratio')
    axs[1,2].legend(loc='upper right', fontsize=7); axs[1,2].grid(axis='y', alpha=0.3)

plt.suptitle("Ma et al. (2024) Dogrulamasi: CD20+ B Hucre Birikimi & Organizasyonel Entropi\n"
             "Ust: B Hucre/Immun Hucre Birikimi | Alt: Shannon Entropi & Hucre Tipi Ratiolari",
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

import os
out_dir = r'C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\saglikli grafikler 35-69'
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, 'ma_bcell_entropy_test.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGrafik kaydedildi: {output_path}")

print("\n" + "=" * 80)
print("OZET: CD20+ B HUCRE BIRIKIMI")
print("=" * 80)
if results_bcell:
    for r in sorted(results_bcell, key=lambda x: x['Yas']):
        print(f"  {r['Yas']}y: CD20 FC={r['CD20_FC']:.2f} (n_sen={r['n_sen']}, n_cd20={r['n_cd20']})")

print("\n" + "=" * 80)
print("OZET: ORGANIZASYONEL ENTROPI")
print("=" * 80)
if results_entropy:
    for r in sorted(results_entropy, key=lambda x: x['Yas']):
        print(f"  {r['Yas']}y: Sen Entropy={r['Sen_Entropy']:.4f}  Normal Entropy={r['Normal_Entropy']:.4f}")

print("\nTamamlandi!")
