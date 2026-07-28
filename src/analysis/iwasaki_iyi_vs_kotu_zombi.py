import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print("Veriler yukleniyor...")
df_list = []
for snt, age, region, label in ACTIVE_COHORT:
    d_clean = load_and_filter_tissue(snt, age)
    if d_clean is not None:
        d_clean['Group_Label'] = label
        d_clean['patient_id'] = label  # Geriye donuk uyumluluk
        df_list.append(d_clean)

df = pd.concat(df_list, ignore_index=True)

global_p16 = df['young_p16_95th'].iloc[0]
global_p21 = df['young_p21_95th'].iloc[0]
print(f"Young Ref Esikleri: p16_robust > {global_p16:.2f}, p21_robust > {global_p21:.2f}")

results_func = []
results_immune = []

RADIUS = 50

for patient_id, group in df.groupby('patient_id'):
    age = group['age'].iloc[0]
    print(f"\n--- {age} Yas Isleniyor ---")

    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95))

    camp_a_mask = is_beta & (group['CH_16_robust'] > global_p16) & (group['CH_9_robust'] <= global_p21)
    camp_b_mask = is_beta & (group['CH_9_robust'] > global_p21) & (group['CH_16_robust'] <= global_p16)

    camp_a_cells = group[camp_a_mask]
    camp_b_cells = group[camp_b_mask]

    n_a = len(camp_a_cells)
    n_b = len(camp_b_cells)

    print(f"  Kamp A (p16+ / p21-): {n_a} hucre")
    print(f"  Kamp B (p21+ / p16-): {n_b} hucre")

    if n_a < 3 or n_b < 3:
        print("  Yeterli hucre yok, atlaniyor...")
        continue

    cpep_a = camp_a_cells['CH_22'].mean()
    cpep_b = camp_b_cells['CH_22'].mean()
    ins_a = camp_a_cells['CH_3'].mean()
    ins_b = camp_b_cells['CH_3'].mean()

    results_func.append({
        'Yas': age,
        'C-Peptide_A(p16)': cpep_a, 'C-Peptide_B(p21)': cpep_b,
        'Insulin_A(p16)': ins_a, 'Insulin_B(p21)': ins_b
    })

    is_cd8   = (group['CH_19'] > group['CH_19'].quantile(0.95))
    is_cd68  = (group['CH_30'] > group['CH_30'].quantile(0.95))
    is_foxp3 = (group['CH_36'] > group['CH_36'].quantile(0.95))

    cd8_coords   = group[is_cd8][['global_x', 'global_y']].values
    cd68_coords  = group[is_cd68][['global_x', 'global_y']].values
    foxp3_coords = group[is_foxp3][['global_x', 'global_y']].values

    def count_neighbors(source_coords, target_coords, radius):
        if len(source_coords) == 0 or len(target_coords) == 0: return 0
        tree = KDTree(target_coords)
        counts = tree.query_ball_point(source_coords, r=radius)
        return np.mean([len(c) for c in counts])

    coords_a = camp_a_cells[['global_x', 'global_y']].values
    coords_b = camp_b_cells[['global_x', 'global_y']].values

    results_immune.append({
        'Yas': age,
        'CD8_A(p16)':   count_neighbors(coords_a, cd8_coords,   RADIUS),
        'CD8_B(p21)':   count_neighbors(coords_b, cd8_coords,   RADIUS),
        'CD68_A(p16)':  count_neighbors(coords_a, cd68_coords,  RADIUS),
        'CD68_B(p21)':  count_neighbors(coords_b, cd68_coords,  RADIUS),
        'FOXP3_A(p16)': count_neighbors(coords_a, foxp3_coords, RADIUS),
        'FOXP3_B(p21)': count_neighbors(coords_b, foxp3_coords, RADIUS)
    })

    gc.collect()

if not results_func:
    print("Yeterli veri bulunamadi.")
    exit()

df_func = pd.DataFrame(results_func).sort_values('Yas')
df_imm  = pd.DataFrame(results_immune).sort_values('Yas')

fig, axs = plt.subplots(2, 2, figsize=(18, 12))

x = np.arange(len(df_func))
w = 0.35
c_p16 = '#2ECC71'  # Iyi zombi (Yesil)
c_p21 = '#E74C3C'  # Kotu zombi (Kirmizi)

axs[0,0].bar(x - w/2, df_func['C-Peptide_A(p16)'], w, label='Kamp A: p16-only (Iyi Zombi)', color=c_p16, edgecolor='black')
axs[0,0].bar(x + w/2, df_func['C-Peptide_B(p21)'], w, label='Kamp B: p21-only (Kotu Zombi)', color=c_p21, edgecolor='black')
axs[0,0].set_xticks(x)
axs[0,0].set_xticklabels([f"{int(a)} yas" for a in df_func['Yas']], fontsize=11)
axs[0,0].set_ylabel('Ortalama C-Peptide Intensitesi')
axs[0,0].set_title('Fonksiyon Testi: C-Peptide (Insulin Sentezi)', fontweight='bold')
axs[0,0].legend()
axs[0,0].grid(axis='y', alpha=0.3)

axs[0,1].bar(x - w/2, df_imm['CD8_A(p16)'], w, label='Kamp A (p16)', color=c_p16, edgecolor='black')
axs[0,1].bar(x + w/2, df_imm['CD8_B(p21)'], w, label='Kamp B (p21)', color=c_p21, edgecolor='black')
axs[0,1].set_xticks(x)
axs[0,1].set_xticklabels([f"{int(a)} yas" for a in df_imm['Yas']], fontsize=11)
axs[0,1].set_ylabel('Ortalama Komsu CD8 Sayisi (R=50px)')
axs[0,1].set_title('SASP Testi: Katil T (CD8) Infiltrasyonu\n(Kotu Zombi Etrafinda Artmali)', fontweight='bold')
axs[0,1].legend()
axs[0,1].grid(axis='y', alpha=0.3)

axs[1,0].bar(x - w/2, df_imm['CD68_A(p16)'], w, label='Kamp A (p16)', color=c_p16, edgecolor='black')
axs[1,0].bar(x + w/2, df_imm['CD68_B(p21)'], w, label='Kamp B (p21)', color=c_p21, edgecolor='black')
axs[1,0].set_xticks(x)
axs[1,0].set_xticklabels([f"{int(a)} yas" for a in df_imm['Yas']], fontsize=11)
axs[1,0].set_ylabel('Ortalama Komsu CD68 Sayisi (R=50px)')
axs[1,0].set_title('SASP Testi: Makrofaj (CD68) Infiltrasyonu\n(Kotu Zombi Etrafinda Artmali)', fontweight='bold')
axs[1,0].legend()
axs[1,0].grid(axis='y', alpha=0.3)

axs[1,1].bar(x - w/2, df_imm['FOXP3_A(p16)'], w, label='Kamp A (p16)', color=c_p16, edgecolor='black')
axs[1,1].bar(x + w/2, df_imm['FOXP3_B(p21)'], w, label='Kamp B (p21)', color=c_p21, edgecolor='black')
axs[1,1].set_xticks(x)
axs[1,1].set_xticklabels([f"{int(a)} yas" for a in df_imm['Yas']], fontsize=11)
axs[1,1].set_ylabel('Ortalama Komsu FOXP3 Sayisi (R=50px)')
axs[1,1].set_title('Immun Duzenleyici: FOXP3 (Treg) Korumasi\n(Iyi Zombi Etrafinda Artmali)', fontweight='bold')
axs[1,1].legend()
axs[1,1].grid(axis='y', alpha=0.3)

plt.suptitle(
    "Iwasaki et al. (2026) Kaniti: Iyi Zombi (p16+ Adaptif) vs Kotu Zombi (p21+ Maladaptif)\n"
    "Kotu Zombi Fonksiyon Kaybedip Katilleri Cekerken, Iyi Zombi Fonksiyonunu Koruyup Kendini Treg ile Savunur\n"
    f"Normalizasyon: Young Reference (SNT648, 35y, Tail Superior)",
    fontsize=13, fontweight='bold', y=0.99
)
plt.tight_layout()

out_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\tail_superior_35_vs_69"
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, "iwasaki_iyi_vs_kotu_zombi.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nGrafik '{output_path}' olarak kaydedildi.")
