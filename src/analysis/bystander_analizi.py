import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import time
import warnings
import os
import sys

warnings.filterwarnings('ignore')

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

RADIUSES = [25, 50, 100, 200]
MIN_SENESCENT = 50

sonuclar_r100 = []
distance_decay = []

print("\nKapsamlı Yayılım (Bystander) Analizi Başlıyor...")

def safe_mannwhitneyu(x, y):
    x_clean = x[~np.isnan(x)]
    y_clean = y[~np.isnan(y)]
    if len(x_clean) > 0 and len(y_clean) > 0:
        stat, p = mannwhitneyu(x_clean, y_clean, alternative='greater')
        return p
    return np.nan

print("Robust Scaling (Zaten filtering.py icinde yapildi)...")

global_p16_robust = df['young_p16_95th'].iloc[0]
global_p21_robust = df['young_p21_95th'].iloc[0]   # Young Reference (SNT648) bazli p21 esigi
global_lamin_robust = df['young_lamin_50th'].iloc[0]
print(f"Esikler: p16>{global_p16_robust:.2f}, p21>{global_p21_robust:.2f}, lamin<{global_lamin_robust:.2f}")

group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df['Group_Label'] = pd.Categorical(df['Group_Label'], categories=group_order, ordered=True)

for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    
    is_sen = ((group['CH_16_robust'] > global_p16_robust) & (group['CH_20_robust'] < global_lamin_robust)).values.astype(bool)
    is_p16 = ((group['CH_16_robust'] > global_p16_robust) & (group['CH_20_robust'] < global_lamin_robust)).values.astype(bool)
    is_p21 = ((group['CH_9_robust'] > global_p21_robust) & (group['CH_20_robust'] < global_lamin_robust)).values.astype(bool)
    
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95)).values.astype(bool)
    is_alpha = (group['CH_6'] > group['CH_6'].quantile(0.95)).values.astype(bool)
    is_fibro = (group['CH_17'] > group['CH_17'].quantile(0.95)).values.astype(bool)
    is_macro = (group['CH_30'] > group['CH_30'].quantile(0.95)).values.astype(bool)
    is_endo = (group['CH_28'] > group['CH_28'].quantile(0.95)).values.astype(bool)
    is_delta = (group['CH_18'] > group['CH_18'].quantile(0.95)).values.astype(bool)
    is_ductal = (group['CH_14'] > group['CH_14'].quantile(0.95)).values.astype(bool)
    is_acinar = (group['CH_5'] > group['CH_5'].quantile(0.95)).values.astype(bool)
    
    n_sen = is_sen.sum()
    
    if n_sen < MIN_SENESCENT:
        print(f"\n---> [ATLANDI] {region}: Sadece {n_sen:,} senesans hücresi bulundu.")
        sonuclar_r100.append({'Group_Label': region, 'n_sen': n_sen})
        continue

    print(f"\n---> {region}: {n_sen:,} senesans hücresi işleniyor...")
    coords = group[['global_x', 'global_y']].values
    tree = cKDTree(coords)
    
    for r in RADIUSES:
        start_r = time.time()
        
        N_cells = len(coords)
        bystander_rates = np.zeros(N_cells)
        alpha_to_beta_rates = np.full(N_cells, np.nan)
        beta_to_alpha_rates = np.full(N_cells, np.nan)
        
        CHUNK_SIZE = 50000
        for start_idx in range(0, N_cells, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, N_cells)
            chunk_coords = coords[start_idx:end_idx]
            
            chunk_neighbors_list = tree.query_ball_point(chunk_coords, r=r)
            
            for local_i, neighbors in enumerate(chunk_neighbors_list):
                i = start_idx + local_i
                
                if len(neighbors) > 1:
                    n_arr = np.array(neighbors, dtype=np.int32)
                    n_arr = n_arr[n_arr != i]
                    if len(n_arr) == 0: continue
                    
                    bystander_rates[i] = np.sum(is_sen[n_arr]) / len(n_arr)
                    
                    if r == 100:
                        if is_alpha[i]:
                            b_neigh = n_arr[is_beta[n_arr]]
                            if len(b_neigh) > 0:
                                alpha_to_beta_rates[i] = np.sum(is_sen[b_neigh]) / len(b_neigh)
                        elif is_beta[i]:
                            a_neigh = n_arr[is_alpha[n_arr]]
                            if len(a_neigh) > 0:
                                beta_to_alpha_rates[i] = np.sum(is_sen[a_neigh]) / len(a_neigh)
                            
        sen_mean = np.mean(bystander_rates[is_sen == 1]) * 100
        norm_mean = np.mean(bystander_rates[is_sen == 0]) * 100
        distance_decay.append({'Group_Label': region, 'radius': r, 'sen_bystander': sen_mean, 'norm_bystander': norm_mean})
        
        if r == 100:
            sen_bystander_mean = sen_mean
            norm_bystander_mean = norm_mean
            p_all = safe_mannwhitneyu(bystander_rates[is_sen == 1], bystander_rates[is_sen == 0])
            
            def calc_metrics(mask):
                sm = np.mean(bystander_rates[(is_sen == 1) & mask]) * 100
                nm = np.mean(bystander_rates[(is_sen == 0) & mask]) * 100
                pval = safe_mannwhitneyu(bystander_rates[(is_sen == 1) & mask], bystander_rates[(is_sen == 0) & mask])
                return sm, nm, pval
                
            beta_s, beta_n, p_beta = calc_metrics(is_beta)
            alpha_s, alpha_n, p_alpha = calc_metrics(is_alpha)
            fibro_s, fibro_n, p_fibro = calc_metrics(is_fibro)
            macro_s, macro_n, p_macro = calc_metrics(is_macro)
            endo_s, endo_n, p_endo = calc_metrics(is_endo)
            delta_s, delta_n, p_delta = calc_metrics(is_delta)
            ductal_s, ductal_n, p_ductal = calc_metrics(is_ductal)
            acinar_s, acinar_n, p_acinar = calc_metrics(is_acinar)
            
            a2b_s = np.nanmean(alpha_to_beta_rates[(is_sen == 1) & is_alpha]) * 100
            a2b_n = np.nanmean(alpha_to_beta_rates[(is_sen == 0) & is_alpha]) * 100
            p_a2b = safe_mannwhitneyu(alpha_to_beta_rates[(is_sen == 1) & is_alpha], alpha_to_beta_rates[(is_sen == 0) & is_alpha])
            
            b2a_s = np.nanmean(beta_to_alpha_rates[(is_sen == 1) & is_beta]) * 100
            b2a_n = np.nanmean(beta_to_alpha_rates[(is_sen == 0) & is_beta]) * 100
            p_b2a = safe_mannwhitneyu(beta_to_alpha_rates[(is_sen == 1) & is_beta], beta_to_alpha_rates[(is_sen == 0) & is_beta])

            sonuclar_r100.append({
                'Group_Label': region, 'n_sen': n_sen,
                'sen_bystander_mean': sen_bystander_mean, 'norm_bystander_mean': norm_bystander_mean, 'p_all': p_all,
                'beta_sen_bystander': beta_s, 'beta_norm_bystander': beta_n, 'p_beta': p_beta,
                'alpha_sen_bystander': alpha_s, 'alpha_norm_bystander': alpha_n, 'p_alpha': p_alpha,
                'fibro_sen_bystander': fibro_s, 'fibro_norm_bystander': fibro_n, 'p_fibro': p_fibro,
                'macro_sen_bystander': macro_s, 'macro_norm_bystander': macro_n, 'p_macro': p_macro,
                'endo_sen_bystander': endo_s, 'endo_norm_bystander': endo_n, 'p_endo': p_endo,
                'delta_sen_bystander': delta_s, 'delta_norm_bystander': delta_n, 'p_delta': p_delta,
                'ductal_sen_bystander': ductal_s, 'ductal_norm_bystander': ductal_n, 'p_ductal': p_ductal,
                'acinar_sen_bystander': acinar_s, 'acinar_norm_bystander': acinar_n, 'p_acinar': p_acinar,
                'a2b_sen': a2b_s, 'a2b_norm': a2b_n, 'p_a2b': p_a2b,
                'b2a_sen': b2a_s, 'b2a_norm': b2a_n, 'p_b2a': p_b2a,
                'p16_beta_bystander': np.mean(bystander_rates[(is_p16 == 1) & is_beta]) * 100,
                'p21_beta_bystander': np.mean(bystander_rates[(is_p21 == 1) & is_beta]) * 100,
                'p_p16vsp21': safe_mannwhitneyu(bystander_rates[(is_p16 == 1) & is_beta], bystander_rates[(is_p21 == 1) & is_beta])
            })
            
            print(f"  [R=100] Çapraz Bulaşıcılık (Cross-Type):")
            print(f"    - Yaşlı Alfa'nın komşu Beta'yı yaşlandırma oranı: %{a2b_s:.2f} (Normal Alfa: %{a2b_n:.2f}) [MW p={p_a2b:.2e}]")
            print(f"    - Yaşlı Beta'nın komşu Alfa'yı yaşlandırma oranı: %{b2a_s:.2f} (Normal Beta: %{b2a_n:.2f}) [MW p={p_b2a:.2e}]")
            
        print(f"  [R={r} px] tamamlandı ({time.time()-start_r:.1f} sn). Yaşlı yayılımı: %{sen_mean:.2f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_r100 = pd.DataFrame(sonuclar_r100)
df_r100['Group_Label'] = pd.Categorical(df_r100['Group_Label'], categories=group_order, ordered=True)
df_r100 = df_r100.sort_values('Group_Label')

df_decay = pd.DataFrame(distance_decay)

if df_decay.empty:
    print("Yeterli senesans hucresi bulunmadigi icin grafikler cizilemiyor.")
    sys.exit(0)

plt.figure(figsize=(10, 6))
for region in df_decay['Group_Label'].unique():
    d = df_decay[df_decay['Group_Label'] == region]
    plt.plot(d['radius'], d['sen_bystander'], marker='o', linewidth=2, label=f'{region}')

plt.title('SASP Etki Alanı Sönümlenmesi (Distance Decay)\nMesafe Arttıkça Yayılımın Logaritmik Düşüşü')
plt.xlabel('Hücreden Uzaklık (Piksel Yarıçapı)')
plt.ylabel('Çevredeki Yaşlı Komşu Ratioı (%)')
plt.xticks(RADIUSES)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_bystander_distance_decay.png"), dpi=300)
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(20, 14))
axes = axes.flatten()

cell_types = [
    ('TÜM HÜCRELER', 'sen_bystander_mean', 'norm_bystander_mean', 'p_all', 'red', 'blue'),
    ('BETA HÜCRELERİ', 'beta_sen_bystander', 'beta_norm_bystander', 'p_beta', 'darkgreen', 'lightgreen'),
    ('ALFA HÜCRELERİ', 'alpha_sen_bystander', 'alpha_norm_bystander', 'p_alpha', 'purple', 'violet'),
    ('DELTA HÜCRELERİ', 'delta_sen_bystander', 'delta_norm_bystander', 'p_delta', 'dodgerblue', 'lightskyblue'),
    ('DUCTAL (KANAL)', 'ductal_sen_bystander', 'ductal_norm_bystander', 'p_ductal', 'saddlebrown', 'burlywood'),
    ('ASİNER (DIŞ SALGI)', 'acinar_sen_bystander', 'acinar_norm_bystander', 'p_acinar', 'navy', 'cornflowerblue'),
    ('FİBROBLASTLAR', 'fibro_sen_bystander', 'fibro_norm_bystander', 'p_fibro', 'darkorange', 'navajowhite'),
    ('MAKROFAJLAR', 'macro_sen_bystander', 'macro_norm_bystander', 'p_macro', 'darkred', 'lightcoral'),
    ('ENDOTEL', 'endo_sen_bystander', 'endo_norm_bystander', 'p_endo', 'teal', 'paleturquoise')
]

x_regions = df_r100['Group_Label'].astype(str).tolist()
x_positions = np.arange(len(x_regions))

for idx, (title, sen_col, norm_col, p_col, c_sen, c_norm) in enumerate(cell_types):
    if sen_col not in df_r100.columns: continue
    
    axes[idx].plot(x_positions, df_r100[sen_col], marker='o', linewidth=2, color=c_sen, label='Yaşlının Çevresi')
    axes[idx].plot(x_positions, df_r100[norm_col], marker='s', linewidth=2, linestyle='--', color=c_norm, label='Normalin Çevresi')
    axes[idx].fill_between(x_positions, df_r100[norm_col].astype(float), df_r100[sen_col].astype(float), color=c_sen, alpha=0.1)
    
    axes[idx].set_title(title)
    axes[idx].set_xlabel('Grup')
    axes[idx].set_ylabel('Çevredeki Yaşlı Ratioı (%)')
    axes[idx].set_xticks(x_positions)
    axes[idx].set_xticklabels(x_regions, rotation=15)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, linestyle='--')
    
    for i, row in df_r100.iterrows():
        pos = x_positions[i]
        if pd.isna(row[sen_col]):
            axes[idx].annotate(f"*n={int(row['n_sen'])} (Yetersiz)", xy=(pos, 0), xytext=(0, 5), 
                               textcoords="offset points", ha='center', fontsize=8, color='gray', style='italic')
        else:
            pval = row[p_col]
            if not pd.isna(pval):
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
                axes[idx].annotate(sig, xy=(pos, row[sen_col]), xytext=(0, 5),
                                   textcoords="offset points", ha='center', fontsize=12, color='black', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_bystander_hucre_tipi_etkisi.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x_positions, df_r100['a2b_sen'], marker='o', linewidth=2, color='darkorange', label='Yaşlı Alfa -> Komşu Beta Bulaşıcılığı')
plt.plot(x_positions, df_r100['b2a_sen'], marker='s', linewidth=2, color='teal', label='Yaşlı Beta -> Komşu Alfa Bulaşıcılığı')
plt.title('Çapraz Hücre Tipi SASP Bulaşıcılığı (Radius = 100px)\nHangi Hücre Tipi Diğerini Daha Çok Yaşlandırıyor?')
plt.xlabel('Grup')
plt.ylabel('Komşu Hedef Hücrenin Yaşlı Olma Ratioı (%)')
plt.xticks(x_positions, x_regions, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')

for i, row in df_r100.iterrows():
    pos = x_positions[i]
    if pd.isna(row['a2b_sen']):
        plt.annotate(f"*n={int(row['n_sen'])} (Yetersiz)", xy=(pos, 0), xytext=(0, 5), 
                     textcoords="offset points", ha='center', fontsize=8, color='gray', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_bystander_capraz_bulasicilik.png"), dpi=300)
plt.close()

print("\nMuazzam Analiz Tamamlandı!")
print(f"Grafikler '{OUTPUT_DIR}' klasörüne kaydedildi.")

plt.figure(figsize=(10, 6))
plt.plot(x_positions, df_r100['p16_beta_bystander'], marker='o', linewidth=2, color='darkred', label='p16-Beta (CDKN2A+ / Kalıcı Zombi)')
plt.plot(x_positions, df_r100['p21_beta_bystander'], marker='s', linewidth=2, linestyle='--', color='darkorange', label='p21-Beta (CDKN1A+ / Stres Zombisi)')
plt.fill_between(x_positions, df_r100['p21_beta_bystander'].astype(float), df_r100['p16_beta_bystander'].astype(float), color='gray', alpha=0.1)

plt.title('Beta Hücrelerinde p16 vs p21 SASP Yayılım Gücü (R=100px)\nHangi Zombi Tipi Daha Bulaşıcı? (Iwasaki Karşılaştırması)')
plt.xlabel('Grup')
plt.ylabel('Çevredeki Yaşlı (SASP) Hücre Ratioı (%)')
plt.xticks(x_positions, x_regions, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')

for i, row in df_r100.iterrows():
    pos = x_positions[i]
    if pd.isna(row.get('p16_beta_bystander')):
        plt.annotate(f"*n={int(row['n_sen'])} (Yetersiz)", xy=(pos, 0), xytext=(0, 10),
                     textcoords="offset points", ha='center', fontsize=8, color='gray', style='italic')
    else:
        pval = row.get('p_p16vsp21', np.nan)
        if not pd.isna(pval):
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            y_pos = max(row['p16_beta_bystander'], row['p21_beta_bystander'])
            plt.annotate(sig, xy=(pos, y_pos), xytext=(0, 5),
                         textcoords="offset points", ha='center', fontsize=12, color='black', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_bystander_p16_vs_p21.png"), dpi=300)
plt.close()
