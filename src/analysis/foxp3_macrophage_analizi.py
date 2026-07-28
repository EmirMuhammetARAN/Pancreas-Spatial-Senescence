import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import sem, t
import time
import os
import sys
import warnings
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

print("Robust Normalizasyon hesaplaniyor...")
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

RADIUS = 100
sonuclar = []

print(f"\n[FOXP3 & MAKROFAJ] Etkilesim ve Guven Araligi (CI) Analizi Basliyor (Radius={RADIUS}px)...")

def calc_ci(data, confidence=0.95):
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, h

global_p16 = df['young_p16_95th'].iloc[0]
global_lamin = df['CH_20_robust'].quantile(0.50)

group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df['Group_Label'] = pd.Categorical(df['Group_Label'], categories=group_order, ordered=True)

for region, group in df.groupby('Group_Label'):
    if len(group) == 0: continue
    
    is_macro = (group['CH_30'] > group['CH_30'].quantile(0.95)).values.astype(int)
    is_foxp3 = (group['CH_36'] > group['CH_36'].quantile(0.95)).values.astype(int)
    
    is_sen = ((group['CH_16_robust'] > global_p16) & (group['CH_20_robust'] < global_lamin)).values.astype(int)
    
    coords = group[['global_x', 'global_y']].values
    tree = cKDTree(coords)
    
    print(f"---> {region}: {is_macro.sum():,} Makrofaj, {is_foxp3.sum():,} FOXP3 isleniyor...")
    
    macro_bystander_rates = np.zeros(len(coords))
    macro_has_foxp3 = np.zeros(len(coords))
    foxp3_has_macro = np.zeros(len(coords))
    
    CHUNK_SIZE = 50000
    for start_idx in range(0, len(coords), CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, len(coords))
        chunk_coords = coords[start_idx:end_idx]
        
        neighbors_list = tree.query_ball_point(chunk_coords, r=RADIUS)
        
        for i_chunk, neighbors in enumerate(neighbors_list):
            i = start_idx + i_chunk
            if len(neighbors) > 1:
                n_list = [n for n in neighbors if n != i]
                if not n_list: continue
                
                if is_macro[i]:
                    sen_count = sum(1 for n in n_list if is_sen[n] == 1)
                    macro_bystander_rates[i] = sen_count / len(n_list)
                    
                    foxp3_count = sum(1 for n in n_list if is_foxp3[n])
                    macro_has_foxp3[i] = foxp3_count / len(n_list)
                    
                if is_foxp3[i]:
                    macro_count = sum(1 for n in n_list if is_macro[n])
                    foxp3_has_macro[i] = macro_count / len(n_list)

    m_with_f = np.mean(macro_has_foxp3[is_macro == 1]) * 100
    f_with_m = np.mean(foxp3_has_macro[is_foxp3 == 1]) * 100
    
    m_rates_around_sen = macro_bystander_rates[(is_macro == 1) & (is_sen == 1)]
    m_rates_around_norm = macro_bystander_rates[(is_macro == 1) & (is_sen == 0)]
    
    mean_sen, ci_sen = calc_ci(m_rates_around_sen)
    mean_norm, ci_norm = calc_ci(m_rates_around_norm)
    
    mean_sen, ci_sen = mean_sen * 100, ci_sen * 100
    mean_norm, ci_norm = mean_norm * 100, ci_norm * 100
    
    sonuclar.append({
        'Group_Label': region,
        'm_with_f': m_with_f,
        'f_with_m': f_with_m,
        'm_sen_mean': mean_sen,
        'm_sen_ci': ci_sen,
        'm_norm_mean': mean_norm,
        'm_norm_ci': ci_norm
    })

df_sonuc = pd.DataFrame(sonuclar)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

regions = df_sonuc['Group_Label'].astype(str).tolist()

ax1.plot(regions, df_sonuc['m_with_f'], marker='o', linewidth=2, color='purple', label='Makrofajin Cevresindeki FOXP3 Ratioi (%)')
ax1.plot(regions, df_sonuc['f_with_m'], marker='s', linewidth=2, linestyle='--', color='green', label="FOXP3'un Cevresindeki Makrofaj Ratioi (%)")
ax1.set_title(f'{COHORT_NAME} - Treg (FOXP3) ve Makrofaj (CD68) Komsulugu (R=100px)')
ax1.set_xlabel('Grup')
ax1.set_ylabel('Etkilesim Ratioi (%)')
ax1.legend()
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=15)

ax2.errorbar(regions, df_sonuc['m_sen_mean'], yerr=df_sonuc['m_sen_ci'], fmt='-o', color='darkred', ecolor='red', capsize=5, elinewidth=2, label='Yasli Makrofajin Etrafindaki Yaslilik (%95 CI)')
ax2.errorbar(regions, df_sonuc['m_norm_mean'], yerr=df_sonuc['m_norm_ci'], fmt='-s', linestyle='--', color='lightcoral', ecolor='lightcoral', capsize=5, elinewidth=2, label='Normal Makrofajin Etrafindaki Yaslilik (%95 CI)')
ax2.set_title('Makrofaj Yayiliminda %95 Guven Araligi (Confidence Interval)')
ax2.set_xlabel('Grup')
ax2.set_ylabel('Cevredeki Yasli Komsu Ratioi (%)')
ax2.legend()
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=15)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_foxp3_macrophage_analizi.png")
plt.savefig(save_path, dpi=300)
plt.close()

print("\n[FOXP3 & Makrofaj CI] Analiz Tamamlandi!")
print(f"Grafik kaydedildi: {save_path}")
