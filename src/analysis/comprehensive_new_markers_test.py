import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr
import gc
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue, get_young_baseline
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print("=" * 60)
print(f"{COHORT_NAME} - Kapsamli Yeni Marker Testi (8 Test)")
print("=" * 60)

columns_to_load = [
    'patient_id', 'age', 'CH_0',
    'CH_1',   # CD45RA (Naif T)
    'CH_4',   # CA9 (Hipoksi)
    'CH_7',   # CD45RO (Hafiza T)
    'CH_8',   # Collagen I
    'CH_11',  # MECOM (Kok hucre)
    'CH_12',  # CFTR
    'CH_13',  # PGP9.5 (Sinir)
    'CH_16',  # p16
    'CH_17',  # Cadherin 11 (Fibroblast)
    'CH_20',  # Lamin B1
    'CH_21',  # CD3e (T hucre genel)
    'CH_24',  # Perilipin (Yag)
    'CH_25',  # Podoplanin (Lenf damari)
    'CH_29',  # CD31 (Kan damari)
    'CH_37',  # SMA (Duz kas/Miyofibroblast)
    'global_x', 'global_y'
]

print("\nVeriler yukleniyor ve DBSCAN filtresi uygulaniyor...")

df_list = []
for snt, age, region, label in ACTIVE_COHORT:
    print(f"{snt} ({region}) isleniyor...")
    d_clean = load_and_filter_tissue(snt, age)
    if d_clean is not None:
        robust_cols = ['CH_16_robust', 'CH_20_robust']
        existing_cols = [c for c in columns_to_load if c in d_clean.columns]
        keep_cols = list(dict.fromkeys(existing_cols + robust_cols))
        keep_cols = [c for c in keep_cols if c in d_clean.columns]
        d_clean = d_clean[keep_cols]
        d_clean['Group_Label'] = label
        df_list.append(d_clean)

if not df_list:
    print("HATA: Hcbir veri yuklenemedi.")
    sys.exit()

df_all = pd.concat(df_list, ignore_index=True)

print("Normalizasyon: filtering.py uzerinden Young Reference (SNT648) kullaniliyor...")
_baseline = get_young_baseline()
p16_sinir  = _baseline['p16_robust_95th']
lamin_sinir = _baseline['lamin_robust_50th']
print(f"Esikler: p16_robust>{p16_sinir:.2f}, LaminB1_robust<{lamin_sinir:.2f}")

RADIUS = 100

results_t_shift = []
results_endotel = []
results_perilipin = []
results_nerve = []
results_fibroblast = []
results_gradient = []
results_lymph = []
results_mecom = []

group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
df_all['Group_Label'] = pd.Categorical(df_all['Group_Label'], categories=group_order, ordered=True)

for region, group in df_all.groupby('Group_Label'):
    if len(group) == 0: continue
    print(f"\n{'='*40} {region} {'='*40}")
    
    is_sen = (group['CH_16_robust'] > p16_sinir) & (group['CH_20_robust'] < lamin_sinir)
    is_normal = ~is_sen
        
    sen_cells = group[is_sen]
    normal_cells = group[is_normal]
    n_sen = len(sen_cells)
    n_normal = len(normal_cells)
    
    if n_sen < 5:
        print("  Yeterli senesans hucre yok, atlaniyor...")
        continue
    
    sen_coords = sen_cells[['global_x', 'global_y']].values
    normal_sample = normal_cells.sample(n=min(n_normal, n_sen * 3), random_state=42)
    normal_coords = normal_sample[['global_x', 'global_y']].values
    
    def get_neighbor_intensity(source_coords, channel, radius=RADIUS):
