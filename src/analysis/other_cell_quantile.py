import pandas as pd
import glob
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print(f"{COHORT_NAME} - Hücre Tipi Eşik Testi")
print("=" * 60)

print("\nVeriler yükleniyor ve DBSCAN filtresi uygulanıyor...")
df_list = []
for snt, age, region, label in ACTIVE_COHORT:
    print(f"{snt} ({region}) isleniyor...")
    d_clean = load_and_filter_tissue(snt, age)
    if d_clean is not None:
        d_clean['Group_Label'] = label
        df_list.append(d_clean)

if not df_list:
    print("HATA: Hiçbir veri yüklenemedi.")
    sys.exit()

df = pd.concat(df_list, ignore_index=True)

print("\n--- Beta & Alfa Çakışma Analizi ---")
beta_alpha_overlaps = []
quantiles = [0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

for q in quantiles:
    df['is_beta_t'] = df.groupby('patient_id')['CH_3'].transform(lambda x: x > x.quantile(q))
    df['is_alpha_t'] = df.groupby('patient_id')['CH_6'].transform(lambda x: x > x.quantile(q))
    
    overlap = ((df['is_beta_t']) & (df['is_alpha_t'])).sum()
    overlap_pct = overlap / len(df) * 100
    
    n_beta = df['is_beta_t'].sum()
    n_alpha = df['is_alpha_t'].sum()
    
    beta_alpha_overlaps.append(overlap_pct)
    
    print(f"q={q}: Beta={n_beta:,} (%{n_beta/len(df)*100:.1f}), Alpha={n_alpha:,} (%{n_alpha/len(df)*100:.1f}), Çakışma=%{overlap_pct:.2f}")

print("\n--- Makrofaj & FOXP3 Çakışma Analizi ---")
macro_foxp3_overlaps = []

for q in quantiles:
    df['is_macro_t'] = df.groupby('patient_id')['CH_30'].transform(lambda x: x > x.quantile(q))
    df['is_foxp3_t'] = df.groupby('patient_id')['CH_36'].transform(lambda x: x > x.quantile(q))
    
    overlap = ((df['is_macro_t']) & (df['is_foxp3_t'])).sum()
    overlap_pct = overlap / len(df) * 100
    
    n_macro = df['is_macro_t'].sum()
    n_foxp3 = df['is_foxp3_t'].sum()
    
    macro_foxp3_overlaps.append(overlap_pct)
    
    print(f"q={q}: Makrofaj={n_macro:,} (%{n_macro/len(df)*100:.1f}), FOXP3={n_foxp3:,} (%{n_foxp3/len(df)*100:.1f}), Çakışma=%{overlap_pct:.2f}")

plt.figure(figsize=(10, 6))

plt.plot(quantiles, beta_alpha_overlaps, marker='o', markersize=8, linewidth=2.5, color='#4C72B0', label='Beta & Alfa Çakışması')
plt.plot(quantiles, macro_foxp3_overlaps, marker='s', markersize=8, linewidth=2.5, color='#C44E52', label='Makrofaj & FOXP3 Çakışması')

plt.title("Hücre Tipi Belirlemede Eşik (Quantile) Seçimi ve Çakışma Ratioı\n(Yalancı Çift Kimlikli Hücrelerin Reddedilmesi)", fontsize=14, fontweight='bold')
plt.xlabel("Quantile Eşiği (q)", fontsize=12, fontweight='bold')
plt.ylabel("Çakışma Ratioı (Tüm Hücrelere Göre %)", fontsize=12, fontweight='bold')
plt.xticks(quantiles)
plt.axvline(x=0.95, color='green', linestyle='--', linewidth=2, label='Seçilen Eşik (q=0.95)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, f"{COHORT_NAME}_cell_type_overlap_test.png")
plt.savefig(save_path, dpi=300)
print(f"\nGrafik '{save_path}' olarak kaydedildi.")
