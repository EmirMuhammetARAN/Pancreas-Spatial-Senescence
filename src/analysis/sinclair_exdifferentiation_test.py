import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

print("=" * 60)
print("Sinclair Exdifferentiation & EMT Testi")
print("=" * 60)

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
global_lam_r = df['young_lamin_50th'].iloc[0]
print(f"Young Ref Esikleri: p16>{global_p16_r:.2f}, Lamin_low<{global_lam_r:.2f}")

results_exdiff = []
results_emt = []
results_chromatin = []

for patient_id, group in df.groupby('patient_id'):
    age = int(group['age'].iloc[0])
    print(f"\n--- {age} Yaş İşleniyor ---")
    
    is_beta = (group['CH_3'] > group['CH_3'].quantile(0.95))
    
    is_sen = (group['CH_16_robust'] > global_p16_r) & (group['CH_20_robust'] < global_lam_r)
        
    beta_cells = group[is_beta]
    sen_beta = group[is_beta & is_sen]
    non_sen_beta = group[is_beta & ~is_sen]
        
    n_sen = len(sen_beta)
    n_non = len(non_sen_beta)
    print(f"  Beta: {len(beta_cells)}, Senescence Beta: {n_sen}, Normal Beta: {n_non}")
    
    if n_sen < 5 or n_non < 5:
        print("  Yeterli hücre yok, atlanıyor...")
        continue
    
    markers_exdiff = {
        'SYCN (CH_5, Exocrine)': 'CH_5',
        'Keratin19 (CH_14, Ductal)': 'CH_14',
        'Insulin (CH_3, Kimlik)': 'CH_3',
    }
    
    row_exdiff = {'Yaş': age, 'n_sen': n_sen, 'n_non': n_non}
    for name, ch in markers_exdiff.items():
        sen_vals = sen_beta[ch].values
        non_vals = non_sen_beta[ch].values
        
        mean_sen = np.mean(sen_vals)
        mean_non = np.mean(non_vals)
        fc = mean_sen / mean_non if mean_non > 0 else np.nan
        
        stat, pval = mannwhitneyu(sen_vals, non_vals, alternative='two-sided')
        
        row_exdiff[f'{name}_Sen'] = mean_sen
        row_exdiff[f'{name}_Non'] = mean_non
        row_exdiff[f'{name}_FC'] = fc
        row_exdiff[f'{name}_p'] = pval
        
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        direction = "↑" if fc > 1 else "↓"
        print(f"  {name}: Sen={mean_sen:.2f} vs Non={mean_non:.2f} | FC={fc:.2f}{direction} | p={pval:.4f} {sig}")
    
    results_exdiff.append(row_exdiff)
    
    sen_all = group[is_sen]
    non_sen_all = group[~is_sen]
    
    markers_emt = {
        'E-cadherin (CH_28, Epitel)': 'CH_28',
        'EpCAM (CH_32, Epitel)': 'CH_32',
        'Cadherin11 (CH_17, Mezenk.)': 'CH_17',
        'SMA (CH_37, Mezenk.)': 'CH_37',
        'CollagenI (CH_8, Fibrozis)': 'CH_8',
    }
    
    row_emt = {'Yaş': age}
    for name, ch in markers_emt.items():
        sen_vals = sen_all[ch].values
        non_vals = non_sen_all[ch].values
        
        mean_sen = np.mean(sen_vals)
        mean_non = np.mean(non_vals)
        fc = mean_sen / mean_non if mean_non > 0 else np.nan
        
        stat, pval = mannwhitneyu(sen_vals, non_vals, alternative='two-sided')
        
        row_emt[f'{name}_Sen'] = mean_sen
        row_emt[f'{name}_Non'] = mean_non
        row_emt[f'{name}_FC'] = fc
        row_emt[f'{name}_p'] = pval
        
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        direction = "↑" if fc > 1 else "↓"
        print(f"  {name}: Sen={mean_sen:.2f} vs Non={mean_non:.2f} | FC={fc:.2f}{direction} | p={pval:.4f} {sig}")
    
    results_emt.append(row_emt)
    
    lamin_vals_all = group['CH_20'].values
    hmgb1_vals_all = group['CH_23'].values
    
    corr = np.corrcoef(lamin_vals_all, hmgb1_vals_all)[0, 1]
    
    hmgb1_sen = sen_all['CH_23'].values
    hmgb1_non = non_sen_all['CH_23'].values
    
    hmgb1_mean_sen = np.mean(hmgb1_sen) if len(hmgb1_sen) > 0 else 0
    hmgb1_mean_non = np.mean(hmgb1_non) if len(hmgb1_non) > 0 else 0
    hmgb1_fc = hmgb1_mean_sen / hmgb1_mean_non if hmgb1_mean_non > 0 else np.nan
    
    stat, pval_hmgb1 = mannwhitneyu(hmgb1_sen, hmgb1_non, alternative='two-sided') if len(hmgb1_sen) > 0 and len(hmgb1_non) > 0 else (0, 1)
    
    results_chromatin.append({
        'Yaş': age,
        'LaminB1_HMGB1_Corr': corr,
        'HMGB1_Sen': hmgb1_mean_sen,
        'HMGB1_Non': hmgb1_mean_non,
        'HMGB1_FC': hmgb1_fc,
        'HMGB1_p': pval_hmgb1
    })
    
    print(f"  Lamin B1 ↔ HMGB1 Korelasyonu: r={corr:.3f}")
    print(f"  HMGB1 Sen={hmgb1_mean_sen:.2f} vs Non={hmgb1_mean_non:.2f} | FC={hmgb1_fc:.2f} | p={pval_hmgb1:.4f}")

if not results_exdiff:
    print("Yeterli veri bulunamadi.")
    exit()

df_exdiff = pd.DataFrame(results_exdiff).sort_values('Yaş')
df_emt = pd.DataFrame(results_emt).sort_values('Yaş')
df_chrom = pd.DataFrame(results_chromatin).sort_values('Yaş')

fig, axs = plt.subplots(2, 3, figsize=(22, 13))

ages = [f"{int(a)} yaş" for a in df_exdiff['Yaş']]
x = np.arange(len(ages))
w = 0.30

c_sen = '#E74C3C'  # Kırmızı (senesans)
c_non = '#3498DB'  # Mavi (normal)

axs[0,0].bar(x - w/2, df_exdiff['SYCN (CH_5, Exocrine)_Sen'], w, label='Senescence Beta', color=c_sen, edgecolor='black')
axs[0,0].bar(x + w/2, df_exdiff['SYCN (CH_5, Exocrine)_Non'], w, label='Normal Beta', color=c_non, edgecolor='black')
for i, row in df_exdiff.iterrows():
    pval = row['SYCN (CH_5, Exocrine)_p']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    fc = row['SYCN (CH_5, Exocrine)_FC']
    idx = list(df_exdiff.index).index(i)
    max_y = max(row['SYCN (CH_5, Exocrine)_Sen'], row['SYCN (CH_5, Exocrine)_Non'])
    axs[0,0].text(idx, max_y * 1.05, f"{sig}\nFC={fc:.2f}", ha='center', fontsize=8, color='red' if pval < 0.05 else 'gray')
axs[0,0].set_xticks(x)
axs[0,0].set_xticklabels(ages)
axs[0,0].set_title('Test 1a: SYCN (Exocrine Marker)\nKimlik Kaybı? Yanlış Protein Kazanımı?', fontweight='bold')
axs[0,0].set_ylabel('Ortalama İntensitesi')
axs[0,0].legend()
axs[0,0].grid(axis='y', alpha=0.3)

axs[0,1].bar(x - w/2, df_exdiff['Keratin19 (CH_14, Ductal)_Sen'], w, label='Senescence Beta', color=c_sen, edgecolor='black')
axs[0,1].bar(x + w/2, df_exdiff['Keratin19 (CH_14, Ductal)_Non'], w, label='Normal Beta', color=c_non, edgecolor='black')
for i, row in df_exdiff.iterrows():
    pval = row['Keratin19 (CH_14, Ductal)_p']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    fc = row['Keratin19 (CH_14, Ductal)_FC']
    idx = list(df_exdiff.index).index(i)
    max_y = max(row['Keratin19 (CH_14, Ductal)_Sen'], row['Keratin19 (CH_14, Ductal)_Non'])
    axs[0,1].text(idx, max_y * 1.05, f"{sig}\nFC={fc:.2f}", ha='center', fontsize=8, color='red' if pval < 0.05 else 'gray')
axs[0,1].set_xticks(x)
axs[0,1].set_xticklabels(ages)
axs[0,1].set_title('Test 1b: Keratin 19 (Ductal Marker)\nKimlik Kaybı? Yanlış Protein Kazanımı?', fontweight='bold')
axs[0,1].set_ylabel('Ortalama İntensitesi')
axs[0,1].legend()
axs[0,1].grid(axis='y', alpha=0.3)

axs[0,2].bar(x - w/2, df_exdiff['Insulin (CH_3, Kimlik)_Sen'], w, label='Senescence Beta', color=c_sen, edgecolor='black')
axs[0,2].bar(x + w/2, df_exdiff['Insulin (CH_3, Kimlik)_Non'], w, label='Normal Beta', color=c_non, edgecolor='black')
for i, row in df_exdiff.iterrows():
    pval = row['Insulin (CH_3, Kimlik)_p']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    fc = row['Insulin (CH_3, Kimlik)_FC']
    idx = list(df_exdiff.index).index(i)
    max_y = max(row['Insulin (CH_3, Kimlik)_Sen'], row['Insulin (CH_3, Kimlik)_Non'])
    axs[0,2].text(idx, max_y * 1.05, f"{sig}\nFC={fc:.2f}", ha='center', fontsize=8, color='red' if pval < 0.05 else 'gray')
axs[0,2].set_xticks(x)
axs[0,2].set_xticklabels(ages)
axs[0,2].set_title('Test 1c: İnsülin (Kimlik Proteini, Kontrol)\nSinclair: Düşmeli → Patra: Artmalı', fontweight='bold')
axs[0,2].set_ylabel('Ortalama İntensitesi')
axs[0,2].legend()
axs[0,2].grid(axis='y', alpha=0.3)

emt_epitel_markers = ['E-cadherin (CH_28, Epitel)', 'EpCAM (CH_32, Epitel)']
fc_epitel = []
for m in emt_epitel_markers:
    fc_epitel.append(df_emt[f'{m}_FC'].values)
fc_epitel = np.array(fc_epitel)

bar_positions = np.arange(len(ages))
bar_w = 0.35
for j, m in enumerate(emt_epitel_markers):
    short_name = m.split('(')[0].strip()
    fc_vals = df_emt[f'{m}_FC'].values
    p_vals = df_emt[f'{m}_p'].values
    bars = axs[1,0].bar(bar_positions + j * bar_w - bar_w/2, fc_vals, bar_w, 
                         label=short_name, edgecolor='black', alpha=0.8)
    for k, (fv, pv) in enumerate(zip(fc_vals, p_vals)):
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        axs[1,0].text(bar_positions[k] + j * bar_w - bar_w/2, fv + 0.02, sig, ha='center', fontsize=8,
                      color='red' if pv < 0.05 else 'gray')

axs[1,0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
axs[1,0].set_xticks(bar_positions)
axs[1,0].set_xticklabels(ages)
axs[1,0].set_title('Test 2a: Epitelyal Marker FC (Sen/Non)\n< 1 = Kayıp (EMT destekler)', fontweight='bold')
axs[1,0].set_ylabel('Fold Change (Sen / Normal)')
axs[1,0].legend()
axs[1,0].grid(axis='y', alpha=0.3)

emt_mesenc_markers = ['Cadherin11 (CH_17, Mezenk.)', 'SMA (CH_37, Mezenk.)', 'CollagenI (CH_8, Fibrozis)']
bar_positions = np.arange(len(ages))
bar_w = 0.25
for j, m in enumerate(emt_mesenc_markers):
    short_name = m.split('(')[0].strip()
    fc_vals = df_emt[f'{m}_FC'].values
    p_vals = df_emt[f'{m}_p'].values
    bars = axs[1,1].bar(bar_positions + j * bar_w - bar_w, fc_vals, bar_w, 
                         label=short_name, edgecolor='black', alpha=0.8)
    for k, (fv, pv) in enumerate(zip(fc_vals, p_vals)):
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        axs[1,1].text(bar_positions[k] + j * bar_w - bar_w, fv + 0.02, sig, ha='center', fontsize=7,
                      color='red' if pv < 0.05 else 'gray')

axs[1,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
axs[1,1].set_xticks(bar_positions)
axs[1,1].set_xticklabels(ages)
axs[1,1].set_title('Test 2b: Mezenkimal Marker FC (Sen/Non)\n> 1 = Artış (EMT destekler)', fontweight='bold')
axs[1,1].set_ylabel('Fold Change (Sen / Normal)')
axs[1,1].legend()
axs[1,1].grid(axis='y', alpha=0.3)

ax6 = axs[1,2]
bar_positions = np.arange(len(ages))
colors_corr = ['#E74C3C' if c < 0 else '#2ECC71' for c in df_chrom['LaminB1_HMGB1_Corr']]
ax6.bar(bar_positions, df_chrom['LaminB1_HMGB1_Corr'], 0.5, color=colors_corr, edgecolor='black')
ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax6.set_xticks(bar_positions)
ax6.set_xticklabels(ages)
ax6.set_title('Test 3: Lamin B1 ↔ HMGB1 Korelasyonu\nNegatif = Çekirdek Çöküşüyle Sızıntı', fontweight='bold')
ax6.set_ylabel('Pearson Korelasyonu (r)')
ax6.grid(axis='y', alpha=0.3)

for k, row in enumerate(df_chrom.itertuples()):
    sig = "***" if row.HMGB1_p < 0.001 else "**" if row.HMGB1_p < 0.01 else "*" if row.HMGB1_p < 0.05 else "ns"
    ax6.text(k, row.LaminB1_HMGB1_Corr + 0.02, f"HMGB1\nFC={row.HMGB1_FC:.2f}\n{sig}", 
             ha='center', fontsize=7, color='red' if row.HMGB1_p < 0.05 else 'gray')

plt.suptitle("Sinclair 'Exdifferentiation' & EMT Hipotez Testi\n"
             "Senescence beta hücreleri kimlik kaybediyor mu? Epitel→Mezenkimal geçiş var mı?",
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()

out_dir = r"C:\Users\emir_\Documents\GitHub\Pancreas-Spatial-Senescence\saglikli grafikler 35-69"
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, 'sinclair_exdifferentiation_emt_test.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGrafik kaydedildi: {output_path}")

print("\n" + "=" * 80)
print("ÖZET: EXDİFFERENTIATION TESTİ (Beta Hücreleri)")
print("=" * 80)
for _, row in df_exdiff.iterrows():
    age = int(row['Yaş'])
    sycn_fc = row['SYCN (CH_5, Exocrine)_FC']
    k19_fc = row['Keratin19 (CH_14, Ductal)_FC']
    ins_fc = row['Insulin (CH_3, Kimlik)_FC']
    print(f"  {age} yaş: SYCN FC={sycn_fc:.2f} | K19 FC={k19_fc:.2f} | Insulin FC={ins_fc:.2f}")

print("\n" + "=" * 80)
print("ÖZET: EMT TESTİ (Tüm Hücreler)")
print("=" * 80)
for _, row in df_emt.iterrows():
    age = int(row['Yaş'])
    ecad_fc = row['E-cadherin (CH_28, Epitel)_FC']
    epcam_fc = row['EpCAM (CH_32, Epitel)_FC']
    cad11_fc = row['Cadherin11 (CH_17, Mezenk.)_FC']
    sma_fc = row['SMA (CH_37, Mezenk.)_FC']
    col_fc = row['CollagenI (CH_8, Fibrozis)_FC']
    print(f"  {age} yaş: E-cad FC={ecad_fc:.2f} | EpCAM FC={epcam_fc:.2f} | "
          f"Cad11 FC={cad11_fc:.2f} | SMA FC={sma_fc:.2f} | ColI FC={col_fc:.2f}")

print("\n" + "=" * 80)
print("ÖZET: KROMATİN ÇÖKÜŞÜ")
print("=" * 80)
for _, row in df_chrom.iterrows():
    age = int(row['Yaş'])
    corr = row['LaminB1_HMGB1_Corr']
    fc = row['HMGB1_FC']
    print(f"  {age} yaş: LaminB1↔HMGB1 r={corr:.3f} | HMGB1 FC(Sen/Non)={fc:.2f}")

print("\nTamamlandı!")
