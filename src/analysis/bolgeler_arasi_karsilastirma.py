import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.filtering import load_and_filter_tissue
from utils.config import ACTIVE_COHORT, COHORT_NAME, OUTPUT_DIR

def process_region(snt_id, age, region, label):
    df = load_and_filter_tissue(snt_id, age, target_region=region)
    if df is None:
        return None
    
    p16_thresh = df['young_p16_95th'].iloc[0]
    lamin_thresh = df['young_lamin_50th'].iloc[0]
    
    is_senescent = (df['CH_16_robust'] > p16_thresh) & (df['CH_20_robust'] < lamin_thresh)
    senescent_cells = df[is_senescent]
    total_cells = len(df)
    senescent_count = len(senescent_cells)
    senescence_ratio = (senescent_count / total_cells) * 100 if total_cells > 0 else 0
    
    print(f"{snt_id} ({label}): Total Cells={total_cells:,}, Senescent={senescent_count:,}, Ratio={senescence_ratio:.4f}%")
    
    return {
        'SNT_ID': snt_id,
        'Group_Label': label,
        'Total_Cells': total_cells,
        'Senescent_Count': senescent_count,
        'Senescence_Ratio': senescence_ratio
    }

def main():
    print(f"{COHORT_NAME} - SENESCENCE COMPARISON")
    print("==============================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    
    print("\n--- Analyzing Groups ---")
    for snt, age, reg, label in ACTIVE_COHORT:
        res = process_region(snt, age, reg, label)
        if res: results.append(res)
        
    if not results:
        print("\nNo data found! Cannot generate plots.")
        return
        
    df_results = pd.DataFrame(results)
    
    group_order = list(dict.fromkeys([item[3] for item in ACTIVE_COHORT]))
    df_results['Group_Label'] = pd.Categorical(df_results['Group_Label'], categories=group_order, ordered=True)
    df_results = df_results.sort_values('Group_Label')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x='Group_Label', y='Senescence_Ratio', hue='Group_Label', palette='viridis', legend=False)
    
    plt.title(f"{COHORT_NAME}\nSenescence Cell Ratio", fontsize=14, fontweight='bold')
    plt.ylabel("Senescence Ratio (%)", fontsize=12)
    plt.xlabel("Group", fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for index, row in df_results.reset_index(drop=True).iterrows():
        plt.text(index, row['Senescence_Ratio'] + 0.05, f"{row['Senescence_Ratio']:.2f}%", 
                 color='black', ha="center", fontweight='bold')
                 
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "senescence_ratio_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nPlot saved: {plot_path}")
    print("\nData Table:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
