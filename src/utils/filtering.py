import os
import pandas as pd
import numpy as np

from utils.config import ACTIVE_COHORT, COHORT_NAME

REGION_MAP = {
    'SNT282': {'Top': 'Head Superior', 'Bottom': 'Tail Superior'}, 
    'SNT484': {'Top': 'Head Superior', 'Bottom': 'Tail Superior'},
    'SNT227': {'Top': 'Unknown', 'Bottom': 'Head Inferior'}, 
    'SNT675': {'Top': 'Body Superior', 'Bottom': 'Unknown'},
    
    'SNT354': {'Top': 'Body Superior', 'Bottom': 'Head Inferior'}, 
    'SNT869': {'Top': 'Body Superior', 'Bottom': 'Head Inferior'},
    'SNT648': {'Top': 'Tail Superior', 'Bottom': 'Head Superior'}, 
    'SNT899': {'Top': 'Tail Superior', 'Bottom': 'Head Superior'},
    
    'SNT393': {'Top': 'Tail Superior', 'Bottom': 'Head Superior'},
    'SNT875': {'Top': 'Tail Superior', 'Bottom': 'Head Superior'},
}

def _load_raw_tissue(snt_id, age, target_region, silent=False):
    path = f"dataset/parquets/{age}/features_{snt_id}_age{age}.parquet"
    if not os.path.exists(path):
        path = f"dataset/parquets/features_{snt_id}_age{age}.parquet"
        if not os.path.exists(path):
            if not silent: print(f"ERROR: {path} not found!")
            return None
            
    df = pd.read_parquet(path)
    
    from sklearn.cluster import DBSCAN
    coords = df[['global_x', 'global_y']]
    clustering = DBSCAN(eps=150, min_samples=50, n_jobs=-1).fit(coords)
    df['dbscan_label'] = clustering.labels_
    cluster_counts = df['dbscan_label'].value_counts()
    valid_clusters = cluster_counts[(cluster_counts > 10000) & (cluster_counts.index != -1)].index
    df = df[df['dbscan_label'].isin(valid_clusters)].copy()

    if target_region and snt_id in REGION_MAP:
        mapping = REGION_MAP[snt_id]
        
        custom_cuts = {
            'SNT354': 28000,
            'SNT648': 22950,
            'SNT282': 25000,
            'SNT227': 26000,
            'SNT675': 29000,
            'SNT348': 33000,
            'SNT393': 25250,
        }
        
        if snt_id in custom_cuts:
            cut_y = custom_cuts[snt_id]
        else:
            hist, bins = np.histogram(df['global_y'], bins=50)
            cut_idx = np.argmin(hist[10:-10]) + 10
            cut_y = bins[cut_idx]
        
        is_top = target_region.lower() in mapping['Top'].lower() or mapping['Top'].lower() in target_region.lower()
        is_bottom = target_region.lower() in mapping['Bottom'].lower() or mapping['Bottom'].lower() in target_region.lower()
        
        if is_top and not is_bottom:
            df = df[df['global_y'] < cut_y]
            if not silent: print(f"{snt_id} filtered: Top region ({mapping['Top']}) extracted.")
        elif is_bottom and not is_top:
            df = df[df['global_y'] >= cut_y]
            if not silent: print(f"{snt_id} filtered: Bottom region ({mapping['Bottom']}) extracted.")
        else:
            if not silent: print(f"Warning: No clean match for '{target_region}' in {snt_id}. Using whole file.")
            
    return df

def load_and_filter_tissue(snt_id, age, target_region=None):
    if target_region is None:
        for row in ACTIVE_COHORT:
            if row[0] == snt_id:
                target_region = row[2]
                break
                
    df = _load_raw_tissue(snt_id, age, target_region, silent=False)
    if df is None: return None
            
    baseline = get_young_baseline(target_region)
    
    df['CH_16_robust'] = (df['CH_16'] - baseline['p16_med']) / baseline['p16_iqr']
    df['P16_robust'] = df['CH_16_robust']
    
    df['CH_20_robust'] = (df['CH_20'] - baseline['lamin_med']) / baseline['lamin_iqr']
    df['LaminB1_robust'] = df['CH_20_robust']
    
    if 'CH_9' in df.columns:
        df['CH_9_robust'] = (df['CH_9'] - baseline['p21_med']) / baseline['p21_iqr']
    
    df['Senescence_Score'] = df['P16_robust'] - df['LaminB1_robust']
    
    df['young_p16_95th']  = baseline['p16_robust_95th']
    df['young_p16_97th']  = baseline['p16_robust_97th']
    df['young_p16_99th']  = baseline['p16_robust_99th']
    df['young_lamin_5th'] = baseline['lamin_robust_5th']
    df['young_lamin_50th']= baseline['lamin_robust_50th']
    df['young_p21_95th']  = baseline['p21_robust_95th']
            
    return df

_YOUNG_BASELINES = {}
def get_young_baseline(target_region):
    global _YOUNG_BASELINES
    
    if target_region is None:
        target_region = 'Tail Superior'
        
    if target_region not in _YOUNG_BASELINES:
        _YOUNG_BASELINES[target_region] = calculate_young_reference_baseline(target_region)
    return _YOUNG_BASELINES[target_region]

def calculate_young_reference_baseline(target_region):
    print(f"\nCalculating specific young reference baseline for [{target_region}]...")
    files = [
        ('SNT354', 35),
        ('SNT648', 35) 
    ]
    
    df_list = []
    for snt, age in files:
        mapping = REGION_MAP.get(snt)
        if not mapping: continue
            
        is_top = target_region.lower() in mapping['Top'].lower() or mapping['Top'].lower() in target_region.lower()
        is_bottom = target_region.lower() in mapping['Bottom'].lower() or mapping['Bottom'].lower() in target_region.lower()
        
        if is_top or is_bottom:
            d = _load_raw_tissue(snt, age, target_region, silent=True)
            if d is not None:
                df_list.append(d)
            
    if not df_list:
        print(f"WARNING: No Young Reference files found for {target_region}. Returning Baseline [0, 1].")
        return {
            'p16_med': 0, 'p16_iqr': 1, 'lamin_med': 0, 'lamin_iqr': 1, 'p21_med': 0, 'p21_iqr': 1,
            'p16_robust_95th': 0, 'p16_robust_97th': 0, 'p16_robust_99th': 0,
            'lamin_robust_5th': 0, 'lamin_robust_50th': 0, 'p21_robust_95th': 0
        }
        
    df_young = pd.concat(df_list, ignore_index=True)
    
    p16_data = df_young['CH_16'].dropna()
    p16_med = p16_data.median()
    p16_iqr = p16_data.quantile(0.75) - p16_data.quantile(0.25)
    p16_95th = p16_data.quantile(0.95)
    p16_97th = p16_data.quantile(0.97)
    p16_99th = p16_data.quantile(0.99)
    if p16_iqr == 0: p16_iqr = 1e-6
    
    lamin_data = df_young['CH_20'].dropna()
    lamin_med = lamin_data.median()
    lamin_iqr = lamin_data.quantile(0.75) - lamin_data.quantile(0.25)
    lamin_5th = lamin_data.quantile(0.05)
    lamin_50th = lamin_data.quantile(0.50)
    if lamin_iqr == 0: lamin_iqr = 1e-6

    p21_data = df_young['CH_9'].dropna() if 'CH_9' in df_young.columns else pd.Series([0])
    p21_med = p21_data.median()
    p21_iqr = p21_data.quantile(0.75) - p21_data.quantile(0.25)
    p21_95th = p21_data.quantile(0.95)
    if p21_iqr == 0: p21_iqr = 1e-6
    
    print(f"Young Reference ({target_region}) Calculated: P16_med={p16_med:.2f}, P16_iqr={p16_iqr:.2f} | Lamin_med={lamin_med:.2f}, Lamin_iqr={lamin_iqr:.2f}")
    
    p16_robust_95th  = (p16_95th  - p16_med) / p16_iqr
    p16_robust_97th  = (p16_97th  - p16_med) / p16_iqr
    p16_robust_99th  = (p16_99th  - p16_med) / p16_iqr
    lamin_robust_5th  = (lamin_5th  - lamin_med) / lamin_iqr
    lamin_robust_50th = (lamin_50th - lamin_med) / lamin_iqr
    p21_robust_95th  = (p21_95th  - p21_med) / p21_iqr
    
    return {
        'p16_med': p16_med,
        'p16_iqr': p16_iqr,
        'lamin_med': lamin_med,
        'lamin_iqr': lamin_iqr,
        'p21_med': p21_med,
        'p21_iqr': p21_iqr,
        'p16_robust_95th':  p16_robust_95th,
        'p16_robust_97th':  p16_robust_97th,
        'p16_robust_99th':  p16_robust_99th,
        'lamin_robust_5th':  lamin_robust_5th,
        'lamin_robust_50th': lamin_robust_50th,
        'p21_robust_95th':  p21_robust_95th,
    }
