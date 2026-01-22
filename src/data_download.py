"""
NHANES Data Download Script
Downloads required datasets for sleep-cardiometabolic analysis
Using correct CDC URL structure
"""

import os
import requests
import pandas as pd
from io import BytesIO
from tqdm import tqdm

# Correct URL pattern for NHANES data
# https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename}.XPT

# NHANES 2017-March 2020 Pre-Pandemic datasets
DATASETS_2017_2020 = {
    'demographics': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.XPT',
    'sleep': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SLQ.XPT',
    'blood_pressure': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.XPT',
    'body_measures': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.XPT',
    'glycohemoglobin': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GHB.XPT',
    'hdl': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_HDL.XPT',
    'triglycerides': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_TRIGLY.XPT',
    'smoking': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SMQ.XPT',
    'alcohol': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_ALQ.XPT',
    'physical_activity': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_PAQ.XPT',
    'diabetes': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DIQ.XPT',
    'blood_pressure_q': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.XPT',
    'medical_conditions': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_MCQ.XPT',
}

# NHANES 2015-2016 datasets (validation set)
DATASETS_2015_2016 = {
    'demographics': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.XPT',
    'sleep': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.XPT',
    'blood_pressure': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.XPT',
    'body_measures': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.XPT',
    'glycohemoglobin': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GHB_I.XPT',
    'hdl': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HDL_I.XPT',
    'triglycerides': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/TRIGLY_I.XPT',
    'smoking': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.XPT',
    'alcohol': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/ALQ_I.XPT',
    'physical_activity': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.XPT',
    'diabetes': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DIQ_I.XPT',
    'blood_pressure_q': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPQ_I.XPT',
    'medical_conditions': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.XPT',
}


def download_xpt(url, save_path=None):
    """Download XPT file and return as DataFrame"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Check if we got actual data (not HTML)
        if b'<!DOCTYPE' in response.content[:100]:
            print(f"  Warning: Got HTML instead of data for {url}")
            return None

        df = pd.read_sas(BytesIO(response.content), format='xport')

        if save_path:
            df.to_pickle(save_path)

        return df

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def download_all_datasets(output_dir, cycle='2017-2020'):
    """Download all required NHANES datasets"""

    datasets = DATASETS_2017_2020 if cycle == '2017-2020' else DATASETS_2015_2016
    cycle_dir = os.path.join(output_dir, cycle.replace('-', '_'))
    os.makedirs(cycle_dir, exist_ok=True)

    downloaded = {}

    print(f"\nDownloading NHANES {cycle} datasets...")
    print("=" * 50)

    for name, url in tqdm(datasets.items(), desc=f"NHANES {cycle}"):
        save_path = os.path.join(cycle_dir, f"{name}.pkl")

        if os.path.exists(save_path):
            print(f"  {name}: Already exists, loading...")
            downloaded[name] = pd.read_pickle(save_path)
        else:
            print(f"  {name}: Downloading...")
            df = download_xpt(url, save_path)
            if df is not None:
                downloaded[name] = df
                print(f"    -> {len(df)} records downloaded")
            else:
                print(f"    -> FAILED")

    return downloaded


def merge_datasets(datasets, cycle='2017-2020'):
    """Merge all datasets on SEQN (participant ID)"""

    if 'demographics' not in datasets:
        raise ValueError("Demographics dataset is required")

    # Start with demographics as base
    merged = datasets['demographics'].copy()
    print(f"\nMerging datasets for {cycle}...")
    print(f"  Base (demographics): {len(merged)} records")

    # Merge other datasets
    for name, df in datasets.items():
        if name != 'demographics':
            merged = merged.merge(df, on='SEQN', how='left', suffixes=('', f'_{name}'))
            print(f"  + {name}: {len(df)} records")

    print(f"\n  Final merged dataset: {len(merged)} records, {len(merged.columns)} columns")

    return merged


if __name__ == "__main__":
    import sys

    # Default output directory
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"

    # Download both cycles
    datasets_2017 = download_all_datasets(output_dir, '2017-2020')
    datasets_2015 = download_all_datasets(output_dir, '2015-2016')

    # Merge datasets
    if datasets_2017:
        merged_2017 = merge_datasets(datasets_2017, '2017-2020')
        merged_2017.to_pickle(os.path.join(output_dir, 'nhanes_2017_2020_merged.pkl'))
        print(f"\nSaved merged 2017-2020 data: {len(merged_2017)} records")

    if datasets_2015:
        merged_2015 = merge_datasets(datasets_2015, '2015-2016')
        merged_2015.to_pickle(os.path.join(output_dir, 'nhanes_2015_2016_merged.pkl'))
        print(f"Saved merged 2015-2016 data: {len(merged_2015)} records")
