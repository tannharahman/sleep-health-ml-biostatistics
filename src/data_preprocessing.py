"""
NHANES Data Preprocessing Script
Cleans and prepares data for sleep-cardiometabolic analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def select_and_rename_variables(df: pd.DataFrame, cycle: str = '2017-2020') -> pd.DataFrame:
    """Select relevant variables and apply consistent naming"""

    # Variable mapping (NHANES code -> analysis name)
    # Note: Variable names may differ slightly between cycles

    var_mapping = {
        # Demographics
        'SEQN': 'id',
        'RIAGENDR': 'sex',
        'RIDAGEYR': 'age',
        'RIDRETH3': 'race_ethnicity',
        'DMDEDUC2': 'education',
        'INDFMPIR': 'poverty_ratio',

        # Survey weights
        'WTMECPRP' if cycle == '2017-2020' else 'WTMEC2YR': 'weight_mec',
        'SDMVPSU': 'psu',
        'SDMVSTRA': 'stratum',

        # Sleep variables
        'SLQ300': 'sleep_weekday_hrs',
        'SLQ310': 'sleep_weekend_hrs',
        'SLD012': 'sleep_hours',  # Average hours of sleep
        'SLQ050': 'sleep_disorder',
        'SLQ120': 'sleepy_frequency',

        # Blood pressure
        'BPXOSY1': 'sbp_1',
        'BPXOSY2': 'sbp_2',
        'BPXOSY3': 'sbp_3',
        'BPXODI1': 'dbp_1',
        'BPXODI2': 'dbp_2',
        'BPXODI3': 'dbp_3',

        # Body measures
        'BMXBMI': 'bmi',
        'BMXWAIST': 'waist_circumference',
        'BMXHT': 'height',
        'BMXWT': 'weight',

        # Lab values
        'LBXGH': 'hba1c',
        'LBDHDD': 'hdl',
        'LBXTR': 'triglycerides',

        # Questionnaire data
        'SMQ020': 'smoked_100',
        'SMQ040': 'current_smoker',
        'ALQ130': 'alcohol_drinks_per_day',
        'PAQ605': 'vigorous_work',
        'PAQ650': 'vigorous_recreation',
        'PAD680': 'sedentary_minutes',

        # Medical conditions
        'BPQ020': 'told_hypertension',
        'BPQ040A': 'taking_bp_meds',
        'DIQ010': 'told_diabetes',
        'DIQ050': 'taking_insulin',
        'DIQ070': 'taking_diabetes_meds',
    }

    # Handle alternative variable names for 2015-2016
    if cycle == '2015-2016':
        # Blood pressure variables have different names
        var_mapping.update({
            'BPXSY1': 'sbp_1',
            'BPXSY2': 'sbp_2',
            'BPXSY3': 'sbp_3',
            'BPXDI1': 'dbp_1',
            'BPXDI2': 'dbp_2',
            'BPXDI3': 'dbp_3',
            'WTMEC2YR': 'weight_mec',
        })

    # Select available variables
    available_vars = {}
    for nhanes_var, analysis_var in var_mapping.items():
        if nhanes_var in df.columns:
            available_vars[nhanes_var] = analysis_var
        else:
            # Try to find similar variable
            similar = [c for c in df.columns if nhanes_var[:4] in c]
            if similar:
                print(f"  Note: {nhanes_var} not found, using {similar[0]}")
                available_vars[similar[0]] = analysis_var

    # Select and rename
    df_selected = df[list(available_vars.keys())].copy()
    df_selected.columns = [available_vars[c] for c in df_selected.columns]

    return df_selected


def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived variables for analysis"""

    df = df.copy()

    # Convert bytes columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Try to convert bytes to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    # =========== EXPOSURE: SLEEP VARIABLES ===========

    # Sleep duration (use SLD012 if available, otherwise calculate)
    if 'sleep_hours' not in df.columns or df['sleep_hours'].isna().all():
        # Calculate weighted average of weekday and weekend sleep
        if 'sleep_weekday_hrs' in df.columns and 'sleep_weekend_hrs' in df.columns:
            df['sleep_hours'] = (df['sleep_weekday_hrs'] * 5 + df['sleep_weekend_hrs'] * 2) / 7

    # Sleep duration categories
    def categorize_sleep(hours):
        if pd.isna(hours):
            return np.nan
        elif hours < 6:
            return 'Very Short (<6h)'
        elif hours < 7:
            return 'Short (6-7h)'
        elif hours <= 8:
            return 'Recommended (7-8h)'
        elif hours <= 9:
            return 'Long (8-9h)'
        else:
            return 'Very Long (>9h)'

    df['sleep_category'] = df['sleep_hours'].apply(categorize_sleep)

    # Sleep irregularity (difference between weekday and weekend)
    if 'sleep_weekday_hrs' in df.columns and 'sleep_weekend_hrs' in df.columns:
        df['sleep_irregularity'] = abs(df['sleep_weekend_hrs'] - df['sleep_weekday_hrs'])
        df['sleep_irregular'] = (df['sleep_irregularity'] >= 2).astype(float)

    # =========== OUTCOMES ===========

    # Mean blood pressure (average of available readings)
    sbp_cols = [c for c in df.columns if c.startswith('sbp_')]
    dbp_cols = [c for c in df.columns if c.startswith('dbp_')]

    if sbp_cols:
        df['sbp_mean'] = df[sbp_cols].mean(axis=1, skipna=True)
    if dbp_cols:
        df['dbp_mean'] = df[dbp_cols].mean(axis=1, skipna=True)

    # Hypertension: SBP >= 140 OR DBP >= 90 OR taking BP meds
    df['hypertension'] = (
        (df['sbp_mean'] >= 140) |
        (df['dbp_mean'] >= 90) |
        (df['taking_bp_meds'] == 1) |
        (df['told_hypertension'] == 1)
    ).astype(float)

    # Diabetes: HbA1c >= 6.5% OR told by doctor OR taking meds/insulin
    df['diabetes'] = (
        (df['hba1c'] >= 6.5) |
        (df['told_diabetes'] == 1) |
        (df['taking_insulin'] == 1) |
        (df['taking_diabetes_meds'] == 1)
    ).astype(float)

    # Obesity: BMI >= 30
    df['obesity'] = (df['bmi'] >= 30).astype(float)

    # Cardiometabolic Index (CMI) = (Waist/Height) * (TG/HDL)
    if all(c in df.columns for c in ['waist_circumference', 'height', 'triglycerides', 'hdl']):
        df['waist_height_ratio'] = df['waist_circumference'] / df['height']
        df['tg_hdl_ratio'] = df['triglycerides'] / df['hdl']
        df['cmi'] = df['waist_height_ratio'] * df['tg_hdl_ratio']

        # CMI categories (based on tertiles)
        df['cmi_high'] = (df['cmi'] > df['cmi'].quantile(0.67)).astype(float)

    # Metabolic syndrome (simplified definition)
    # 3 or more of: high waist, high TG, low HDL, high BP, high glucose
    ms_criteria = pd.DataFrame()
    ms_criteria['high_waist'] = ((df['sex'] == 1) & (df['waist_circumference'] >= 102)) | \
                                 ((df['sex'] == 2) & (df['waist_circumference'] >= 88))
    ms_criteria['high_tg'] = df['triglycerides'] >= 150
    ms_criteria['low_hdl'] = ((df['sex'] == 1) & (df['hdl'] < 40)) | \
                              ((df['sex'] == 2) & (df['hdl'] < 50))
    ms_criteria['high_bp'] = (df['sbp_mean'] >= 130) | (df['dbp_mean'] >= 85) | (df['taking_bp_meds'] == 1)
    ms_criteria['high_glucose'] = df['hba1c'] >= 5.7

    df['metabolic_syndrome_count'] = ms_criteria.sum(axis=1)
    df['metabolic_syndrome'] = (df['metabolic_syndrome_count'] >= 3).astype(float)

    # =========== COVARIATES ===========

    # Age categories
    df['age_category'] = pd.cut(df['age'],
                                 bins=[0, 40, 60, 120],
                                 labels=['<40', '40-60', '>60'])

    # Sex (recode to string)
    df['sex_str'] = df['sex'].map({1: 'Male', 2: 'Female'})

    # Race/ethnicity (recode)
    race_map = {
        1: 'Mexican American',
        2: 'Other Hispanic',
        3: 'Non-Hispanic White',
        4: 'Non-Hispanic Black',
        6: 'Non-Hispanic Asian',
        7: 'Other/Multi'
    }
    df['race_str'] = df['race_ethnicity'].map(race_map)

    # Education (recode)
    edu_map = {
        1: 'Less than 9th grade',
        2: '9-11th grade',
        3: 'High school/GED',
        4: 'Some college',
        5: 'College graduate+'
    }
    df['education_str'] = df['education'].map(edu_map)

    # Smoking status
    df['smoking_status'] = 'Never'
    df.loc[df['smoked_100'] == 1, 'smoking_status'] = 'Former'
    df.loc[(df['smoked_100'] == 1) & (df['current_smoker'].isin([1, 2])), 'smoking_status'] = 'Current'

    # Physical activity (any vigorous activity)
    df['physical_activity'] = (
        (df['vigorous_work'] == 1) | (df['vigorous_recreation'] == 1)
    ).map({True: 'Active', False: 'Inactive'})

    # Poverty status
    df['poverty_status'] = pd.cut(df['poverty_ratio'],
                                   bins=[0, 1, 2, 5, 10],
                                   labels=['Below poverty', 'Near poverty', 'Above poverty', 'High income'])

    return df


def apply_exclusion_criteria(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Apply study exclusion criteria and return filtered data with counts"""

    exclusion_log = {'initial': len(df)}

    # 1. Age >= 20 (adults only)
    df = df[df['age'] >= 20]
    exclusion_log['after_age_filter'] = len(df)

    # 2. Non-missing sleep data
    df = df[df['sleep_hours'].notna()]
    exclusion_log['after_sleep_filter'] = len(df)

    # 3. Valid survey weights
    df = df[df['weight_mec'] > 0]
    exclusion_log['after_weight_filter'] = len(df)

    # 4. Plausible sleep values (2-16 hours)
    df = df[(df['sleep_hours'] >= 2) & (df['sleep_hours'] <= 16)]
    exclusion_log['after_plausible_sleep'] = len(df)

    # Calculate exclusion counts
    exclusion_log['excluded_age'] = exclusion_log['initial'] - exclusion_log['after_age_filter']
    exclusion_log['excluded_sleep'] = exclusion_log['after_age_filter'] - exclusion_log['after_sleep_filter']
    exclusion_log['excluded_weight'] = exclusion_log['after_sleep_filter'] - exclusion_log['after_weight_filter']
    exclusion_log['excluded_plausible'] = exclusion_log['after_weight_filter'] - exclusion_log['after_plausible_sleep']
    exclusion_log['final'] = len(df)

    return df, exclusion_log


def preprocess_nhanes(raw_data: pd.DataFrame, cycle: str = '2017-2020') -> Tuple[pd.DataFrame, dict]:
    """Main preprocessing pipeline"""

    print(f"\n{'='*60}")
    print(f"Preprocessing NHANES {cycle} Data")
    print('='*60)

    # Step 1: Select and rename variables
    print("\n1. Selecting and renaming variables...")
    df = select_and_rename_variables(raw_data, cycle)
    print(f"   Selected {len(df.columns)} variables")

    # Step 2: Create derived variables
    print("\n2. Creating derived variables...")
    df = create_derived_variables(df)
    print(f"   Created exposure, outcome, and covariate variables")

    # Step 3: Apply exclusion criteria
    print("\n3. Applying exclusion criteria...")
    df, exclusion_log = apply_exclusion_criteria(df)

    print(f"\n   Exclusion Summary:")
    print(f"   - Initial: {exclusion_log['initial']}")
    print(f"   - Excluded (age < 20): {exclusion_log['excluded_age']}")
    print(f"   - Excluded (missing sleep): {exclusion_log['excluded_sleep']}")
    print(f"   - Excluded (invalid weights): {exclusion_log['excluded_weight']}")
    print(f"   - Excluded (implausible sleep): {exclusion_log['excluded_plausible']}")
    print(f"   - Final analytic sample: {exclusion_log['final']}")

    # Step 4: Summary of key variables
    print("\n4. Key Variable Summary:")
    print(f"   - Sleep hours: mean={df['sleep_hours'].mean():.1f}, sd={df['sleep_hours'].std():.1f}")
    print(f"   - Hypertension prevalence: {df['hypertension'].mean()*100:.1f}%")
    print(f"   - Diabetes prevalence: {df['diabetes'].mean()*100:.1f}%")
    print(f"   - Obesity prevalence: {df['obesity'].mean()*100:.1f}%")
    if 'metabolic_syndrome' in df.columns:
        print(f"   - Metabolic syndrome prevalence: {df['metabolic_syndrome'].mean()*100:.1f}%")

    return df, exclusion_log


if __name__ == "__main__":
    # Example usage
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/processed"

    # Load merged data
    for cycle in ['2017-2020', '2015-2016']:
        merged_path = f"{data_dir}/nhanes_{cycle.replace('-', '_')}_merged.pkl"
        try:
            raw_data = pd.read_pickle(merged_path)
            processed_data, exclusion_log = preprocess_nhanes(raw_data, cycle)

            # Save processed data
            output_path = f"{output_dir}/nhanes_{cycle.replace('-', '_')}_processed.pkl"
            processed_data.to_pickle(output_path)
            print(f"\nSaved processed data to {output_path}")
        except FileNotFoundError:
            print(f"File not found: {merged_path}")
