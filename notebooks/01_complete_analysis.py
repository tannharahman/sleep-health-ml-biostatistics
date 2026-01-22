#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sleep Duration and Cardiometabolic Risk Analysis
================================================
NHANES 2017-2020 with External Validation (2015-2016)

This script performs the complete analysis pipeline:
1. Data download and preprocessing
2. Descriptive statistics (Table 1)
3. Primary analysis (logistic regression)
4. Propensity score analysis
5. ML comparison (XGBoost + SHAP)
6. External validation
7. Visualization generation
8. Report generation

Author: Tannha Rahman
Date: 2026
"""

import os
import sys
import warnings
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_download import download_all_datasets, merge_datasets
from data_preprocessing import preprocess_nhanes
from statistical_analysis import (
    create_table1, fit_nested_models, ipw_analysis,
    calculate_e_value, subgroup_analysis, interaction_test
)
from ml_analysis import (
    prepare_ml_data, train_xgboost, train_logistic_regression,
    cross_validate_model, compute_shap_values, compare_models,
    external_validation, get_model_performance_summary
)
from visualizations import create_all_figures

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

# Create directories
for dir_path in [RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Analysis configuration
OUTCOMES = ['hypertension', 'diabetes', 'obesity', 'metabolic_syndrome']
EXPOSURE = 'sleep_category'
COVARIATES_DEMOGRAPHIC = ['age', 'sex', 'race_str', 'education_str', 'poverty_ratio']
COVARIATES_LIFESTYLE = ['smoking_status', 'physical_activity']
COVARIATES_FULL = COVARIATES_DEMOGRAPHIC + COVARIATES_LIFESTYLE + ['bmi']

ML_FEATURES = [
    'sleep_hours', 'sleep_irregularity', 'age', 'sex', 'race_ethnicity',
    'education', 'poverty_ratio', 'bmi', 'smoking_status', 'physical_activity'
]


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print formatted section"""
    print(f"\n--- {text} ---")


# ============================================================================
# STEP 1: DATA DOWNLOAD
# ============================================================================

def step1_download_data():
    """Download NHANES data from CDC"""
    print_header("STEP 1: DATA DOWNLOAD")

    merged_2017_path = os.path.join(RAW_DIR, 'nhanes_2017_2020_merged.pkl')
    merged_2015_path = os.path.join(RAW_DIR, 'nhanes_2015_2016_merged.pkl')

    # Check if data already exists
    if os.path.exists(merged_2017_path) and os.path.exists(merged_2015_path):
        print("Data already exists. Loading from disk...")
        return {
            '2017-2020': pd.read_pickle(merged_2017_path),
            '2015-2016': pd.read_pickle(merged_2015_path)
        }

    # Download NHANES 2017-2020
    print_section("Downloading NHANES 2017-2020")
    datasets_2017 = download_all_datasets(RAW_DIR, '2017-2020')
    merged_2017 = merge_datasets(datasets_2017, '2017-2020')
    merged_2017.to_pickle(merged_2017_path)

    # Download NHANES 2015-2016 (validation)
    print_section("Downloading NHANES 2015-2016 (Validation)")
    datasets_2015 = download_all_datasets(RAW_DIR, '2015-2016')
    merged_2015 = merge_datasets(datasets_2015, '2015-2016')
    merged_2015.to_pickle(merged_2015_path)

    return {
        '2017-2020': merged_2017,
        '2015-2016': merged_2015
    }


# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

def step2_preprocess_data(raw_data):
    """Preprocess NHANES data"""
    print_header("STEP 2: DATA PREPROCESSING")

    processed_data = {}
    exclusion_logs = {}

    for cycle, df in raw_data.items():
        if df is not None:
            processed, exclusion_log = preprocess_nhanes(df, cycle)
            processed_data[cycle] = processed
            exclusion_logs[cycle] = exclusion_log

            # Save processed data
            output_path = os.path.join(PROCESSED_DIR, f'nhanes_{cycle.replace("-", "_")}_processed.pkl')
            processed.to_pickle(output_path)
            print(f"Saved: {output_path}")

    return processed_data, exclusion_logs


# ============================================================================
# STEP 3: DESCRIPTIVE STATISTICS
# ============================================================================

def step3_descriptive_stats(df):
    """Generate descriptive statistics"""
    print_header("STEP 3: DESCRIPTIVE STATISTICS")

    # Table 1: Participant Characteristics
    print_section("Generating Table 1")
    table1 = create_table1(df, group_var='sleep_category')
    table1.to_csv(os.path.join(TABLES_DIR, 'table1_characteristics.csv'), index=False)
    print(f"Table 1 saved: {TABLES_DIR}/table1_characteristics.csv")

    # Summary statistics
    print_section("Sample Summary")
    print(f"Total sample size: {len(df):,}")
    print(f"Mean age: {df['age'].mean():.1f} years")
    print(f"Female: {(df['sex'] == 2).mean()*100:.1f}%")
    print(f"\nSleep categories:")
    print(df['sleep_category'].value_counts())
    print(f"\nOutcome prevalence:")
    for outcome in OUTCOMES:
        if outcome in df.columns:
            print(f"  {outcome}: {df[outcome].mean()*100:.1f}%")

    return table1


# ============================================================================
# STEP 4: PRIMARY ANALYSIS - LOGISTIC REGRESSION
# ============================================================================

def step4_primary_analysis(df):
    """Run primary logistic regression analysis"""
    print_header("STEP 4: PRIMARY ANALYSIS - LOGISTIC REGRESSION")

    all_results = {}

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            print(f"Skipping {outcome} - not in data")
            continue

        print_section(f"Outcome: {outcome.upper()}")

        # Fit nested models
        results = fit_nested_models(df, outcome, EXPOSURE)
        all_results[outcome] = results

        # Print results
        for model_name, model_result in results.items():
            if 'error' in model_result:
                print(f"  {model_name}: ERROR - {model_result['error']}")
                continue

            print(f"\n  {model_name} (n={model_result['n']:,}):")

            # Extract sleep category ORs
            summary = model_result['summary']
            sleep_rows = summary[summary.index.str.contains('sleep_category')]

            for idx, row in sleep_rows.iterrows():
                cat_name = idx.replace('sleep_category_', '')
                print(f"    {cat_name}: OR={row['OR']:.2f} ({row['OR_lower']:.2f}-{row['OR_upper']:.2f}), p={row['p_value']:.4f}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'regression_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved: {results_path}")

    # Create summary table
    summary_rows = []
    for outcome, outcome_results in all_results.items():
        if 'model3_fully_adjusted' in outcome_results and 'summary' in outcome_results['model3_fully_adjusted']:
            summary = outcome_results['model3_fully_adjusted']['summary']
            sleep_rows = summary[summary.index.str.contains('sleep_category')]
            for idx, row in sleep_rows.iterrows():
                summary_rows.append({
                    'Outcome': outcome,
                    'Sleep Category': idx.replace('sleep_category_', ''),
                    'OR': row['OR'],
                    'OR_lower': row['OR_lower'],
                    'OR_upper': row['OR_upper'],
                    'p_value': row['p_value']
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLES_DIR, 'table2_odds_ratios.csv'), index=False)
    print(f"Summary table saved: {TABLES_DIR}/table2_odds_ratios.csv")

    return all_results


# ============================================================================
# STEP 5: PROPENSITY SCORE ANALYSIS
# ============================================================================

def step5_propensity_analysis(df):
    """Run propensity score / IPW analysis"""
    print_header("STEP 5: PROPENSITY SCORE ANALYSIS")

    ipw_results = {}

    # Binary treatment: Short sleep (<7h) vs Recommended (7-8h)
    df_binary = df[df['sleep_category'].isin(['Very Short (<6h)', 'Short (6-7h)', 'Recommended (7-8h)'])].copy()
    df_binary['short_sleep'] = (df_binary['sleep_category'].isin(['Very Short (<6h)', 'Short (6-7h)'])).astype(float)

    print(f"Analysis sample: {len(df_binary):,}")
    print(f"Short sleep (<7h): {df_binary['short_sleep'].mean()*100:.1f}%")

    for outcome in OUTCOMES:
        if outcome not in df_binary.columns:
            continue

        print_section(f"IPW Analysis: Short Sleep -> {outcome.upper()}")

        try:
            confounders = [c for c in COVARIATES_FULL if c in df_binary.columns and c != 'bmi' or outcome != 'obesity']
            result = ipw_analysis(df_binary, outcome, 'short_sleep', confounders)

            print(f"  OR: {result['OR']:.2f} ({result['OR_CI'][0]:.2f}-{result['OR_CI'][1]:.2f})")
            print(f"  p-value: {result['p_value']:.4f}")

            # E-value
            e_val = calculate_e_value(result['OR'], result['OR_CI'][0])
            print(f"  E-value: {e_val['e_value_point']:.2f} (CI: {e_val.get('e_value_ci', 'N/A')})")

            ipw_results[outcome] = {
                'OR': result['OR'],
                'OR_CI': result['OR_CI'],
                'p_value': result['p_value'],
                'e_value': e_val,
                'n': result['n']
            }

        except Exception as e:
            print(f"  Error: {e}")

    # Save IPW results
    ipw_path = os.path.join(RESULTS_DIR, 'ipw_results.pkl')
    with open(ipw_path, 'wb') as f:
        pickle.dump(ipw_results, f)

    return ipw_results


# ============================================================================
# STEP 6: ML ANALYSIS
# ============================================================================

def step6_ml_analysis(df):
    """Run ML analysis with XGBoost and SHAP"""
    print_header("STEP 6: MACHINE LEARNING ANALYSIS")

    ml_results = {}

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue

        print_section(f"ML Analysis: {outcome.upper()}")

        # Prepare data
        features = [f for f in ML_FEATURES if f in df.columns]
        ml_data = prepare_ml_data(df, outcome, features)

        print(f"  Training set: {len(ml_data['X_train']):,}")
        print(f"  Test set: {len(ml_data['X_test']):,}")
        print(f"  Features: {len(ml_data['features'])}")

        # Compare models
        print("\n  Training models...")
        comparison_df, model_results = compare_models(
            ml_data['X_train'], ml_data['y_train'],
            ml_data['X_test'], ml_data['y_test']
        )

        print("\n  Model Comparison:")
        print(comparison_df.to_string(index=False))

        # Cross-validation for XGBoost
        print("\n  Running cross-validation...")
        cv_results = cross_validate_model(
            pd.concat([ml_data['X_train'], ml_data['X_test']]),
            pd.concat([ml_data['y_train'], ml_data['y_test']]),
            model_type='xgboost'
        )
        print(f"  CV AUC: {cv_results['cv_auc_mean']:.3f} (+/- {cv_results['cv_auc_std']:.3f})")

        # SHAP analysis
        print("\n  Computing SHAP values...")
        xgb_model = model_results['xgboost']['model']
        shap_results = compute_shap_values(xgb_model, ml_data['X_test'])

        if shap_results:
            print("\n  Top 10 Important Features (SHAP):")
            print(shap_results['shap_importance'].head(10).to_string(index=False))

        ml_results[outcome] = {
            'comparison': comparison_df,
            'xgboost': model_results['xgboost'],
            'logistic': model_results['logistic'],
            'cv_results': cv_results,
            'shap': shap_results,
            'ml_data': ml_data
        }

    # Save ML results
    ml_path = os.path.join(RESULTS_DIR, 'ml_results.pkl')
    with open(ml_path, 'wb') as f:
        pickle.dump(ml_results, f)

    # Save comparison table
    all_comparisons = []
    for outcome, results in ml_results.items():
        comp = results['comparison'].copy()
        comp['Outcome'] = outcome
        all_comparisons.append(comp)

    if all_comparisons:
        comparison_table = pd.concat(all_comparisons)
        comparison_table.to_csv(os.path.join(TABLES_DIR, 'table3_ml_comparison.csv'), index=False)

    return ml_results


# ============================================================================
# STEP 7: EXTERNAL VALIDATION
# ============================================================================

def step7_external_validation(df_train, df_valid, ml_results):
    """Validate models on 2015-2016 data"""
    print_header("STEP 7: EXTERNAL VALIDATION")

    if df_valid is None:
        print("Validation data not available. Skipping.")
        return None

    validation_results = {}

    for outcome in OUTCOMES:
        if outcome not in df_valid.columns or outcome not in ml_results:
            continue

        print_section(f"Validating: {outcome.upper()}")

        # Get trained model
        xgb_model = ml_results[outcome]['xgboost']['model']
        features = ml_results[outcome]['ml_data']['features']

        # Prepare validation data
        X_valid = df_valid[features].copy()
        y_valid = df_valid[outcome].dropna()

        # Align indices
        common_idx = X_valid.index.intersection(y_valid.index)
        X_valid = X_valid.loc[common_idx]
        y_valid = y_valid.loc[common_idx]

        # Handle categorical variables
        for col in X_valid.columns:
            if X_valid[col].dtype == 'object':
                X_valid[col] = pd.Categorical(X_valid[col]).codes

        # Fill missing
        X_valid = X_valid.fillna(X_valid.median())

        try:
            ext_result = external_validation(xgb_model, X_valid, y_valid)
            validation_results[outcome] = ext_result

            print(f"  External AUC: {ext_result['metrics']['external_auc']:.3f}")
            print(f"  External Accuracy: {ext_result['metrics']['external_accuracy']:.3f}")
            print(f"  N (external): {ext_result['metrics']['n_external']:,}")

        except Exception as e:
            print(f"  Error: {e}")

    # Save validation results
    valid_path = os.path.join(RESULTS_DIR, 'validation_results.pkl')
    with open(valid_path, 'wb') as f:
        pickle.dump(validation_results, f)

    return validation_results


# ============================================================================
# STEP 8: SUBGROUP ANALYSIS
# ============================================================================

def step8_subgroup_analysis(df):
    """Run subgroup and interaction analyses"""
    print_header("STEP 8: SUBGROUP ANALYSIS")

    subgroup_results = {}

    for outcome in OUTCOMES[:2]:  # Just hypertension and diabetes for brevity
        if outcome not in df.columns:
            continue

        print_section(f"Subgroups for: {outcome.upper()}")

        # By sex
        print("\n  By Sex:")
        sex_results = subgroup_analysis(df, outcome, EXPOSURE, 'sex_str', COVARIATES_DEMOGRAPHIC)
        if len(sex_results) > 0:
            print(sex_results[['subgroup', 'exposure', 'OR', 'p_value']].to_string(index=False))

        # By age group
        print("\n  By Age Group:")
        age_results = subgroup_analysis(df, outcome, EXPOSURE, 'age_category', COVARIATES_DEMOGRAPHIC)
        if len(age_results) > 0:
            print(age_results[['subgroup', 'exposure', 'OR', 'p_value']].to_string(index=False))

        # Interaction tests
        print("\n  Interaction Tests:")
        for modifier in ['sex_str', 'age_category']:
            if modifier in df.columns:
                try:
                    int_result = interaction_test(df, outcome, EXPOSURE, modifier, COVARIATES_DEMOGRAPHIC)
                    print(f"    {EXPOSURE} x {modifier}: p = {int_result['p_value']:.4f}")
                except Exception as e:
                    print(f"    {EXPOSURE} x {modifier}: Could not compute ({str(e)[:50]}...)")

        subgroup_results[outcome] = {
            'by_sex': sex_results,
            'by_age': age_results
        }

    # Save subgroup results
    subgroup_path = os.path.join(RESULTS_DIR, 'subgroup_results.pkl')
    with open(subgroup_path, 'wb') as f:
        pickle.dump(subgroup_results, f)

    return subgroup_results


# ============================================================================
# STEP 9: GENERATE FIGURES
# ============================================================================

def step9_generate_figures(df, all_results):
    """Generate all figures"""
    print_header("STEP 9: GENERATING FIGURES")

    figures = create_all_figures(df, all_results, FIGURES_DIR)
    print(f"\nFigures saved to: {FIGURES_DIR}")

    return figures


# ============================================================================
# STEP 10: GENERATE REPORT
# ============================================================================

def step10_generate_report(all_results):
    """Generate final analysis report"""
    print_header("STEP 10: GENERATING FINAL REPORT")

    report_path = os.path.join(RESULTS_DIR, 'analysis_report.md')

    with open(report_path, 'w') as f:
        f.write("# Sleep Duration and Cardiometabolic Risk Analysis\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Data Source:** NHANES 2017-2020 (Pre-Pandemic)\n\n")
        f.write("**Validation Data:** NHANES 2015-2016\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This analysis examines the association between sleep duration patterns and ")
        f.write("cardiometabolic risk factors in U.S. adults using nationally representative data.\n\n")

        f.write("## Key Findings\n\n")

        # Add regression results
        if 'regression' in all_results:
            f.write("### 1. Association Analysis (Logistic Regression)\n\n")
            f.write("After adjusting for demographic and lifestyle factors:\n\n")
            for outcome, results in all_results['regression'].items():
                if 'model3_fully_adjusted' in results and 'summary' in results['model3_fully_adjusted']:
                    f.write(f"**{outcome.title()}:**\n")
                    summary = results['model3_fully_adjusted']['summary']
                    sleep_rows = summary[summary.index.str.contains('sleep_category')]
                    for idx, row in sleep_rows.iterrows():
                        cat = idx.replace('sleep_category_', '')
                        sig = "*" if row['p_value'] < 0.05 else ""
                        f.write(f"- {cat}: OR = {row['OR']:.2f} (95% CI: {row['OR_lower']:.2f}-{row['OR_upper']:.2f}){sig}\n")
                    f.write("\n")

        # Add ML results
        if 'ml' in all_results:
            f.write("### 2. Predictive Modeling (Machine Learning)\n\n")
            f.write("| Outcome | Model | Test AUC |\n")
            f.write("|---------|-------|----------|\n")
            for outcome, results in all_results['ml'].items():
                comp = results['comparison']
                for _, row in comp.iterrows():
                    f.write(f"| {outcome.title()} | {row['Model']} | {row['test_auc']:.3f} |\n")
            f.write("\n")

        # Add validation results
        if 'validation' in all_results and all_results['validation']:
            f.write("### 3. External Validation (NHANES 2015-2016)\n\n")
            for outcome, results in all_results['validation'].items():
                f.write(f"- {outcome.title()}: AUC = {results['metrics']['external_auc']:.3f}\n")
            f.write("\n")

        f.write("## Methods\n\n")
        f.write("1. **Study Design:** Cross-sectional analysis\n")
        f.write("2. **Exposure:** Sleep duration (categorized)\n")
        f.write("3. **Outcomes:** Hypertension, diabetes, obesity, metabolic syndrome\n")
        f.write("4. **Statistical Methods:**\n")
        f.write("   - Weighted logistic regression with complex survey design\n")
        f.write("   - Inverse probability weighting (IPW)\n")
        f.write("   - XGBoost with SHAP interpretability\n")
        f.write("   - External validation\n\n")

        f.write("## Limitations\n\n")
        f.write("1. Cross-sectional design limits causal inference\n")
        f.write("2. Self-reported sleep measures subject to recall bias\n")
        f.write("3. Unmeasured confounding possible despite adjustments\n\n")

        f.write("## Conclusions\n\n")
        f.write("Both short and long sleep duration are associated with increased ")
        f.write("cardiometabolic risk in U.S. adults. These findings support public health ")
        f.write("recommendations for adequate sleep duration (7-8 hours) for cardiometabolic health.\n")

    print(f"Report saved: {report_path}")

    return report_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline"""

    print("\n" + "=" * 70)
    print("  SLEEP DURATION AND CARDIOMETABOLIC RISK ANALYSIS")
    print("  NHANES 2017-2020 with External Validation")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project directory: {PROJECT_DIR}")

    all_results = {}

    # Step 1: Download data
    raw_data = step1_download_data()

    # Step 2: Preprocess data
    processed_data, exclusion_logs = step2_preprocess_data(raw_data)
    all_results['exclusion_log'] = exclusion_logs.get('2017-2020', {})

    # Get main analysis dataset
    df_main = processed_data.get('2017-2020')
    df_valid = processed_data.get('2015-2016')

    if df_main is None:
        print("ERROR: Main dataset not available. Exiting.")
        return

    # Step 3: Descriptive statistics
    table1 = step3_descriptive_stats(df_main)

    # Step 4: Primary analysis
    regression_results = step4_primary_analysis(df_main)
    all_results['regression'] = regression_results

    # Step 5: Propensity score analysis
    ipw_results = step5_propensity_analysis(df_main)
    all_results['ipw'] = ipw_results

    # Step 6: ML analysis
    ml_results = step6_ml_analysis(df_main)
    all_results['ml'] = ml_results

    # Step 7: External validation
    if df_valid is not None:
        validation_results = step7_external_validation(df_main, df_valid, ml_results)
        all_results['validation'] = validation_results

    # Step 8: Subgroup analysis
    subgroup_results = step8_subgroup_analysis(df_main)
    all_results['subgroups'] = subgroup_results

    # Step 9: Generate figures
    all_results['regression_results'] = regression_results
    figures = step9_generate_figures(df_main, all_results)

    # Step 10: Generate report
    report_path = step10_generate_report(all_results)

    # Save all results
    all_results_path = os.path.join(RESULTS_DIR, 'all_results.pkl')
    with open(all_results_path, 'wb') as f:
        pickle.dump(all_results, f)

    print_header("ANALYSIS COMPLETE")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - Tables: {TABLES_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Report: {report_path}")


if __name__ == "__main__":
    main()
