"""
Statistical Analysis Module
Implements regression models, propensity scores, and inferential statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class SurveyWeightedAnalysis:
    """Class for survey-weighted statistical analysis"""

    def __init__(self, data: pd.DataFrame, weight_col: str = 'weight_mec'):
        self.data = data.copy()
        self.weight_col = weight_col
        self.results = {}

    def weighted_mean(self, var: str) -> Tuple[float, float]:
        """Calculate weighted mean and standard error"""
        weights = self.data[self.weight_col]
        values = self.data[var]

        mask = ~(values.isna() | weights.isna())
        w = weights[mask]
        v = values[mask]

        weighted_mean = np.average(v, weights=w)
        weighted_var = np.average((v - weighted_mean) ** 2, weights=w)
        weighted_se = np.sqrt(weighted_var / len(v))

        return weighted_mean, weighted_se

    def weighted_proportion(self, var: str, value: float = 1) -> Tuple[float, float]:
        """Calculate weighted proportion and standard error"""
        weights = self.data[self.weight_col]
        values = (self.data[var] == value).astype(float)

        mask = ~(values.isna() | weights.isna())
        w = weights[mask]
        v = values[mask]

        p = np.average(v, weights=w)
        se = np.sqrt(p * (1 - p) / len(v))

        return p, se


def create_table1(df: pd.DataFrame,
                  group_var: str = 'sleep_category',
                  weight_col: str = 'weight_mec') -> pd.DataFrame:
    """Generate Table 1: Participant Characteristics by Sleep Category"""

    # Define variables for Table 1
    continuous_vars = ['age', 'bmi', 'sleep_hours', 'sbp_mean', 'dbp_mean',
                       'hba1c', 'hdl', 'triglycerides', 'cmi']

    categorical_vars = ['sex_str', 'race_str', 'education_str',
                        'smoking_status', 'physical_activity',
                        'hypertension', 'diabetes', 'obesity', 'metabolic_syndrome']

    results = []
    groups = df[group_var].dropna().unique()
    groups = sorted([g for g in groups if pd.notna(g)])

    # Overall and by group
    for var in continuous_vars:
        if var not in df.columns:
            continue

        row = {'Variable': var, 'Type': 'continuous'}

        # Overall
        mask = df[var].notna() & df[weight_col].notna()
        if mask.sum() > 0:
            mean = np.average(df.loc[mask, var], weights=df.loc[mask, weight_col])
            std = np.sqrt(np.average((df.loc[mask, var] - mean)**2, weights=df.loc[mask, weight_col]))
            row['Overall'] = f"{mean:.1f} ({std:.1f})"

        # By group
        for group in groups:
            mask = (df[group_var] == group) & df[var].notna() & df[weight_col].notna()
            if mask.sum() > 0:
                mean = np.average(df.loc[mask, var], weights=df.loc[mask, weight_col])
                std = np.sqrt(np.average((df.loc[mask, var] - mean)**2, weights=df.loc[mask, weight_col]))
                row[str(group)] = f"{mean:.1f} ({std:.1f})"

        results.append(row)

    for var in categorical_vars:
        if var not in df.columns:
            continue

        categories = df[var].dropna().unique()

        for cat in sorted([c for c in categories if pd.notna(c)]):
            row = {'Variable': f"{var}: {cat}", 'Type': 'categorical'}

            # Overall
            mask = df[var].notna() & df[weight_col].notna()
            if mask.sum() > 0:
                prop = np.average(df.loc[mask, var] == cat, weights=df.loc[mask, weight_col])
                row['Overall'] = f"{prop*100:.1f}%"

            # By group
            for group in groups:
                mask = (df[group_var] == group) & df[var].notna() & df[weight_col].notna()
                if mask.sum() > 0:
                    prop = np.average(df.loc[mask, var] == cat, weights=df.loc[mask, weight_col])
                    row[str(group)] = f"{prop*100:.1f}%"

            results.append(row)

    return pd.DataFrame(results)


def fit_logistic_regression(df: pd.DataFrame,
                            outcome: str,
                            exposure: str = 'sleep_category',
                            covariates: List[str] = None,
                            reference_category: str = 'Recommended (7-8h)',
                            weight_col: str = 'weight_mec') -> Dict:
    """Fit weighted logistic regression model"""

    # Prepare data
    df_model = df.copy()

    # Create dummy variables for sleep category with reference
    if exposure == 'sleep_category':
        df_model = pd.get_dummies(df_model, columns=[exposure], drop_first=False)
        exposure_cols = [c for c in df_model.columns if c.startswith('sleep_category_')]
        ref_col = f'sleep_category_{reference_category}'
        exposure_cols = [c for c in exposure_cols if c != ref_col]
    else:
        exposure_cols = [exposure]

    # Build covariate list
    if covariates is None:
        covariates = []

    # Create categorical dummies for string covariates
    cat_covariates = []
    for cov in covariates:
        if df_model[cov].dtype == 'object' or cov.endswith('_str'):
            df_model = pd.get_dummies(df_model, columns=[cov], drop_first=True)
            cat_covariates.extend([c for c in df_model.columns if c.startswith(cov + '_')])
        else:
            cat_covariates.append(cov)

    # Remove reference columns
    cat_covariates = [c for c in cat_covariates if c in df_model.columns]

    # Build model
    all_predictors = exposure_cols + cat_covariates
    all_predictors = [p for p in all_predictors if p in df_model.columns]

    # Drop missing values
    model_vars = [outcome] + all_predictors + [weight_col]
    df_clean = df_model[model_vars].dropna()

    if len(df_clean) < 100:
        return {'error': 'Insufficient data after removing missing values'}

    # Fit model
    X = df_clean[all_predictors]
    X = sm.add_constant(X)
    y = df_clean[outcome]
    weights = df_clean[weight_col]

    try:
        model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
        result = model.fit()

        # Extract results
        summary_df = pd.DataFrame({
            'coef': result.params,
            'se': result.bse,
            'z': result.tvalues,
            'p_value': result.pvalues,
            'OR': np.exp(result.params),
            'OR_lower': np.exp(result.conf_int()[0]),
            'OR_upper': np.exp(result.conf_int()[1])
        })

        return {
            'model': result,
            'summary': summary_df,
            'n': len(df_clean),
            'aic': result.aic,
            'pseudo_r2': 1 - result.deviance / result.null_deviance
        }

    except Exception as e:
        return {'error': str(e)}


def fit_nested_models(df: pd.DataFrame,
                      outcome: str,
                      exposure: str = 'sleep_category') -> Dict:
    """Fit nested models (unadjusted, demographic-adjusted, fully adjusted)"""

    results = {}

    # Model 1: Unadjusted
    results['model1_unadjusted'] = fit_logistic_regression(
        df, outcome, exposure, covariates=[]
    )

    # Model 2: Demographic-adjusted
    demo_covariates = ['age', 'sex', 'race_str', 'education_str', 'poverty_ratio']
    demo_covariates = [c for c in demo_covariates if c in df.columns]
    results['model2_demographic'] = fit_logistic_regression(
        df, outcome, exposure, covariates=demo_covariates
    )

    # Model 3: Fully adjusted
    full_covariates = demo_covariates + ['smoking_status', 'physical_activity', 'bmi']
    full_covariates = [c for c in full_covariates if c in df.columns]

    # Don't adjust for BMI if outcome is obesity
    if outcome == 'obesity':
        full_covariates = [c for c in full_covariates if c != 'bmi']

    results['model3_fully_adjusted'] = fit_logistic_regression(
        df, outcome, exposure, covariates=full_covariates
    )

    return results


def calculate_propensity_scores(df: pd.DataFrame,
                                treatment: str,
                                confounders: List[str]) -> pd.Series:
    """Calculate propensity scores for treatment"""

    df_ps = df.copy()

    # Create dummies for categorical confounders
    for conf in confounders:
        if df_ps[conf].dtype == 'object':
            df_ps = pd.get_dummies(df_ps, columns=[conf], drop_first=True)

    # Get confounder columns
    ps_predictors = []
    for conf in confounders:
        if conf in df_ps.columns:
            ps_predictors.append(conf)
        else:
            ps_predictors.extend([c for c in df_ps.columns if c.startswith(conf + '_')])

    # Fit propensity score model
    df_clean = df_ps[[treatment] + ps_predictors].dropna()

    X = df_clean[ps_predictors]
    X = sm.add_constant(X)
    y = df_clean[treatment]

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    # Return propensity scores
    ps = result.predict(X)
    return pd.Series(ps.values, index=df_clean.index)


def ipw_analysis(df: pd.DataFrame,
                 outcome: str,
                 treatment: str,
                 confounders: List[str],
                 weight_col: str = 'weight_mec') -> Dict:
    """Inverse Probability Weighting analysis"""

    # Calculate propensity scores
    ps = calculate_propensity_scores(df, treatment, confounders)

    # Calculate IPW weights
    df_ipw = df.loc[ps.index].copy()
    df_ipw['ps'] = ps

    # Stabilized weights
    p_treated = df_ipw[treatment].mean()
    df_ipw['ipw'] = np.where(
        df_ipw[treatment] == 1,
        p_treated / df_ipw['ps'],
        (1 - p_treated) / (1 - df_ipw['ps'])
    )

    # Trim extreme weights (1st and 99th percentile)
    lower = df_ipw['ipw'].quantile(0.01)
    upper = df_ipw['ipw'].quantile(0.99)
    df_ipw['ipw_trimmed'] = df_ipw['ipw'].clip(lower, upper)

    # Combine with survey weights
    df_ipw['combined_weight'] = df_ipw[weight_col] * df_ipw['ipw_trimmed']

    # Fit weighted outcome model
    df_clean = df_ipw[[outcome, treatment, 'combined_weight']].dropna()

    X = sm.add_constant(df_clean[[treatment]])
    y = df_clean[outcome]
    weights = df_clean['combined_weight']

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    return {
        'model': result,
        'OR': np.exp(result.params[treatment]),
        'OR_CI': (np.exp(result.conf_int().loc[treatment, 0]),
                  np.exp(result.conf_int().loc[treatment, 1])),
        'p_value': result.pvalues[treatment],
        'propensity_scores': ps,
        'n': len(df_clean)
    }


def calculate_e_value(or_estimate: float, or_lower: float = None) -> Dict:
    """Calculate E-value for sensitivity to unmeasured confounding"""

    if or_estimate >= 1:
        e_value = or_estimate + np.sqrt(or_estimate * (or_estimate - 1))
    else:
        rr = 1 / or_estimate
        e_value = rr + np.sqrt(rr * (rr - 1))

    result = {'e_value_point': e_value}

    if or_lower is not None:
        if or_lower >= 1:
            e_value_ci = or_lower + np.sqrt(or_lower * (or_lower - 1))
        elif or_lower > 0:
            rr_ci = 1 / or_lower
            e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
        else:
            e_value_ci = 1

        result['e_value_ci'] = e_value_ci

    return result


def subgroup_analysis(df: pd.DataFrame,
                      outcome: str,
                      exposure: str,
                      subgroup_var: str,
                      covariates: List[str]) -> pd.DataFrame:
    """Perform subgroup analysis"""

    results = []
    subgroups = df[subgroup_var].dropna().unique()

    for subgroup in sorted(subgroups):
        df_sub = df[df[subgroup_var] == subgroup]

        model_result = fit_logistic_regression(
            df_sub, outcome, exposure, covariates
        )

        if 'error' not in model_result:
            # Extract exposure effect
            exposure_rows = model_result['summary'][
                model_result['summary'].index.str.contains('sleep_category')
            ]

            for idx, row in exposure_rows.iterrows():
                results.append({
                    'subgroup_var': subgroup_var,
                    'subgroup': subgroup,
                    'exposure': idx.replace('sleep_category_', ''),
                    'OR': row['OR'],
                    'OR_lower': row['OR_lower'],
                    'OR_upper': row['OR_upper'],
                    'p_value': row['p_value'],
                    'n': model_result['n']
                })

    return pd.DataFrame(results)


def interaction_test(df: pd.DataFrame,
                     outcome: str,
                     exposure: str,
                     modifier: str,
                     covariates: List[str]) -> Dict:
    """Test for interaction between exposure and effect modifier"""

    df_int = df.copy()

    # Create interaction term
    if df_int[exposure].dtype == 'object':
        df_int = pd.get_dummies(df_int, columns=[exposure], drop_first=True)
        exposure_cols = [c for c in df_int.columns if c.startswith(exposure + '_')]
    else:
        exposure_cols = [exposure]

    if df_int[modifier].dtype == 'object':
        df_int = pd.get_dummies(df_int, columns=[modifier], drop_first=True)
        modifier_cols = [c for c in df_int.columns if c.startswith(modifier + '_')]
    else:
        modifier_cols = [modifier]

    # Create interaction terms
    for exp_col in exposure_cols:
        for mod_col in modifier_cols:
            df_int[f'{exp_col}_x_{mod_col}'] = df_int[exp_col] * df_int[mod_col]

    interaction_cols = [c for c in df_int.columns if '_x_' in c]

    # Fit model without interaction
    all_predictors = exposure_cols + modifier_cols + covariates
    all_predictors = [p for p in all_predictors if p in df_int.columns]

    df_clean = df_int[[outcome] + all_predictors + interaction_cols + ['weight_mec']].dropna()

    X_reduced = sm.add_constant(df_clean[all_predictors])
    X_full = sm.add_constant(df_clean[all_predictors + interaction_cols])
    y = df_clean[outcome]
    weights = df_clean['weight_mec']

    # Fit both models
    model_reduced = sm.GLM(y, X_reduced, family=sm.families.Binomial(), freq_weights=weights).fit()
    model_full = sm.GLM(y, X_full, family=sm.families.Binomial(), freq_weights=weights).fit()

    # Likelihood ratio test
    lr_stat = model_reduced.deviance - model_full.deviance
    df_diff = len(interaction_cols)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

    return {
        'lr_statistic': lr_stat,
        'df': df_diff,
        'p_value': p_value,
        'interaction_significant': p_value < 0.05
    }


if __name__ == "__main__":
    # Test with sample data
    print("Statistical Analysis Module loaded successfully")
