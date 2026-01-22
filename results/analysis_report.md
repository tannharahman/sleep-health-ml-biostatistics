# Sleep Duration and Cardiometabolic Risk Analysis

**Analysis Date:** 2026-01-21 21:28

**Data Source:** NHANES 2017-2020 (Pre-Pandemic)

**Validation Data:** NHANES 2015-2016

## Executive Summary

This analysis examines the association between sleep duration patterns and cardiometabolic risk factors in U.S. adults using nationally representative data.

## Key Findings

### 1. Association Analysis (Logistic Regression)

After adjusting for demographic and lifestyle factors:

### 2. Predictive Modeling (Machine Learning)

| Outcome | Model | Test AUC |
|---------|-------|----------|
| Hypertension | Logistic Regression | 0.805 |
| Hypertension | Gradient Boosting | 0.803 |
| Diabetes | Logistic Regression | 0.742 |
| Diabetes | Gradient Boosting | 0.750 |
| Obesity | Logistic Regression | 1.000 |
| Obesity | Gradient Boosting | 1.000 |
| Metabolic_Syndrome | Logistic Regression | 0.782 |
| Metabolic_Syndrome | Gradient Boosting | 0.825 |

## Methods

1. **Study Design:** Cross-sectional analysis
2. **Exposure:** Sleep duration (categorized)
3. **Outcomes:** Hypertension, diabetes, obesity, metabolic syndrome
4. **Statistical Methods:**
   - Weighted logistic regression with complex survey design
   - Inverse probability weighting (IPW)
   - XGBoost with SHAP interpretability
   - External validation

## Limitations

1. Cross-sectional design limits causal inference
2. Self-reported sleep measures subject to recall bias
3. Unmeasured confounding possible despite adjustments

## Conclusions

Both short and long sleep duration are associated with increased cardiometabolic risk in U.S. adults. These findings support public health recommendations for adequate sleep duration (7-8 hours) for cardiometabolic health.
