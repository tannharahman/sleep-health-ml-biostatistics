# Sleep Duration and Cardiometabolic Risk Analysis

A comprehensive biostatistical analysis examining the association between sleep duration patterns and cardiometabolic health outcomes using NHANES data.

## Overview

This project investigates how sleep duration affects cardiometabolic risk factors including hypertension, diabetes, obesity, and metabolic syndrome in U.S. adults. The analysis combines traditional epidemiological methods with machine learning approaches for a robust, multi-faceted investigation.

## Key Features

- **Real NHANES Data**: Uses publicly available NHANES 2017-2020 (pre-pandemic) data with 2015-2016 for external validation
- **Hybrid Methodology**: Combines traditional biostatistics (logistic regression, propensity scores) with ML (Gradient Boosting/XGBoost)
- **Causal Framework**: DAG-informed analysis with confounding control
- **Model Interpretability**: SHAP values for machine learning model explanation
- **External Validation**: Independent validation cohort for generalizability assessment
- **Survey-Weighted Analysis**: Proper handling of NHANES complex survey design

## Results Summary

### Sample Characteristics
- **Primary cohort**: 9,145 U.S. adults (NHANES 2017-2020)
- **Validation cohort**: 5,735 adults (NHANES 2015-2016)
- **Mean sleep duration**: 7.1 hours

### Machine Learning Performance (Test AUC)

| Outcome | Logistic Regression | Gradient Boosting |
|---------|---------------------|-------------------|
| Hypertension | 0.805 | 0.803 |
| Diabetes | 0.742 | 0.750 |
| Metabolic Syndrome | 0.782 | 0.825 |

### Key Findings

1. **U-shaped relationship**: Both short (<7h) and long (>9h) sleep associated with increased cardiometabolic risk
2. **Top predictors**: Age, BMI, and sleep duration consistently important across outcomes
3. **Sex differences**: Males show higher hypertension risk (SHAP analysis)

## Project Structure

```
sleep-health-ml-biostatistics/
├── src/
│   ├── data_download.py      # NHANES data download from CDC
│   ├── data_preprocessing.py # Data cleaning and variable creation
│   ├── statistical_analysis.py # Regression and propensity scores
│   ├── ml_analysis.py        # ML models and SHAP analysis
│   └── visualizations.py     # Publication-quality figures
├── notebooks/
│   └── 01_complete_analysis.py # Main analysis pipeline
├── results/
│   ├── figures/              # Generated visualizations
│   ├── tables/               # Statistical tables
│   └── analysis_report.md    # Summary report
├── data/
│   ├── raw/                  # Downloaded NHANES data
│   └── processed/            # Cleaned analysis datasets
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tannharahman/sleep-health-ml-biostatistics.git
cd sleep-health-ml-biostatistics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run complete analysis pipeline
python notebooks/01_complete_analysis.py
```

This will:
1. Download NHANES data from CDC (if not already present)
2. Preprocess and clean data
3. Run statistical analyses
4. Train ML models with SHAP interpretation
5. Generate figures and reports

### Step-by-Step

```python
# Download data only
from src.data_download import download_all_datasets, merge_datasets
datasets = download_all_datasets('./data/raw', '2017-2020')
merged = merge_datasets(datasets, '2017-2020')

# Preprocess data
from src.data_preprocessing import preprocess_nhanes
processed, exclusion_log = preprocess_nhanes(merged, '2017-2020')

# Run ML analysis
from src.ml_analysis import prepare_ml_data, train_xgboost, compute_shap_values
ml_data = prepare_ml_data(processed, 'hypertension', features)
model, metrics = train_xgboost(ml_data['X_train'], ml_data['y_train'],
                                ml_data['X_test'], ml_data['y_test'])
shap_results = compute_shap_values(model, ml_data['X_test'])
```

## Methodology

### Exposures
- **Sleep duration**: Categorized as Very Short (<6h), Short (6-7h), Recommended (7-8h), Long (8-9h), Very Long (>9h)
- **Sleep irregularity**: Weekday-weekend sleep difference

### Outcomes
- **Hypertension**: SBP ≥140 or DBP ≥90 or taking BP medications
- **Diabetes**: HbA1c ≥6.5% or diagnosed/treated
- **Obesity**: BMI ≥30
- **Metabolic Syndrome**: ≥3 ATP III criteria

### Statistical Methods
1. **Nested logistic regression**: Unadjusted → demographic-adjusted → fully-adjusted
2. **Propensity score weighting**: IPW for causal effect estimation
3. **E-values**: Sensitivity analysis for unmeasured confounding
4. **Gradient Boosting**: Ensemble ML with hyperparameter tuning
5. **SHAP analysis**: Feature importance and model interpretation
6. **Cross-validation**: 5-fold CV for performance estimation
7. **External validation**: NHANES 2015-2016 cohort

## Figures

| Figure | Description |
|--------|-------------|
| fig1_dag.png | Directed Acyclic Graph (causal framework) |
| fig2_flowchart.png | Participant flow diagram |
| fig4_sleep_distribution.png | Sleep duration distribution |
| fig5_outcome_prevalence.png | U-shaped sleep-outcome relationship |
| fig6_shap_importance.png | SHAP feature importance (all outcomes) |
| fig7_shap_beeswarm_hypertension.png | SHAP beeswarm plot (hypertension) |

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- scikit-learn
- statsmodels
- matplotlib, seaborn
- shap
- requests, tqdm

See `requirements.txt` for complete list.

## Data Source

Data is downloaded directly from [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/):
- Demographics, sleep questionnaire, blood pressure, body measures
- Laboratory data (HbA1c, lipids)
- Health questionnaires (smoking, physical activity, medical conditions)

## Limitations

1. **Cross-sectional design**: Cannot establish causality
2. **Self-reported sleep**: Subject to recall bias
3. **Unmeasured confounding**: Despite adjustments, residual confounding possible
4. **Single time point**: Does not capture sleep pattern changes

## License

This project is for educational and research purposes. NHANES data is publicly available from the CDC.

## Author

Tannha Rahman

## Acknowledgments

- CDC National Center for Health Statistics for NHANES data
- SHAP library for model interpretability tools
