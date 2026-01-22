"""
Machine Learning Analysis Module
Implements XGBoost prediction model with SHAP interpretability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                            recall_score, f1_score, brier_score_loss,
                            roc_curve, precision_recall_curve, confusion_matrix,
                            classification_report)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, use fallback if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, using GradientBoosting as fallback")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available, skipping interpretability analysis")


def prepare_ml_data(df: pd.DataFrame,
                    outcome: str,
                    features: List[str],
                    test_size: float = 0.2,
                    random_state: int = 42) -> Dict:
    """Prepare data for ML modeling"""

    df_ml = df.copy()

    # Handle categorical variables
    label_encoders = {}
    for col in features:
        if col in df_ml.columns and (df_ml[col].dtype == 'object' or df_ml[col].dtype.name == 'category'):
            le = LabelEncoder()
            df_ml[col] = df_ml[col].astype(str)
            df_ml[col] = le.fit_transform(df_ml[col].fillna('Missing'))
            label_encoders[col] = le

    # Select features that exist
    available_features = [f for f in features if f in df_ml.columns]

    # Drop rows with missing outcome
    df_ml = df_ml[df_ml[outcome].notna()]

    # Prepare X and y
    X = df_ml[available_features].copy()
    y = df_ml[outcome].copy()

    # Fill remaining missing values with median for numeric
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)
        else:
            # For any remaining object columns, fill with mode or 0
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': available_features,
        'label_encoders': label_encoders
    }


def train_logistic_regression(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_test: pd.DataFrame,
                              y_test: pd.Series) -> Dict:
    """Train logistic regression as baseline"""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_proba_train),
        'test_auc': roc_auc_score(y_test, y_pred_proba_test),
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'brier_score': brier_score_loss(y_test, y_pred_proba_test)
    }

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)

    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'y_pred_proba': y_pred_proba_test,
        'y_pred': y_pred_test,
        'roc_curve': {'fpr': fpr, 'tpr': tpr}
    }


def train_gradient_boosting(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            params: Dict = None) -> Dict:
    """Train Gradient Boosting model (fallback for XGBoost)"""

    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_proba_train),
        'test_auc': roc_auc_score(y_test, y_pred_proba_test),
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'brier_score': brier_score_loss(y_test, y_pred_proba_test)
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)

    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'y_pred_proba': y_pred_proba_test,
        'y_pred': y_pred_test,
        'roc_curve': {'fpr': fpr, 'tpr': tpr}
    }


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  params: Dict = None) -> Dict:
    """Train XGBoost model (or fallback to GradientBoosting)"""

    if not XGBOOST_AVAILABLE:
        print("  Using GradientBoosting as fallback for XGBoost")
        return train_gradient_boosting(X_train, y_train, X_test, y_test, params)

    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False
        }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_proba_train),
        'test_auc': roc_auc_score(y_test, y_pred_proba_test),
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'brier_score': brier_score_loss(y_test, y_pred_proba_test)
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)

    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'y_pred_proba': y_pred_proba_test,
        'y_pred': y_pred_test,
        'roc_curve': {'fpr': fpr, 'tpr': tpr}
    }


def cross_validate_model(X: pd.DataFrame,
                         y: pd.Series,
                         model_type: str = 'xgboost',
                         n_folds: int = 5) -> Dict:
    """Perform k-fold cross-validation"""

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    if model_type == 'xgboost' and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            use_label_encoder=False,
            eval_metric='auc'
        )
    elif model_type == 'gradient_boosting' or not XGBOOST_AVAILABLE:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        model = LogisticRegression(max_iter=1000, random_state=42)

    # Cross-validation predictions
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics per fold
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]

        fold_metrics.append({
            'fold': fold + 1,
            'auc': roc_auc_score(y_val_fold, y_val_pred_proba)
        })

    fold_df = pd.DataFrame(fold_metrics)

    return {
        'cv_auc_mean': fold_df['auc'].mean(),
        'cv_auc_std': fold_df['auc'].std(),
        'fold_metrics': fold_df,
        'y_pred_proba': y_pred_proba,
        'overall_auc': roc_auc_score(y, y_pred_proba)
    }


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 500) -> Dict:
    """Compute SHAP values for model interpretation"""

    if not SHAP_AVAILABLE:
        print("  SHAP not available, skipping interpretability analysis")
        return None

    try:
        # Sample data if too large
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        # Create SHAP explainer
        if hasattr(model, 'get_booster'):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        # Summary statistics
        shap_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'X_sample': X_sample,
            'shap_importance': shap_importance,
            'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
        }

    except Exception as e:
        print(f"  SHAP analysis error: {e}")
        return None


def compare_models(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> pd.DataFrame:
    """Compare logistic regression vs Gradient Boosting"""

    results = []

    # Logistic Regression
    lr_result = train_logistic_regression(X_train, y_train, X_test, y_test)
    results.append({
        'Model': 'Logistic Regression',
        **lr_result['metrics']
    })

    # Gradient Boosting / XGBoost
    gb_result = train_xgboost(X_train, y_train, X_test, y_test)
    model_name = 'XGBoost' if XGBOOST_AVAILABLE else 'Gradient Boosting'
    results.append({
        'Model': model_name,
        **gb_result['metrics']
    })

    comparison_df = pd.DataFrame(results)

    return comparison_df, {'logistic': lr_result, 'xgboost': gb_result}


def external_validation(model,
                        X_external: pd.DataFrame,
                        y_external: pd.Series,
                        model_type: str = 'xgboost') -> Dict:
    """Validate model on external dataset"""

    # Handle missing columns
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_external.columns

    for col in model_features:
        if col not in X_external.columns:
            X_external[col] = 0

    # Ensure same column order
    if hasattr(model, 'feature_names_in_'):
        X_external = X_external[list(model.feature_names_in_)]

    # Fill missing values
    for col in X_external.columns:
        if X_external[col].dtype in ['float64', 'int64']:
            X_external[col] = X_external[col].fillna(X_external[col].median())

    # Predictions
    y_pred_proba = model.predict_proba(X_external)[:, 1]
    y_pred = model.predict(X_external)

    # Metrics
    metrics = {
        'external_auc': roc_auc_score(y_external, y_pred_proba),
        'external_accuracy': accuracy_score(y_external, y_pred),
        'external_precision': precision_score(y_external, y_pred, zero_division=0),
        'external_recall': recall_score(y_external, y_pred, zero_division=0),
        'external_f1': f1_score(y_external, y_pred, zero_division=0),
        'external_brier': brier_score_loss(y_external, y_pred_proba),
        'n_external': len(y_external)
    }

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_external, y_pred_proba)

    # Calibration curve data
    prob_true, prob_pred = calibration_curve(y_external, y_pred_proba, n_bins=10, strategy='uniform')

    return {
        'metrics': metrics,
        'roc_curve': {'fpr': fpr, 'tpr': tpr},
        'calibration': {'prob_true': prob_true, 'prob_pred': prob_pred},
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def get_model_performance_summary(train_metrics: Dict,
                                  cv_metrics: Dict,
                                  external_metrics: Dict = None) -> pd.DataFrame:
    """Create summary table of model performance"""

    summary = []

    summary.append({
        'Dataset': 'Training',
        'AUC': train_metrics.get('train_auc', np.nan),
        'Accuracy': train_metrics.get('accuracy', np.nan),
        'Precision': train_metrics.get('precision', np.nan),
        'Recall': train_metrics.get('recall', np.nan),
        'F1': train_metrics.get('f1', np.nan),
        'Brier Score': train_metrics.get('brier_score', np.nan)
    })

    summary.append({
        'Dataset': 'Test',
        'AUC': train_metrics.get('test_auc', np.nan),
        'Accuracy': train_metrics.get('accuracy', np.nan),
        'Precision': train_metrics.get('precision', np.nan),
        'Recall': train_metrics.get('recall', np.nan),
        'F1': train_metrics.get('f1', np.nan),
        'Brier Score': train_metrics.get('brier_score', np.nan)
    })

    summary.append({
        'Dataset': f"CV ({cv_metrics.get('cv_auc_std', 0):.3f} SD)",
        'AUC': cv_metrics.get('cv_auc_mean', np.nan),
        'Accuracy': np.nan,
        'Precision': np.nan,
        'Recall': np.nan,
        'F1': np.nan,
        'Brier Score': np.nan
    })

    if external_metrics:
        summary.append({
            'Dataset': 'External Validation',
            'AUC': external_metrics.get('external_auc', np.nan),
            'Accuracy': external_metrics.get('external_accuracy', np.nan),
            'Precision': external_metrics.get('external_precision', np.nan),
            'Recall': external_metrics.get('external_recall', np.nan),
            'F1': external_metrics.get('external_f1', np.nan),
            'Brier Score': external_metrics.get('external_brier', np.nan)
        })

    return pd.DataFrame(summary)


if __name__ == "__main__":
    print("ML Analysis Module loaded successfully")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"SHAP available: {SHAP_AVAILABLE}")
