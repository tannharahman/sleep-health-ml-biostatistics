"""
Visualization Module
Creates publication-quality figures for the analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color palette
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#3498DB',
    'quaternary': '#2ECC71',
    'neutral': '#95A5A6',
    'sleep_categories': ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB', '#9B59B6']
}


def create_dag_figure(save_path: str = None) -> plt.Figure:
    """Create DAG (Directed Acyclic Graph) visualization"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Node positions
    nodes = {
        'Sleep\nPatterns': (0.2, 0.5),
        'Cardiometabolic\nOutcomes': (0.8, 0.5),
        'Age': (0.3, 0.85),
        'Sex': (0.5, 0.9),
        'Race/\nEthnicity': (0.7, 0.85),
        'SES': (0.15, 0.75),
        'Smoking': (0.35, 0.15),
        'Physical\nActivity': (0.5, 0.1),
        'Alcohol': (0.65, 0.15),
        'BMI': (0.5, 0.5),
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        if name in ['Sleep\nPatterns', 'Cardiometabolic\nOutcomes']:
            color = COLORS['primary'] if name == 'Sleep\nPatterns' else COLORS['secondary']
            circle = plt.Circle((x, y), 0.08, color=color, alpha=0.8)
            ax.add_patch(circle)
            ax.annotate(name, (x, y), ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
        else:
            rect = plt.Rectangle((x-0.06, y-0.04), 0.12, 0.08,
                                 color=COLORS['neutral'], alpha=0.6)
            ax.add_patch(rect)
            ax.annotate(name, (x, y), ha='center', va='center', fontsize=8)

    # Draw edges (arrows)
    edges = [
        ('Sleep\nPatterns', 'Cardiometabolic\nOutcomes', 'causal'),
        ('Age', 'Sleep\nPatterns', 'confounder'),
        ('Age', 'Cardiometabolic\nOutcomes', 'confounder'),
        ('Sex', 'Sleep\nPatterns', 'confounder'),
        ('Sex', 'Cardiometabolic\nOutcomes', 'confounder'),
        ('Race/\nEthnicity', 'Cardiometabolic\nOutcomes', 'confounder'),
        ('SES', 'Sleep\nPatterns', 'confounder'),
        ('SES', 'Cardiometabolic\nOutcomes', 'confounder'),
        ('Smoking', 'Cardiometabolic\nOutcomes', 'confounder'),
        ('Physical\nActivity', 'Cardiometabolic\nOutcomes', 'confounder'),
        ('Sleep\nPatterns', 'BMI', 'mediator'),
        ('BMI', 'Cardiometabolic\nOutcomes', 'mediator'),
    ]

    for start, end, edge_type in edges:
        start_pos = nodes[start]
        end_pos = nodes[end]

        color = COLORS['primary'] if edge_type == 'causal' else \
                COLORS['tertiary'] if edge_type == 'mediator' else COLORS['neutral']

        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', color=color,
                                  lw=2 if edge_type == 'causal' else 1,
                                  alpha=0.7))

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['primary'], lw=2, label='Causal path'),
        plt.Line2D([0], [0], color=COLORS['tertiary'], lw=1.5, label='Mediation path'),
        plt.Line2D([0], [0], color=COLORS['neutral'], lw=1, label='Confounding path'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Figure 1: Directed Acyclic Graph (DAG) for Sleep-Cardiometabolic Analysis',
                fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_flowchart(exclusion_log: Dict, save_path: str = None) -> plt.Figure:
    """Create participant flowchart"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Box positions and text
    boxes = [
        (0.5, 0.95, f"NHANES 2017-2020\nTotal participants\nn = {exclusion_log.get('initial', 'N/A'):,}"),
        (0.5, 0.75, f"Adults ≥20 years\nn = {exclusion_log.get('after_age_filter', 'N/A'):,}"),
        (0.5, 0.55, f"With sleep data\nn = {exclusion_log.get('after_sleep_filter', 'N/A'):,}"),
        (0.5, 0.35, f"Valid survey weights\nn = {exclusion_log.get('after_weight_filter', 'N/A'):,}"),
        (0.5, 0.15, f"Final analytic sample\nn = {exclusion_log.get('final', 'N/A'):,}"),
    ]

    # Exclusion text
    exclusions = [
        (0.85, 0.85, f"Excluded: Age <20\nn = {exclusion_log.get('excluded_age', 'N/A'):,}"),
        (0.85, 0.65, f"Excluded: Missing sleep\nn = {exclusion_log.get('excluded_sleep', 'N/A'):,}"),
        (0.85, 0.45, f"Excluded: Invalid weights\nn = {exclusion_log.get('excluded_weight', 'N/A'):,}"),
        (0.85, 0.25, f"Excluded: Implausible values\nn = {exclusion_log.get('excluded_plausible', 'N/A'):,}"),
    ]

    # Draw boxes
    for x, y, text in boxes:
        rect = plt.Rectangle((x-0.15, y-0.06), 0.3, 0.12,
                             fill=True, facecolor='white',
                             edgecolor=COLORS['primary'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)

    # Draw exclusion boxes
    for x, y, text in exclusions:
        rect = plt.Rectangle((x-0.12, y-0.04), 0.24, 0.08,
                             fill=True, facecolor='#FFF5F5',
                             edgecolor=COLORS['secondary'], linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color=COLORS['secondary'])

    # Draw arrows
    arrow_y = [0.89, 0.69, 0.49, 0.29]
    for y in arrow_y:
        ax.annotate('', xy=(0.5, y-0.08), xytext=(0.5, y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))
        ax.annotate('', xy=(0.73, y-0.04), xytext=(0.65, y-0.04),
                   arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Figure 2: Participant Flow Diagram', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_forest_plot(results_df: pd.DataFrame,
                       outcome: str,
                       save_path: str = None) -> plt.Figure:
    """Create forest plot for odds ratios"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Filter for sleep category results
    df_plot = results_df[results_df.index.str.contains('sleep_category')].copy()

    if len(df_plot) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig

    # Clean category names
    df_plot['category'] = df_plot.index.str.replace('sleep_category_', '')

    y_positions = range(len(df_plot))

    # Plot points and CI
    ax.errorbar(df_plot['OR'], y_positions,
               xerr=[df_plot['OR'] - df_plot['OR_lower'],
                     df_plot['OR_upper'] - df_plot['OR']],
               fmt='o', color=COLORS['primary'], capsize=4, capthick=2,
               markersize=8, linewidth=2)

    # Reference line at OR = 1
    ax.axvline(x=1, color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.7)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot['category'])
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
    ax.set_title(f'Figure 3: Association Between Sleep Duration and {outcome.title()}\n(Reference: 7-8 hours)',
                fontsize=12, fontweight='bold')

    # Add OR values as text
    for i, row in df_plot.iterrows():
        idx = list(df_plot.index).index(i)
        or_text = f"{row['OR']:.2f} ({row['OR_lower']:.2f}-{row['OR_upper']:.2f})"
        ax.text(ax.get_xlim()[1] * 0.95, idx, or_text, va='center', ha='right', fontsize=9)

    ax.set_xlim(0.3, max(df_plot['OR_upper'].max() * 1.3, 3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_sleep_distribution(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Create sleep duration distribution plot"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(df['sleep_hours'].dropna(), bins=30, color=COLORS['primary'],
             alpha=0.7, edgecolor='white')
    ax1.axvline(x=7, color=COLORS['secondary'], linestyle='--', linewidth=2, label='7 hours')
    ax1.axvline(x=8, color=COLORS['secondary'], linestyle='--', linewidth=2, label='8 hours')
    ax1.set_xlabel('Sleep Duration (hours)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('A. Distribution of Sleep Duration')
    ax1.legend()

    # Category bar plot
    ax2 = axes[1]
    if 'sleep_category' in df.columns:
        cat_order = ['Very Short (<6h)', 'Short (6-7h)', 'Recommended (7-8h)',
                     'Long (8-9h)', 'Very Long (>9h)']
        cat_counts = df['sleep_category'].value_counts()
        cat_counts = cat_counts.reindex([c for c in cat_order if c in cat_counts.index])

        bars = ax2.bar(range(len(cat_counts)), cat_counts.values,
                      color=COLORS['sleep_categories'][:len(cat_counts)])
        ax2.set_xticks(range(len(cat_counts)))
        ax2.set_xticklabels([c.replace(' ', '\n') for c in cat_counts.index],
                           fontsize=9)
        ax2.set_ylabel('Frequency')
        ax2.set_title('B. Sleep Duration Categories')

        # Add percentage labels
        total = cat_counts.sum()
        for i, (bar, val) in enumerate(zip(bars, cat_counts.values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val/total*100:.1f}%', ha='center', fontsize=9)

    fig.suptitle('Figure 4: Sleep Duration Distribution', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_outcome_prevalence(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Create outcome prevalence by sleep category"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    outcomes = ['hypertension', 'diabetes', 'obesity', 'metabolic_syndrome']
    titles = ['Hypertension', 'Diabetes', 'Obesity', 'Metabolic Syndrome']

    cat_order = ['Very Short (<6h)', 'Short (6-7h)', 'Recommended (7-8h)',
                 'Long (8-9h)', 'Very Long (>9h)']

    for ax, outcome, title in zip(axes.flatten(), outcomes, titles):
        if outcome not in df.columns:
            ax.text(0.5, 0.5, f'{title}\nData not available',
                   ha='center', va='center')
            continue

        # Calculate prevalence by category
        prev_by_cat = df.groupby('sleep_category')[outcome].mean() * 100
        prev_by_cat = prev_by_cat.reindex([c for c in cat_order if c in prev_by_cat.index])

        bars = ax.bar(range(len(prev_by_cat)), prev_by_cat.values,
                     color=COLORS['sleep_categories'][:len(prev_by_cat)])

        ax.set_xticks(range(len(prev_by_cat)))
        ax.set_xticklabels([c.split('(')[0].strip() for c in prev_by_cat.index],
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Prevalence (%)')
        ax.set_title(title, fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, prev_by_cat.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontsize=9)

    fig.suptitle('Figure 5: Outcome Prevalence by Sleep Duration Category',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_roc_curves(results: Dict, save_path: str = None) -> plt.Figure:
    """Create ROC curves comparing models"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for i, (name, result) in enumerate(results.items()):
        if 'roc_curve' in result:
            fpr = result['roc_curve']['fpr']
            tpr = result['roc_curve']['tpr']
            auc_val = result.get('metrics', {}).get('test_auc', auc(fpr, tpr))

            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                   linewidth=2, label=f'{name} (AUC = {auc_val:.3f})')

    # Diagonal line
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], linestyle='--', linewidth=1.5)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Figure 6: ROC Curves - Model Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_shap_summary(shap_values, X_sample: pd.DataFrame,
                        save_path: str = None) -> plt.Figure:
    """Create SHAP summary plot"""

    try:
        import shap

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)

        plt.title('Figure 7: SHAP Feature Importance', fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig

    except ImportError:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.text(0.5, 0.5, 'SHAP not available', ha='center', va='center')
        return fig


def create_calibration_plot(y_true: np.ndarray,
                            y_pred_proba: np.ndarray,
                            model_name: str = 'Model',
                            save_path: str = None) -> plt.Figure:
    """Create calibration plot"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', color=COLORS['primary'],
           linewidth=2, markersize=8, label=model_name)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color=COLORS['neutral'],
           linewidth=1.5, label='Perfect calibration')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Figure 8: Calibration Plot', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_subgroup_forest(subgroup_results: pd.DataFrame,
                           save_path: str = None) -> plt.Figure:
    """Create forest plot for subgroup analysis"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create labels
    subgroup_results['label'] = subgroup_results['subgroup_var'] + ': ' + \
                                 subgroup_results['subgroup'].astype(str)

    y_positions = range(len(subgroup_results))

    # Plot
    ax.errorbar(subgroup_results['OR'], y_positions,
               xerr=[subgroup_results['OR'] - subgroup_results['OR_lower'],
                     subgroup_results['OR_upper'] - subgroup_results['OR']],
               fmt='o', color=COLORS['primary'], capsize=4, capthick=2,
               markersize=8, linewidth=2)

    ax.axvline(x=1, color=COLORS['neutral'], linestyle='--', linewidth=1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(subgroup_results['label'])
    ax.set_xlabel('Odds Ratio (95% CI)')
    ax.set_title('Figure 9: Subgroup Analysis - Effect Modification',
                fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def create_all_figures(df: pd.DataFrame,
                       results: Dict,
                       output_dir: str) -> Dict:
    """Generate all figures for the analysis"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    print("\nGenerating figures...")

    # 1. DAG
    print("  - Creating DAG...")
    figures['dag'] = create_dag_figure(f"{output_dir}/fig1_dag.png")

    # 2. Flowchart
    if 'exclusion_log' in results:
        print("  - Creating flowchart...")
        figures['flowchart'] = create_flowchart(
            results['exclusion_log'], f"{output_dir}/fig2_flowchart.png"
        )

    # 3. Sleep distribution
    print("  - Creating sleep distribution...")
    figures['sleep_dist'] = create_sleep_distribution(
        df, f"{output_dir}/fig4_sleep_distribution.png"
    )

    # 4. Outcome prevalence
    print("  - Creating outcome prevalence...")
    figures['prevalence'] = create_outcome_prevalence(
        df, f"{output_dir}/fig5_outcome_prevalence.png"
    )

    # 5. Forest plot (if results available)
    if 'regression_results' in results:
        print("  - Creating forest plot...")
        for outcome, outcome_results in results['regression_results'].items():
            if 'model3_fully_adjusted' in outcome_results:
                if 'summary' in outcome_results['model3_fully_adjusted']:
                    figures[f'forest_{outcome}'] = create_forest_plot(
                        outcome_results['model3_fully_adjusted']['summary'],
                        outcome,
                        f"{output_dir}/fig3_forest_{outcome}.png"
                    )

    print(f"\n  Figures saved to {output_dir}/")

    return figures


if __name__ == "__main__":
    print("Visualization Module loaded successfully")
