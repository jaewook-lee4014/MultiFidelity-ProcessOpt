"""
Visualization Module for Model Comparison

Creates various plots comparing DNGO and BNN performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional


def set_plot_style():
    """Set consistent plot style"""
    # Try different seaborn style names for compatibility
    seaborn_styles = ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']
    for style in seaborn_styles:
        try:
            plt.style.use(style)
            break
        except (OSError, IOError):
            continue

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 100
    })


def extract_metrics_dataframe(results: Dict) -> pd.DataFrame:
    """
    Extract metrics from results into a pandas DataFrame

    Args:
        results: Evaluation results dictionary

    Returns:
        DataFrame with all metrics (both HIFI and LOFI)
    """
    records = []

    for config_name, config_results in results.items():
        for set_idx, set_result in enumerate(config_results):
            for model_type, model_result in set_result.items():
                metrics_hifi = model_result['metrics_all']
                metrics_lofi = model_result.get('metrics_all_lofi')
                dataset_info = model_result['dataset_info']

                record = {
                    'config': config_name,
                    'set_idx': set_idx,
                    'model': model_type.upper(),
                    'n_lofi': dataset_info['n_lofi'],
                    'n_hifi': dataset_info['n_hifi'],
                    'cost': dataset_info['actual_cost'],
                    # HIFI metrics (original)
                    'rmse': metrics_hifi.rmse,
                    'mae': metrics_hifi.mae,
                    'r2': metrics_hifi.r2,
                    'spearman': metrics_hifi.spearman_corr,
                    'unc_calibration': metrics_hifi.uncertainty_calibration,
                    'coverage_90': metrics_hifi.coverage_90,
                    'best_pred': metrics_hifi.best_prediction,
                    'best_actual': metrics_hifi.best_actual
                }

                # Add LOFI metrics if available
                if metrics_lofi is not None:
                    record.update({
                        'rmse_lofi': metrics_lofi.rmse,
                        'mae_lofi': metrics_lofi.mae,
                        'r2_lofi': metrics_lofi.r2,
                        'spearman_lofi': metrics_lofi.spearman_corr,
                    })

                records.append(record)

    return pd.DataFrame(records)


def plot_metric_comparison_boxplot(
    df: pd.DataFrame,
    metric: str,
    output_path: str = None,
    title: str = None
):
    """
    Create boxplot comparing models across configurations

    Args:
        df: Metrics DataFrame
        metric: Metric column name to plot
        output_path: Path to save figure
        title: Plot title
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped boxplot
    x_labels = df['config'].unique()
    models = df['model'].unique()
    colors = {'DNGO': '#2ecc71', 'BNN': '#3498db'}

    positions = []
    for i, config in enumerate(x_labels):
        for j, model in enumerate(models):
            positions.append(i * 3 + j)

    # Manual boxplot
    width = 0.8
    for i, config in enumerate(x_labels):
        for j, model in enumerate(models):
            data = df[(df['config'] == config) & (df['model'] == model)][metric]
            pos = i * 3 + j
            bp = ax.boxplot(
                data,
                positions=[pos],
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=colors[model], alpha=0.7),
                medianprops=dict(color='black', linewidth=2)
            )

    # Set labels
    ax.set_xticks([i * 3 + 0.5 for i in range(len(x_labels))])
    ax.set_xticklabels(x_labels)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[m], alpha=0.7, label=m) for m in models]
    ax.legend(handles=legend_elements, loc='upper right')

    # Labels and title
    metric_labels = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R² Score',
        'spearman': 'Spearman Correlation',
        'unc_calibration': 'Uncertainty Calibration',
        'coverage_90': '90% CI Coverage (%)'
    }
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_xlabel('Dataset Configuration')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{metric_labels.get(metric, metric)} Comparison: DNGO vs BNN')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def plot_all_metrics_comparison(
    df: pd.DataFrame,
    output_dir: str
):
    """
    Create comparison plots for all metrics

    Args:
        df: Metrics DataFrame
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['rmse', 'mae', 'r2', 'spearman', 'unc_calibration', 'coverage_90']

    for metric in metrics:
        output_path = os.path.join(output_dir, f'{metric}_comparison.png')
        plot_metric_comparison_boxplot(df, metric, output_path)
        plt.close()


def plot_summary_bar_chart(
    df: pd.DataFrame,
    output_path: str = None
):
    """
    Create summary bar chart with mean ± std for key metrics

    Args:
        df: Metrics DataFrame
        output_path: Path to save figure
    """
    set_plot_style()

    # Aggregate metrics
    summary = df.groupby(['config', 'model']).agg({
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'spearman': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [('rmse', 'RMSE'), ('r2', 'R² Score'), ('spearman', 'Spearman Correlation')]
    colors = {'DNGO': '#2ecc71', 'BNN': '#3498db'}

    for ax, (metric, label) in zip(axes, metrics):
        configs = summary['config'].unique()
        x = np.arange(len(configs))
        width = 0.35

        for i, model in enumerate(['DNGO', 'BNN']):
            model_data = summary[summary['model'] == model]
            means = model_data[f'{metric}_mean'].values
            stds = model_data[f'{metric}_std'].values

            offset = -width/2 + i * width
            bars = ax.bar(x + offset, means, width, label=model,
                         color=colors[model], alpha=0.8, yerr=stds, capsize=5)

        ax.set_xlabel('Configuration')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def plot_prediction_scatter(
    results: Dict,
    config_name: str,
    set_idx: int,
    all_y: np.ndarray,
    output_path: str = None
):
    """
    Create scatter plot of predictions vs actual values

    Args:
        results: Evaluation results
        config_name: Configuration name
        set_idx: Dataset set index
        all_y: True y values for all combinations
        output_path: Path to save figure
    """
    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    set_result = results[config_name][set_idx]

    for ax, (model, model_result) in zip(axes, set_result.items()):
        y_pred = model_result['y_pred_all']
        y_std = model_result['y_std_all']

        # Scatter plot
        ax.scatter(all_y, y_pred, alpha=0.5, s=30, c='blue', label='Predictions')

        # Error bars for some points (randomly sample for clarity)
        n_show = min(50, len(all_y))
        indices = np.random.choice(len(all_y), n_show, replace=False)
        ax.errorbar(all_y[indices], y_pred[indices], yerr=y_std[indices],
                   fmt='none', alpha=0.3, color='blue')

        # Perfect prediction line
        min_val, max_val = min(all_y.min(), y_pred.min()), max(all_y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        # Labels
        ax.set_xlabel('Actual Bandgap (eV)')
        ax.set_ylabel('Predicted Bandgap (eV)')
        ax.set_title(f'{model.upper()} - {config_name}')
        ax.legend()

        # Add metrics
        metrics = model_result['metrics_all']
        text = f'RMSE: {metrics.rmse:.4f}\nR²: {metrics.r2:.4f}\nρ: {metrics.spearman_corr:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def plot_uncertainty_calibration(
    results: Dict,
    config_name: str,
    set_idx: int,
    all_y: np.ndarray,
    output_path: str = None
):
    """
    Create uncertainty calibration plot

    Args:
        results: Evaluation results
        config_name: Configuration name
        set_idx: Dataset set index
        all_y: True y values
        output_path: Path to save figure
    """
    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    set_result = results[config_name][set_idx]

    for ax, (model, model_result) in zip(axes, set_result.items()):
        y_pred = model_result['y_pred_all']
        y_std = model_result['y_std_all']

        errors = np.abs(y_pred - all_y)

        # Scatter: uncertainty vs error
        ax.scatter(y_std, errors, alpha=0.4, s=30)

        # Fit line
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_std, errors)

        # Trend line
        z = np.polyfit(y_std, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_std.min(), y_std.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', lw=2, label=f'Trend (ρ={corr:.3f})')

        ax.set_xlabel('Predicted Uncertainty (σ)')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'{model.upper()} Uncertainty Calibration')
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def create_all_visualizations(
    results_path: str,
    output_dir: str,
    all_y: np.ndarray = None
):
    """
    Create all visualizations from results

    Args:
        results_path: Path to evaluation_results.pkl
        output_dir: Directory to save visualizations
        all_y: True y values for all combinations
    """
    # Load results
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics DataFrame
    df = extract_metrics_dataframe(results)
    df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    print(f"Saved metrics to: {os.path.join(output_dir, 'metrics_summary.csv')}")

    # 1. All metrics boxplots
    plot_all_metrics_comparison(df, output_dir)

    # 2. Summary bar chart
    plot_summary_bar_chart(df, os.path.join(output_dir, 'summary_comparison.png'))
    plt.close()

    # 3. Prediction scatter plots (for first set of each config)
    if all_y is not None:
        for config_name in results.keys():
            scatter_path = os.path.join(output_dir, f'scatter_{config_name}.png')
            plot_prediction_scatter(results, config_name, 0, all_y, scatter_path)
            plt.close()

            calib_path = os.path.join(output_dir, f'calibration_{config_name}.png')
            plot_uncertainty_calibration(results, config_name, 0, all_y, calib_path)
            plt.close()

    print(f"\nAll visualizations saved to: {output_dir}")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    has_lofi = 'rmse_lofi' in df.columns

    for config in df['config'].unique():
        print(f"\n{config}:")
        for model in df['model'].unique():
            subset = df[(df['config'] == config) & (df['model'] == model)]
            print(f"  {model}:")
            print(f"    [HIFI Target] (Finetune quality)")
            print(f"      RMSE: {subset['rmse'].mean():.4f} ± {subset['rmse'].std():.4f}")
            print(f"      R²:   {subset['r2'].mean():.4f} ± {subset['r2'].std():.4f}")
            print(f"      ρ:    {subset['spearman'].mean():.4f} ± {subset['spearman'].std():.4f}")
            if has_lofi and 'rmse_lofi' in subset.columns and not subset['rmse_lofi'].isna().all():
                print(f"    [LOFI Target] (Pretrain quality)")
                print(f"      RMSE: {subset['rmse_lofi'].mean():.4f} ± {subset['rmse_lofi'].std():.4f}")
                print(f"      R²:   {subset['r2_lofi'].mean():.4f} ± {subset['r2_lofi'].std():.4f}")
                print(f"      ρ:    {subset['spearman_lofi'].mean():.4f} ± {subset['spearman_lofi'].std():.4f}")


if __name__ == '__main__':
    import sys
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent / 'Pure_TL_BO'))

    results_path = SCRIPT_DIR / 'results' / 'evaluation_results.pkl'
    output_dir = SCRIPT_DIR / 'visualizations'

    # Load lookup for all_y
    import json
    lookup_path = SCRIPT_DIR.parent.parent / '0.Data' / 'lookup_table.pkl'
    with open(lookup_path, 'rb') as f:
        lookup = pickle.load(f)

    with open(SCRIPT_DIR / 'datasets' / 'param_space.json', 'r') as f:
        param_space = json.load(f)

    # Generate all_y
    all_y = []
    for org in param_space['organic']:
        for cat in param_space['cation']:
            for ani in param_space['anion']:
                bandgap = np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06'])
                all_y.append(bandgap)
    all_y = np.array(all_y)

    create_all_visualizations(str(results_path), str(output_dir), all_y)
