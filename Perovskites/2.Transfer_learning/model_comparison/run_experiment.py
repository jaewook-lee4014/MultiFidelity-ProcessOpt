#!/usr/bin/env python
"""
Main Script for Model Comparison Experiments

This script runs the complete pipeline:
1. Generate datasets (lofi + hifi samples)
2. Train and evaluate DNGO and BNN models
3. Create visualizations

Usage:
    python run_experiment.py                    # Run full pipeline
    python run_experiment.py --generate-only    # Only generate datasets
    python run_experiment.py --evaluate-only    # Only run evaluation (datasets must exist)
    python run_experiment.py --visualize-only   # Only create visualizations (results must exist)
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR.parent / 'Pure_TL_BO'))

# Import modules
from dataset_generator import generate_all_datasets, load_datasets
from model_evaluators import run_all_evaluations
from visualizations import create_all_visualizations, extract_metrics_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description='Model Comparison Experiment')

    # Mode selection
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate datasets')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only create visualizations')

    # Dataset parameters
    parser.add_argument('--n-sets', type=int, default=5,
                       help='Number of random sets per configuration (default: 5)')

    # Model parameters
    parser.add_argument('--hidden-layers', type=int, default=2,
                       help='Number of hidden layers (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension (default: 64)')
    parser.add_argument('--pretrain-epochs', type=int, default=200,
                       help='Pretrain epochs (default: 200)')
    parser.add_argument('--finetune-epochs', type=int, default=100,
                       help='Finetune epochs (default: 100)')

    # Output
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    return parser.parse_args()


def main():
    args = parse_args()

    # Directories
    datasets_dir = SCRIPT_DIR / 'datasets'
    results_dir = SCRIPT_DIR / 'results'
    viz_dir = SCRIPT_DIR / 'visualizations'

    print("="*60)
    print("Model Comparison Experiment")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Determine what to run
    run_all = not (args.generate_only or args.evaluate_only or args.visualize_only)

    # Step 1: Generate datasets
    if run_all or args.generate_only:
        print("\n" + "="*60)
        print("Step 1: Generating Datasets")
        print("="*60)

        datasets = generate_all_datasets(
            str(datasets_dir),
            n_sets_per_config=args.n_sets
        )

        if args.generate_only:
            print("\nDataset generation complete.")
            return

    # Step 2: Run evaluation
    if run_all or args.evaluate_only:
        print("\n" + "="*60)
        print("Step 2: Running Model Evaluation")
        print("="*60)

        # Model configurations
        model_configs = {
            'dngo': {
                'hidden_layers': args.hidden_layers,
                'hidden_dim': args.hidden_dim,
                'pretrain_epochs': args.pretrain_epochs,
                'finetune_epochs': args.finetune_epochs,
                'pretrain_lr': 1e-3,
                'finetune_lr': 1e-4
            },
            'bnn': {
                'hidden_layers': args.hidden_layers,
                'hidden_dim': args.hidden_dim,
                'pretrain_epochs': args.pretrain_epochs,
                'finetune_epochs': args.finetune_epochs,
                'pretrain_lr': 1e-3,
                'finetune_lr': 1e-4,
                'kl_weight': 1.0
            }
        }

        results = run_all_evaluations(
            str(datasets_dir),
            str(results_dir),
            model_configs=model_configs,
            verbose=args.verbose
        )

        if args.evaluate_only:
            print("\nEvaluation complete.")
            return

    # Step 3: Create visualizations
    if run_all or args.visualize_only:
        print("\n" + "="*60)
        print("Step 3: Creating Visualizations")
        print("="*60)

        # Load lookup for all_y
        lookup_path = SCRIPT_DIR.parent.parent / '0.Data' / 'lookup_table.pkl'
        with open(lookup_path, 'rb') as f:
            lookup = pickle.load(f)

        with open(datasets_dir / 'param_space.json', 'r') as f:
            param_space = json.load(f)

        # Generate all_y
        all_y = []
        for org in param_space['organic']:
            for cat in param_space['cation']:
                for ani in param_space['anion']:
                    bandgap = np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06'])
                    all_y.append(bandgap)
        all_y = np.array(all_y)

        results_path = results_dir / 'evaluation_results.pkl'
        create_all_visualizations(str(results_path), str(viz_dir), all_y)

    # Final summary
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Datasets:       {datasets_dir}/")
    print(f"  Results:        {results_dir}/")
    print(f"  Visualizations: {viz_dir}/")

    # Print quick summary if results exist
    results_path = results_dir / 'evaluation_results.pkl'
    if results_path.exists():
        with open(results_path, 'rb') as f:
            results = pickle.load(f)

        df = extract_metrics_dataframe(results)

        print("\n" + "="*60)
        print("Quick Summary (Mean ± Std)")
        print("="*60)

        has_lofi = 'rmse_lofi' in df.columns

        for config in df['config'].unique():
            print(f"\n{config}:")
            for model in df['model'].unique():
                subset = df[(df['config'] == config) & (df['model'] == model)]
                print(f"  {model}:")
                print(f"    [HIFI Target] (Finetune quality)")
                print(f"      RMSE:     {subset['rmse'].mean():.4f} ± {subset['rmse'].std():.4f}")
                print(f"      R²:       {subset['r2'].mean():.4f} ± {subset['r2'].std():.4f}")
                print(f"      Spearman: {subset['spearman'].mean():.4f} ± {subset['spearman'].std():.4f}")
                if has_lofi and 'rmse_lofi' in subset.columns and not subset['rmse_lofi'].isna().all():
                    print(f"    [LOFI Target] (Pretrain quality)")
                    print(f"      RMSE:     {subset['rmse_lofi'].mean():.4f} ± {subset['rmse_lofi'].std():.4f}")
                    print(f"      R²:       {subset['r2_lofi'].mean():.4f} ± {subset['r2_lofi'].std():.4f}")
                    print(f"      Spearman: {subset['spearman_lofi'].mean():.4f} ± {subset['spearman_lofi'].std():.4f}")


if __name__ == '__main__':
    main()
