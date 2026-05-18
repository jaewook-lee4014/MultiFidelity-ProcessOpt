#!/usr/bin/env python
"""
Standalone runner for non-learning baselines (HF-Only Random + LF-Screening).

Runs 2 baselines × 7 tasks × 20 seeds on CPU only.
Does NOT re-run existing surrogate-based models.

Usage:
    python benchmark_random_baselines.py [--n-seeds 20] [--n-workers 14]
"""

import numpy as np
import pandas as pd
import time
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime

from benchmark_parallel import (
    SyntheticBenchmark, ChemistryBenchmark,
    furthest_point_sampling, latin_hypercube_sampling, find_nearest_candidates,
)
from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf
from baselines import run_hf_random_search, run_lf_screening


def run_single(args):
    """Worker: run one (baseline, benchmark, seed) combination."""
    bench_name, bench_config, model_name, budget, seed = args

    # Create benchmark
    data_dir = Path(__file__).parent / 'data'
    if bench_config['type'] == 'synthetic':
        benchmark = SyntheticBenchmark(
            bench_name,
            bench_config['hf_func'],
            bench_config['lf_func'],
            bench_config['dim'],
            bench_config['alpha'],
            bench_config['cost_ratio'],
            bench_config['f_star'],
            bench_config['grid_size'],
        )
    else:
        csv_path = data_dir / bench_config['csv_file']
        benchmark = ChemistryBenchmark(
            bench_name, csv_path,
            bench_config['cost_ratio'],
            bench_config['use_smiles'],
            bench_config['minimize'],
            bench_config.get('negate', False),
        )

    t0 = time.time()
    if model_name == 'HF-Only Random':
        result = run_hf_random_search(benchmark, budget, seed)
    else:
        result = run_lf_screening(benchmark, budget, seed)
    elapsed = time.time() - t0

    # Summary row
    summary = {
        'benchmark': bench_name,
        'model': model_name,
        'seed': seed,
        'final_regret': result['final_regret'],
        'n_hf': result['n_hf'],
        'n_lf': result['n_lf'],
        'best_y': result['best_y'],
        'elapsed_sec': round(elapsed, 3),
    }

    # Trajectory rows
    trajectory = []
    for b, r in zip(result['budgets'], result['regrets']):
        trajectory.append({
            'benchmark': bench_name,
            'model': model_name,
            'seed': seed,
            'budget': round(b, 2),
            'regret': r,
        })

    return summary, trajectory


def main():
    parser = argparse.ArgumentParser(description='Non-learning baselines runner')
    parser.add_argument('--n-seeds', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=14)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'random_baselines_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(__file__).parent / 'data'

    bench_configs = {
        'Branin-Fav': {
            'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
            'dim': 2, 'alpha': 0.8, 'cost_ratio': 0.1, 'f_star': 0.397887, 'grid_size': 50,
        },
        'Branin-Unfav': {
            'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
            'dim': 2, 'alpha': 0.1, 'cost_ratio': 0.5, 'f_star': 0.397887, 'grid_size': 50,
        },
        'Park-Fav': {
            'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
            'dim': 4, 'alpha': 0.6, 'cost_ratio': 0.1, 'f_star': 0.0, 'grid_size': 10,
        },
        'Park-Unfav': {
            'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
            'dim': 4, 'alpha': 0.0, 'cost_ratio': 0.5, 'f_star': 0.0, 'grid_size': 10,
        },
        'COFs': {
            'type': 'chemistry', 'csv_file': 'cofs.csv',
            'cost_ratio': 0.065, 'use_smiles': False, 'minimize': True, 'negate': True,
        },
        'FreeSolv': {
            'type': 'chemistry', 'csv_file': 'freesolv.csv',
            'cost_ratio': 0.1, 'use_smiles': True, 'minimize': True, 'negate': False,
        },
        'Polarizability': {
            'type': 'chemistry', 'csv_file': 'polarizability.csv',
            'cost_ratio': 0.167, 'use_smiles': True, 'minimize': True, 'negate': True,
        },
    }

    budgets = {
        'Branin-Fav': 50, 'Branin-Unfav': 50,
        'Park-Fav': 50, 'Park-Unfav': 50,
        'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30,
    }

    baselines = ['HF-Only Random', 'LF-Screening']
    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    # Build task list
    tasks = []
    for bench_name, bench_config in bench_configs.items():
        for model_name in baselines:
            for seed in seeds:
                tasks.append((bench_name, bench_config, model_name, budgets[bench_name], seed))

    total = len(tasks)
    n_combos = len(bench_configs) * len(baselines)
    print("=" * 70)
    print("Non-Learning Baselines Runner")
    print("=" * 70)
    print(f"Baselines: {baselines}")
    print(f"Tasks: {len(bench_configs)} benchmarks × {len(baselines)} baselines × {args.n_seeds} seeds = {total} runs")
    print(f"Workers: {args.n_workers}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    start = time.time()
    all_summary = []
    all_trajectory = []

    with Pool(processes=args.n_workers) as pool:
        for i, (summary, trajectory) in enumerate(pool.imap_unordered(run_single, tasks)):
            all_summary.append(summary)
            all_trajectory.extend(trajectory)
            done = i + 1
            if done % args.n_seeds == 0 or done == total:
                elapsed = time.time() - start
                eta = (elapsed / done) * (total - done)
                print(f"[{done}/{total}] {summary['benchmark']} + {summary['model']} "
                      f"seed={summary['seed']}: regret={summary['final_regret']:.4f} "
                      f"| ETA: {eta/60:.1f}min")

    total_time = time.time() - start

    df_summary = pd.DataFrame(all_summary)
    df_summary.to_csv(output_dir / 'results_summary.csv', index=False)

    df_trajectory = pd.DataFrame(all_trajectory)
    df_trajectory.to_csv(output_dir / 'results_trajectory.csv', index=False)

    print(f"\nCompleted in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results saved to: {output_dir}/")

    # Summary table
    print("\nMean Final Regret (± std):")
    print("-" * 70)
    for bench in sorted(df_summary['benchmark'].unique()):
        print(f"\n{bench}:")
        s = df_summary[df_summary['benchmark'] == bench].groupby('model')['final_regret']
        stats = s.agg(['mean', 'std']).sort_values('mean')
        for model, row in stats.iterrows():
            print(f"  {model:<20}: {row['mean']:.4f} ± {row['std']:.4f}")

    # Budget allocation verification
    print("\nBudget Allocation (n_lf, n_hf):")
    print("-" * 70)
    for bench in sorted(df_summary['benchmark'].unique()):
        sub = df_summary[(df_summary['benchmark'] == bench) & (df_summary['model'] == 'LF-Screening')]
        if not sub.empty:
            row = sub.iloc[0]
            print(f"  {bench:<20}: n_lf={int(row['n_lf'])}, n_hf={int(row['n_hf'])}")


if __name__ == '__main__':
    main()
