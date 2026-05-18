#!/bin/bash
#SBATCH --job-name=new_baselines
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_new_baselines_%j.out
#SBATCH --error=slurm_new_baselines_%j.err

echo "========================================"
echo "Full Sweep: 4 New Baselines × 7 Benchmarks × 20 Seeds"
echo "========================================"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi 2>/dev/null || echo "No GPU info available"

export PYTHONUNBUFFERED=1
export PATH="/scratch/users/k23070952/vllm_env310/bin:$PATH"

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

# Run only the 4 new baselines (not the existing 12)
python -u -c "
import sys, time, os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

from benchmark_parallel import (
    SyntheticBenchmark, ChemistryBenchmark, run_bo, run_combination,
    branin_hf, branin_lf, park_hf, park_lf
)
from baselines import SparseMFGP, NARGP, DKLMultiFidelity

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path(f'new_baselines_{timestamp}')
output_dir.mkdir(exist_ok=True)

data_dir = Path('data')

bench_configs = {
    'Branin-Fav': {
        'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
        'dim': 2, 'alpha': 0.8, 'cost_ratio': 0.1, 'f_star': 0.397887, 'grid_size': 50
    },
    'Branin-Unfav': {
        'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
        'dim': 2, 'alpha': 0.1, 'cost_ratio': 0.5, 'f_star': 0.397887, 'grid_size': 50
    },
    'Park-Fav': {
        'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
        'dim': 4, 'alpha': 0.6, 'cost_ratio': 0.1, 'f_star': 0.0, 'grid_size': 10
    },
    'Park-Unfav': {
        'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
        'dim': 4, 'alpha': 0.0, 'cost_ratio': 0.5, 'f_star': 0.0, 'grid_size': 10
    },
    'COFs': {
        'type': 'chemistry', 'csv_path': data_dir / 'cofs.csv',
        'cost_ratio': 0.065, 'use_smiles': False, 'minimize': True, 'negate': True
    },
    'FreeSolv': {
        'type': 'chemistry', 'csv_path': data_dir / 'freesolv.csv',
        'cost_ratio': 0.1, 'use_smiles': True, 'minimize': True, 'negate': False
    },
    'Polarizability': {
        'type': 'chemistry', 'csv_path': data_dir / 'polarizability.csv',
        'cost_ratio': 0.167, 'use_smiles': True, 'minimize': True, 'negate': True
    },
}

budgets = {
    'Branin-Fav': 50, 'Branin-Unfav': 50,
    'Park-Fav': 50, 'Park-Unfav': 50,
    'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30,
}

new_models = {
    'Sparse MFGP': SparseMFGP,
    'NARGP': NARGP,
    'DKL Multi-Fidelity': DKLMultiFidelity,
    'Successive Halving': None,
}

seeds = [42 + i for i in range(20)]
n_workers = 16

tasks = []
worker_id = 0
for bench_name, bench_config in bench_configs.items():
    for model_name, model_class in new_models.items():
        tasks.append((
            bench_name, bench_config, model_name, model_class,
            budgets[bench_name], seeds, output_dir, worker_id
        ))
        worker_id += 1

total = len(tasks)
print(f'Running {total} combinations ({len(bench_configs)} benchmarks x {len(new_models)} models) x {len(seeds)} seeds')
print(f'Workers: {n_workers}, Output: {output_dir}')

start = time.time()
with Pool(processes=n_workers) as pool:
    results = []
    for i, result in enumerate(pool.imap_unordered(run_combination, tasks)):
        results.append(result)
        elapsed = time.time() - start
        done = i + 1
        eta = (elapsed / done) * (total - done)
        print(f'[{done}/{total}] {result[\"bench_name\"]} + {result[\"model_name\"]}: {result[\"elapsed\"]:.1f}s | ETA: {eta/60:.1f}min')

total_time = time.time() - start

all_summary = []
all_trajectory = []
for r in results:
    all_summary.extend(r['results_summary'])
    all_trajectory.extend(r['results_trajectory'])

df_s = pd.DataFrame(all_summary)
df_s.to_csv(output_dir / 'results_summary.csv', index=False)
df_t = pd.DataFrame(all_trajectory)
df_t.to_csv(output_dir / 'results_trajectory.csv', index=False)

print(f'\nCompleted in {total_time/60:.1f} min')
print(f'Results: {output_dir}')

# Summary table
print('\nMean Final Regret:')
for bench in sorted(df_s['benchmark'].unique()):
    print(f'\n{bench}:')
    s = df_s[df_s['benchmark']==bench].groupby('model')['final_regret'].agg(['mean','std']).sort_values('mean')
    for m, row in s.iterrows():
        print(f'  {m:<25}: {row[\"mean\"]:.4f} +/- {row[\"std\"]:.4f}')
"

echo ""
echo "========================================"
echo "Sensitivity Analysis: Sparse MFGP n_inducing"
echo "========================================"

python -u -c "
import numpy as np, pandas as pd, torch, time
from benchmark_parallel import SyntheticBenchmark, run_bo, branin_hf, branin_lf
from baselines import SparseMFGP

benchmark = SyntheticBenchmark('Branin-Fav', branin_hf, branin_lf, 2, 0.8, 0.1, 0.397887, 50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = []
for n_ind in [50, 100, 200]:
    print(f'n_inducing={n_ind}...')
    class SparseMFGP_Custom(SparseMFGP):
        def __init__(self, input_dim, hidden_dim=64, device=None):
            super().__init__(input_dim, hidden_dim, device, n_inducing=n_ind)
    for seed in [42 + i for i in range(20)]:
        t0 = time.time()
        r = run_bo(benchmark, SparseMFGP_Custom, 50, seed, device)
        results.append({
            'n_inducing': n_ind, 'seed': seed,
            'final_regret': r['final_regret'], 'best_y': r['best_y'],
            'time_sec': round(time.time()-t0, 2)
        })
    vals = [x['final_regret'] for x in results if x['n_inducing']==n_ind]
    print(f'  regret: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

df = pd.DataFrame(results)
df.to_csv('sensitivity_sparse_mfgp_inducing.csv', index=False)
print('Saved sensitivity_sparse_mfgp_inducing.csv')
"

echo "End: $(date)"
echo "========================================"
