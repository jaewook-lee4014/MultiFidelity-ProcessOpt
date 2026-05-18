#!/bin/bash
#SBATCH --job-name=sh_rerun
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_sh_rerun_%j.out
#SBATCH --error=slurm_sh_rerun_%j.err

echo "========================================"
echo "Re-run Successive Halving (fixed schedule)"
echo "========================================"
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONUNBUFFERED=1
export PATH="/scratch/users/k23070952/vllm_env310/bin:$PATH"

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

python -u -c "
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf
from benchmark_parallel import SyntheticBenchmark, ChemistryBenchmark
from baselines import run_successive_halving

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path(f'sh_rerun_{timestamp}')
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

seeds = [42 + i for i in range(20)]

all_summary = []
all_trajectory = []

for bench_name, cfg in bench_configs.items():
    if cfg['type'] == 'synthetic':
        benchmark = SyntheticBenchmark(
            bench_name, cfg['hf_func'], cfg['lf_func'],
            cfg['dim'], cfg['alpha'], cfg['cost_ratio'], cfg['f_star'], cfg['grid_size']
        )
    else:
        benchmark = ChemistryBenchmark(
            bench_name, cfg['csv_path'], cfg['cost_ratio'],
            cfg['use_smiles'], cfg['minimize'], cfg.get('negate', False)
        )

    budget = budgets[bench_name]
    regrets_all = []
    for seed in seeds:
        t0 = time.time()
        result = run_successive_halving(benchmark, budget, seed)
        elapsed = time.time() - t0
        regrets_all.append(result['final_regret'])
        all_summary.append({
            'benchmark': bench_name, 'model': 'Successive Halving', 'seed': seed,
            'final_regret': result['final_regret'], 'n_hf': result['n_hf'],
            'n_lf': result['n_lf'], 'best_y': result['best_y'],
            'elapsed_sec': round(elapsed, 3),
        })
        for b, r in zip(result['budgets'], result['regrets']):
            all_trajectory.append({
                'benchmark': bench_name, 'model': 'Successive Halving',
                'seed': seed, 'budget': round(b, 2), 'regret': r,
            })
        if result.get('step_records'):
            df_steps = pd.DataFrame(result['step_records'])
            df_steps.to_csv(output_dir / f'results_Successive_Halving_{bench_name}_{seed}.csv', index=False)

    print(f'{bench_name}: regret={np.mean(regrets_all):.4f} +/- {np.std(regrets_all):.4f}, n_hf={result[\"n_hf\"]}, n_lf={result[\"n_lf\"]}')

df_s = pd.DataFrame(all_summary)
df_s.to_csv(output_dir / 'results_summary.csv', index=False)
df_t = pd.DataFrame(all_trajectory)
df_t.to_csv(output_dir / 'results_trajectory.csv', index=False)

print(f'\nResults saved to {output_dir}')
"

echo "End: $(date)"
echo "========================================"
