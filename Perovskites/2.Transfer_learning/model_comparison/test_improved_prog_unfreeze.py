"""
Test improved Progressive Unfreezing (lr_boost=2.0, lr_decay=0.7)
Compare with existing results
"""
import sys
import numpy as np
import torch
import pickle
import json
import copy
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from DNGO.models import TransferLearningDNN, BayesianLinearRegression

# Configuration - same as full comparison
BO_TRIALS = 100
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_data():
    data_path = Path(__file__).parent.parent.parent / '0.Data'
    with open(data_path / 'lookup_table.pkl', 'rb') as f:
        lookup = pickle.load(f)
    with open(data_path / 'organics.json', 'r') as f:
        organics_map = json.load(f)
    with open(data_path / 'cations.json', 'r') as f:
        cations_map = json.load(f)
    with open(data_path / 'anions.json', 'r') as f:
        anions_map = json.load(f)

    param_space = {
        'organic': list(organics_map.keys()),
        'cation': list(cations_map.keys()),
        'anion': list(anions_map.keys())
    }

    all_combinations = []
    for i, org in enumerate(param_space['organic'], 1):
        for j, cat in enumerate(param_space['cation'], 1):
            for k, ani in enumerate(param_space['anion'], 1):
                all_combinations.append({
                    'labels': [i, j, k],
                    'names': [org, cat, ani],
                    'org_idx': i, 'cat_idx': j, 'ani_idx': k
                })

    return lookup, all_combinations, param_space


def generate_data(lookup, all_combinations, n_lofi=72, n_hifi=9, seed=42):
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    n_total = len(all_combinations)
    lofi_idx = rng.choice(n_total, size=n_lofi, replace=False)
    hifi_idx = rng.choice(n_total, size=n_hifi, replace=False)

    X_low, y_low = [], []
    for idx in lofi_idx:
        c = all_combinations[idx]
        X_low.append(c['labels'])
        y_low.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))

    X_high, y_high = [], []
    for idx in hifi_idx:
        c = all_combinations[idx]
        X_high.append(c['labels'])
        y_high.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))

    X_all, y_all_hf = [], []
    for c in all_combinations:
        X_all.append(c['labels'])
        y_all_hf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_all': np.array(X_all, dtype=np.float32),
        'y_all': np.array(y_all_hf, dtype=np.float32),
        'hifi_idx': hifi_idx
    }


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return rmse, r2


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"BO Trials: {BO_TRIALS}")
    print(f"Testing improved Progressive Unfreezing (lr_boost=2.0, lr_decay=0.7)")

    lookup, all_combinations, param_space = load_base_data()

    # Load existing results for comparison
    existing_results_path = Path(__file__).parent / 'visualizations' / '20251204_235816_full_comparison_bo100' / 'results_summary.csv'
    existing_df = pd.read_csv(existing_results_path)
    print(f"\nExisting results loaded from: {existing_results_path}")

    results = []
    total_start = time.time()

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} (seed={seed})")
        print('='*60)

        data = generate_data(lookup, all_combinations, seed=seed)

        train_idx = data['hifi_idx']
        n_all = len(data['X_all'])
        test_mask = np.ones(n_all, dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]

        # Pretrain (shared)
        t0 = time.time()
        set_seeds(seed)
        pretrained_model = TransferLearningDNN(
            input_dim=data['X_low'].shape[1],
            hidden_dim=64,
            device=device,
            use_hyperparameter_bo=True
        )
        pretrained_model.pretrain(data['X_low'], data['y_low'], bo_trials=BO_TRIALS, verbose=False)
        pretrain_time = time.time() - t0
        print(f"  Pretrain in {pretrain_time:.2f}s")

        # DNGO-ProgUnfreeze (improved)
        t0 = time.time()
        set_seeds(seed)
        dngo_prog = copy.deepcopy(pretrained_model)
        dngo_prog.finetune(data['X_high'], data['y_high'], bo_trials=BO_TRIALS,
                          use_progressive_unfreezing=True, verbose=False)

        features_prog = dngo_prog.extract_features(data['X_high'])
        blr_prog = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr_prog.fit(features_prog, data['y_high'])

        features_all_prog = dngo_prog.extract_features(data['X_all'])
        prog_pred_all, _ = blr_prog.predict_batch(features_all_prog)
        prog_time = time.time() - t0

        # Calculate metrics
        y_test_hf = data['y_all'][test_idx]
        prog_rmse, prog_r2 = calc_metrics(y_test_hf, prog_pred_all[test_idx])

        # Get existing results for comparison
        existing_row = existing_df[existing_df['fold'] == fold_idx].iloc[0]
        old_prog_r2 = existing_row['prog_r2']
        dngo_r2 = existing_row['dngo_r2']
        mfgp_r2 = existing_row['mfgp_r2']

        improvement = prog_r2 - old_prog_r2

        print(f"  DNGO-ProgUnfreeze (improved): RMSE={prog_rmse:.4f}, R²={prog_r2:.4f} (time: {prog_time:.2f}s)")
        print(f"  Comparison:")
        print(f"    Old ProgUnfreeze R²: {old_prog_r2:.4f}")
        print(f"    New ProgUnfreeze R²: {prog_r2:.4f} ({'+' if improvement > 0 else ''}{improvement:.4f})")
        print(f"    DNGO R²:            {dngo_r2:.4f}")
        print(f"    MFGP R²:            {mfgp_r2:.4f}")

        results.append({
            'fold': fold_idx,
            'seed': seed,
            'old_prog_r2': old_prog_r2,
            'new_prog_r2': prog_r2,
            'new_prog_rmse': prog_rmse,
            'dngo_r2': dngo_r2,
            'mfgp_r2': mfgp_r2,
            'improvement': improvement
        })

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    df = pd.DataFrame(results)
    print(f"\nOld ProgUnfreeze Avg R²: {df['old_prog_r2'].mean():.4f}")
    print(f"New ProgUnfreeze Avg R²: {df['new_prog_r2'].mean():.4f}")
    print(f"DNGO Avg R²:             {df['dngo_r2'].mean():.4f}")
    print(f"MFGP Avg R²:             {df['mfgp_r2'].mean():.4f}")
    print(f"\nAverage Improvement: {df['improvement'].mean():.4f}")
    print(f"Folds improved: {(df['improvement'] > 0).sum()}/{len(df)}")

    print(f"\nTotal time: {total_time/60:.2f} minutes")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(__file__).parent / f'improved_prog_unfreeze_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
