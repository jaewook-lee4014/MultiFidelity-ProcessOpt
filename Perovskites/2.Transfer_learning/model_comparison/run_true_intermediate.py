"""
True Intermediate DNGO vs GP-mimic vs MFGP 비교 실험

비교 모델 (5개):
1. MFGP - vs HF
2. DNGO-Base (기존 Transfer Learning) - vs HF
3. GP-mimic (기존 Intermediate, 공유 trunk) - vs HF
4. True-Intermediate (새 구조, y_LF를 HF 입력에 포함) - vs HF
5. Pretrain-Base (LF만 학습) - vs LF

하이퍼파라미터 튜닝: Optuna 300 trials
"""
import sys
import numpy as np
import torch
import pickle
import json
import copy
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from DNGO.models import TransferLearningDNN, BayesianLinearRegression

# GP-mimic (기존)
from dngo_intermediate import IntermediateDNGO as GPmimicDNGO

# True Intermediate (새 구조)
from dngo_intermediate_v2 import TrueIntermediateDNGO, BayesianLinearRegression as BLR_v2

# MFGP
try:
    from mfgp_model import MultiFidelityGP
    MFGP_AVAILABLE = True
except ImportError:
    MFGP_AVAILABLE = False
    print("Warning: MFGP not available")

# Configuration
N_LOFI = 72
N_HIFI = 9
BO_TRIALS = 300
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

    X_all, y_all_hf, y_all_lf = [], [], []
    for c in all_combinations:
        X_all.append(c['labels'])
        y_all_hf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))
        y_all_lf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_all': np.array(X_all, dtype=np.float32),
        'y_all': np.array(y_all_hf, dtype=np.float32),
        'y_all_lf': np.array(y_all_lf, dtype=np.float32),
        'hifi_idx': hifi_idx,
        'lofi_idx': lofi_idx
    }


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rmse, r2


# ============== MFGP ==============

def run_mfgp(data, device, seed):
    if not MFGP_AVAILABLE:
        return None, None
    set_seeds(seed)
    mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred, mfgp_std = mfgp.predict(data['X_all'], return_std=True)
    return mfgp_pred, mfgp_std


# ============== DNGO-Base ==============

def optimize_dngo_base(data, device, seed, n_trials=BO_TRIALS):
    set_seeds(seed)

    def objective(trial):
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        pretrain_lr = trial.suggest_float('pretrain_lr', 1e-4, 1e-2, log=True)
        finetune_lr = trial.suggest_float('finetune_lr', 1e-5, 1e-3, log=True)
        pretrain_epochs = trial.suggest_int('pretrain_epochs', 100, 500)
        finetune_epochs = trial.suggest_int('finetune_epochs', 50, 300)

        model = TransferLearningDNN(
            input_dim=data['X_low'].shape[1],
            hidden_dim=hidden_dim,
            device=device,
            use_hyperparameter_bo=False,
            activation='tanh'
        )

        model.pretrain(data['X_low'], data['y_low'], epochs=pretrain_epochs, lr=pretrain_lr, verbose=False)
        model.finetune(data['X_high'], data['y_high'], epochs=finetune_epochs, lr=finetune_lr, verbose=False)

        features_high = model.extract_features(data['X_high'])
        blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr.fit(features_high, data['y_high'])

        pred, _ = blr.predict_batch(features_high)
        rmse = np.sqrt(np.mean((pred - data['y_high'])**2))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_dngo_base(data, device, seed, params=None):
    set_seeds(seed)

    if params is None:
        params = {'hidden_dim': 64, 'pretrain_lr': 1e-3, 'finetune_lr': 1e-4,
                  'pretrain_epochs': 300, 'finetune_epochs': 150}

    model = TransferLearningDNN(
        input_dim=data['X_low'].shape[1],
        hidden_dim=params['hidden_dim'],
        device=device,
        use_hyperparameter_bo=False,
        activation='tanh'
    )

    model.pretrain(data['X_low'], data['y_low'], epochs=params['pretrain_epochs'], lr=params['pretrain_lr'], verbose=False)
    model.finetune(data['X_high'], data['y_high'], epochs=params['finetune_epochs'], lr=params['finetune_lr'], verbose=False)

    features_high = model.extract_features(data['X_high'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features_high, data['y_high'])

    features_all = model.extract_features(data['X_all'])
    pred_all, var_all = blr.predict_batch(features_all)

    return pred_all, np.sqrt(var_all), model


def run_pretrain_base(data, device, seed, params=None):
    set_seeds(seed)

    if params is None:
        params = {'hidden_dim': 64, 'pretrain_lr': 1e-3, 'pretrain_epochs': 300}

    model = TransferLearningDNN(
        input_dim=data['X_low'].shape[1],
        hidden_dim=params.get('hidden_dim', 64),
        device=device,
        use_hyperparameter_bo=False,
        activation='tanh'
    )

    model.pretrain(data['X_low'], data['y_low'],
                   epochs=params.get('pretrain_epochs', 300),
                   lr=params.get('pretrain_lr', 1e-3),
                   verbose=False)

    features_low = model.extract_features(data['X_low'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features_low, data['y_low'])

    features_all = model.extract_features(data['X_all'])
    pred_all, var_all = blr.predict_batch(features_all)

    return pred_all, np.sqrt(var_all), model


# ============== GP-mimic (기존 Intermediate) ==============

def optimize_gpmimic(data, device, seed, n_trials=BO_TRIALS):
    set_seeds(seed)

    def objective(trial):
        shared_width = trial.suggest_categorical('shared_width', [32, 64, 128])
        num_shared_layers = trial.suggest_int('num_shared_layers', 2, 4)
        hf_width = trial.suggest_categorical('hf_width', [16, 32, 64])
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 100, 500)

        model = GPmimicDNGO(
            input_dim=data['X_low'].shape[1],
            shared_width=shared_width,
            num_shared_layers=num_shared_layers,
            hf_hidden_dims=[hf_width],
            device=device,
            alpha=alpha
        )

        history = model.train_joint(
            data['X_low'], data['y_low'],
            data['X_high'], data['y_high'],
            epochs=epochs, lr=lr, verbose=False
        )

        return history['best_val_loss']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_gpmimic(data, device, seed, params=None):
    set_seeds(seed)

    if params is None:
        params = {'shared_width': 64, 'num_shared_layers': 3, 'hf_width': 32,
                  'alpha': 0.7, 'lr': 1e-3, 'epochs': 300}

    model = GPmimicDNGO(
        input_dim=data['X_low'].shape[1],
        shared_width=params.get('shared_width', 64),
        num_shared_layers=params.get('num_shared_layers', 3),
        hf_hidden_dims=[params.get('hf_width', 32)],
        device=device,
        alpha=params.get('alpha', 0.7)
    )

    model.train_joint(
        data['X_low'], data['y_low'],
        data['X_high'], data['y_high'],
        epochs=params.get('epochs', 300),
        lr=params.get('lr', 1e-3),
        verbose=False
    )

    features_high = model.extract_features(data['X_high'], for_fidelity='high')
    blr = BLR_v2(alpha=1.0, beta=25.0)
    blr.fit(features_high, data['y_high'])

    features_all = model.extract_features(data['X_all'], for_fidelity='high')
    pred_all, var_all = blr.predict_batch(features_all)

    return pred_all, np.sqrt(var_all), model


# ============== True Intermediate (새 구조) ==============

def optimize_true_intermediate(data, device, seed, n_trials=BO_TRIALS, save_dir=None):
    set_seeds(seed)

    def objective(trial):
        # LF 네트워크 (넓은 범위)
        lf_width = trial.suggest_categorical('lf_width', [16, 32, 64, 128])
        lf_layers = trial.suggest_int('lf_layers', 1, 4)
        lf_dropout = trial.suggest_float('lf_dropout', 0.0, 0.5)

        # HF 네트워크 (넓은 범위 - 작은 것 포함)
        hf_width = trial.suggest_categorical('hf_width', [4, 8, 16, 32, 64])
        hf_layers = trial.suggest_int('hf_layers', 1, 3)
        hf_dropout = trial.suggest_float('hf_dropout', 0.0, 0.5)

        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 50, 500)

        # HF 정규화 파라미터
        hf_lr_ratio = trial.suggest_float('hf_lr_ratio', 0.001, 1.0, log=True)
        hf_weight_decay = trial.suggest_float('hf_weight_decay', 1e-6, 1e-1, log=True)

        model = TrueIntermediateDNGO(
            input_dim=data['X_low'].shape[1],
            lf_hidden_dims=[lf_width] * lf_layers,
            hf_hidden_dims=[hf_width] * hf_layers,
            device=device,
            alpha=alpha,
            lf_dropout=lf_dropout,
            hf_dropout=hf_dropout
        )

        history = model.train_joint(
            data['X_low'], data['y_low'],
            data['X_high'], data['y_high'],
            epochs=epochs, lr=lr,
            hf_lr_ratio=hf_lr_ratio,
            hf_weight_decay=hf_weight_decay,
            verbose=False
        )

        return history['best_val_loss']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # BO 결과 저장
    if save_dir is not None:
        save_optuna_results(study, save_dir, f'true_intermediate_seed{seed}')

    return study.best_params


def save_optuna_results(study, save_dir, prefix):
    """Optuna 탐색 결과 저장"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 모든 trial 결과를 DataFrame으로 저장
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'value': trial.value,
            'state': str(trial.state),
        }
        trial_dict.update(trial.params)
        trials_data.append(trial_dict)

    df_trials = pd.DataFrame(trials_data)
    df_trials.to_csv(save_path / f'{prefix}_all_trials.csv', index=False)

    # Best trial 정보 저장
    best_info = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params
    }
    with open(save_path / f'{prefix}_best.json', 'w') as f:
        json.dump(best_info, f, indent=2)

    print(f"    Saved BO results: {prefix}_all_trials.csv, {prefix}_best.json")


def run_true_intermediate(data, device, seed, params=None):
    set_seeds(seed)

    if params is None:
        params = {'lf_width': 64, 'lf_layers': 2, 'hf_width': 16, 'hf_layers': 1,
                  'alpha': 0.5, 'lr': 1e-3, 'epochs': 300,
                  'hf_lr_ratio': 0.1, 'hf_weight_decay': 1e-3,
                  'lf_dropout': 0.0, 'hf_dropout': 0.0}

    model = TrueIntermediateDNGO(
        input_dim=data['X_low'].shape[1],
        lf_hidden_dims=[params.get('lf_width', 64)] * params.get('lf_layers', 2),
        hf_hidden_dims=[params.get('hf_width', 16)] * params.get('hf_layers', 1),
        device=device,
        alpha=params.get('alpha', 0.5),
        lf_dropout=params.get('lf_dropout', 0.0),
        hf_dropout=params.get('hf_dropout', 0.0)
    )

    model.train_joint(
        data['X_low'], data['y_low'],
        data['X_high'], data['y_high'],
        epochs=params.get('epochs', 300),
        lr=params.get('lr', 1e-3),
        hf_lr_ratio=params.get('hf_lr_ratio', 0.1),
        hf_weight_decay=params.get('hf_weight_decay', 1e-3),
        verbose=False
    )

    features_high = model.extract_features(data['X_high'], for_fidelity='high')
    blr = BLR_v2(alpha=1.0, beta=25.0)
    blr.fit(features_high, data['y_high'])

    features_all = model.extract_features(data['X_all'], for_fidelity='high')
    pred_all, var_all = blr.predict_batch(features_all)

    return pred_all, np.sqrt(var_all), model


# ============== Visualization ==============

def get_composition_sort_idx(all_combinations):
    return np.arange(len(all_combinations))


def create_5panel_visualization(fold_idx, seed, data, results, stds, param_space, output_dir):
    n_all = len(data['X_all'])
    hifi_idx = data['hifi_idx']
    lofi_idx = data['lofi_idx']

    y_all_hf = data['y_all']
    y_all_lf = data['y_all_lf']

    colors = {
        'mfgp': 'blue',
        'dngo_base': 'orange',
        'gpmimic': 'green',
        'true_inter': 'purple',
        'pretrain_base': 'red'
    }

    fold_dir = output_dir / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ============== 1. 조합순 정렬 ==============
    sort_idx_comp = get_composition_sort_idx(data['X_all'])

    org_names = param_space['organic']
    cat_names = param_space['cation']
    ani_names = param_space['anion']
    n_cat = len(cat_names)
    n_ani = len(ani_names)
    group_size = n_cat * n_ani

    comp_labels = []
    for i in range(len(org_names)):
        for j in range(len(cat_names)):
            for k in range(len(ani_names)):
                org_short = org_names[i][:4].capitalize()
                cat_short = cat_names[j][:2].capitalize()
                ani_short = ani_names[k][:2].capitalize()
                comp_labels.append(f"{org_short}-{cat_short}-{ani_short}")

    _plot_5panel(
        fold_idx, seed, sort_idx_comp, y_all_hf, y_all_lf,
        results, stds, hifi_idx, lofi_idx, param_space,
        comp_labels, colors, fold_dir, 'predictions_by_composition.png',
        'Sorted by Composition'
    )

    # ============== 2. 실제값순 정렬 (HF 기준) ==============
    sort_idx_value = np.argsort(y_all_hf)
    value_labels = [comp_labels[i] for i in sort_idx_value]

    _plot_5panel(
        fold_idx, seed, sort_idx_value, y_all_hf, y_all_lf,
        results, stds, hifi_idx, lofi_idx, param_space,
        value_labels, colors, fold_dir, 'predictions_by_value.png',
        'Sorted by HF Value'
    )

    return fold_dir


def _plot_5panel(fold_idx, seed, sort_idx, y_all_hf, y_all_lf,
                 results, stds, hifi_idx, lofi_idx, param_space,
                 x_labels, colors, output_dir, filename, sort_type):

    y_sorted_hf = y_all_hf[sort_idx]
    y_sorted_lf = y_all_lf[sort_idx]
    x_axis = np.arange(len(y_sorted_hf))

    hf_train_mask = np.isin(sort_idx, hifi_idx)
    lf_train_mask = np.isin(sort_idx, lofi_idx)

    n_cat = len(param_space['cation'])
    n_ani = len(param_space['anion'])
    group_size = n_cat * n_ani

    fig, axes = plt.subplots(5, 1, figsize=(24, 30))

    models_info = [
        ('MFGP', 'mfgp', y_sorted_hf, hf_train_mask, colors['mfgp'], 'vs HF'),
        ('DNGO-Base', 'dngo_base', y_sorted_hf, hf_train_mask, colors['dngo_base'], 'vs HF'),
        ('GP-mimic', 'gpmimic', y_sorted_hf, hf_train_mask, colors['gpmimic'], 'vs HF'),
        ('True-Intermediate', 'true_inter', y_sorted_hf, hf_train_mask, colors['true_inter'], 'vs HF'),
        ('Pretrain-Base', 'pretrain_base', y_sorted_lf, lf_train_mask, colors['pretrain_base'], 'vs LF'),
    ]

    for ax_idx, (name, key, y_true, train_mask, color, compare_type) in enumerate(models_info):
        ax = axes[ax_idx]

        pred = results.get(f'{key}_pred')
        std = stds.get(f'{key}_std')

        if pred is None:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{name} ({compare_type})')
            continue

        pred_sorted = pred[sort_idx]
        std_sorted = std[sort_idx] if std is not None else np.zeros_like(pred_sorted)

        ax.fill_between(x_axis, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                        alpha=0.3, color=color, label='±2σ')

        ax.plot(x_axis, pred_sorted, color=color, linewidth=0.8, alpha=0.7, label=f'{name} Predicted')

        ax.scatter(x_axis[~train_mask], y_true[~train_mask], c='black', s=15, zorder=5,
                   label=f'Test {compare_type.split()[1]}', alpha=0.6)

        if compare_type == 'vs HF':
            ax.scatter(x_axis[train_mask], y_true[train_mask], c='red', s=100, marker='*',
                       zorder=6, label=f'Train HF ({np.sum(train_mask)})', edgecolors='darkred')
        else:
            ax.scatter(x_axis[train_mask], y_true[train_mask], c='red', s=40, marker='o',
                       zorder=6, label=f'Train LF ({np.sum(train_mask)})', edgecolors='darkred', alpha=0.7)

        test_mask = ~train_mask
        rmse, r2 = calc_metrics(y_true[test_mask], pred_sorted[test_mask])

        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'{name} ({compare_type}): RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if 'composition' in filename:
            for i in range(1, len(param_space['organic'])):
                ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)

    axes[-1].set_xticks(x_axis)
    axes[-1].set_xticklabels(x_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel('Composition (Organic-Cation-Anion)', fontsize=12)

    if 'composition' in filename:
        org_names = param_space['organic']
        for i, org in enumerate(org_names):
            mid_x = i * group_size + group_size / 2
            y_top = axes[0].get_ylim()[1]
            y_range = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
            axes[0].text(mid_x, y_top + y_range * 0.08, org.capitalize(),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - True Intermediate vs GP-mimic ({sort_type})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF, {N_HIFI} HF")
    print(f"Optuna trials: {BO_TRIALS}")
    print(f"Models: MFGP, DNGO-Base, GP-mimic, True-Intermediate, Pretrain-Base")
    print(f"Folds: {len(SEEDS)}")

    lookup, all_combinations, param_space = load_base_data()
    print(f"Total combinations: {len(all_combinations)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_true_intermediate'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results_list = []
    total_start = time.time()

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{len(SEEDS)} (seed={seed})")
        print('='*60)

        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        train_idx = data['hifi_idx']
        n_all = len(data['X_all'])
        test_mask = np.ones(n_all, dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]
        y_test_hf = data['y_all'][test_idx]
        y_test_lf = data['y_all_lf'][test_idx]

        fold_results = {'fold': fold_idx, 'seed': seed}
        preds = {}
        stds_dict = {}

        # 1. MFGP
        print(f"\n  [1/5] MFGP...")
        t0 = time.time()
        mfgp_pred, mfgp_std = run_mfgp(data, device, seed)
        mfgp_time = time.time() - t0
        if mfgp_pred is not None:
            mfgp_rmse, mfgp_r2 = calc_metrics(y_test_hf, mfgp_pred[test_idx])
            print(f"        RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f} ({mfgp_time:.1f}s)")
            fold_results.update({'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2})
            preds['mfgp_pred'] = mfgp_pred
            stds_dict['mfgp_std'] = mfgp_std
        else:
            print(f"        Not available")
            fold_results.update({'mfgp_rmse': None, 'mfgp_r2': None})

        # 2. DNGO-Base
        print(f"\n  [2/5] DNGO-Base (HP tuning {BO_TRIALS} trials)...")
        t0 = time.time()
        dngo_base_params = optimize_dngo_base(data, device, seed, n_trials=BO_TRIALS)
        print(f"        Best params: hd={dngo_base_params['hidden_dim']}, "
              f"plr={dngo_base_params['pretrain_lr']:.2e}, flr={dngo_base_params['finetune_lr']:.2e}")
        dngo_base_pred, dngo_base_std, _ = run_dngo_base(data, device, seed, params=dngo_base_params)
        dngo_base_time = time.time() - t0
        dngo_base_rmse, dngo_base_r2 = calc_metrics(y_test_hf, dngo_base_pred[test_idx])
        print(f"        RMSE={dngo_base_rmse:.4f}, R²={dngo_base_r2:.4f} ({dngo_base_time:.1f}s)")
        fold_results.update({'dngo_base_rmse': dngo_base_rmse, 'dngo_base_r2': dngo_base_r2})
        preds['dngo_base_pred'] = dngo_base_pred
        stds_dict['dngo_base_std'] = dngo_base_std

        # 3. GP-mimic (기존 Intermediate)
        print(f"\n  [3/5] GP-mimic (HP tuning {BO_TRIALS} trials)...")
        t0 = time.time()
        gpmimic_params = optimize_gpmimic(data, device, seed, n_trials=BO_TRIALS)
        print(f"        Best params: sw={gpmimic_params['shared_width']}, "
              f"nl={gpmimic_params['num_shared_layers']}, alpha={gpmimic_params['alpha']:.2f}")
        gpmimic_pred, gpmimic_std, _ = run_gpmimic(data, device, seed, params=gpmimic_params)
        gpmimic_time = time.time() - t0
        gpmimic_rmse, gpmimic_r2 = calc_metrics(y_test_hf, gpmimic_pred[test_idx])
        print(f"        RMSE={gpmimic_rmse:.4f}, R²={gpmimic_r2:.4f} ({gpmimic_time:.1f}s)")
        fold_results.update({'gpmimic_rmse': gpmimic_rmse, 'gpmimic_r2': gpmimic_r2})
        preds['gpmimic_pred'] = gpmimic_pred
        stds_dict['gpmimic_std'] = gpmimic_std

        # 4. True-Intermediate (새 구조)
        print(f"\n  [4/5] True-Intermediate (HP tuning {BO_TRIALS} trials)...")
        t0 = time.time()
        bo_save_dir = output_dir / 'bo_results'
        true_inter_params = optimize_true_intermediate(data, device, seed, n_trials=BO_TRIALS, save_dir=bo_save_dir)
        print(f"        Best params: lfw={true_inter_params['lf_width']}, lfl={true_inter_params['lf_layers']}, "
              f"hfw={true_inter_params['hf_width']}, hfl={true_inter_params['hf_layers']}, "
              f"alpha={true_inter_params['alpha']:.2f}, hf_lr_ratio={true_inter_params['hf_lr_ratio']:.4f}, "
              f"hf_wd={true_inter_params['hf_weight_decay']:.2e}, lf_do={true_inter_params['lf_dropout']:.2f}, "
              f"hf_do={true_inter_params['hf_dropout']:.2f}")
        true_inter_pred, true_inter_std, _ = run_true_intermediate(data, device, seed, params=true_inter_params)
        true_inter_time = time.time() - t0
        true_inter_rmse, true_inter_r2 = calc_metrics(y_test_hf, true_inter_pred[test_idx])
        print(f"        RMSE={true_inter_rmse:.4f}, R²={true_inter_r2:.4f} ({true_inter_time:.1f}s)")
        fold_results.update({'true_inter_rmse': true_inter_rmse, 'true_inter_r2': true_inter_r2})
        fold_results.update({f'true_inter_{k}': v for k, v in true_inter_params.items()})
        preds['true_inter_pred'] = true_inter_pred
        stds_dict['true_inter_std'] = true_inter_std

        # 5. Pretrain-Base (LF only)
        print(f"\n  [5/5] Pretrain-Base...")
        t0 = time.time()
        pretrain_base_pred, pretrain_base_std, _ = run_pretrain_base(data, device, seed, params=dngo_base_params)
        pretrain_base_time = time.time() - t0
        pretrain_base_rmse, pretrain_base_r2 = calc_metrics(y_test_lf, pretrain_base_pred[test_idx])
        print(f"        RMSE={pretrain_base_rmse:.4f}, R²={pretrain_base_r2:.4f} (vs LF) ({pretrain_base_time:.1f}s)")
        fold_results.update({'pretrain_base_rmse': pretrain_base_rmse, 'pretrain_base_r2': pretrain_base_r2})
        preds['pretrain_base_pred'] = pretrain_base_pred
        stds_dict['pretrain_base_std'] = pretrain_base_std

        # Visualization
        fold_dir = create_5panel_visualization(fold_idx, seed, data, preds, stds_dict, param_space, output_dir)
        print(f"\n  Visualization saved: {fold_dir}")

        results_list.append(fold_results)

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<20} {'Avg RMSE':<12} {'Avg R²':<12} {'Compare'}")
    print('-' * 60)

    if df['mfgp_r2'].notna().any():
        print(f"{'MFGP':<20} {df['mfgp_rmse'].mean():.4f}       {df['mfgp_r2'].mean():.4f}       vs HF")
    print(f"{'DNGO-Base':<20} {df['dngo_base_rmse'].mean():.4f}       {df['dngo_base_r2'].mean():.4f}       vs HF")
    print(f"{'GP-mimic':<20} {df['gpmimic_rmse'].mean():.4f}       {df['gpmimic_r2'].mean():.4f}       vs HF")
    print(f"{'True-Intermediate':<20} {df['true_inter_rmse'].mean():.4f}       {df['true_inter_r2'].mean():.4f}       vs HF")
    print(f"{'Pretrain-Base':<20} {df['pretrain_base_rmse'].mean():.4f}       {df['pretrain_base_r2'].mean():.4f}       vs LF")

    # GP-mimic vs True-Intermediate
    print(f"\n{'='*60}")
    print("True-Intermediate vs GP-mimic")
    print('='*60)
    improvement = df['true_inter_r2'].mean() - df['gpmimic_r2'].mean()
    true_wins = (df['true_inter_r2'] > df['gpmimic_r2']).sum()
    print(f"True-Intermediate Avg R²: {df['true_inter_r2'].mean():.4f}")
    print(f"GP-mimic Avg R²:          {df['gpmimic_r2'].mean():.4f}")
    print(f"Improvement: {'+' if improvement > 0 else ''}{improvement:.4f}")
    print(f"Folds where True > GP-mimic: {true_wins}/{len(df)}")

    print(f"\nTotal time: {total_time/60:.2f} minutes")

    # Save results
    df.to_csv(output_dir / 'results_summary.csv', index=False)
    print(f"Results saved to: {output_dir / 'results_summary.csv'}")

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    models = ['MFGP', 'DNGO-Base', 'GP-mimic', 'True-Intermediate']
    r2_means = [
        df['mfgp_r2'].mean() if df['mfgp_r2'].notna().any() else 0,
        df['dngo_base_r2'].mean(),
        df['gpmimic_r2'].mean(),
        df['true_inter_r2'].mean()
    ]
    r2_stds = [
        df['mfgp_r2'].std() if df['mfgp_r2'].notna().any() else 0,
        df['dngo_base_r2'].std(),
        df['gpmimic_r2'].std(),
        df['true_inter_r2'].std()
    ]

    colors_bar = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    bars = ax.bar(models, r2_means, yerr=r2_stds, capsize=5, color=colors_bar, alpha=0.7)
    ax.set_ylabel('R² Score (vs High-Fidelity)', fontsize=12)
    ax.set_title(f'HF Prediction: True-Intermediate vs GP-mimic ({len(SEEDS)}-fold, Optuna {BO_TRIALS} trials)', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    for bar, mean in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary chart saved: {output_dir / 'summary_comparison.png'}")


if __name__ == '__main__':
    main()
