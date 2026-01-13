"""
DNGO-AllInOne vs Baseline 모델 비교 실험

비교 모델 (5개):
1. MFGP (Multi-Fidelity Gaussian Process) - vs HF
2. DNGO-Base (기존 Transfer Learning DNGO) - vs HF
3. DNGO-AllInOne (논문 기반 All-in-one MFNN) - vs HF
4. Pretrain-Base (DNGO pretrain only) - vs LF
5. Pretrain-AllInOne (AllInOne LF output only) - vs LF

하이퍼파라미터 튜닝: Optuna (alpha 포함)
평가 방식: 3-fold cross validation
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

# DNGO-AllInOne import
from dngo_intermediate import IntermediateDNGO as AllInOneDNGO, IntermediateMFNN as AllInOneMFNN, BayesianLinearRegression as AllInOneBLR

# MFGP import
try:
    from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
    from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
    import GPy
    MFGP_AVAILABLE = True
except ImportError:
    MFGP_AVAILABLE = False
    print("Warning: MFGP not available")

# Configuration
N_LOFI = 72   # Low-fidelity 샘플 수
N_HIFI = 9    # High-fidelity 샘플 수
BO_TRIALS = 300  # Optuna trials for hyperparameter tuning
SEEDS = [42, 123, 456]  # 3-fold for testing


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_data():
    """Perovskite 데이터 로드"""
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
    """LF/HF 데이터 생성"""
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
    """RMSE, R² 계산"""
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rmse, r2


def run_mfgp(data, seed):
    """MFGP 실행"""
    if not MFGP_AVAILABLE:
        return None, None, None

    set_seeds(seed)

    X_train, Y_train = convert_xy_lists_to_arrays(
        [data['X_low'], data['X_high']],
        [data['y_low'].reshape(-1, 1), data['y_high'].reshape(-1, 1)]
    )

    kernels = [GPy.kern.RBF(data['X_low'].shape[1]), GPy.kern.RBF(data['X_low'].shape[1])]
    lin_mf_kernel = GPy.util.multioutput.LCM(input_dim=data['X_low'].shape[1], num_outputs=2, kernels_list=kernels)

    mfgp_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    mfgp_model.mixed_noise.Gaussian_noise.fix(0)
    mfgp_model.mixed_noise.Gaussian_noise_1.fix(0)
    mfgp_model.optimize()

    X_all_hf = np.column_stack([data['X_all'], np.ones(len(data['X_all']))])
    mfgp_pred, mfgp_var = mfgp_model.predict(X_all_hf)

    return mfgp_pred.flatten(), np.sqrt(mfgp_var.flatten()), mfgp_model


# ============== DNGO-Base (기존 Transfer Learning) ==============

def optimize_dngo_base(data, device, seed, n_trials=BO_TRIALS):
    """DNGO-Base 하이퍼파라미터 최적화"""
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

        # Validation: HF 예측 성능
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
    """DNGO-Base 실행"""
    set_seeds(seed)

    if params is None:
        params = {
            'hidden_dim': 64,
            'pretrain_lr': 1e-3,
            'finetune_lr': 1e-4,
            'pretrain_epochs': 300,
            'finetune_epochs': 150
        }

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
    """Pretrain-Base (LF만 학습) 실행"""
    set_seeds(seed)

    if params is None:
        params = {
            'hidden_dim': 64,
            'pretrain_lr': 1e-3,
            'pretrain_epochs': 300
        }

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


# ============== DNGO-AllInOne ==============

def optimize_dngo_allinone(data, device, seed, n_trials=BO_TRIALS):
    """DNGO-AllInOne 하이퍼파라미터 최적화 (alpha 포함)"""
    set_seeds(seed)

    def objective(trial):
        shared_width = trial.suggest_categorical('shared_width', [32, 64, 128])
        num_shared_layers = trial.suggest_int('num_shared_layers', 2, 4)
        hf_width = trial.suggest_categorical('hf_width', [16, 32, 64])
        alpha = trial.suggest_float('alpha', 0.0, 1.0)  # HF loss weight
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 100, 500)
        lambda_reg = trial.suggest_float('lambda_reg', 1e-6, 1e-3, log=True)

        model = AllInOneDNGO(
            input_dim=data['X_low'].shape[1],
            shared_width=shared_width,
            num_shared_layers=num_shared_layers,
            hf_hidden_dims=[hf_width],
            device=device,
            alpha=alpha,
            lambda_reg=lambda_reg
        )

        history = model.train_joint(
            data['X_low'], data['y_low'],
            data['X_high'], data['y_high'],
            epochs=epochs,
            lr=lr,
            verbose=False
        )

        return history['best_val_loss']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def run_dngo_allinone(data, device, seed, params=None):
    """DNGO-AllInOne 실행"""
    set_seeds(seed)

    if params is None:
        params = {
            'shared_width': 64,
            'num_shared_layers': 3,
            'hf_width': 32,
            'alpha': 0.7,
            'lr': 1e-3,
            'epochs': 300,
            'lambda_reg': 1e-4
        }

    model = AllInOneDNGO(
        input_dim=data['X_low'].shape[1],
        shared_width=params.get('shared_width', 64),
        num_shared_layers=params.get('num_shared_layers', 3),
        hf_hidden_dims=[params.get('hf_width', 32)],
        device=device,
        alpha=params.get('alpha', 0.7),
        lambda_reg=params.get('lambda_reg', 1e-4)
    )

    model.train_joint(
        data['X_low'], data['y_low'],
        data['X_high'], data['y_high'],
        epochs=params.get('epochs', 300),
        lr=params.get('lr', 1e-3),
        verbose=False
    )

    # HF prediction with BLR
    features_high = model.extract_features(data['X_high'], for_fidelity='high')
    blr = AllInOneBLR(alpha=1.0, beta=25.0)
    blr.fit(features_high, data['y_high'])

    features_all = model.extract_features(data['X_all'], for_fidelity='high')
    pred_all, var_all = blr.predict_batch(features_all)

    return pred_all, np.sqrt(var_all), model


def run_pretrain_allinone(data, device, seed, params=None):
    """Pretrain-AllInOne (LF output만 사용) 실행"""
    set_seeds(seed)

    if params is None:
        params = {
            'shared_width': 64,
            'num_shared_layers': 3,
            'hf_width': 32,
            'alpha': 0.7,
            'lr': 1e-3,
            'epochs': 300,
            'lambda_reg': 1e-4
        }

    model = AllInOneDNGO(
        input_dim=data['X_low'].shape[1],
        shared_width=params.get('shared_width', 64),
        num_shared_layers=params.get('num_shared_layers', 3),
        hf_hidden_dims=[params.get('hf_width', 32)],
        device=device,
        alpha=params.get('alpha', 0.7),
        lambda_reg=params.get('lambda_reg', 1e-4)
    )

    model.train_joint(
        data['X_low'], data['y_low'],
        data['X_high'], data['y_high'],
        epochs=params.get('epochs', 300),
        lr=params.get('lr', 1e-3),
        verbose=False
    )

    # LF prediction (shared features + LF head)
    features_low = model.extract_features(data['X_low'], for_fidelity='low')
    blr = AllInOneBLR(alpha=1.0, beta=25.0)
    blr.fit(features_low, data['y_low'])

    features_all = model.extract_features(data['X_all'], for_fidelity='low')
    pred_all, var_all = blr.predict_batch(features_all)

    return pred_all, np.sqrt(var_all), model


# ============== Visualization ==============

def get_composition_sort_idx(all_combinations):
    """조합순 정렬 인덱스 반환 (organic > cation > anion)"""
    return np.arange(len(all_combinations))


def create_5panel_visualization(fold_idx, seed, data, results, stds, param_space, output_dir):
    """
    5패널 시각화 생성 (2장: 조합순, 실제값순)

    Args:
        fold_idx: fold 번호
        seed: random seed
        data: 데이터 딕셔너리
        results: 예측 결과 딕셔너리
        stds: 불확실성(std) 딕셔너리
        param_space: 파라미터 공간
        output_dir: 출력 디렉토리
    """
    n_all = len(data['X_all'])
    hifi_idx = data['hifi_idx']
    lofi_idx = data['lofi_idx']

    y_all_hf = data['y_all']
    y_all_lf = data['y_all_lf']

    # 모델별 색상
    colors = {
        'mfgp': 'blue',
        'dngo_base': 'orange',
        'dngo_aio': 'green',
        'pretrain_base': 'red',
        'pretrain_aio': 'purple'
    }

    # fold 디렉토리 생성
    fold_dir = output_dir / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ============== 1. 조합순 정렬 ==============
    sort_idx_comp = get_composition_sort_idx(data['X_all'])

    # 조합 라벨 생성
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
    """5패널 플롯 생성"""

    y_sorted_hf = y_all_hf[sort_idx]
    y_sorted_lf = y_all_lf[sort_idx]
    x_axis = np.arange(len(y_sorted_hf))

    # train mask
    hf_train_mask = np.isin(sort_idx, hifi_idx)
    lf_train_mask = np.isin(sort_idx, lofi_idx)

    n_cat = len(param_space['cation'])
    n_ani = len(param_space['anion'])
    group_size = n_cat * n_ani

    fig, axes = plt.subplots(5, 1, figsize=(24, 30))

    # 모델 정보: (이름, pred_key, std_key, y_true, train_mask, color, compare_type)
    models_info = [
        ('MFGP', 'mfgp', y_sorted_hf, hf_train_mask, colors['mfgp'], 'vs HF'),
        ('DNGO-Base', 'dngo_base', y_sorted_hf, hf_train_mask, colors['dngo_base'], 'vs HF'),
        ('DNGO-AllInOne', 'dngo_aio', y_sorted_hf, hf_train_mask, colors['dngo_aio'], 'vs HF'),
        ('Pretrain-Base', 'pretrain_base', y_sorted_lf, lf_train_mask, colors['pretrain_base'], 'vs LF'),
        ('Pretrain-AllInOne', 'pretrain_aio', y_sorted_lf, lf_train_mask, colors['pretrain_aio'], 'vs LF'),
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

        # 불확실성 밴드
        ax.fill_between(x_axis, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                        alpha=0.3, color=color, label='±2σ')

        # 예측선
        ax.plot(x_axis, pred_sorted, color=color, linewidth=0.8, alpha=0.7, label=f'{name} Predicted')

        # 테스트 포인트
        ax.scatter(x_axis[~train_mask], y_true[~train_mask], c='black', s=15, zorder=5,
                   label=f'Test {compare_type.split()[1]}', alpha=0.6)

        # 훈련 포인트
        if compare_type == 'vs HF':
            ax.scatter(x_axis[train_mask], y_true[train_mask], c='red', s=100, marker='*',
                       zorder=6, label=f'Train HF ({np.sum(train_mask)})', edgecolors='darkred')
        else:
            ax.scatter(x_axis[train_mask], y_true[train_mask], c='red', s=40, marker='o',
                       zorder=6, label=f'Train LF ({np.sum(train_mask)})', edgecolors='darkred', alpha=0.7)

        # 메트릭 계산
        test_mask = ~train_mask
        rmse, r2 = calc_metrics(y_true[test_mask], pred_sorted[test_mask])

        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'{name} ({compare_type}): RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Organic 구분선 (조합순일 때만)
        if 'composition' in filename:
            for i in range(1, len(param_space['organic'])):
                ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)

    # X축 라벨 (마지막 패널만)
    axes[-1].set_xticks(x_axis)
    axes[-1].set_xticklabels(x_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel('Composition (Organic-Cation-Anion)', fontsize=12)

    # Organic 라벨 (조합순일 때만)
    if 'composition' in filename:
        org_names = param_space['organic']
        for i, org in enumerate(org_names):
            mid_x = i * group_size + group_size / 2
            y_top = axes[0].get_ylim()[1]
            y_range = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
            axes[0].text(mid_x, y_top + y_range * 0.08, org.capitalize(),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - 5 Model Comparison ({sort_type})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF samples, {N_HIFI} HF samples")
    print(f"Optuna trials: {BO_TRIALS}")
    print(f"Models: MFGP, DNGO-Base, DNGO-AllInOne, Pretrain-Base, Pretrain-AllInOne")
    print(f"Folds: {len(SEEDS)}")

    lookup, all_combinations, param_space = load_base_data()
    print(f"Total combinations: {len(all_combinations)}")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_allinone_comparison'
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
        stds = {}

        # 1. MFGP
        print(f"\n  [1/5] MFGP...")
        t0 = time.time()
        mfgp_pred, mfgp_std, _ = run_mfgp(data, seed)
        mfgp_time = time.time() - t0
        if mfgp_pred is not None:
            mfgp_rmse, mfgp_r2 = calc_metrics(y_test_hf, mfgp_pred[test_idx])
            print(f"        RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f} ({mfgp_time:.1f}s)")
            fold_results.update({'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2})
            preds['mfgp_pred'] = mfgp_pred
            stds['mfgp_std'] = mfgp_std
        else:
            print(f"        Not available")
            fold_results.update({'mfgp_rmse': None, 'mfgp_r2': None})

        # 2. DNGO-Base with HP optimization
        print(f"\n  [2/5] DNGO-Base (HP tuning {BO_TRIALS} trials)...")
        t0 = time.time()
        dngo_base_params = optimize_dngo_base(data, device, seed, n_trials=BO_TRIALS)
        print(f"        Best params: {dngo_base_params}")
        dngo_base_pred, dngo_base_std, dngo_base_model = run_dngo_base(data, device, seed, params=dngo_base_params)
        dngo_base_time = time.time() - t0
        dngo_base_rmse, dngo_base_r2 = calc_metrics(y_test_hf, dngo_base_pred[test_idx])
        print(f"        RMSE={dngo_base_rmse:.4f}, R²={dngo_base_r2:.4f} ({dngo_base_time:.1f}s)")
        fold_results.update({'dngo_base_rmse': dngo_base_rmse, 'dngo_base_r2': dngo_base_r2})
        fold_results.update({f'dngo_base_{k}': v for k, v in dngo_base_params.items()})
        preds['dngo_base_pred'] = dngo_base_pred
        stds['dngo_base_std'] = dngo_base_std

        # 3. Pretrain-Base (uses same params as DNGO-Base)
        print(f"\n  [3/5] Pretrain-Base...")
        t0 = time.time()
        pretrain_base_pred, pretrain_base_std, _ = run_pretrain_base(data, device, seed, params=dngo_base_params)
        pretrain_base_time = time.time() - t0
        pretrain_base_rmse, pretrain_base_r2 = calc_metrics(y_test_lf, pretrain_base_pred[test_idx])
        print(f"        RMSE={pretrain_base_rmse:.4f}, R²={pretrain_base_r2:.4f} (vs LF) ({pretrain_base_time:.1f}s)")
        fold_results.update({'pretrain_base_rmse': pretrain_base_rmse, 'pretrain_base_r2': pretrain_base_r2})
        preds['pretrain_base_pred'] = pretrain_base_pred
        stds['pretrain_base_std'] = pretrain_base_std

        # 4. DNGO-AllInOne with HP optimization (alpha 포함)
        print(f"\n  [4/5] DNGO-AllInOne (HP tuning {BO_TRIALS} trials, alpha included)...")
        t0 = time.time()
        dngo_aio_params = optimize_dngo_allinone(data, device, seed, n_trials=BO_TRIALS)
        print(f"        Best params: {dngo_aio_params}")
        print(f"        Best alpha: {dngo_aio_params.get('alpha', 0.7):.3f}")
        dngo_aio_pred, dngo_aio_std, dngo_aio_model = run_dngo_allinone(data, device, seed, params=dngo_aio_params)
        dngo_aio_time = time.time() - t0
        dngo_aio_rmse, dngo_aio_r2 = calc_metrics(y_test_hf, dngo_aio_pred[test_idx])
        print(f"        RMSE={dngo_aio_rmse:.4f}, R²={dngo_aio_r2:.4f} ({dngo_aio_time:.1f}s)")
        fold_results.update({'dngo_aio_rmse': dngo_aio_rmse, 'dngo_aio_r2': dngo_aio_r2})
        fold_results.update({f'dngo_aio_{k}': v for k, v in dngo_aio_params.items()})
        preds['dngo_aio_pred'] = dngo_aio_pred
        stds['dngo_aio_std'] = dngo_aio_std

        # 5. Pretrain-AllInOne (uses same params as DNGO-AllInOne)
        print(f"\n  [5/5] Pretrain-AllInOne...")
        t0 = time.time()
        pretrain_aio_pred, pretrain_aio_std, _ = run_pretrain_allinone(data, device, seed, params=dngo_aio_params)
        pretrain_aio_time = time.time() - t0
        pretrain_aio_rmse, pretrain_aio_r2 = calc_metrics(y_test_lf, pretrain_aio_pred[test_idx])
        print(f"        RMSE={pretrain_aio_rmse:.4f}, R²={pretrain_aio_r2:.4f} (vs LF) ({pretrain_aio_time:.1f}s)")
        fold_results.update({'pretrain_aio_rmse': pretrain_aio_rmse, 'pretrain_aio_r2': pretrain_aio_r2})
        preds['pretrain_aio_pred'] = pretrain_aio_pred
        stds['pretrain_aio_std'] = pretrain_aio_std

        # 5-panel visualization (2장: 조합순, 실제값순)
        fold_dir = create_5panel_visualization(fold_idx, seed, data, preds, stds, param_space, output_dir)
        print(f"\n  Visualization saved: {fold_dir}")

        results_list.append(fold_results)

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<25} {'Avg RMSE':<12} {'Avg R²':<12} {'Compare'}")
    print('-' * 65)

    if df['mfgp_r2'].notna().any():
        print(f"{'MFGP':<25} {df['mfgp_rmse'].mean():.4f}       {df['mfgp_r2'].mean():.4f}       vs HF")
    print(f"{'DNGO-Base':<25} {df['dngo_base_rmse'].mean():.4f}       {df['dngo_base_r2'].mean():.4f}       vs HF")
    print(f"{'DNGO-AllInOne':<25} {df['dngo_aio_rmse'].mean():.4f}       {df['dngo_aio_r2'].mean():.4f}       vs HF")
    print(f"{'Pretrain-Base':<25} {df['pretrain_base_rmse'].mean():.4f}       {df['pretrain_base_r2'].mean():.4f}       vs LF")
    print(f"{'Pretrain-AllInOne':<25} {df['pretrain_aio_rmse'].mean():.4f}       {df['pretrain_aio_r2'].mean():.4f}       vs LF")

    # HF comparison
    print(f"\n{'='*60}")
    print("DNGO-AllInOne vs DNGO-Base (HF prediction)")
    print('='*60)
    improvement_r2 = df['dngo_aio_r2'].mean() - df['dngo_base_r2'].mean()
    aio_wins = (df['dngo_aio_r2'] > df['dngo_base_r2']).sum()
    print(f"DNGO-AllInOne Avg R²: {df['dngo_aio_r2'].mean():.4f}")
    print(f"DNGO-Base Avg R²: {df['dngo_base_r2'].mean():.4f}")
    print(f"Improvement: {'+' if improvement_r2 > 0 else ''}{improvement_r2:.4f}")
    print(f"Folds where AllInOne > Base: {aio_wins}/{len(df)}")

    # Alpha analysis
    print(f"\n{'='*60}")
    print("Optimized Alpha Values (DNGO-AllInOne)")
    print('='*60)
    for _, row in df.iterrows():
        print(f"  Fold {int(row['fold'])}: alpha = {row.get('dngo_aio_alpha', 'N/A'):.3f}")
    if 'dngo_aio_alpha' in df.columns:
        print(f"  Mean alpha: {df['dngo_aio_alpha'].mean():.3f}")

    print(f"\nTotal time: {total_time/60:.2f} minutes")

    # Save results
    df.to_csv(output_dir / 'results_summary.csv', index=False)
    print(f"Results saved to: {output_dir / 'results_summary.csv'}")

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    models = ['MFGP', 'DNGO-Base', 'DNGO-AllInOne']
    r2_means = [
        df['mfgp_r2'].mean() if df['mfgp_r2'].notna().any() else 0,
        df['dngo_base_r2'].mean(),
        df['dngo_aio_r2'].mean()
    ]
    r2_stds = [
        df['mfgp_r2'].std() if df['mfgp_r2'].notna().any() else 0,
        df['dngo_base_r2'].std(),
        df['dngo_aio_r2'].std()
    ]

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    bars = ax.bar(models, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel('R² Score (vs High-Fidelity)', fontsize=12)
    ax.set_title(f'HF Prediction Comparison ({len(SEEDS)}-fold, Optuna {BO_TRIALS} trials)', fontsize=14)
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
