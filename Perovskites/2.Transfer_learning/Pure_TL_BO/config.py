"""
Transfer Learning Bayesian Optimization 실험 설정
"""

# 실험 파라미터
NUM_RUNS = 100
COST_BUDGET = 50.0
NUM_INIT_DESIGN = 10
HIGH_FIDELITY_RATIO = 0.2

# 파라미터 공간 정의
PARAM_SPACE = {
    "organic": [
        "ethylammonium", "propylammonium", "butylammonium", "isopropylammonium",
        "dimethylammonium", "acetamidinium", "methylammonium", "guanidinium",
        "hydroxylammonium", "formamidinium", "tetramethylammonium", "hydrazinium",
        "ammonium", "trimethylammonium", "azetidinium", "imidazolium"
    ],
    "cation": ["Ge", "Sn", "Pb"],
    "anion": ["F", "Cl", "Br", "I"]
}

# 라벨 범위 (1부터 시작)
PARAM_RANGES = [
    range(1, 17),  # organic_label: 1~16
    range(1, 4),   # cation_label: 1~3
    range(1, 5),   # anion_label: 1~4
]

# 목표값
MIN_HSE06_BANDGAP = 1.5249

# 모델 하이퍼파라미터
MODEL_CONFIG = {
    'input_dim': 3,
    'hidden_dim': 64,
    'pretrain_epochs': 300,
    'finetune_epochs': 150,
    'pretrain_lr': 1e-3,
    'finetune_lr': 1e-3
}

# BLR 하이퍼파라미터
BLR_CONFIG = {
    'alpha': 1.0,
    'beta': 25.0
}

# EI 파라미터
EI_CONFIG = {
    'xi': 0.01
}

# 파일 경로
DATA_PATHS = {
    'lookup_table': '../../0.Data/lookup_table.pkl'
}

# 출력 파일명
OUTPUT_FILES = {
    'timing_results': 'TL_timing_results.csv',
    'cost_results': 'TL_cumulative_cost.csv',
    'best_so_far_curve': 'TL_best_so_far_curve.csv',
    'all_results': 'TL_all_iter_results.csv',
    'multiple_runs_costs': 'transfer_learning_costs.csv'
}

# 시각화 설정
PLOT_CONFIG = {
    'figsize_large': (18, 7),
    'figsize_medium': (10, 6),
    'figsize_small': (8, 8),
    'dpi': 100
}

# GPU 설정
DEVICE_CONFIG = {
    'use_gpu': True,  # GPU 사용 여부
    'device': 'cpu'   # 기본값은 CPU, GPU 사용 시 자동으로 변경
} 