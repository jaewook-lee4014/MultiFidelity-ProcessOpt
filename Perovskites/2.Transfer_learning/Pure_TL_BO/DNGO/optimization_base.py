import numpy as np
import itertools
import time
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
# Universal imports - model specific imports are conditional based on model_type
from .models import BayesianLinearRegression  # BLR is used by both models
try:
    from .models import TransferLearningDNN  # Only needed for DNGO
except ImportError:
    TransferLearningDNN = None
from tqdm import tqdm
import torch.nn as nn


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement 계산
    
    Args:
        mu: 예측 평균
        sigma: 예측 표준편차
        y_best: 현재까지의 최적값
        xi: exploration parameter
        
    Returns:
        Expected Improvement 값
    """
    sigma = np.maximum(sigma, 1e-8)
    z = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei


def penalized_expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, 
                                  xi: float = 0.01, penalty_scale: float = 1.0, 
                                  penalty_func=None) -> np.ndarray:
    """
    Penalized Expected Improvement 계산
    """
    ei = expected_improvement(mu, sigma, y_best, xi)
    if penalty_func is not None:
        penalty = penalty_func(mu, sigma)
        ei = ei - penalty_scale * penalty
    return ei


def train_model(X_low: np.ndarray, y_low: np.ndarray, X_high: np.ndarray, y_high: np.ndarray,
                input_dim: int = 3, hidden_dim: int = 64, device: str = 'cpu',
                pretrain_epochs: int = 200, finetune_epochs: int = 100,
                pretrain_lr: float = 1e-3, finetune_lr: float = 1e-4,
                use_hyperparameter_bo: bool = False, pretrain_bo_trials: int = 0,
                finetune_bo_trials: int = 0, data_size: str = 'small',
                verbose: bool = False, model_type: str = 'DNGO', hidden_dims: list = None,
                incremental_params: Dict = None,
                use_loocv: bool = False, use_uncertainty_loss: bool = False,
                uncertainty_weight: float = 0.3,
                use_freeze: bool = False, unfreeze_ratio: float = 1.0):
    """
    Universal Transfer Learning 모델 학습 (DNGO/BNN 모두 지원, 하이퍼파라미터 BO 지원)

    Args:
        X_low: low-fidelity 입력 데이터
        y_low: low-fidelity 출력 데이터
        X_high: high-fidelity 입력 데이터
        y_high: high-fidelity 출력 데이터
        input_dim: 입력 차원
        hidden_dim: 기본 hidden 차원 (BO 사용 시 무시됨)
        device: 디바이스
        pretrain_epochs: 기본 pretrain epochs (BO 사용 시 무시됨)
        finetune_epochs: 기본 finetune epochs (BO 사용 시 무시됨)
        pretrain_lr: 기본 pretrain 학습률 (BO 사용 시 무시됨)
        finetune_lr: 기본 finetune 학습률 (BO 사용 시 무시됨)
        use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
        pretrain_bo_trials: pretrain BO 시행 횟수
        finetune_bo_trials: finetune BO 시행 횟수
        data_size: 데이터 크기 ('small', 'medium', 'large')
        verbose: 상세 출력
        model_type: 모델 타입 ('DNGO' 또는 'BNN')
        hidden_dims: BNN용 hidden 차원 리스트 (BNN 모델에만 사용)
        incremental_params: 점진적 학습 파라미터 딕셔너리
        use_loocv: HP 최적화 시 LOOCV 사용 여부
        use_uncertainty_loss: HP 최적화 시 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치
        use_freeze: Freeze 기법 사용 여부 (finetune 시 레이어 부분 동결)
        unfreeze_ratio: 해동할 레이어 비율 (0.0~1.0, use_freeze=True일 때만 적용)

    Returns:
        학습된 TransferLearningDNN 또는 TransferLearningBNN 모델
    """
    # 모델 타입에 따른 모델 생성
    if model_type.upper() == 'DNGO':
        if TransferLearningDNN is None:
            raise ImportError("TransferLearningDNN not found. Please ensure DNGO module is available.")
        model = TransferLearningDNN(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            device=device,
            use_hyperparameter_bo=use_hyperparameter_bo
        )
        # 점진적 학습 파라미터 설정
        if incremental_params:
            model.incremental_params = incremental_params
    elif model_type.upper() == 'BNN':
        # BNN 모델 임포트 및 생성
        try:
            from BNN.bnn_models import TransferLearningBNN
            # BNN은 hidden_dims 리스트를 사용, 제공되지 않으면 기본값 사용
            bnn_hidden_dims = hidden_dims if hidden_dims is not None else [hidden_dim, hidden_dim]
            model = TransferLearningBNN(
                input_dim=input_dim,
                hidden_dims=bnn_hidden_dims,
                device=device,
                use_hyperparameter_bo=use_hyperparameter_bo
            )
            # 점진적 학습 파라미터 설정
            if incremental_params:
                model.incremental_params = incremental_params
        except ImportError as e:
            raise ImportError(f"BNN model not found. Please ensure BNN module is available. Original error: {e}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'DNGO' or 'BNN'.")
    
    # Pretrain with low-fidelity data
    if len(X_low) > 0:
        if use_hyperparameter_bo and pretrain_bo_trials > 0:
            model.pretrain(
                X_low, y_low,
                epochs=pretrain_epochs, lr=pretrain_lr, verbose=verbose,
                bo_trials=pretrain_bo_trials, data_size=data_size,
                use_loocv=use_loocv, use_uncertainty_loss=use_uncertainty_loss,
                uncertainty_weight=uncertainty_weight
            )
        else:
            model.pretrain(
                X_low, y_low,
                epochs=pretrain_epochs, lr=pretrain_lr, verbose=verbose
            )

    # Finetune with high-fidelity data
    if len(X_high) > 0:
        if use_hyperparameter_bo and finetune_bo_trials > 0:
            model.finetune(
                X_high, y_high,
                epochs=finetune_epochs, lr=finetune_lr, verbose=verbose,
                bo_trials=finetune_bo_trials, data_size=data_size,
                use_loocv=use_loocv, use_uncertainty_loss=use_uncertainty_loss,
                uncertainty_weight=uncertainty_weight,
                use_freeze=use_freeze, unfreeze_ratio=unfreeze_ratio
            )
        else:
            model.finetune(
                X_high, y_high,
                epochs=finetune_epochs, lr=finetune_lr, verbose=verbose,
                use_freeze=use_freeze, unfreeze_ratio=unfreeze_ratio
            )

    return model


def fit_blr(model, X_low: np.ndarray, X_high: np.ndarray, 
            y_low: np.ndarray, y_high: np.ndarray, alpha: float = 1.0, beta: float = 25.0,
            use_incremental: bool = False, new_X_low: np.ndarray = None, new_y_low: np.ndarray = None,
            new_X_high: np.ndarray = None, new_y_high: np.ndarray = None):
    """
    Bayesian Linear Regression 학습 (DNGO/BNN 범용, 점진적 학습 지원)
    Transfer Learning 구조: LOFI 모델 → HIFI 모델 (fine-tuned)
    
    Args:
        model: 특성 추출 모델
        X_low, y_low: 저보정도 데이터
        X_high, y_high: 고보정도 데이터
        alpha, beta: BLR 하이퍼파라미터
        use_incremental: 점진적 학습 사용 여부
        new_X_low, new_y_low: 새로운 저보정도 데이터 (점진적 학습용)
        new_X_high, new_y_high: 새로운 고보정도 데이터 (점진적 학습용)
    """
    # 모델에 저장된 BLR 인스턴스가 있으면 재사용 (점진적 학습용)
    blr_low = getattr(model, 'blr_low', None)
    blr_high = getattr(model, 'blr_high', None)
    
    if use_incremental and blr_low is not None and blr_high is not None:
        # 점진적 업데이트 수행
        if new_X_low is not None and len(new_X_low) > 0:
            new_features_low = model.extract_features(new_X_low)
            for i in range(len(new_features_low)):
                blr_low.incremental_update(new_features_low[i:i+1], new_y_low[i:i+1])
        
        if new_X_high is not None and len(new_X_high) > 0:
            new_features_high = model.extract_features(new_X_high)
            for i in range(len(new_features_high)):
                blr_high.incremental_update(new_features_high[i:i+1], new_y_high[i:i+1])
    else:
        # 전체 재학습 (기존 방식)
        # LOFI 모델용 BLR (low-fidelity 데이터만 사용)
        blr_low = None
        if len(X_low) > 0:
            features_low = model.extract_features(X_low)
            blr_low = BayesianLinearRegression(alpha=alpha, beta=beta)
            blr_low.fit(features_low, y_low)
        
        # HIFI 모델용 BLR (high-fidelity 데이터만 사용)
        # Transfer Learning: DNN은 pretrain(LOFI) + finetune(HIFI) 완료 상태
        # BLR은 각 fidelity 데이터를 분리하여 학습 (DNGO-OL과 동일한 방식)
        blr_high = None
        if len(X_high) > 0:
            # High-fidelity 데이터만 사용하여 BLR 학습
            features_high = model.extract_features(X_high)
            blr_high = BayesianLinearRegression(alpha=alpha, beta=beta)
            blr_high.fit(features_high, y_high)
        else:
            # HIFI 데이터가 없으면 LOFI 모델을 그대로 사용
            blr_high = blr_low
        
        # 모델에 BLR 인스턴스 저장 (다음 점진적 업데이트를 위해)
        model.blr_low = blr_low
        model.blr_high = blr_high
    
    return blr_low, blr_high


def recommend_next(model, blr_low: BayesianLinearRegression, blr_high: BayesianLinearRegression,
                   param_ranges: List[range], X_low: np.ndarray, X_high: np.ndarray,
                   y_low: np.ndarray, y_high: np.ndarray, s: float) -> Tuple:
    """
    다음 실험점 추천 (Expected Improvement 최대화, DNGO/BNN 범용)
    각 fidelity level에 대해 독립적으로 EI 계산
    """
    # 전체 조합 생성
    all_combinations = list(itertools.product(*param_ranges))
    X_grid = np.array(all_combinations, dtype=np.float32)
    
    # Feature 추출
    features_grid = model.extract_features(X_grid)
    
    # 현재 fidelity에 따라 적절한 모델 선택
    if s == 1.0:  # High-fidelity
        blr = blr_high
        y_best = np.min(y_high) if len(y_high) > 0 else np.inf
    else:  # Low-fidelity
        blr = blr_low if blr_low is not None else blr_high
        y_best = np.min(y_low) if len(y_low) > 0 else np.inf
    
    # 선택된 모델로 예측
    y_pred, y_std = [], []
    for phi in features_grid:
        mu, var = blr.predict(phi)
        y_pred.append(mu)
        y_std.append(np.sqrt(var))
    
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)
    
    # Expected Improvement 계산
    ei = expected_improvement(y_pred, y_std, y_best)
    
    # 각 모델별 예측값도 저장 (시각화용)
    y_pred_low, y_std_low, ei_low = [], [], []
    y_pred_high, y_std_high, ei_high = [], [], []
    
    # LOFI 모델 예측
    if blr_low is not None:
        y_best_low = np.min(y_low) if len(y_low) > 0 else np.inf
        for phi in features_grid:
            mu, var = blr_low.predict(phi)
            y_pred_low.append(mu)
            y_std_low.append(np.sqrt(var))
        y_pred_low = np.array(y_pred_low)
        y_std_low = np.array(y_std_low)
        ei_low = expected_improvement(y_pred_low, y_std_low, y_best_low)
    
    # HIFI 모델 예측
    if blr_high is not None:
        y_best_high = np.min(y_high) if len(y_high) > 0 else np.inf
        for phi in features_grid:
            mu, var = blr_high.predict(phi)
            y_pred_high.append(mu)
            y_std_high.append(np.sqrt(var))
        y_pred_high = np.array(y_pred_high)
        y_std_high = np.array(y_std_high)
        ei_high = expected_improvement(y_pred_high, y_std_high, y_best_high)
    
    # 이미 측정된 점들을 fidelity별로 구분하여 저장
    measured_points = set()
    for x in X_low:
        measured_points.add((tuple(x.astype(int)), 'low'))
    for x in X_high:
        measured_points.add((tuple(x.astype(int)), 'high'))
    
    # 현재 요청되는 fidelity 결정
    current_fidelity = 'high' if s == 1.0 else 'low'
    
    # 유효한 후보 필터링: 같은 fidelity로 이미 측정된 점만 제외
    valid_indices = []
    for i, combo in enumerate(X_grid):
        combo_tuple = tuple(combo.astype(int))
        # 현재 fidelity로 이미 측정되었는지 확인
        if (combo_tuple, current_fidelity) not in measured_points:
            valid_indices.append(i)
    
    if not valid_indices:
        # 현재 fidelity로 모든 점이 측정된 경우
        # (이론적으로 발생하지 않아야 하지만) 최대 EI 선택
        best_idx = np.argmax(ei)
    else:
        # 유효한 점들 중에서 최대 EI 선택
        valid_ei = ei[valid_indices]
        best_valid_idx = np.argmax(valid_ei)
        best_idx = valid_indices[best_valid_idx]
    
    next_x_label = list(X_grid[best_idx].astype(int))
    
    # 각 모델별 예측 결과 반환
    predictions = {
        'y_pred': y_pred,      # 현재 fidelity 모델의 예측
        'y_std': y_std,
        'ei': ei,
        'y_pred_low': np.array(y_pred_low) if y_pred_low is not None else None,
        'y_std_low': np.array(y_std_low) if y_std_low is not None else None,
        'ei_low': np.array(ei_low) if ei_low is not None else None,
        'y_pred_high': np.array(y_pred_high) if y_pred_high is not None else None,
        'y_std_high': np.array(y_std_high) if y_std_high is not None else None,
        'ei_high': np.array(ei_high) if ei_high is not None else None
    }
    
    return next_x_label, predictions, best_idx, X_grid


def single_optimization_run(param_space: Dict, label_maps: Dict, lookup: Dict,
                           cost_budget: float = 50.0, num_init_design: int = 10,
                           high_fidelity_ratio: float = 0.2, min_target: float = 1.5249,
                           random_state: int = 42, verbose: bool = True,
                           model_config: Dict = None,
                           use_hyperparameter_bo: bool = False,
                           pretrain_bo_trials: int = 0, finetune_bo_trials: int = 0,
                           data_size: str = 'small', model_type: str = 'DNGO',
                           use_incremental_learning: bool = False,
                           incremental_params: Dict = None,
                           use_loocv: bool = False,
                           use_uncertainty_loss: bool = False,
                           uncertainty_weight: float = 0.3,
                           use_freeze: bool = False,
                           unfreeze_ratio: float = 1.0) -> Dict:
    """
    범용 단일 최적화 실행 (DNGO/BNN 모두 지원, 하이퍼파라미터 BO 지원)

    Args:
        param_space: 파라미터 공간
        label_maps: 라벨 매핑
        lookup: lookup table
        cost_budget: 비용 예산
        num_init_design: 초기 설계점 개수
        high_fidelity_ratio: high-fidelity 비율
        min_target: 목표 최솟값
        random_state: 랜덤 시드
        verbose: 상세 출력
        model_config: 모델 설정
        use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
        pretrain_bo_trials: pretrain BO 시행 횟수
        finetune_bo_trials: finetune BO 시행 횟수
        data_size: 데이터 크기
        model_type: 모델 타입 ('DNGO' 또는 'BNN')
        use_incremental_learning: 점진적 학습 사용 여부
        incremental_params: 점진적 학습 파라미터
        use_loocv: HP 최적화 시 LOOCV 사용 여부
        use_uncertainty_loss: HP 최적화 시 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치
        use_freeze: Freeze 기법 사용 여부 (finetune 시 레이어 부분 동결)
        unfreeze_ratio: 해동할 레이어 비율 (0.0~1.0, BO 사용 시 자동 최적화)

    Returns:
        결과 딕셔너리 (비용, best_so_far 곡선, 시간 등)
    """
    from common.data_utils import (
        sample_param_space, assign_fidelities, prepare_initial_data,
        measure_from_label, append_measurement_to_data
    )
    
    if model_config is None:
        model_config = {
            'input_dim': 3,
            'hidden_dim': 64,
            'pretrain_epochs': 200,
            'finetune_epochs': 100,
            'device': 'cpu'
        }
    
    # 파라미터 범위
    param_ranges = [
        range(1, len(param_space['organic']) + 1),
        range(1, len(param_space['cation']) + 1),
        range(1, len(param_space['anion']) + 1),
    ]
    
    # 초기 설계
    init_samples = sample_param_space(param_space, num_init_design, random_state=random_state)
    init_fids = assign_fidelities(num_init_design, high_fidelity_ratio, random_state=random_state)
    
    # 초기 데이터 준비
    X_low, y_low, X_high, y_high = prepare_initial_data(init_samples, init_fids, label_maps, lookup)
    
    # 초기 비용 계산
    total_cost = sum(init_fids)
    
    # 추적 변수들
    best_so_far = np.inf  # high-fidelity 측정값만으로 추적
    best_so_far_curve = []
    timing_data = []
    cost_data = []
    iter_ = 0
    hyperparameter_history = []  # 하이퍼파라미터 기록
    visualization_data = []  # 시각화용 데이터
    
    # 하이퍼파라미터 최적화 관련 변수
    last_optimized_params = None  # 마지막으로 최적화된 파라미터 저장
    last_optimization_iter = 0  # 마지막 최적화 시점
    optimization_interval = 50  # 50개 데이터마다 최적화
    
    # 점진적 학습 관련 변수
    previous_X_low, previous_y_low = X_low.copy(), y_low.copy()
    previous_X_high, previous_y_high = X_high.copy(), y_high.copy()
    model = None  # 모델 인스턴스 유지를 위해
    
    # 초기 best_so_far 설정
    if len(y_high) > 0:
        best_so_far = np.min(y_high)
    
    if verbose:
        print(f"Initial cost: {total_cost:.2f}, Initial best_so_far: {best_so_far}")
        if use_hyperparameter_bo:
            print(f"🔧 Using hyperparameter BO: pretrain={pretrain_bo_trials}, finetune={finetune_bo_trials}")
        if use_incremental_learning:
            print(f"🔄 Using incremental learning with params: {incremental_params}")
    
    # 예상 iteration 수 계산 (대략적인 추정)
    estimated_iterations = int(cost_budget - total_cost) + 1
    
    # 메인 최적화 루프 with progress bar
    pbar = tqdm(total=cost_budget, initial=total_cost, desc="Optimization Progress", 
                unit="cost", disable=not verbose)
    
    while total_cost < cost_budget:
        iter_ += 1
        iter_start = time.time()
        
        # Progress bar에 iteration 정보 표시
        pbar.set_description(f"Optimization [Iter {iter_}]")
        
        # 출력 구분을 위한 구분선
        if verbose and iter_ > 1:
            print("\n" + "="*60)
        
        if verbose:
            print(f"\n📍 Iteration {iter_}")
            print(f"  Current data: {len(X_low)} low-fidelity, {len(X_high)} high-fidelity")
        
        # Fidelity 스케줄링: 8번 중 1번만 high-fidelity
        s = 1.0 if (iter_ % 8 == 0) else 0.1
        
        # 전체 데이터 수 계산
        total_data_points = len(X_low) + len(X_high)
        
        # 하이퍼파라미터 최적화 여부 결정
        should_optimize_hp = False
        if use_hyperparameter_bo:
            # 첫 iteration이거나, 10개씩 증가할 때마다 최적화
            if (iter_ == 1) or \
               (total_data_points > 0 and total_data_points % optimization_interval == 0):
                should_optimize_hp = True
                if verbose:
                    print(f"\n  🔧 Hyperparameter optimization triggered (data points: {total_data_points})")
        
        # 새로운 데이터 확인 (점진적 학습용)
        new_X_low = X_low[len(previous_X_low):] if len(X_low) > len(previous_X_low) else np.array([])
        new_y_low = y_low[len(previous_y_low):] if len(y_low) > len(previous_y_low) else np.array([])
        new_X_high = X_high[len(previous_X_high):] if len(X_high) > len(previous_X_high) else np.array([])
        new_y_high = y_high[len(previous_y_high):] if len(y_high) > len(previous_y_high) else np.array([])
        
        has_new_data = len(new_X_low) > 0 or len(new_X_high) > 0
        
        # 모델 학습 결정
        use_incremental_this_iter = (use_incremental_learning and 
                                   incremental_params is not None and 
                                   model is not None and 
                                   has_new_data and 
                                   not should_optimize_hp)
        
        # 모델 학습
        if should_optimize_hp:
            # 하이퍼파라미터 최적화 수행
            # 모델 타입에 따른 hidden_dim 설정
            if model_type.upper() == 'BNN':
                # BNN은 hidden_dims (list)를 사용
                hidden_dim_param = model_config.get('hidden_dims', [64, 64])[0]  # 첫 번째 차원 사용
            else:
                # DNGO는 hidden_dim (int)를 사용
                hidden_dim_param = model_config['hidden_dim']
            
            # BNN의 경우 hidden_dims 전달
            hidden_dims_param = None
            if model_type.upper() == 'BNN':
                hidden_dims_param = model_config.get('hidden_dims', [64, 64])
            
            model = train_model(
                X_low, y_low, X_high, y_high,
                input_dim=model_config['input_dim'],
                hidden_dim=hidden_dim_param,
                device=model_config['device'],
                pretrain_epochs=model_config['pretrain_epochs'],
                finetune_epochs=model_config['finetune_epochs'],
                use_hyperparameter_bo=True,
                pretrain_bo_trials=pretrain_bo_trials,
                finetune_bo_trials=finetune_bo_trials,
                data_size=data_size,
                verbose=verbose,
                model_type=model_type,
                hidden_dims=hidden_dims_param,
                incremental_params=incremental_params,
                use_loocv=use_loocv,
                use_uncertainty_loss=use_uncertainty_loss,
                uncertainty_weight=uncertainty_weight,
                use_freeze=use_freeze,
                unfreeze_ratio=unfreeze_ratio
            )
            
            # 최적화된 파라미터 저장
            hp_summary = model.get_hyperparameter_summary()
            last_optimized_params = {
                'pretrain_params': hp_summary['pretrain_best_params'],
                'finetune_params': hp_summary['finetune_best_params']
            }
            last_optimization_iter = total_data_points
            
            # 하이퍼파라미터 기록
            hyperparameter_history.append({
                'iteration': iter_,
                'data_points': total_data_points,
                'pretrain_params': hp_summary['pretrain_best_params'],
                'finetune_params': hp_summary['finetune_best_params'],
                'pretrain_trials': hp_summary.get('pretrain_bo_history', []),
                'finetune_trials': hp_summary.get('finetune_bo_history', [])
            })
        else:
            # 이전에 최적화된 파라미터 사용 또는 기본 파라미터 사용
            if verbose and use_hyperparameter_bo and last_optimized_params:
                print(f"\n  📌 Using previously optimized hyperparameters")
            
            if last_optimized_params and use_hyperparameter_bo:
                # 이전 파라미터 사용
                pretrain_params = last_optimized_params['pretrain_params']
                finetune_params = last_optimized_params['finetune_params']
                
                # 모델 타입별 모델 생성
                if model_type.upper() == 'DNGO':
                    if TransferLearningDNN is None:
                        raise ImportError("TransferLearningDNN not found. Please ensure DNGO module is available.")
                    model = TransferLearningDNN(
                        input_dim=model_config['input_dim'],
                        hidden_dim=model_config['hidden_dim'],  # 기본값 사용 (동적 구조는 pretrain에서 설정)
                        device=model_config['device'],
                        use_hyperparameter_bo=False
                    )
                    
                    # Pretrain with optimized parameters (DNGO)
                    if len(X_low) > 0 and pretrain_params:
                        # 최적화된 구조로 모델 재구성
                        model._build_dynamic_model(pretrain_params)
                        model.pretrain(
                            X_low, y_low,
                            epochs=pretrain_params['epochs'],
                            lr=pretrain_params['learning_rate'],
                            verbose=False
                        )
                    
                    # Finetune with optimized parameters (DNGO)
                    if len(X_high) > 0 and finetune_params:
                        # Feature extractor는 유지하고 출력층만 재구성
                        # feature_net[-1]은 nn.Sequential(Linear, ReLU)이므로 [0]으로 Linear 접근
                        feature_dim = model.feature_net[-1][0].out_features  # 마지막 hidden layer의 출력 차원
                        model.out_layer = nn.Linear(feature_dim, 1, bias=False).to(model.device).float()
                        model.model = nn.Sequential(model.feature_net, model.out_layer)
                        
                        model.finetune(
                            X_high, y_high,
                            epochs=finetune_params['epochs'],
                            lr=finetune_params['learning_rate'],
                            verbose=False
                        )
                        
                elif model_type.upper() == 'BNN':
                    # BNN 모델 생성
                    from BNN.bnn_models import TransferLearningBNN
                    
                    # BNN은 하이퍼파라미터가 다를 수 있으므로 기본 파라미터 사용
                    hidden_dims = pretrain_params.get('hidden_dims', model_config.get('hidden_dims', [64, 64]))
                    
                    model = TransferLearningBNN(
                        input_dim=model_config['input_dim'],
                        hidden_dims=hidden_dims,
                        device=model_config['device'],
                        use_hyperparameter_bo=False
                    )
                    
                    # Pretrain with optimized parameters (BNN)
                    if len(X_low) > 0 and pretrain_params:
                        model.pretrain(
                            X_low, y_low,
                            epochs=pretrain_params.get('epochs', model_config['pretrain_epochs']),
                            lr=pretrain_params.get('learning_rate', model_config.get('pretrain_lr', 1e-3)),
                            verbose=False
                        )
                    
                    # Finetune with optimized parameters (BNN)  
                    if len(X_high) > 0 and finetune_params:
                        model.finetune(
                            X_high, y_high,
                            epochs=finetune_params.get('epochs', model_config['finetune_epochs']),
                            lr=finetune_params.get('learning_rate', model_config.get('finetune_lr', 1e-4)),
                            verbose=False
                        )
            else:
                # 기본 파라미터로 모델 학습
                # 모델 타입에 따른 hidden_dim 설정
                if model_type.upper() == 'BNN':
                    # BNN은 hidden_dims (list)를 사용
                    hidden_dim_param = model_config.get('hidden_dims', [64, 64])[0]  # 첫 번째 차원 사용
                else:
                    # DNGO는 hidden_dim (int)를 사용
                    hidden_dim_param = model_config['hidden_dim']
                
                # BNN의 경우 hidden_dims 전달
                hidden_dims_param = None
                if model_type.upper() == 'BNN':
                    hidden_dims_param = model_config.get('hidden_dims', [64, 64])
                
                model = train_model(
                    X_low, y_low, X_high, y_high,
                    input_dim=model_config['input_dim'],
                    hidden_dim=hidden_dim_param,
                    device=model_config['device'],
                    pretrain_epochs=model_config['pretrain_epochs'],
                    finetune_epochs=model_config['finetune_epochs'],
                    use_hyperparameter_bo=False,
                    verbose=False,
                    model_type=model_type,
                    hidden_dims=hidden_dims_param,
                    incremental_params=incremental_params,
                    use_freeze=use_freeze,
                    unfreeze_ratio=unfreeze_ratio
                )
            
        
        # 점진적 학습 처리 (별도 블록)
        if use_incremental_this_iter:
            # 점진적 학습 수행
            if verbose:
                print(f"  🔄 Performing incremental learning with {len(new_X_low)} low + {len(new_X_high)} high new data points")
            
            # 모델 점진적 업데이트
            if hasattr(model, 'incremental_update'):
                if len(new_X_low) > 0:
                    model.incremental_update(new_X_low, new_y_low, fidelity='low')
                if len(new_X_high) > 0:
                    model.incremental_update(new_X_high, new_y_high, fidelity='high')
        
        # BLR 학습 (Transfer Learning: LOFI → HIFI)
        if use_incremental_this_iter:
            # 점진적 BLR 업데이트
            blr_low, blr_high = fit_blr(
                model, X_low, X_high, y_low, y_high,
                use_incremental=True,
                new_X_low=new_X_low if len(new_X_low) > 0 else None,
                new_y_low=new_y_low if len(new_y_low) > 0 else None,
                new_X_high=new_X_high if len(new_X_high) > 0 else None,
                new_y_high=new_y_high if len(new_y_high) > 0 else None
            )
        else:
            # 전체 재학습
            blr_low, blr_high = fit_blr(model, X_low, X_high, y_low, y_high)
        
        # 다음 실험점 추천 (각 모델별로 독립적 예측)
        next_x_label, predictions, best_idx, X_grid = recommend_next(
            model, blr_low, blr_high, param_ranges, X_low, X_high, y_low, y_high, s
        )
        
        # 실제값 수집 (X_grid의 각 점에 대한 실제 bandgap 값)
        def get_actual_bandgap(label_arr, lookup, label_maps, fidelity=1.0):
            """조합 라벨로부터 실제 bandgap 값 가져오기"""
            # 라벨 역변환
            reverse_maps = {
                "organic": {v: k for k, v in label_maps["organic"].items()},
                "cation": {v: k for k, v in label_maps["cation"].items()},
                "anion": {v: k for k, v in label_maps["anion"].items()},
            }
            
            try:
                organic = reverse_maps["organic"][int(label_arr[0])]
                cation = reverse_maps["cation"][int(label_arr[1])]
                anion = reverse_maps["anion"][int(label_arr[2])]
                
                # High-fidelity 값 사용 (HSE06)
                if fidelity == 1.0:
                    return np.amin(lookup[organic.capitalize()][cation][anion]['bandgap_hse06'])
                else:
                    return np.amin(lookup[organic.capitalize()][cation][anion]['bandgap_gga'])
            except (KeyError, IndexError):
                return np.nan
        
        y_actual = np.array([get_actual_bandgap(combo, lookup, label_maps, fidelity=1.0) 
                            for combo in X_grid])
        
        # predictions에서 값 추출
        y_pred = predictions['y_pred']
        y_std = predictions['y_std']
        ei = predictions['ei']
        
        # 시각화용 데이터 저장
        viz_entry = {
            'iteration': iter_,
            'y_pred': y_pred.copy(),
            'y_std': y_std.copy(),
            'y_actual': y_actual.copy(),  # 실제값 추가
            'ei': ei.copy(),
            'best_idx': best_idx,
            'X_grid': X_grid.copy(),
            'fidelity': s,
            'recommended_point': next_x_label.copy(),
            'model': model,
            'blr': blr_high if blr_high is not None else blr_low,  # 하위 호환성을 위해 유지
            'blr_low': blr_low,    # LOFI 전용 모델
            'blr_high': blr_high,  # HIFI 전용 모델
            'X_low': X_low.copy(),
            'y_low': y_low.copy(),
            'X_high': X_high.copy(),
            'y_high': y_high.copy(),
            # LOFI와 HIFI 모델의 독립적인 예측 추가
            'y_pred_low': predictions.get('y_pred_low').copy() if predictions.get('y_pred_low') is not None else None,
            'y_std_low': predictions.get('y_std_low').copy() if predictions.get('y_std_low') is not None else None,
            'ei_low': predictions.get('ei_low').copy() if predictions.get('ei_low') is not None else None,
            'y_pred_high': predictions.get('y_pred_high').copy() if predictions.get('y_pred_high') is not None else None,
            'y_std_high': predictions.get('y_std_high').copy() if predictions.get('y_std_high') is not None else None,
            'ei_high': predictions.get('ei_high').copy() if predictions.get('ei_high') is not None else None
        }
        
        # 하이퍼파라미터 정보 추가 (베이지안 최적화 사용 시)
        if use_hyperparameter_bo:
            hp_summary = model.get_hyperparameter_summary()
            viz_entry['hyperparameters'] = hp_summary  # 전체 summary 저장 (history 포함)
        
        visualization_data.append(viz_entry)
        
        # 측정
        measurement = measure_from_label(next_x_label, s, label_maps, lookup)

        # BO iteration 로그 (항상 출력)
        fid_str = "H" if s == 1.0 else "L"
        print(f"  [Iter {iter_:3d}] point={next_x_label}, fid={fid_str}, "
              f"EI={ei[best_idx]:.4f}, pred={y_pred[best_idx]:.3f}±{y_std[best_idx]:.3f}, "
              f"actual={measurement:.4f}", flush=True)

        # 이전 데이터 상태 업데이트 (다음 점진적 학습을 위해)
        previous_X_low, previous_y_low = X_low.copy(), y_low.copy()
        previous_X_high, previous_y_high = X_high.copy(), y_high.copy()
        
        # 데이터 업데이트
        X_low, y_low, X_high, y_high = append_measurement_to_data(
            X_low, y_low, X_high, y_high, next_x_label, s, label_maps, lookup
        )
        
        # 비용 및 시간 업데이트
        iter_end = time.time()
        time_taken = iter_end - iter_start
        total_cost += s
        
        # best_so_far 업데이트 (high-fidelity만)
        if s == 1.0:
            if measurement < best_so_far:
                best_so_far = measurement
        
        # 기록
        timing_data.append([0, iter_, time_taken])
        cost_data.append([0, iter_, total_cost])
        best_so_far_curve.append([0, iter_, s, best_so_far])
        
        # Progress bar 업데이트
        pbar.update(s)
        pbar.set_postfix({'best': f'{best_so_far:.4f}', 'current': f'{measurement:.4f}', 'EI': f'{ei[best_idx]:.3f}'})
        
        # 조기 종료 조건
        if s == 1.0 and np.isclose(measurement, min_target, atol=1e-6):
            if verbose:
                print('\n✅ Found the minimum target value!')
            break
    
    # Progress bar 닫기
    pbar.close()
    
    # 결과 호환성을 위한 데이터 변환
    best_values_history = [x[3] for x in best_so_far_curve]
    cost_history = [x[2] for x in cost_data]
    fidelity_history = [data['fidelity'] for data in visualization_data]
    ei_history = [data['ei'][data['best_idx']] for data in visualization_data]
    
    return {
        'total_cost': total_cost,
        'best_so_far': best_so_far,
        'iterations': iter_,
        'best_so_far_curve': best_so_far_curve,
        'timing_data': timing_data,
        'cost_data': cost_data,
        'final_X_low': X_low,
        'final_y_low': y_low,
        'final_X_high': X_high,
        'final_y_high': y_high,
        'hyperparameter_history': hyperparameter_history,
        'use_hyperparameter_bo': use_hyperparameter_bo,
        'visualization_data': visualization_data,
        'best_values_history': best_values_history,
        'cost_history': cost_history,
        'fidelity_history': fidelity_history,
        'ei_history': ei_history,
        'model_type': model_type,
        'use_incremental_learning': use_incremental_learning,
        'incremental_params': incremental_params
    }


def multiple_optimization_runs(param_space: Dict, label_maps: Dict, lookup: Dict,
                              num_runs: int = 100, cost_budget: float = 50.0,
                              num_init_design: int = 10, high_fidelity_ratio: float = 0.2,
                              min_target: float = 1.5249, model_config: Dict = None,
                              save_results: bool = True, results_filename: str = 'tl_bo_results.csv',
                              use_hyperparameter_bo: bool = False, pretrain_bo_trials: int = 0,
                              finetune_bo_trials: int = 0, data_size: str = 'small',
                              save_visualizations: bool = False, visualizations_dir: str = 'images',
                              model_type: str = 'DNGO', use_incremental_learning: bool = False,
                              incremental_params: Dict = None,
                              use_loocv: bool = False,
                              use_uncertainty_loss: bool = False,
                              uncertainty_weight: float = 0.3,
                              use_freeze: bool = False,
                              unfreeze_ratio: float = 1.0) -> List[Dict]:
    """
    범용 다중 최적화 실행 (DNGO/BNN 모두 지원, 하이퍼파라미터 BO 지원)
    """
    import pandas as pd
    
    all_results = []
    all_costs = []
    
    print(f"Starting {num_runs} optimization runs...")
    if use_hyperparameter_bo:
        print(f"🔧 Using hyperparameter BO: pretrain={pretrain_bo_trials}, finetune={finetune_bo_trials}")
    
    for run in range(num_runs):
        print(f"\n===== Run {run+1}/{num_runs} =====")
        
        result = single_optimization_run(
            param_space=param_space,
            label_maps=label_maps,
            lookup=lookup,
            cost_budget=cost_budget,
            num_init_design=num_init_design,
            high_fidelity_ratio=high_fidelity_ratio,
            min_target=min_target,
            random_state=run,  # 각 런마다 다른 시드
            verbose=False,  # 다중 실행 시 상세 출력 비활성화
            model_config=model_config,
            use_hyperparameter_bo=use_hyperparameter_bo,
            pretrain_bo_trials=pretrain_bo_trials,
            finetune_bo_trials=finetune_bo_trials,
            data_size=data_size,
            model_type=model_type,
            use_incremental_learning=use_incremental_learning,
            incremental_params=incremental_params,
            use_loocv=use_loocv,
            use_uncertainty_loss=use_uncertainty_loss,
            uncertainty_weight=uncertainty_weight,
            use_freeze=use_freeze,
            unfreeze_ratio=unfreeze_ratio
        )
        
        all_results.append(result)
        all_costs.append(result['total_cost'])
        
        # 각 실행에 대한 시각화 저장
        if save_visualizations and 'visualization_data' in result:
            import os
            from datetime import datetime
            from .visualization import plot_optimization_progress
            
            # 각 실행별 디렉토리 생성
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            run_dir = os.path.join(visualizations_dir, f"{model_type.lower()}_{timestamp}_run{run+1:02d}")
            
            if not os.path.exists(run_dir):
                os.makedirs(run_dir, exist_ok=True)
            
            print(f"  📸 Saving visualizations to {run_dir}/")
            plot_optimization_progress(result, save_dir=run_dir)
            print(f"  ✅ Visualization saved for run {run+1}")
            
            # 베이지안 최적화 결과 저장
            try:
                from ..common.result_saver import save_optimization_results
                save_optimization_results(result, run_dir, model_type.lower())
            except ImportError:
                # 상대 import 실패 시 절대 import 시도
                import sys
                sys.path.append('..')
                from common.result_saver import save_optimization_results
                save_optimization_results(result, run_dir, model_type.lower())
        
        if result['best_so_far'] <= min_target:
            print(f"Run {run+1}: Found target! Cost: {result['total_cost']:.2f}")
        else:
            print(f"Run {run+1}: Completed. Cost: {result['total_cost']:.2f}, Best: {result['best_so_far']:.4f}")

        # 매 run 완료 후 실시간 결과 저장
        if save_results:
            results_df = pd.DataFrame({
                'run': range(1, len(all_results) + 1),
                'total_cost': all_costs,
                'best_so_far': [r['best_so_far'] for r in all_results],
                'iterations': [r['iterations'] for r in all_results],
                'use_hyperparameter_bo': [r['use_hyperparameter_bo'] for r in all_results]
            })
            results_df.to_csv(results_filename, index=False)
            print(f"  💾 Results saved ({len(all_results)}/{num_runs} runs)", flush=True)

    # 최종 결과 저장 (중복이지만 안전을 위해 유지)
    if save_results:
        results_df = pd.DataFrame({
            'run': range(1, num_runs + 1),
            'total_cost': all_costs,
            'best_so_far': [r['best_so_far'] for r in all_results],
            'iterations': [r['iterations'] for r in all_results],
            'use_hyperparameter_bo': [r['use_hyperparameter_bo'] for r in all_results]
        })
        results_df.to_csv(results_filename, index=False)
        print(f"\nResults saved to {results_filename}")
        
        # 하이퍼파라미터 기록 저장 (BO 사용 시)
        if use_hyperparameter_bo:
            all_hp_records = []
            for run_idx, result in enumerate(all_results):
                for hp_record in result['hyperparameter_history']:
                    hp_record['run'] = run_idx + 1
                    all_hp_records.append(hp_record)
            
            if all_hp_records:
                hp_df = pd.DataFrame(all_hp_records)
                hp_filename = results_filename.replace('.csv', '_hyperparameters.csv')
                hp_df.to_csv(hp_filename, index=False)
                print(f"Hyperparameter history saved to {hp_filename}")
    
    # 요약 통계
    success_rate = sum(1 for r in all_results if r['best_so_far'] <= min_target) / num_runs
    avg_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    
    print(f"\n=== Summary Statistics ===")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average cost: {avg_cost:.2f} ± {std_cost:.2f}")
    print(f"Min cost: {np.min(all_costs):.2f}")
    print(f"Max cost: {np.max(all_costs):.2f}")
    
    return all_results 