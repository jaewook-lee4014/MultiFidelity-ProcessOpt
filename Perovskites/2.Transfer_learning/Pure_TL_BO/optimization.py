import numpy as np
import itertools
import time
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
from models import TransferLearningDNN, BayesianLinearRegression


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
                verbose: bool = False) -> TransferLearningDNN:
    """
    Transfer Learning DNN 모델 학습 (하이퍼파라미터 BO 지원)
    
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
        
    Returns:
        학습된 TransferLearningDNN 모델
    """
    model = TransferLearningDNN(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        device=device,
        use_hyperparameter_bo=use_hyperparameter_bo
    )
    
    # Pretrain with low-fidelity data
    if len(X_low) > 0:
        if use_hyperparameter_bo and pretrain_bo_trials > 0:
            model.pretrain(
                X_low, y_low, 
                epochs=pretrain_epochs, lr=pretrain_lr, verbose=verbose,
                bo_trials=pretrain_bo_trials, data_size=data_size
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
                bo_trials=finetune_bo_trials, data_size=data_size
            )
        else:
            model.finetune(
                X_high, y_high, 
                epochs=finetune_epochs, lr=finetune_lr, verbose=verbose
            )
    
    return model


def fit_blr(model: TransferLearningDNN, X_low: np.ndarray, X_high: np.ndarray, 
            y_low: np.ndarray, y_high: np.ndarray, alpha: float = 1.0, beta: float = 25.0):
    """
    Bayesian Linear Regression 학습
    """
    # 전체 데이터 결합
    X_all = np.vstack([X_low, X_high]) if len(X_high) > 0 else X_low
    y_all = np.concatenate([y_low, y_high]) if len(y_high) > 0 else y_low
    
    # Feature 추출
    features_all = model.extract_features(X_all)
    
    # BLR 학습
    blr = BayesianLinearRegression(alpha=alpha, beta=beta)
    blr.fit(features_all, y_all)
    
    return blr, X_all, y_all


def recommend_next(model: TransferLearningDNN, blr: BayesianLinearRegression, 
                   param_ranges: List[range], X_low: np.ndarray, X_high: np.ndarray,
                   y_low: np.ndarray, y_high: np.ndarray, s: float) -> Tuple:
    """
    다음 실험점 추천 (Expected Improvement 최대화)
    """
    # 전체 조합 생성
    all_combinations = list(itertools.product(*param_ranges))
    X_grid = np.array(all_combinations, dtype=np.float32)
    
    # 현재까지의 최적값 (high-fidelity만 고려)
    if len(y_high) > 0:
        y_best = np.min(y_high)
    else:
        y_best = np.inf
    
    # 전체 조합에 대한 예측
    features_grid = model.extract_features(X_grid)
    y_pred, y_std = [], []
    
    for phi in features_grid:
        mu, var = blr.predict(phi)
        y_pred.append(mu)
        y_std.append(np.sqrt(var))
    
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)
    
    # Expected Improvement 계산
    ei = expected_improvement(y_pred, y_std, y_best)
    
    # 이미 측정된 점들은 제외
    measured_points = set()
    for x in X_low:
        measured_points.add(tuple(x.astype(int)))
    for x in X_high:
        measured_points.add(tuple(x.astype(int)))
    
    # 측정되지 않은 점들만 고려
    valid_indices = []
    for i, combo in enumerate(X_grid):
        if tuple(combo.astype(int)) not in measured_points:
            valid_indices.append(i)
    
    if not valid_indices:
        # 모든 점이 측정된 경우 (이론적으로 발생하지 않아야 함)
        best_idx = np.argmax(ei)
    else:
        # 유효한 점들 중에서 최대 EI 선택
        valid_ei = ei[valid_indices]
        best_valid_idx = np.argmax(valid_ei)
        best_idx = valid_indices[best_valid_idx]
    
    next_x_label = list(X_grid[best_idx].astype(int))
    
    return next_x_label, y_pred, y_std, ei, best_idx, X_grid


def single_optimization_run(param_space: Dict, label_maps: Dict, lookup: Dict,
                           cost_budget: float = 50.0, num_init_design: int = 10,
                           high_fidelity_ratio: float = 0.2, min_target: float = 1.5249,
                           random_state: int = 42, verbose: bool = True,
                           model_config: Dict = None, 
                           use_hyperparameter_bo: bool = False,
                           pretrain_bo_trials: int = 0, finetune_bo_trials: int = 0,
                           data_size: str = 'small') -> Dict:
    """
    단일 최적화 실행 (하이퍼파라미터 BO 지원)
    
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
        
    Returns:
        결과 딕셔너리 (비용, best_so_far 곡선, 시간 등)
    """
    from data_utils import (
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
    
    # 초기 best_so_far 설정
    if len(y_high) > 0:
        best_so_far = np.min(y_high)
    
    if verbose:
        print(f"Initial cost: {total_cost:.2f}, Initial best_so_far: {best_so_far}")
        if use_hyperparameter_bo:
            print(f"🔧 Using hyperparameter BO: pretrain={pretrain_bo_trials}, finetune={finetune_bo_trials}")
    
    # 메인 최적화 루프
    while total_cost < cost_budget:
        iter_ += 1
        iter_start = time.time()
        
        if verbose:
            print(f"\n==== Iteration {iter_} ====")
        
        # Fidelity 스케줄링: 8번 중 1번만 high-fidelity
        s = 1.0 if (iter_ % 8 == 0) else 0.1
        
        # 모델 학습 (하이퍼파라미터 BO 포함)
        model = train_model(
            X_low, y_low, X_high, y_high, 
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            device=model_config['device'],
            pretrain_epochs=model_config['pretrain_epochs'],
            finetune_epochs=model_config['finetune_epochs'],
            use_hyperparameter_bo=use_hyperparameter_bo,
            pretrain_bo_trials=pretrain_bo_trials,
            finetune_bo_trials=finetune_bo_trials,
            data_size=data_size,
            verbose=verbose
        )
        
        # 하이퍼파라미터 기록
        if use_hyperparameter_bo:
            hp_summary = model.get_hyperparameter_summary()
            hyperparameter_history.append({
                'iteration': iter_,
                'pretrain_params': hp_summary['pretrain_best_params'],
                'finetune_params': hp_summary['finetune_best_params']
            })
            
            if verbose and hp_summary['pretrain_best_params']:
                print(f"🔧 Pretrain params: {hp_summary['pretrain_best_params']}")
            if verbose and hp_summary['finetune_best_params']:
                print(f"🔧 Finetune params: {hp_summary['finetune_best_params']}")
        
        # BLR 학습
        blr, X_all, y_all = fit_blr(model, X_low, X_high, y_low, y_high)
        
        # 다음 실험점 추천
        next_x_label, y_pred, y_std, ei, best_idx, X_grid = recommend_next(
            model, blr, param_ranges, X_low, X_high, y_low, y_high, s
        )
        
        # 측정
        measurement = measure_from_label(next_x_label, s, label_maps, lookup)
        
        if verbose:
            print(f"Recommended: {next_x_label} (fidelity: {s})")
            print(f"Measurement: {measurement:.4f}")
            print(f"Max EI: {ei[best_idx]:.6f}")
        
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
        
        if verbose:
            print(f"Cumulative cost: {total_cost:.2f}, best_so_far: {best_so_far:.4f}")
        
        # 조기 종료 조건
        if s == 1.0 and np.isclose(measurement, min_target, atol=1e-6):
            if verbose:
                print('Found the minimum target value!')
            break
    
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
        'use_hyperparameter_bo': use_hyperparameter_bo
    }


def multiple_optimization_runs(param_space: Dict, label_maps: Dict, lookup: Dict,
                              num_runs: int = 100, cost_budget: float = 50.0,
                              num_init_design: int = 10, high_fidelity_ratio: float = 0.2,
                              min_target: float = 1.5249, model_config: Dict = None,
                              save_results: bool = True, results_filename: str = 'tl_bo_results.csv',
                              use_hyperparameter_bo: bool = False, pretrain_bo_trials: int = 0,
                              finetune_bo_trials: int = 0, data_size: str = 'small') -> List[Dict]:
    """
    다중 최적화 실행 (하이퍼파라미터 BO 지원)
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
            data_size=data_size
        )
        
        all_results.append(result)
        all_costs.append(result['total_cost'])
        
        if result['best_so_far'] <= min_target:
            print(f"Run {run+1}: Found target! Cost: {result['total_cost']:.2f}")
        else:
            print(f"Run {run+1}: Completed. Cost: {result['total_cost']:.2f}, Best: {result['best_so_far']:.4f}")
    
    # 결과 저장
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