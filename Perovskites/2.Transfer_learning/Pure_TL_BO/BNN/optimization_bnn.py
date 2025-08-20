import numpy as np
import itertools
import time
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from DNGO.models import TransferLearningDNN
from BNN.bnn_models import TransferLearningBNN


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


def train_dual_bnn_models(X_low: np.ndarray, y_low: np.ndarray, X_high: np.ndarray, y_high: np.ndarray,
                         input_dim: int = 3, hidden_dims: List[int] = [64, 64], device: str = 'cpu',
                         pretrain_epochs: int = 200, finetune_epochs: int = 100,
                         pretrain_lr: float = 1e-3, finetune_lr: float = 1e-4,
                         kl_weight: float = 1.0, kl_warmup_epochs: int = 10,
                         prior_std: float = 1.0, noise_type: str = 'homoscedastic',
                         use_hyperparameter_bo: bool = False, pretrain_bo_trials: int = 0,
                         finetune_bo_trials: int = 0, data_size: str = 'small',
                         verbose: bool = False) -> Tuple[TransferLearningBNN, TransferLearningBNN]:
    """
    듀얼 BNN 모델 학습 (LOW/HIGH 서로게이트 분리)
    
    Args:
        X_low: low-fidelity 입력 데이터
        y_low: low-fidelity 출력 데이터
        X_high: high-fidelity 입력 데이터
        y_high: high-fidelity 출력 데이터
        input_dim: 입력 차원
        hidden_dims: BNN hidden layer 차원들
        device: 디바이스
        pretrain_epochs: pretrain epochs
        finetune_epochs: finetune epochs
        pretrain_lr: pretrain 학습률
        finetune_lr: finetune 학습률
        kl_weight: KL divergence 가중치
        kl_warmup_epochs: KL warm-up epochs
        prior_std: 사전분포 표준편차
        noise_type: 'homoscedastic' or 'heteroscedastic'
        use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
        pretrain_bo_trials: pretrain BO 시행 횟수
        finetune_bo_trials: finetune BO 시행 횟수
        data_size: 데이터 크기
        verbose: 상세 출력
        
    Returns:
        (surrogate_L, surrogate_H): 로우파이/하이파이 전용 모델들
    """
    # Surrogate_L: 로우파이 전용 모델
    surrogate_L = TransferLearningBNN(
        input_dim=input_dim, 
        hidden_dims=hidden_dims, 
        device=device,
        prior_std=prior_std,
        noise_type=noise_type,
        use_hyperparameter_bo=use_hyperparameter_bo
    )
    
    # Surrogate_H: 하이파이 전용 모델 (전이학습)
    surrogate_H = TransferLearningBNN(
        input_dim=input_dim, 
        hidden_dims=hidden_dims, 
        device=device,
        prior_std=prior_std,
        noise_type=noise_type,
        use_hyperparameter_bo=use_hyperparameter_bo
    )
    
    # 1) Surrogate_L: 로우파이 데이터로 BNN 학습
    if len(X_low) > 0:
        surrogate_L.train_bnn(X_low, y_low, 
                            epochs=pretrain_epochs, lr=pretrain_lr,
                            kl_weight=kl_weight, kl_warmup_epochs=kl_warmup_epochs, 
                            verbose=verbose, pretrained_model=None)
        if verbose:
            print("✅ Surrogate_L 학습 완료 (LOW 데이터)")
    
    # 2) Surrogate_H: Surrogate_L의 가중치로 시작 → 하이파이로 미세조정
    if len(X_high) > 0:
        # Transfer learning: Surrogate_L의 학습된 가중치를 복사해서 시작
        surrogate_H.train_bnn(X_high, y_high,
                            epochs=finetune_epochs, lr=finetune_lr,
                            kl_weight=kl_weight, kl_warmup_epochs=kl_warmup_epochs,
                            verbose=verbose, 
                            pretrained_model=surrogate_L if len(X_low) > 0 else None)
        if verbose:
            print("✅ Surrogate_H 학습 완료 (Transfer from Surrogate_L → HIGH 미세조정)")
    else:
        # 하이파이 데이터 없으면 Surrogate_L을 그대로 복사
        if len(X_low) > 0:
            surrogate_H.train_bnn(X_low, y_low,
                                epochs=0, lr=finetune_lr,  # epochs=0으로 가중치만 복사
                                kl_weight=kl_weight, kl_warmup_epochs=kl_warmup_epochs,
                                verbose=False, pretrained_model=surrogate_L)
        if verbose:
            print("⚠️  Surrogate_H: 하이파이 데이터 없음, Surrogate_L 복사")
    
    return surrogate_L, surrogate_H


def recommend_next_bnn(surrogate_L: TransferLearningBNN, surrogate_H: TransferLearningBNN, 
                       param_ranges: List[range], X_low: np.ndarray, X_high: np.ndarray,
                       y_low: np.ndarray, y_high: np.ndarray, s: float,
                       n_samples: int = 100, verbose: bool = False) -> Tuple:
    """
    Dual BNN을 사용한 다음 실험점 추천 (Expected Improvement 최대화)
    
    Args:
        surrogate_L: 저신뢰도 전용 BNN 모델
        surrogate_H: 고신뢰도 전용 BNN 모델 (전이학습)
        param_ranges: 파라미터 범위
        X_low, X_high: 측정된 데이터
        y_low, y_high: 측정된 값
        s: 현재 요청 fidelity (0.1=low, 1.0=high)
        n_samples: 몬테카를로 샘플 수
        verbose: 상세 출력
        
    Returns:
        추천점, 예측값, 표준편차, EI, 최적 인덱스, 그리드
    """
    # 전체 조합 생성
    all_combinations = list(itertools.product(*param_ranges))
    X_grid = np.array(all_combinations, dtype=np.float32)
    
    
    
    # Fidelity에 따라 적절한 모델 선택
    if s == 1.0:  # 고신뢰도 요청
        model = surrogate_H
        if verbose:
            print("🎯 고신뢰도 EI 계산: Surrogate_H 사용")
    else:  # 저신뢰도 요청
        model = surrogate_L
        if verbose:
            print("🔍 저신뢰도 EI 계산: Surrogate_L 사용")
    
    # 전체 조합에 대한 예측
    y_pred, y_std = model.predict(X_grid, n_samples=n_samples)
    
    # 현재까지의 최적값 (high-fidelity 우선, 수치 안정성 개선)
    if len(y_high) > 0:
        y_best = np.min(y_high)
    elif len(y_low) > 0:
        y_best = np.min(y_low)
    else:
        # 데이터가 없으면 예측값 기반으로 초기화
        y_best = np.min(y_pred) - 0.1
        
    # 수치 안정성 체크: y_best가 무한대나 매우 큰 경우
    if np.isinf(y_best) or y_best > 1000:
        if len(y_pred) > 0:
            y_best = np.min(y_pred) - 0.1
        else:
            y_best = 1.5  # 기본 목표값
        
    # Expected Improvement 계산
    ei = expected_improvement(y_pred, y_std, y_best)
    
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
        if verbose:
            print(f"⚠️  Warning: All points measured at {current_fidelity} fidelity, selecting max EI anyway")
    else:
        # 유효한 점들 중에서 최대 EI 선택
        valid_ei = ei[valid_indices]
        best_valid_idx = np.argmax(valid_ei)
        best_idx = valid_indices[best_valid_idx]
    
    next_x_label = list(X_grid[best_idx].astype(int))
    
    return next_x_label, y_pred, y_std, ei, best_idx, X_grid


def single_optimization_run_bnn(param_space: Dict, label_maps: Dict, lookup: Dict,
                                cost_budget: float = 50.0, num_init_design: int = 10,
                                high_fidelity_ratio: float = 0.2, min_target: float = 1.5249,
                                random_state: int = 42, verbose: bool = True,
                                model_config: Dict = None, use_hyperparameter_bo: bool = False,
                                pretrain_bo_trials: int = 0, finetune_bo_trials: int = 0,
                                data_size: str = 'small', save_images: bool = False,
                                images_dir: str = 'images') -> Dict:
    """
    BNN을 사용한 단일 최적화 실행
    
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
        model_config: BNN 모델 설정
        use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
        pretrain_bo_trials: pretrain BO 시행 횟수
        finetune_bo_trials: finetune BO 시행 횟수
        data_size: 데이터 크기
        save_images: 이미지 저장 여부
        images_dir: 이미지 저장 디렉토리
        
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
            'hidden_dims': [64, 64],
            'pretrain_epochs': 200,
            'finetune_epochs': 100,
            'pretrain_lr': 1e-3,
            'finetune_lr': 1e-4,
            'kl_weight': 1.0,
            'kl_warmup_epochs': 10,
            'prior_std': 1.0,
            'noise_type': 'homoscedastic',
            'n_samples': 100,
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
    
    # 시각화용 데이터 추가
    visualization_data = []  # 각 iteration의 y_pred, y_std, ei, best_idx, X_grid 저장
    
    # 이미지 저장 폴더 설정 (한 번만)
    run_dir = None
    if save_images:
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_dir = os.path.join(images_dir, f"BNN_cost{int(cost_budget)}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        if verbose:
            print(f"💾 Images will be saved to: {run_dir}")
    
    # 초기 best_so_far 설정
    if len(y_high) > 0:
        best_so_far = np.min(y_high)
    
    if verbose:
        print(f"🧠 Using Bayesian Neural Network (BNN)")
        print(f"Initial cost: {total_cost:.2f}, Initial best_so_far: {best_so_far}")
        print(f"BNN Config: hidden_dims={model_config['hidden_dims']}, noise_type={model_config['noise_type']}")
        if use_hyperparameter_bo:
            print(f"🔧 Using BNN hyperparameter BO: finetune={finetune_bo_trials} trials")
    
    # 메인 최적화 루프
    while total_cost < cost_budget:
        iter_ += 1
        iter_start = time.time()
        
        if verbose:
            print(f"\n==== Iteration {iter_} ====")
        
        # Fidelity 스케줄링: 8번 중 1번만 high-fidelity
        s = 1.0 if (iter_ % 8 == 0) else 0.1
        
        # Dual BNN 모델 학습 (LOW/HIGH 분리)
        surrogate_L, surrogate_H = train_dual_bnn_models(
            X_low, y_low, X_high, y_high, 
            input_dim=model_config['input_dim'],
            hidden_dims=model_config['hidden_dims'],
            device=model_config['device'],
            pretrain_epochs=model_config['pretrain_epochs'],
            finetune_epochs=model_config['finetune_epochs'],
            pretrain_lr=model_config['pretrain_lr'],
            finetune_lr=model_config['finetune_lr'],
            kl_weight=model_config['kl_weight'],
            kl_warmup_epochs=model_config['kl_warmup_epochs'],
            prior_std=model_config['prior_std'],
            noise_type=model_config['noise_type'],
            use_hyperparameter_bo=use_hyperparameter_bo,
            pretrain_bo_trials=pretrain_bo_trials,
            finetune_bo_trials=finetune_bo_trials,
            data_size=data_size,
            verbose=False
        )
        
        # 하이퍼파라미터 기록 (Surrogate_H에서 가져오기)
        if use_hyperparameter_bo:
            hp_summary = surrogate_H.get_hyperparameter_summary()
            hyperparameter_history.append({
                'iteration': iter_,
                'finetune_params': hp_summary['finetune_best_params']
            })
            
            if verbose and hp_summary['finetune_best_params']:
                print(f"🔧 Optimal BNN params: {hp_summary['finetune_best_params']}")
        
        # 다음 실험점 추천 (Dual Surrogate 사용)
        next_x_label, y_pred, y_std, ei, best_idx, X_grid = recommend_next_bnn(
            surrogate_L, surrogate_H, param_ranges, X_low, X_high, y_low, y_high, s,
            n_samples=model_config['n_samples'], verbose=False
        )
        
        # 두 모델의 예측값 모두 계산 (시각화용)
        import itertools
        all_combinations = list(itertools.product(*param_ranges))
        X_grid_full = np.array(all_combinations, dtype=np.float32)
        
        # Surrogate_L 예측
        y_pred_L, y_std_L = surrogate_L.predict(X_grid_full, n_samples=model_config['n_samples'])
        # Surrogate_H 예측
        y_pred_H, y_std_H = surrogate_H.predict(X_grid_full, n_samples=model_config['n_samples'])
        
        # 시각화용 데이터 저장 (두 모델 모두)
        visualization_data.append({
            'iteration': iter_,
            'y_pred': y_pred.copy(),  # 현재 사용된 모델의 예측
            'y_std': y_std.copy(),
            'y_pred_L': y_pred_L.copy(),  # Surrogate_L 예측
            'y_std_L': y_std_L.copy(),
            'y_pred_H': y_pred_H.copy(),  # Surrogate_H 예측
            'y_std_H': y_std_H.copy(),
            'ei': ei.copy(),
            'best_idx': best_idx,
            'X_grid': X_grid.copy(),
            'fidelity': s,
            'recommended_point': next_x_label.copy()
        })
        
        # 이미지 저장 (요청된 경우)
        if save_images and run_dir is not None:
            from visualization import plot_bnn_iteration_results
            from data_utils import create_all_combinations_data
            
            # 전체 조합 데이터 생성 
            param_ranges = [range(1, 17), range(1, 4), range(1, 5)]
            organic_options = param_space['organic']  # 이미 리스트임
            cation_options = param_space['cation']    # 이미 리스트임
            anion_options = param_space['anion']      # 이미 리스트임
            original_data = create_all_combinations_data(
                param_ranges, lookup, organic_options, cation_options, anion_options
            )
            
            # iteration 시각화 (plt.show() 대신 저장)
            import matplotlib
            matplotlib.use('Agg')  # 화면 출력 없이 저장만
            import matplotlib.pyplot as plt
            
            plot_bnn_iteration_results(
                iteration_data=visualization_data[-1],
                original_data=original_data,
                X_low=X_low,
                X_high=X_high,
                show=False  # 백그라운드 저장 시에는 show하지 않음
            )
            
            # 파일 저장
            filename = f"iteration_{iter_:03d}_fidelity_{s}.png"
            plt.savefig(os.path.join(run_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()  # 메모리 정리
            
            if verbose:
                print(f"💾 Image saved: {filename}")
        
        # 측정
        measurement = measure_from_label(next_x_label, s, label_maps, lookup)
        
        if verbose:
            print(f"Recommended: {next_x_label} (fidelity: {s})")
            print(f"Measurement: {measurement:.4f}")
            print(f"Max EI: {ei[best_idx]:.6f}")
            print(f"Predicted: {y_pred[best_idx]:.4f} ± {y_std[best_idx]:.4f}")
        
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
    
    # 기존 DNGO 시각화와 호환성을 위한 데이터 변환
    best_values_history = [x[3] for x in best_so_far_curve]  # best-so-far 값들
    cost_history = [x[2] for x in cost_data]  # 비용 기록
    fidelity_history = [data['fidelity'] for data in visualization_data]  # fidelity 기록
    ei_history = [data['ei'][data['best_idx']] for data in visualization_data]  # EI 기록
    
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
        'model_type': 'BNN',
        'model_config': model_config,
        'visualization_data': visualization_data,
        # 기존 DNGO 시각화 호환용
        'best_values_history': best_values_history,
        'cost_history': cost_history,
        'fidelity_history': fidelity_history,
        'ei_history': ei_history
    }


def multiple_optimization_runs_bnn(param_space: Dict, label_maps: Dict, lookup: Dict,
                                   num_runs: int = 100, cost_budget: float = 50.0,
                                   num_init_design: int = 10, high_fidelity_ratio: float = 0.2,
                                   min_target: float = 1.5249, model_config: Dict = None,
                                   save_results: bool = True, results_filename: str = 'bnn_bo_results.csv') -> List[Dict]:
    """
    BNN을 사용한 다중 최적화 실행
    """
    import pandas as pd
    
    all_results = []
    all_costs = []
    
    print(f"🧠 Starting {num_runs} optimization runs with Bayesian Neural Network...")
    
    for run in range(num_runs):
        print(f"\n===== Run {run+1}/{num_runs} =====")
        
        result = single_optimization_run_bnn(
            param_space=param_space,
            label_maps=label_maps,
            lookup=lookup,
            cost_budget=cost_budget,
            num_init_design=num_init_design,
            high_fidelity_ratio=high_fidelity_ratio,
            min_target=min_target,
            random_state=run,  # 각 런마다 다른 시드
            verbose=False,  # 다중 실행 시 상세 출력 비활성화
            model_config=model_config
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
            'model_type': [r['model_type'] for r in all_results]
        })
        results_df.to_csv(results_filename, index=False)
        print(f"\nResults saved to {results_filename}")
    
    # 요약 통계
    success_rate = sum(1 for r in all_results if r['best_so_far'] <= min_target) / num_runs
    avg_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    
    print(f"\n=== Summary Statistics (BNN) ===")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average cost: {avg_cost:.2f} ± {std_cost:.2f}")
    print(f"Min cost: {np.min(all_costs):.2f}")
    print(f"Max cost: {np.max(all_costs):.2f}")
    
    return all_results