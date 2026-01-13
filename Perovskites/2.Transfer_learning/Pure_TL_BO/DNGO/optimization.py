import numpy as np
import itertools
import time
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from DNGO.models import TransferLearningDNN, BayesianLinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class OnlineBayesianLinearRegression:
    """온라인 학습을 위한 Bayesian Linear Regression 모델"""

    def __init__(self, alpha: float = 1.0, beta: float = 25.0,
                 forgetting_factor: float = 0.99, memory_size: int = 100):
        """
        Args:
            alpha: 사전분포 정밀도 (precision)
            beta: 노이즈 정밀도
            forgetting_factor: 망각 계수 (0~1, 클수록 이전 데이터 영향 증가)
            memory_size: 메모리 버퍼 크기
        """
        self.alpha = alpha
        self.beta = beta
        self.forgetting_factor = forgetting_factor
        self.memory_size = memory_size

        self.mean = None
        self.cov = None
        self.memory_buffer = deque(maxlen=memory_size)
        self.n_samples = 0

        # 학습 메트릭 추적
        self.training_history = {
            'prediction_errors': [],      # 예측 오차
            'uncertainties': [],          # 불확실성 (분산)
            'log_likelihoods': [],        # 로그 우도
            'cov_trace': [],              # 공분산 행렬 trace (모델 확신도)
            'update_timestamps': [],      # 업데이트 시점
            'n_samples_history': []       # 샘플 수 기록
        }
        
    def fit(self, Phi: np.ndarray, t: np.ndarray):
        """전체 데이터로 초기 학습"""
        N, M = Phi.shape
        
        # 사전분포
        S0_inv = self.alpha * np.eye(M)
        
        # 사후분포 계산
        SN_inv = S0_inv + self.beta * Phi.T @ Phi
        self.cov = np.linalg.inv(SN_inv)
        self.mean = self.beta * self.cov @ Phi.T @ t
        
        # 메모리 버퍼 초기화
        self.memory_buffer.clear()
        for i in range(N):
            self.memory_buffer.append((Phi[i], t[i]))
        self.n_samples = N
        
    def update_online(self, phi_new: np.ndarray, t_new: float):
        """새로운 데이터 포인트로 온라인 업데이트"""
        if self.mean is None:
            # 첫 번째 데이터 포인트인 경우
            M = len(phi_new)
            self.cov = (1.0 / self.alpha) * np.eye(M)
            self.mean = np.zeros(M)

        # 업데이트 전 예측 (메트릭 기록용)
        phi_flat = phi_new.flatten()
        pred_before, var_before = self.predict(phi_flat)
        prediction_error = float(t_new - pred_before)

        # 망각 계수 적용: 이전 정보의 영향력을 감소시킴
        self.cov = self.cov / self.forgetting_factor

        # Sequential Bayesian update (Recursive Least Squares 형태)
        phi_new = phi_new.reshape(-1, 1)

        # Kalman gain 계산
        k = self.beta * self.cov @ phi_new
        denominator = 1 + self.beta * phi_new.T @ self.cov @ phi_new
        k = k / denominator

        # 평균 업데이트
        self.mean = self.mean + k.flatten() * prediction_error

        # 공분산 업데이트 (Joseph form for numerical stability)
        I = np.eye(len(self.mean))
        self.cov = (I - k @ phi_new.T) @ self.cov

        # 메모리 버퍼에 추가
        self.memory_buffer.append((phi_new.flatten(), t_new))
        self.n_samples += 1

        # 학습 메트릭 기록
        log_likelihood = -0.5 * (prediction_error**2 * self.beta + np.log(2 * np.pi / self.beta))
        self.training_history['prediction_errors'].append(prediction_error)
        self.training_history['uncertainties'].append(float(var_before))
        self.training_history['log_likelihoods'].append(float(log_likelihood))
        self.training_history['cov_trace'].append(float(np.trace(self.cov)))
        self.training_history['update_timestamps'].append(self.n_samples)
        self.training_history['n_samples_history'].append(self.n_samples)
        
    def predict(self, phi: np.ndarray) -> Tuple[float, float]:
        """예측 평균과 분산 반환"""
        if self.mean is None:
            return 0.0, 1.0
            
        mean = phi @ self.mean
        var = 1.0/self.beta + phi @ self.cov @ phi.T
        return mean, var
        
    def periodic_refit(self):
        """메모리 버퍼의 데이터로 주기적으로 재학습"""
        if len(self.memory_buffer) > 0:
            Phi = np.array([x[0] for x in self.memory_buffer])
            t = np.array([x[1] for x in self.memory_buffer])
            self.fit(Phi, t)


class OnlineTransferLearningDNN(TransferLearningDNN):
    """온라인 학습을 위한 Transfer Learning DNN"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 device: str = 'cpu', replay_buffer_size: int = 100,
                 online_batch_size: int = 16, online_epochs: int = 5,
                 use_hyperparameter_bo: bool = False):
        """
        Args:
            replay_buffer_size: 리플레이 버퍼 크기
            online_batch_size: 온라인 학습 배치 크기
            online_epochs: 온라인 업데이트 시 epoch 수
            use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
        """
        super().__init__(input_dim, hidden_dim, device, use_hyperparameter_bo=use_hyperparameter_bo)

        self.replay_buffer_size = replay_buffer_size
        self.online_batch_size = online_batch_size
        self.online_epochs = online_epochs

        # 리플레이 버퍼
        self.replay_buffer = {
            'low': deque(maxlen=replay_buffer_size),
            'high': deque(maxlen=replay_buffer_size)
        }

        # 온라인 학습 메트릭 추적
        self.online_training_history = {
            'losses': [],                 # 온라인 업데이트 시 loss
            'losses_per_epoch': [],       # 각 epoch별 loss
            'update_counts': [],          # 업데이트 횟수
            'fidelities': [],             # 각 업데이트의 fidelity
            'buffer_sizes': [],           # 리플레이 버퍼 크기
            'learning_rates': []          # 사용된 학습률
        }
        self.online_update_count = 0
        
    def update_online(self, X_new: np.ndarray, y_new: np.ndarray,
                     fidelity: str = 'high', lr: float = 1e-4):
        """새로운 데이터로 온라인 업데이트"""
        self.online_update_count += 1

        # 리플레이 버퍼에 추가
        for x, y in zip(X_new, y_new):
            self.replay_buffer[fidelity].append((x, y))

        # 리플레이 버퍼에서 샘플링하여 미니배치 학습
        epoch_losses = []
        final_loss = None

        if len(self.replay_buffer[fidelity]) >= self.online_batch_size:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            self.model.train()

            for epoch in range(self.online_epochs):
                # 랜덤 샘플링
                indices = np.random.choice(
                    len(self.replay_buffer[fidelity]),
                    size=min(self.online_batch_size, len(self.replay_buffer[fidelity])),
                    replace=False
                )

                batch_data = [self.replay_buffer[fidelity][i] for i in indices]
                X_batch = np.array([x for x, _ in batch_data])
                y_batch = np.array([y for _, y in batch_data])

                # Tensor 변환
                X_tensor = torch.FloatTensor(X_batch).to(self.device)
                y_tensor = torch.FloatTensor(y_batch).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X_tensor).squeeze()
                loss = criterion(outputs, y_tensor)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                # Loss 기록
                epoch_losses.append(loss.item())

            self.model.eval()
            final_loss = epoch_losses[-1] if epoch_losses else None

        # 메트릭 기록
        self.online_training_history['losses'].append(final_loss)
        self.online_training_history['losses_per_epoch'].append(epoch_losses)
        self.online_training_history['update_counts'].append(self.online_update_count)
        self.online_training_history['fidelities'].append(fidelity)
        self.online_training_history['buffer_sizes'].append(len(self.replay_buffer[fidelity]))
        self.online_training_history['learning_rates'].append(lr)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement 계산"""
    sigma = np.maximum(sigma, 1e-8)
    z = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei


def train_dngo_ol_models(X_low: np.ndarray, y_low: np.ndarray, X_high: np.ndarray, y_high: np.ndarray,
                        input_dim: int = 3, hidden_dim: int = 64, device: str = 'cpu',
                        pretrain_epochs: int = 200, finetune_epochs: int = 100,
                        pretrain_lr: float = 1e-3, finetune_lr: float = 1e-4,
                        online_lr: float = 1e-5, forgetting_factor: float = 0.99,
                        memory_size: int = 100, replay_buffer_size: int = 100,
                        online_batch_size: int = 16, online_epochs: int = 5,
                        verbose: bool = False,
                        use_hyperparameter_bo: bool = False, pretrain_bo_trials: int = 0,
                        finetune_bo_trials: int = 0, data_size: str = 'small',
                        use_loocv: bool = False, use_uncertainty_loss: bool = False,
                        uncertainty_weight: float = 0.3,
                        use_freeze: bool = False, unfreeze_ratio: float = 1.0) -> Tuple:
    """
    DNGO-OL 모델 학습 (온라인 학습 지원, 하이퍼파라미터 BO 지원)

    Args:
        use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
        pretrain_bo_trials: pretrain BO 시행 횟수
        finetune_bo_trials: finetune BO 시행 횟수
        data_size: 데이터 크기 ('small', 'medium', 'large')
        use_loocv: HP 최적화 시 LOOCV 사용 여부
        use_uncertainty_loss: HP 최적화 시 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치
        use_freeze: Freeze 기법 사용 여부
        unfreeze_ratio: 해동할 레이어 비율 (0.0~1.0)

    Returns:
        (model, blr_L, blr_H): DNN 모델과 LOW/HIGH BLR 모델들
    """
    # 온라인 DNN 모델 생성
    model = OnlineTransferLearningDNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        device=device,
        replay_buffer_size=replay_buffer_size,
        online_batch_size=online_batch_size,
        online_epochs=online_epochs,
        use_hyperparameter_bo=use_hyperparameter_bo
    )

    # 초기 학습 (기존 데이터가 있는 경우)
    if len(X_low) > 0:
        if use_hyperparameter_bo and pretrain_bo_trials > 0:
            model.pretrain(X_low, y_low, epochs=pretrain_epochs, lr=pretrain_lr, verbose=verbose,
                          bo_trials=pretrain_bo_trials, data_size=data_size,
                          use_loocv=use_loocv, use_uncertainty_loss=use_uncertainty_loss,
                          uncertainty_weight=uncertainty_weight)
        else:
            model.pretrain(X_low, y_low, epochs=pretrain_epochs, lr=pretrain_lr, verbose=verbose)
        if verbose:
            print("✅ DNGO-OL: Low-fidelity 초기 학습 완료")

    if len(X_high) > 0:
        if use_hyperparameter_bo and finetune_bo_trials > 0:
            model.finetune(X_high, y_high, epochs=finetune_epochs, lr=finetune_lr, verbose=verbose,
                          bo_trials=finetune_bo_trials, data_size=data_size,
                          use_loocv=use_loocv, use_uncertainty_loss=use_uncertainty_loss,
                          uncertainty_weight=uncertainty_weight,
                          use_freeze=use_freeze, unfreeze_ratio=unfreeze_ratio)
        else:
            model.finetune(X_high, y_high, epochs=finetune_epochs, lr=finetune_lr, verbose=verbose,
                          use_freeze=use_freeze, unfreeze_ratio=unfreeze_ratio)
        if verbose:
            print("✅ DNGO-OL: High-fidelity 미세조정 완료")
    
    # 온라인 BLR 모델 생성 (LOW/HIGH 분리)
    blr_L = OnlineBayesianLinearRegression(
        alpha=1.0, beta=25.0,
        forgetting_factor=forgetting_factor,
        memory_size=memory_size
    )
    
    blr_H = OnlineBayesianLinearRegression(
        alpha=1.0, beta=25.0,
        forgetting_factor=forgetting_factor,
        memory_size=memory_size
    )
    
    # BLR 초기 학습
    if len(X_low) > 0:
        features_low = model.extract_features(X_low)
        blr_L.fit(features_low, y_low)
    
    if len(X_high) > 0:
        features_high = model.extract_features(X_high)
        blr_H.fit(features_high, y_high)
    
    return model, blr_L, blr_H


def update_dngo_ol_online(model: OnlineTransferLearningDNN, 
                         blr_L: OnlineBayesianLinearRegression,
                         blr_H: OnlineBayesianLinearRegression,
                         new_x: np.ndarray, new_y: float, fidelity: float,
                         online_lr: float = 1e-5, verbose: bool = False):
    """새로운 데이터 포인트로 DNGO-OL 모델 온라인 업데이트"""
    
    # 적절한 BLR 모델 선택
    if fidelity == 1.0:
        fidelity_str = 'high'
        blr = blr_H
    else:
        fidelity_str = 'low'
        blr = blr_L
    
    # DNN 온라인 업데이트
    model.update_online(
        new_x.reshape(1, -1), 
        np.array([new_y]), 
        fidelity=fidelity_str,
        lr=online_lr
    )
    
    # 새로운 특징 추출
    new_features = model.extract_features(new_x.reshape(1, -1))
    
    # BLR 온라인 업데이트
    blr.update_online(new_features.flatten(), new_y)
    
    if verbose:
        print(f"✅ DNGO-OL 온라인 업데이트 완료 ({fidelity_str} fidelity)")


def recommend_next_dngo_ol(model: OnlineTransferLearningDNN,
                          blr_L: OnlineBayesianLinearRegression,
                          blr_H: OnlineBayesianLinearRegression,
                          param_ranges: List[range], X_low: np.ndarray, X_high: np.ndarray,
                          y_low: np.ndarray, y_high: np.ndarray, s: float,
                          verbose: bool = False) -> Tuple:
    """DNGO-OL을 사용한 다음 실험점 추천"""
    
    # 전체 조합 생성
    all_combinations = list(itertools.product(*param_ranges))
    X_grid = np.array(all_combinations, dtype=np.float32)
    
    # Fidelity에 따라 적절한 BLR 모델 선택
    if s == 1.0:
        blr = blr_H
        if verbose:
            print("🎯 고신뢰도 EI 계산: BLR_H 사용")
    else:
        blr = blr_L
        if verbose:
            print("🔍 저신뢰도 EI 계산: BLR_L 사용")
    
    # 현재까지의 최적값
    if len(y_high) > 0:
        y_best = np.min(y_high)
    elif len(y_low) > 0:
        y_best = np.min(y_low)
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
    
    # 이미 측정된 점들 추적
    measured_points = set()
    for x in X_low:
        measured_points.add((tuple(x.astype(int)), 'low'))
    for x in X_high:
        measured_points.add((tuple(x.astype(int)), 'high'))
    
    current_fidelity = 'high' if s == 1.0 else 'low'
    
    # 유효한 후보 필터링
    valid_indices = []
    for i, combo in enumerate(X_grid):
        combo_tuple = tuple(combo.astype(int))
        if (combo_tuple, current_fidelity) not in measured_points:
            valid_indices.append(i)
    
    if not valid_indices:
        best_idx = np.argmax(ei)
        if verbose:
            print(f"⚠️  Warning: All points measured at {current_fidelity} fidelity")
    else:
        valid_ei = ei[valid_indices]
        best_valid_idx = np.argmax(valid_ei)
        best_idx = valid_indices[best_valid_idx]
    
    next_x_label = list(X_grid[best_idx].astype(int))
    
    return next_x_label, y_pred, y_std, ei, best_idx, X_grid


def single_optimization_run_dngo_ol(param_space: Dict, label_maps: Dict, lookup: Dict,
                                   cost_budget: float = 50.0, num_init_design: int = 10,
                                   high_fidelity_ratio: float = 0.2, min_target: float = 1.5249,
                                   random_state: int = 42, verbose: bool = True,
                                   model_config: Dict = None, save_images: bool = False,
                                   images_dir: str = 'images',
                                   use_hyperparameter_bo: bool = False,
                                   pretrain_bo_trials: int = 0, finetune_bo_trials: int = 0,
                                   data_size: str = 'small',
                                   use_loocv: bool = False, use_uncertainty_loss: bool = False,
                                   uncertainty_weight: float = 0.3,
                                   use_freeze: bool = False, unfreeze_ratio: float = 1.0) -> Dict:
    """
    DNGO-OL을 사용한 단일 최적화 실행 (하이퍼파라미터 BO 지원)

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
        save_images: 이미지 저장 여부
        images_dir: 이미지 저장 디렉토리
        use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
        pretrain_bo_trials: pretrain BO 시행 횟수
        finetune_bo_trials: finetune BO 시행 횟수
        data_size: 데이터 크기 ('small', 'medium', 'large')
        use_loocv: HP 최적화 시 LOOCV 사용 여부
        use_uncertainty_loss: HP 최적화 시 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치
        use_freeze: Freeze 기법 사용 여부
        unfreeze_ratio: 해동할 레이어 비율 (0.0~1.0)

    Returns:
        결과 딕셔너리
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
            'pretrain_lr': 1e-3,
            'finetune_lr': 1e-4,
            'online_lr': 1e-5,
            'forgetting_factor': 0.99,
            'memory_size': 100,
            'replay_buffer_size': 100,
            'online_batch_size': 16,
            'online_epochs': 5,
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
    best_so_far = np.inf
    best_so_far_curve = []
    timing_data = []
    cost_data = []
    iter_ = 0
    visualization_data = []
    
    # 이미지 저장 폴더 설정
    run_dir = None
    if save_images:
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_dir = os.path.join(images_dir, f"DNGO-OL_cost{int(cost_budget)}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        if verbose:
            print(f"💾 Images will be saved to: {run_dir}")
    
    # 초기 best_so_far 설정
    if len(y_high) > 0:
        best_so_far = np.min(y_high)
    
    # 초기 모델 학습
    model, blr_L, blr_H = train_dngo_ol_models(
        X_low, y_low, X_high, y_high,
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        device=model_config['device'],
        pretrain_epochs=model_config['pretrain_epochs'],
        finetune_epochs=model_config['finetune_epochs'],
        pretrain_lr=model_config['pretrain_lr'],
        finetune_lr=model_config['finetune_lr'],
        online_lr=model_config['online_lr'],
        forgetting_factor=model_config['forgetting_factor'],
        memory_size=model_config['memory_size'],
        replay_buffer_size=model_config['replay_buffer_size'],
        online_batch_size=model_config['online_batch_size'],
        online_epochs=model_config['online_epochs'],
        verbose=verbose,
        use_hyperparameter_bo=use_hyperparameter_bo,
        pretrain_bo_trials=pretrain_bo_trials,
        finetune_bo_trials=finetune_bo_trials,
        data_size=data_size,
        use_loocv=use_loocv,
        use_uncertainty_loss=use_uncertainty_loss,
        uncertainty_weight=uncertainty_weight,
        use_freeze=use_freeze,
        unfreeze_ratio=unfreeze_ratio
    )
    
    if verbose:
        print(f"🚀 Using DNGO-OL (Online Learning)")
        print(f"Initial cost: {total_cost:.2f}, Initial best_so_far: {best_so_far}")
        print(f"Config: forgetting_factor={model_config['forgetting_factor']}, memory_size={model_config['memory_size']}")
    
    # 메인 최적화 루프
    while total_cost < cost_budget:
        iter_ += 1
        iter_start = time.time()
        
        if verbose:
            print(f"\n==== Iteration {iter_} ====")
        
        # Fidelity 스케줄링: 8번 중 1번만 high-fidelity
        s = 1.0 if (iter_ % 8 == 0) else 0.1
        
        # 주기적으로 BLR 재학습 (안정성 향상)
        if iter_ % 20 == 0:
            if verbose:
                print("🔄 Periodic BLR refit")
            blr_L.periodic_refit()
            blr_H.periodic_refit()
        
        # 다음 실험점 추천
        next_x_label, y_pred, y_std, ei, best_idx, X_grid = recommend_next_dngo_ol(
            model, blr_L, blr_H, param_ranges, X_low, X_high, y_low, y_high, s,
            verbose=False
        )
        
        # 시각화용 데이터 저장
        visualization_data.append({
            'iteration': iter_,
            'y_pred': y_pred.copy(),
            'y_std': y_std.copy(),
            'ei': ei.copy(),
            'best_idx': best_idx,
            'X_grid': X_grid.copy(),
            'fidelity': s,
            'recommended_point': next_x_label.copy()
        })
        
        # 측정
        measurement = measure_from_label(next_x_label, s, label_maps, lookup)

        # BO iteration 로그 (항상 출력)
        fid_str = "H" if s == 1.0 else "L"
        print(f"  [Iter {iter_:3d}] point={next_x_label}, fid={fid_str}, "
              f"EI={ei[best_idx]:.4f}, pred={y_pred[best_idx]:.3f}±{y_std[best_idx]:.3f}, "
              f"actual={measurement:.4f}", flush=True)

        if verbose:
            print(f"Recommended: {next_x_label} (fidelity: {s})")
            print(f"Measurement: {measurement:.4f}")
            print(f"Max EI: {ei[best_idx]:.6f}")
            print(f"Predicted: {y_pred[best_idx]:.4f} ± {y_std[best_idx]:.4f}")
        
        # 온라인 모델 업데이트
        new_x = np.array(next_x_label, dtype=np.float32)
        update_dngo_ol_online(
            model, blr_L, blr_H, new_x, measurement, s,
            online_lr=model_config['online_lr'],
            verbose=False
        )
        
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
    
    # 결과 호환성을 위한 데이터 변환
    best_values_history = [x[3] for x in best_so_far_curve]
    cost_history = [x[2] for x in cost_data]
    fidelity_history = [data['fidelity'] for data in visualization_data]
    ei_history = [data['ei'][data['best_idx']] for data in visualization_data]

    # 학습 메트릭 수집
    training_history = {
        'pretrain_losses': model.pretrain_losses if hasattr(model, 'pretrain_losses') else [],
        'finetune_losses': model.finetune_losses if hasattr(model, 'finetune_losses') else [],
        'online_dnn_history': model.online_training_history if hasattr(model, 'online_training_history') else {},
        'blr_L_history': blr_L.training_history if hasattr(blr_L, 'training_history') else {},
        'blr_H_history': blr_H.training_history if hasattr(blr_H, 'training_history') else {},
    }

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
        'model_type': 'DNGO-OL',
        'model_config': model_config,
        'visualization_data': visualization_data,
        'best_values_history': best_values_history,
        'cost_history': cost_history,
        'fidelity_history': fidelity_history,
        'ei_history': ei_history,
        # 학습 메트릭 추가
        'training_history': training_history,
        'pretrain_losses': training_history['pretrain_losses'],
        'finetune_losses': training_history['finetune_losses'],
        'online_dnn_losses': training_history['online_dnn_history'].get('losses', []),
        'blr_L_errors': training_history['blr_L_history'].get('prediction_errors', []),
        'blr_H_errors': training_history['blr_H_history'].get('prediction_errors', []),
    }


def multiple_optimization_runs_dngo_ol(param_space: Dict, label_maps: Dict, lookup: Dict,
                                      num_runs: int = 100, cost_budget: float = 50.0,
                                      num_init_design: int = 10, high_fidelity_ratio: float = 0.2,
                                      min_target: float = 1.5249, model_config: Dict = None,
                                      save_results: bool = True,
                                      results_filename: str = 'dngo_ol_results.csv',
                                      use_hyperparameter_bo: bool = False,
                                      pretrain_bo_trials: int = 0, finetune_bo_trials: int = 0,
                                      data_size: str = 'small',
                                      use_loocv: bool = False, use_uncertainty_loss: bool = False,
                                      uncertainty_weight: float = 0.3,
                                      use_freeze: bool = False, unfreeze_ratio: float = 1.0) -> List[Dict]:
    """
    DNGO-OL을 사용한 다중 최적화 실행 (하이퍼파라미터 BO 지원)
    """
    import pandas as pd

    all_results = []
    all_costs = []

    print(f"🚀 Starting {num_runs} optimization runs with DNGO-OL (Online Learning)...")
    if use_hyperparameter_bo:
        print(f"   Hyperparameter BO: pretrain_trials={pretrain_bo_trials}, finetune_trials={finetune_bo_trials}")
        print(f"   LOOCV: {use_loocv}, Uncertainty Loss: {use_uncertainty_loss} (weight={uncertainty_weight})")

    for run in range(num_runs):
        print(f"\n===== Run {run+1}/{num_runs} =====")

        result = single_optimization_run_dngo_ol(
            param_space=param_space,
            label_maps=label_maps,
            lookup=lookup,
            cost_budget=cost_budget,
            num_init_design=num_init_design,
            high_fidelity_ratio=high_fidelity_ratio,
            min_target=min_target,
            random_state=run,
            verbose=False,
            model_config=model_config,
            use_hyperparameter_bo=use_hyperparameter_bo,
            pretrain_bo_trials=pretrain_bo_trials,
            finetune_bo_trials=finetune_bo_trials,
            data_size=data_size,
            use_loocv=use_loocv,
            use_uncertainty_loss=use_uncertainty_loss,
            uncertainty_weight=uncertainty_weight,
            use_freeze=use_freeze,
            unfreeze_ratio=unfreeze_ratio
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
    
    print(f"\n=== Summary Statistics (DNGO-OL) ===")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average cost: {avg_cost:.2f} ± {std_cost:.2f}")
    print(f"Min cost: {np.min(all_costs):.2f}")
    print(f"Max cost: {np.max(all_costs):.2f}")
    
    return all_results