"""
BNN 하이퍼파라미터 베이지안 최적화 모듈

BNN의 다양한 하이퍼파라미터들을 베이지안 최적화로 자동 튜닝합니다.
- 모델 구조 (hidden_dims, noise_type)
- 학습 설정 (epochs, learning_rate)
- 베이지안 설정 (kl_weight, kl_warmup_epochs, prior_std)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .bnn_models import TransferLearningBNN
import time


class BNNHyperparameterSpace:
    """BNN 핵심 하이퍼파라미터 탐색 공간 정의 (축소 버전)"""
    
    def __init__(self, data_size: str = 'small'):
        """
        Args:
            data_size: 'small', 'medium', 'large' - 데이터 크기에 따른 탐색 공간 조정
        """
        # 핵심 하이퍼파라미터만 최적화 (4개로 축소)
        
        if data_size == 'small':
            # 작은 데이터셋용 - 간단한 구조만
            self.hidden_layer_configs = [
                [32], [64],                      # Single layer
                [32, 32], [64, 32]               # Double layer (4개만)
            ]
            self.finetune_epochs_range = (50, 150)  # pretrain은 고정
            self.finetune_lrs = [1e-4, 5e-4, 1e-3]  # 3개만
            
        elif data_size == 'medium':
            self.hidden_layer_configs = [
                [32], [64], [128],               # Single layer
                [64, 64], [128, 64]              # Double layer (5개만)
            ]
            self.finetune_epochs_range = (75, 200)
            self.finetune_lrs = [5e-5, 1e-4, 5e-4, 1e-3]  # 4개만
            
        else:  # large
            self.hidden_layer_configs = [
                [64], [128],                     # Single layer
                [64, 64], [128, 64], [128, 128]  # Double layer (5개만)
            ]
            self.finetune_epochs_range = (100, 250)
            self.finetune_lrs = [1e-5, 5e-5, 1e-4, 5e-4]  # 4개만
        
        # BNN 특화 하이퍼파라미터 (가장 중요한 것만)
        self.kl_weights = [0.1, 1.0, 10.0]  # 3개만 (0.01, 5.0 제거)
        
        # 고정 하이퍼파라미터 (최적화하지 않음)
        self.fixed_params = {
            'pretrain_epochs': 200,  # 고정
            'pretrain_lr': 1e-3,     # 고정
            'kl_warmup_epochs': 10,  # 고정
            'prior_std': 1.0,        # 고정
            'noise_type': 'homoscedastic'  # 고정 (heteroscedastic은 복잡함)
        }
    
    def sample_random(self) -> Dict:
        """랜덤 BNN 하이퍼파라미터 샘플링 (핵심 4개만)"""
        # hidden_dims 랜덤 선택 수정
        config_idx = np.random.randint(0, len(self.hidden_layer_configs))
        selected_config = self.hidden_layer_configs[config_idx]
        
        params = {
            'hidden_dims': list(selected_config),
            'finetune_epochs': np.random.randint(self.finetune_epochs_range[0], self.finetune_epochs_range[1] + 1),
            'finetune_lr': np.random.choice(self.finetune_lrs),
            'kl_weight': np.random.choice(self.kl_weights),
        }
        # 고정 파라미터 추가
        params.update(self.fixed_params)
        return params
    
    def normalize_params(self, params: Dict) -> np.ndarray:
        """하이퍼파라미터를 [0,1] 범위로 정규화 (핵심 4개만)"""
        normalized = np.zeros(4)
        
        # hidden_dims 정규화 (config index)
        config_tuples = [tuple(config) for config in self.hidden_layer_configs]
        config_idx = config_tuples.index(tuple(params['hidden_dims']))
        normalized[0] = config_idx / (len(self.hidden_layer_configs) - 1) if len(self.hidden_layer_configs) > 1 else 0.0
        
        # finetune_epochs 정규화
        normalized[1] = (params['finetune_epochs'] - self.finetune_epochs_range[0]) / \
                       (self.finetune_epochs_range[1] - self.finetune_epochs_range[0])
        
        # finetune_lr 정규화 (log scale)
        log_finetune_lrs = np.log10(self.finetune_lrs)
        normalized[2] = (np.log10(params['finetune_lr']) - min(log_finetune_lrs)) / \
                       (max(log_finetune_lrs) - min(log_finetune_lrs)) if len(log_finetune_lrs) > 1 else 0.0
        
        # kl_weight 정규화 (log scale)
        log_kl_weights = np.log10(self.kl_weights)
        normalized[3] = (np.log10(params['kl_weight']) - min(log_kl_weights)) / \
                       (max(log_kl_weights) - min(log_kl_weights)) if len(log_kl_weights) > 1 else 0.0
        
        return normalized
    
    def denormalize_params(self, normalized: np.ndarray) -> Dict:
        """정규화된 값을 실제 하이퍼파라미터로 변환 (핵심 4개만)"""
        params = {}
        
        # hidden_dims
        config_idx = int(np.round(normalized[0] * (len(self.hidden_layer_configs) - 1)))
        config_idx = np.clip(config_idx, 0, len(self.hidden_layer_configs) - 1)
        params['hidden_dims'] = list(self.hidden_layer_configs[config_idx])
        
        # finetune_epochs
        params['finetune_epochs'] = int(np.round(
            normalized[1] * (self.finetune_epochs_range[1] - self.finetune_epochs_range[0]) + 
            self.finetune_epochs_range[0]
        ))
        
        # finetune_lr
        log_finetune_lrs = np.log10(self.finetune_lrs)
        if len(log_finetune_lrs) > 1:
            log_lr = normalized[2] * (max(log_finetune_lrs) - min(log_finetune_lrs)) + min(log_finetune_lrs)
            params['finetune_lr'] = float(10 ** log_lr)
            params['finetune_lr'] = min(self.finetune_lrs, key=lambda x: abs(x - params['finetune_lr']))
        else:
            params['finetune_lr'] = self.finetune_lrs[0]
        
        # kl_weight
        log_kl_weights = np.log10(self.kl_weights)
        if len(log_kl_weights) > 1:
            log_kl = normalized[3] * (max(log_kl_weights) - min(log_kl_weights)) + min(log_kl_weights)
            params['kl_weight'] = float(10 ** log_kl)
            params['kl_weight'] = min(self.kl_weights, key=lambda x: abs(x - params['kl_weight']))
        else:
            params['kl_weight'] = self.kl_weights[0]
        
        # 고정 파라미터 추가
        params.update(self.fixed_params)
        
        return params


def evaluate_bnn_hyperparameters(params: Dict, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray, input_dim: int,
                                device: str = 'cpu', verbose: bool = False) -> float:
    """BNN 하이퍼파라미터 평가"""
    try:
        # BNN 모델 생성
        bnn = TransferLearningBNN(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            device=device,
            prior_std=params['prior_std'],
            noise_type=params['noise_type']
        )
        
        # 학습 (pretrain + finetune을 하나로 통합)
        # 여기서는 validation을 위해 train 데이터만으로 학습
        bnn.pretrain(X_train, y_train, 
                    epochs=params['pretrain_epochs'], 
                    lr=params['pretrain_lr'], 
                    verbose=False)
        
        bnn.finetune(X_train, y_train,
                    epochs=params['finetune_epochs'],
                    lr=params['finetune_lr'],
                    kl_weight=params['kl_weight'],
                    kl_warmup_epochs=params['kl_warmup_epochs'],
                    verbose=False)
        
        # 검증 데이터로 성능 평가
        pred_mean, pred_std = bnn.predict(X_val, n_samples=50)  # 빠른 평가를 위해 샘플 수 줄임
        
        # MSE + 불확실성 품질을 고려한 점수
        mse = np.mean((pred_mean - y_val) ** 2)
        
        # 불확실성 품질: 예측 오차와 불확실성의 상관관계
        errors = np.abs(pred_mean - y_val)
        uncertainty_quality = np.corrcoef(errors, pred_std)[0, 1] if len(errors) > 1 else 0
        
        # 낮을수록 좋은 점수 (MSE 중심, 불확실성 품질 보정)
        score = mse - 0.1 * max(0, uncertainty_quality)  # 좋은 불확실성은 점수 개선
        
        # 수치 안정성 체크: 무한대나 너무 큰 값 방지
        if np.isnan(score) or np.isinf(score) or score > 1000:
            score = 1000.0
        elif score < 0:
            score = max(0, score)  # 음수 점수도 방지
        
        if verbose:
            print(f"  Params: {params}")
            print(f"  MSE: {mse:.4f}, Uncertainty quality: {uncertainty_quality:.3f}, Score: {score:.4f}")
        
        return score
        
    except Exception as e:
        if verbose:
            print(f"  Error with params {params}: {e}")
        return 1000.0  # 실패한 경우 매우 나쁜 점수 (무한대 대신 큰 수치)


def optimize_bnn_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: np.ndarray, y_val: np.ndarray,
                                input_dim: int, n_trials: int = 20,
                                data_size: str = 'small', device: str = 'cpu',
                                verbose: bool = False, stage: str = 'finetune') -> Tuple[Dict, float, List]:
    """
    BNN 하이퍼파라미터 베이지안 최적화
    
    Args:
        X_train, y_train: 훈련 데이터
        X_val, y_val: 검증 데이터  
        input_dim: 입력 차원
        n_trials: BO 시행 횟수
        data_size: 데이터 크기 카테고리
        device: 디바이스
        verbose: 상세 출력
        stage: 'pretrain' 또는 'finetune' - 최적화 단계
        
    Returns:
        최적 파라미터, 최적 성능, 전체 히스토리
    """
    if verbose:
        print(f"🔍 BNN 하이퍼파라미터 최적화 시작 (trials={n_trials}, data_size={data_size})")
    
    # 탐색 공간 초기화
    param_space = BNNHyperparameterSpace(data_size=data_size)
    
    # GP 모델 초기화
    kernel = Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
    
    # 히스토리 저장
    history = []
    X_observed = []
    y_observed = []
    
    # 초기 랜덤 샘플링 (n_trials의 1/3)
    n_initial = max(3, n_trials // 3)
    
    if verbose:
        print(f"📊 초기 랜덤 샘플링: {n_initial}개")
    
    for i in range(n_initial):
        params = param_space.sample_random()
        score = evaluate_bnn_hyperparameters(params, X_train, y_train, X_val, y_val, 
                                           input_dim, device, verbose=verbose)
        
        X_observed.append(param_space.normalize_params(params))
        y_observed.append(score)
        history.append({'params': params.copy(), 'score': score})
        
        if verbose:
            print(f"  Trial {i+1}/{n_initial}: Score = {score:.4f}")
    
    # GP 기반 베이지안 최적화
    if verbose:
        print(f"🎯 베이지안 최적화: {n_trials - n_initial}개")
    
    for i in range(n_initial, n_trials):
        # GP 학습 전 데이터 안전성 체크
        X_observed_array = np.array(X_observed)
        y_observed_array = np.array(y_observed)
        
        # 무한대나 NaN 값 필터링
        valid_mask = np.isfinite(y_observed_array) & (y_observed_array < 1000)
        if not np.any(valid_mask):
            if verbose:
                print(f"  Warning: All scores are invalid, using fallback")
            break
            
        X_observed_clean = X_observed_array[valid_mask]
        y_observed_clean = y_observed_array[valid_mask]
        
        if len(y_observed_clean) < 2:
            if verbose:
                print(f"  Warning: Not enough valid data points, using random selection")
            # 랜덤 선택으로 대체
            next_params = param_space.sample_random()
            normalized_params = param_space.normalize_params(next_params)
        else:
            gp.fit(X_observed_clean, y_observed_clean)
        
            # Expected Improvement으로 다음 점 찾기
            def negative_ei(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                best_y = np.min(y_observed_clean)  # 정리된 데이터 사용
                ei = expected_improvement(mu, sigma, best_y, xi=0.01)
                return -ei[0]
            
            # 최적화
            bounds = [(0, 1)] * 4  # 4개 하이퍼파라미터만
            try:
                result = minimize(negative_ei, x0=np.random.rand(4), bounds=bounds, method='L-BFGS-B')
                next_params = param_space.denormalize_params(result.x)
                normalized_params = result.x
            except Exception as e:
                if verbose:
                    print(f"  Optimization failed: {e}, using random selection")
                next_params = param_space.sample_random()
                # 랜덤 선택 시에도 정규화된 파라미터 생성
                normalized_params = param_space.normalize_params(next_params)
        score = evaluate_bnn_hyperparameters(next_params, X_train, y_train, X_val, y_val,
                                           input_dim, device, verbose=verbose)
        
        X_observed.append(normalized_params)
        y_observed.append(score)
        history.append({'params': next_params.copy(), 'score': score})
        
        if verbose:
            print(f"  Trial {i+1}/{n_trials}: Score = {score:.4f}")
    
    # 최적 결과 찾기
    best_idx = np.argmin(y_observed)
    best_params = history[best_idx]['params']
    best_score = history[best_idx]['score']
    
    if verbose:
        print(f"✅ 최적화 완료!")
        print(f"   최적 점수: {best_score:.4f}")
        print(f"   최적 파라미터: {best_params}")
    
    return best_params, best_score, history