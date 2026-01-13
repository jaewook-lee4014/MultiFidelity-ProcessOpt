import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split


class TransferLearningDNN:
    """
    Transfer Learning을 위한 Deep Neural Network 클래스
    - Pretrain: low-fidelity 데이터로 전체 네트워크 학습
    - Finetune: high-fidelity 데이터로 unfreeze_ratio에 따라 부분 학습
    - 하이퍼파라미터 베이지안 최적화 지원 (구조 + unfreeze_ratio)

    네트워크 구조:
        feature_net: 범용 표현 학습 레이어들 (pretrain에서 학습, finetune에서 부분 동결)
        out_layer: 출력층 (항상 학습)
    """

    def __init__(self, input_dim, hidden_dim=64, device='cpu', use_hyperparameter_bo=False,
                 activation='relu'):
        """
        Transfer Learning을 위한 Deep Neural Network

        Args:
            input_dim: 입력 차원
            hidden_dim: 은닉층 차원
            device: 디바이스 ('cpu' or 'cuda')
            use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
            activation: 활성화 함수 ('relu', 'tanh', 'relu_tanh')
                - 'relu': 모든 레이어에 ReLU (기본값)
                - 'tanh': 모든 레이어에 tanh (DNGO 논문 권장)
                - 'relu_tanh': 앞쪽 레이어 ReLU, 마지막 레이어 tanh (혼합형)
        """
        self.input_dim = input_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_hyperparameter_bo = use_hyperparameter_bo
        self.activation = activation
        self.pretrain_losses = []
        self.finetune_losses = []

        # BO 관련 변수
        self.pretrain_best_params = None
        self.finetune_best_params = None
        self.pretrain_bo_history = []
        self.finetune_bo_history = []

        # Freeze 관련 변수
        self.num_layers = 2  # 기본 레이어 수 (BO로 결정될 수 있음)
        self.layer_list = []  # 개별 레이어 리스트 (freeze 제어용)

        # Incremental learning 관련 변수
        self.incremental_params = None  # BO로 설정될 파라미터
        self.data_buffer = {'X_low': [], 'y_low': [], 'X_high': [], 'y_high': []}
        self.update_counter = 0
        self.last_learning_rates = {'pretrain': 1e-3, 'finetune': 1e-4}

        # 기본 모델 구조 (BO 사용하지 않을 때)
        if not use_hyperparameter_bo:
            self._build_default_model(hidden_dim)
    
    def _get_activation(self, layer_idx: int, total_layers: int):
        """레이어별 활성화 함수 반환"""
        if self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'relu_tanh':
            # 마지막 레이어만 tanh, 나머지는 ReLU
            if layer_idx == total_layers - 1:
                return nn.Tanh()
            else:
                return nn.ReLU()
        else:  # 'relu' (기본값)
            return nn.ReLU()

    def _build_default_model(self, hidden_dim):
        """기본 모델 구조 생성 (레이어별 분리 저장)"""
        self.num_layers = 2
        self.hidden_dim = hidden_dim

        # 레이어별로 분리하여 저장 (freeze 제어용)
        self.layer_list = nn.ModuleList([
            nn.Sequential(nn.Linear(self.input_dim, hidden_dim), self._get_activation(0, 2)),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self._get_activation(1, 2)),
        ]).to(self.device)

        # feature_net: layer_list를 순차적으로 적용하는 wrapper
        self.feature_net = nn.Sequential(*self.layer_list).to(self.device)

        # 출력층 (항상 학습)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False).to(self.device)

        # 전체 모델
        self.model = nn.Sequential(self.feature_net, self.out_layer)

        # float32로 설정
        self.layer_list = self.layer_list.float()
        self.feature_net = self.feature_net.float()
        self.out_layer = self.out_layer.float()
        self.model = self.model.float()
    
    def _build_dynamic_model(self, params: Dict):
        """동적 모델 구조 생성 (BO 결과 기반, 레이어별 분리 저장)"""
        self.num_layers = params['hidden_layers']
        self.hidden_dim = params['hidden_dim']
        total_layers = params['hidden_layers']

        # 레이어별로 분리하여 저장 (freeze 제어용)
        layer_modules = []

        # 첫 번째 레이어
        layer_modules.append(
            nn.Sequential(nn.Linear(self.input_dim, params['hidden_dim']),
                         self._get_activation(0, total_layers))
        )

        # 중간 레이어들
        for i in range(1, params['hidden_layers']):
            layer_modules.append(
                nn.Sequential(nn.Linear(params['hidden_dim'], params['hidden_dim']),
                             self._get_activation(i, total_layers))
            )

        self.layer_list = nn.ModuleList(layer_modules).to(self.device)
        self.feature_net = nn.Sequential(*self.layer_list).to(self.device)
        self.out_layer = nn.Linear(params['hidden_dim'], 1, bias=False).to(self.device)
        self.model = nn.Sequential(self.feature_net, self.out_layer)

        # float32로 설정
        self.layer_list = self.layer_list.float()
        self.feature_net = self.feature_net.float()
        self.out_layer = self.out_layer.float()
        self.model = self.model.float()
    
    def _split_validation_data(self, X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2) -> Tuple:
        """검증 데이터 분할"""
        if len(X) < 3:  # 데이터가 너무 적으면 분할하지 않음
            return X, y, X, y

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )
        return X_train, y_train, X_val, y_val

    def _apply_unfreeze_ratio(self, unfreeze_ratio: float):
        """
        unfreeze_ratio에 따라 레이어 동결/해동 적용

        Args:
            unfreeze_ratio: 0.0 ~ 1.0
                - 0.0: 모든 feature_net 레이어 동결 (out_layer만 학습)
                - 0.5: 뒤쪽 50% 레이어만 해동
                - 1.0: 전체 해동 (Full fine-tuning)

        동결 방식:
            - 앞쪽 레이어부터 동결 (입력에 가까운 범용 표현)
            - 뒤쪽 레이어는 해동 (task-specific 표현)
        """
        n_layers = len(self.layer_list)
        n_unfreeze = int(n_layers * unfreeze_ratio)

        # 앞쪽 레이어 동결, 뒤쪽 레이어 해동
        for i, layer in enumerate(self.layer_list):
            freeze = (i < n_layers - n_unfreeze)
            for param in layer.parameters():
                param.requires_grad = not freeze

        # out_layer는 항상 학습
        for param in self.out_layer.parameters():
            param.requires_grad = True

    def _get_trainable_parameters(self):
        """현재 requires_grad=True인 파라미터만 반환"""
        params = []
        for layer in self.layer_list:
            for param in layer.parameters():
                if param.requires_grad:
                    params.append(param)
        for param in self.out_layer.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def _progressive_unfreeze_step(self, current_phase: int, total_phases: int, verbose: bool = False):
        """
        Progressive Unfreezing: 현재 phase에 맞게 레이어 해동

        Args:
            current_phase: 현재 phase (0부터 시작)
            total_phases: 총 phase 수 (보통 레이어 수 + 1)
            verbose: 상세 출력

        Phase 진행:
            - Phase 0: out_layer만 학습
            - Phase 1: out_layer + 마지막 레이어
            - Phase 2: out_layer + 마지막 2개 레이어
            - ...
            - Phase n: 전체 레이어 학습
        """
        n_layers = len(self.layer_list)

        # Phase에 따라 해동할 레이어 수 계산
        # Phase 0: 0개 (out_layer만), Phase 1: 1개, ..., Phase n: n개
        n_unfreeze = min(current_phase, n_layers)

        # 모든 feature_net 레이어 동결
        for layer in self.layer_list:
            for param in layer.parameters():
                param.requires_grad = False

        # 뒤쪽부터 n_unfreeze개 레이어 해동
        for i in range(n_layers - n_unfreeze, n_layers):
            for param in self.layer_list[i].parameters():
                param.requires_grad = True

        # out_layer는 항상 학습
        for param in self.out_layer.parameters():
            param.requires_grad = True

        if verbose:
            unfrozen_layers = [i for i in range(n_layers) if any(p.requires_grad for p in self.layer_list[i].parameters())]
            print(f"      Phase {current_phase}: Unfrozen layers = {unfrozen_layers if unfrozen_layers else 'None (out_layer only)'}")

    def _finetune_progressive(self, X_tensor, y_tensor, epochs: int, lr: float,
                              verbose: bool = False,
                              lr_boost: float = 2.0, lr_decay: float = 0.7):
        """
        Progressive Unfreezing을 적용한 finetune

        전략:
            - 총 epochs를 (n_layers + 1) phase로 나눔
            - 각 phase마다 레이어를 하나씩 해동하며 학습

        Args:
            lr_boost: 초기 lr에 적용할 boost factor (default: 2.0)
            lr_decay: phase마다 적용할 lr decay factor (default: 0.7)
        """
        n_layers = len(self.layer_list)
        n_phases = n_layers + 1  # out_layer만 → 전체 레이어
        epochs_per_phase = max(epochs // n_phases, 1)

        # Progressive Unfreezing용 lr boost 적용
        base_lr = lr * lr_boost

        if verbose:
            print(f"    Progressive Unfreezing: {n_phases} phases, {epochs_per_phase} epochs/phase")
            print(f"    LR: {lr:.2e} -> {base_lr:.2e} (boost={lr_boost}x), decay={lr_decay}")

        loss_fn = nn.MSELoss()
        self.model.train()

        for phase in range(n_phases):
            # 현재 phase에 맞게 레이어 해동
            self._progressive_unfreeze_step(phase, n_phases, verbose=verbose)

            # 현재 학습 가능한 파라미터로 optimizer 생성
            trainable_params = self._get_trainable_parameters()

            # Phase가 진행될수록 학습률 감소 (더 완만하게: 0.7배)
            phase_lr = base_lr * (lr_decay ** phase)
            optimizer = optim.Adam(trainable_params, lr=phase_lr)

            # 해당 phase의 epochs만큼 학습
            for epoch in range(epochs_per_phase):
                optimizer.zero_grad()
                features = self.feature_net(X_tensor)
                pred = self.out_layer(features)
                loss = loss_fn(pred, y_tensor)
                loss.backward()
                optimizer.step()
                self.finetune_losses.append(loss.item())

        # 남은 epochs 처리 (전체 레이어 학습)
        remaining_epochs = epochs - (n_phases * epochs_per_phase)
        if remaining_epochs > 0:
            trainable_params = self._get_trainable_parameters()
            final_lr = base_lr * (lr_decay ** n_phases)
            optimizer = optim.Adam(trainable_params, lr=final_lr)

            for epoch in range(remaining_epochs):
                optimizer.zero_grad()
                features = self.feature_net(X_tensor)
                pred = self.out_layer(features)
                loss = loss_fn(pred, y_tensor)
                loss.backward()
                optimizer.step()
                self.finetune_losses.append(loss.item())

        # 학습 후 모든 파라미터 requires_grad 복원
        for layer in self.layer_list:
            for param in layer.parameters():
                param.requires_grad = True

    def pretrain(self, X_low, y_low, epochs=50, lr=1e-3, verbose=False,
                 bo_trials=None, data_size='small',
                 use_loocv=False, use_uncertainty_loss=False, uncertainty_weight=0.3):
        """
        Low-fidelity 데이터로 pretrain

        Args:
            X_low: low-fidelity 입력 데이터
            y_low: low-fidelity 출력 데이터
            epochs: 기본 epoch 수 (BO 사용 시 무시됨)
            lr: 기본 학습률 (BO 사용 시 무시됨)
            verbose: 상세 출력
            bo_trials: BO 시행 횟수 (None이면 BO 사용 안함)
            data_size: 데이터 크기 ('small', 'medium', 'large')
            use_loocv: LOOCV 사용 여부 (Pretrain에서는 무시됨, Finetune에서만 적용)
            use_uncertainty_loss: 불확실성 손실 사용 여부
            uncertainty_weight: 불확실성 손실 가중치
        """
        self.pretrain_losses = []
        X_low = np.asarray(X_low, dtype=np.float32)
        y_low = np.asarray(y_low, dtype=np.float32).flatten()

        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            # 베이지안 최적화로 하이퍼파라미터 찾기
            if verbose:
                print(f"    - Running Pretrain BO with {bo_trials} trials...")

            # 검증 데이터 분할
            X_train, y_train, X_val, y_val = self._split_validation_data(X_low, y_low)

            # BO 실행 (Optuna 사용) - Pretrain 단계: 모든 파라미터 탐색
            # NOTE: Pretrain에서는 LOOCV 사용 안함 (use_loocv=False 강제)
            from .hyperparameter_optimization_optuna import optimize_dnn_hyperparameters_optuna
            best_params, best_performance, history = optimize_dnn_hyperparameters_optuna(
                X_train, y_train, X_val, y_val,
                input_dim=self.input_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                stage='pretrain',
                verbose=verbose,
                use_loocv=False,  # Pretrain에서는 LOOCV 사용 안함
                use_uncertainty_loss=use_uncertainty_loss,
                uncertainty_weight=uncertainty_weight
            )
            
            self.pretrain_best_params = best_params
            self.pretrain_bo_history = history
            
            # 최적 하이퍼파라미터로 모델 구성
            self._build_dynamic_model(best_params)
            epochs = best_params['epochs']
            lr = best_params['learning_rate']
        else:
            # 기본 하이퍼파라미터 사용
            if not hasattr(self, 'model'):
                self._build_default_model(self.hidden_dim)
        
        # 전체 데이터로 최종 학습
        X_tensor = torch.tensor(X_low, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_low, dtype=torch.float32).view(-1, 1).to(self.device)
        optimizer = optim.Adam(list(self.feature_net.parameters()) + list(self.out_layer.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            features = self.feature_net(X_tensor)
            pred = self.out_layer(features)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            self.pretrain_losses.append(loss.item())
            

    def finetune(self, X_high, y_high, epochs=50, lr=1e-4, verbose=False,
                 bo_trials=None, data_size='small',
                 use_loocv=False, use_uncertainty_loss=False, uncertainty_weight=0.3,
                 use_freeze=False, unfreeze_ratio=1.0,
                 use_progressive_unfreezing=False):
        """
        High-fidelity 데이터로 finetune

        Args:
            X_high: high-fidelity 입력 데이터
            y_high: high-fidelity 출력 데이터
            epochs: 기본 epoch 수 (BO 사용 시 무시됨)
            lr: 기본 학습률 (BO 사용 시 무시됨)
            verbose: 상세 출력
            bo_trials: BO 시행 횟수 (None이면 BO 사용 안함)
            data_size: 데이터 크기 ('small', 'medium', 'large')
            use_loocv: LOOCV 사용 여부
            use_uncertainty_loss: 불확실성 손실 사용 여부
            uncertainty_weight: 불확실성 손실 가중치
            use_freeze: Freeze 기법 사용 여부 (False면 기존 방식: 전체 fine-tuning)
            unfreeze_ratio: 해동할 레이어 비율 (0.0~1.0, use_freeze=True일 때만 적용)
                - 0.0: 모든 feature_net 동결, out_layer만 학습
                - 0.5: 뒤쪽 50% 레이어 해동
                - 1.0: 전체 해동 (Full fine-tuning, 기존과 동일)
            use_progressive_unfreezing: Progressive Unfreezing 사용 여부
                - True: 점진적으로 레이어를 해동하며 학습 (ULMFiT 스타일)
                - use_freeze와 함께 사용 불가 (둘 중 하나만 선택)
        """
        self.finetune_losses = []
        X_high = np.asarray(X_high, dtype=np.float32)
        y_high = np.asarray(y_high, dtype=np.float32).flatten()

        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            # 베이지안 최적화로 하이퍼파라미터 찾기
            if verbose:
                print(f"    - Running Finetune BO with {bo_trials} trials...")

            # 검증 데이터 분할
            X_train, y_train, X_val, y_val = self._split_validation_data(X_high, y_high)

            # 현재 feature extractor의 출력 차원 확인
            with torch.no_grad():
                sample_input = torch.tensor(X_train[:1], dtype=torch.float32).to(self.device)
                feature_dim = self.feature_net(sample_input).shape[1]

            # Pretrain에서 결정된 구조를 고정
            fixed_structure = {
                'hidden_layers': self.pretrain_best_params.get('hidden_layers', 2) if self.pretrain_best_params else 2,
                'hidden_dim': self.pretrain_best_params.get('hidden_dim', self.hidden_dim) if self.pretrain_best_params else self.hidden_dim
            }

            from .hyperparameter_optimization_optuna import optimize_dnn_hyperparameters_optuna
            best_params, best_performance, history = optimize_dnn_hyperparameters_optuna(
                X_train, y_train, X_val, y_val,
                input_dim=self.input_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                stage='finetune',
                fixed_structure=fixed_structure,
                verbose=verbose,
                use_loocv=use_loocv,
                use_uncertainty_loss=use_uncertainty_loss,
                uncertainty_weight=uncertainty_weight,
                use_freeze=use_freeze,  # freeze 모드 전달
                model_for_freeze=self if use_freeze else None  # freeze 모드일 때 모델 전달
            )

            self.finetune_best_params = best_params
            self.finetune_bo_history = history

            epochs = best_params['epochs']
            lr = best_params['learning_rate']

            # use_freeze 모드에서 unfreeze_ratio가 BO로 결정됨
            if use_freeze and 'unfreeze_ratio' in best_params:
                unfreeze_ratio = best_params['unfreeze_ratio']
                if verbose:
                    print(f"    - Optimal unfreeze_ratio: {unfreeze_ratio:.2f}")

        # 전체 데이터로 최종 학습
        X_tensor = torch.tensor(X_high, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_high, dtype=torch.float32).view(-1, 1).to(self.device)

        # Progressive Unfreezing 모드
        if use_progressive_unfreezing:
            if use_freeze:
                raise ValueError("use_freeze와 use_progressive_unfreezing은 동시에 사용할 수 없습니다.")
            if verbose:
                print(f"    - Using Progressive Unfreezing")
            self._finetune_progressive(X_tensor, y_tensor, epochs, lr, verbose=verbose)
            return

        # Static Freeze 적용 (use_freeze=True일 때만)
        if use_freeze:
            self._apply_unfreeze_ratio(unfreeze_ratio)
            trainable_params = self._get_trainable_parameters()
            if verbose:
                n_frozen = sum(1 for layer in self.layer_list for p in layer.parameters() if not p.requires_grad)
                n_total = sum(1 for layer in self.layer_list for p in layer.parameters())
                print(f"    - Freeze applied: {n_frozen}/{n_total} layer params frozen (unfreeze_ratio={unfreeze_ratio:.2f})")
        else:
            # 기존 방식: 모든 파라미터 학습
            trainable_params = list(self.feature_net.parameters()) + list(self.out_layer.parameters())

        optimizer = optim.Adam(trainable_params, lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            features = self.feature_net(X_tensor)
            pred = self.out_layer(features)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            self.finetune_losses.append(loss.item())

        # 학습 후 모든 파라미터 requires_grad 복원 (다음 iteration을 위해)
        if use_freeze:
            for layer in self.layer_list:
                for param in layer.parameters():
                    param.requires_grad = True
            

    def predict(self, X):
        """예측"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(X_tensor)
            return pred.cpu().numpy().flatten()

    def extract_features(self, X):
        """Feature 추출"""
        self.feature_net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            features = self.feature_net(X_tensor)
            return features.cpu().numpy()
    
    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str = 'high'):
        """
        점진적 업데이트 (BO로 최적화된 파라미터 사용)
        
        Args:
            X_new: 새로운 입력 데이터
            y_new: 새로운 출력 데이터  
            fidelity: 'high' 또는 'low'
        """
        self.update_counter += 1
        
        # BO로 최적화된 파라미터 또는 기본값 사용
        if self.incremental_params:
            mode = self.incremental_params.get('mode', 'incremental')
            lr_boost = self.incremental_params.get('lr_boost_factor', 2.0)
            inc_epochs = self.incremental_params.get('incremental_epochs', 10)
            retrain_interval = self.incremental_params.get('full_retrain_interval', 5)
            replay_ratio = self.incremental_params.get('replay_ratio', 0.3)
            weight_decay = self.incremental_params.get('weight_decay_factor', 0.95)
        else:
            # 기본값
            mode = 'incremental'
            lr_boost = 2.0
            inc_epochs = 10
            retrain_interval = 5
            replay_ratio = 0.3
            weight_decay = 0.95
        
        # 모드에 따른 업데이트 전략
        if mode == 'full':
            # 항상 전체 재학습
            self._full_retrain(X_new, y_new, fidelity)
        elif mode == 'hybrid' and self.update_counter % retrain_interval == 0:
            # 주기적 전체 재학습
            self._full_retrain(X_new, y_new, fidelity)
        else:
            # 점진적 업데이트
            self._incremental_train(X_new, y_new, fidelity, lr_boost, inc_epochs, 
                                  replay_ratio, weight_decay)
        
        # 데이터 버퍼 업데이트
        self._update_buffer(X_new, y_new, fidelity)
    
    def _full_retrain(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str):
        """전체 재학습"""
        if fidelity == 'high':
            self.finetune(X_new, y_new, 
                         epochs=self.finetune_best_params.get('epochs', 50) if self.finetune_best_params else 50,
                         lr=self.finetune_best_params.get('learning_rate', 1e-4) if self.finetune_best_params else 1e-4)
        else:
            self.pretrain(X_new, y_new,
                         epochs=self.pretrain_best_params.get('epochs', 50) if self.pretrain_best_params else 50, 
                         lr=self.pretrain_best_params.get('learning_rate', 1e-3) if self.pretrain_best_params else 1e-3)
    
    def _incremental_train(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str,
                          lr_boost: float, inc_epochs: int, replay_ratio: float, weight_decay: float):
        """점진적 학습"""
        if not hasattr(self, 'model') or self.model is None:
            # 모델이 없으면 전체 학습
            self._full_retrain(X_new, y_new, fidelity)
            return
        
        # 기본 학습률에 부스트 적용
        base_lr = self.last_learning_rates.get('finetune' if fidelity == 'high' else 'pretrain', 1e-4)
        boosted_lr = base_lr * lr_boost
        
        # Experience Replay: 이전 데이터 일부 재사용
        X_combined, y_combined, sample_weights = self._prepare_replay_data(
            X_new, y_new, fidelity, replay_ratio, weight_decay
        )
        
        # 점진적 학습
        optimizer = optim.Adam(self.model.parameters(), lr=boosted_lr)
        loss_fn = nn.MSELoss(reduction='none')  # 가중치 적용을 위해
        
        X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_combined, dtype=torch.float32).view(-1, 1).to(self.device)
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(self.device)
        
        self.model.train()
        for epoch in range(inc_epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            losses = loss_fn(pred, y_tensor).squeeze()
            weighted_loss = (losses * weights_tensor).mean()
            weighted_loss.backward()
            
            # Gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        
        # 학습률 기록
        self.last_learning_rates[fidelity] = boosted_lr
    
    def _prepare_replay_data(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str,
                            replay_ratio: float, weight_decay: float):
        """Experience Replay를 위한 데이터 준비"""
        buffer_key_x = f'X_{fidelity}'
        buffer_key_y = f'y_{fidelity}'
        
        # 새 데이터
        X_combined = X_new.copy()
        y_combined = y_new.copy()
        sample_weights = np.ones(len(X_new))  # 새 데이터는 가중치 1.0
        
        # 이전 데이터 재사용
        if len(self.data_buffer[buffer_key_x]) > 0 and replay_ratio > 0:
            n_replay = int(len(X_new) * replay_ratio)
            buffer_size = len(self.data_buffer[buffer_key_x])
            n_replay = min(n_replay, buffer_size)
            
            if n_replay > 0:
                # 랜덤 샘플링
                indices = np.random.choice(buffer_size, n_replay, replace=False)
                X_replay = np.array([self.data_buffer[buffer_key_x][i] for i in indices])
                y_replay = np.array([self.data_buffer[buffer_key_y][i] for i in indices])
                
                # 이전 데이터는 감소된 가중치
                replay_weights = np.full(n_replay, weight_decay)
                
                # 결합
                X_combined = np.vstack([X_combined, X_replay])
                y_combined = np.concatenate([y_combined, y_replay])
                sample_weights = np.concatenate([sample_weights, replay_weights])
        
        return X_combined, y_combined, sample_weights
    
    def _update_buffer(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str, max_buffer_size: int = 100):
        """데이터 버퍼 업데이트"""
        buffer_key_x = f'X_{fidelity}'
        buffer_key_y = f'y_{fidelity}'
        
        # 새 데이터 추가
        for x, y in zip(X_new, y_new):
            self.data_buffer[buffer_key_x].append(x)
            self.data_buffer[buffer_key_y].append(y)
        
        # 버퍼 크기 제한 (FIFO)
        if len(self.data_buffer[buffer_key_x]) > max_buffer_size:
            excess = len(self.data_buffer[buffer_key_x]) - max_buffer_size
            self.data_buffer[buffer_key_x] = self.data_buffer[buffer_key_x][excess:]
            self.data_buffer[buffer_key_y] = self.data_buffer[buffer_key_y][excess:]
    
    def get_hyperparameter_summary(self) -> Dict:
        """하이퍼파라미터 최적화 결과 요약"""
        summary = {
            'use_hyperparameter_bo': self.use_hyperparameter_bo,
            'pretrain_best_params': self.pretrain_best_params,
            'finetune_best_params': self.finetune_best_params,
            'pretrain_trials': len(self.pretrain_bo_history),
            'finetune_trials': len(self.finetune_bo_history),
            'pretrain_bo_history': self.pretrain_bo_history,  # 모든 시행 기록
            'finetune_bo_history': self.finetune_bo_history   # 모든 시행 기록
        }
        return summary


class BayesianLinearRegression:
    """
    베이지안 선형 회귀 모델
    불확실성을 포함한 예측을 제공합니다.
    """
    
    def __init__(self, alpha=1.0, beta=25.0):
        """
        Args:
            alpha: 가중치의 정밀도 (precision) 파라미터
            beta: 노이즈의 정밀도 파라미터
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.fitted = False
    
    def fit(self, X, y):
        """
        베이지안 선형 회귀 학습
        
        Args:
            X: 입력 특성 (N x D)
            y: 타겟 값 (N,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()
        
        # 편향 항 추가
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # 사전 분포: w ~ N(0, α^(-1)I)
        S0_inv = self.alpha * np.eye(X_with_bias.shape[1])
        
        # 사후 분포 계산
        S_N_inv = S0_inv + self.beta * X_with_bias.T @ X_with_bias
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X_with_bias.T @ y
        
        self.fitted = True
    
    def predict(self, x):
        """
        단일 점에 대한 예측
        
        Args:
            x: 입력 특성 벡터 (D,)
            
        Returns:
            평균, 분산
        """
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        x = np.asarray(x, dtype=np.float32).flatten()
        x_with_bias = np.concatenate([[1], x])
        
        # 예측 평균
        mu = x_with_bias @ self.mean
        
        # 예측 분산
        var = (1/self.beta) + x_with_bias @ self.cov @ x_with_bias
        
        return mu, var
    
    def predict_batch(self, X):
        """
        배치 예측
        
        Args:
            X: 입력 특성 행렬 (N x D)
            
        Returns:
            평균들, 분산들
        """
        X = np.asarray(X, dtype=np.float32)
        means = []
        variances = []
        
        for x in X:
            mu, var = self.predict(x)
            means.append(mu)
            variances.append(var)
        
        return np.array(means), np.array(variances) 
    
    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray, weight: float = 1.0):
        """
        Sherman-Morrison-Woodbury formula를 사용한 효율적 점진적 업데이트
        
        Args:
            X_new: 새로운 입력 특성 행렬 (N x D)
            y_new: 새로운 타겟 값 (N,)
            weight: 새 데이터의 가중치 (기본값 1.0)
        """
        X_new = np.asarray(X_new, dtype=np.float32)
        y_new = np.asarray(y_new, dtype=np.float32).flatten()
        
        if not self.fitted:
            # 첫 업데이트면 일반 fit 사용
            self.fit(X_new, y_new)
            return
        
        # 편향 항 추가
        X_new_bias = np.column_stack([np.ones(len(X_new)), X_new])
        
        # 가중치 적용
        if weight != 1.0:
            beta_weighted = self.beta * weight
        else:
            beta_weighted = self.beta
        
        # Sherman-Morrison-Woodbury 공식으로 효율적 업데이트
        # 여러 점을 한번에 업데이트
        for i, (x_new, y_val) in enumerate(zip(X_new_bias, y_new)):
            x_new = x_new.reshape(-1, 1)
            
            # 공분산 행렬 업데이트: S_new = S_old - (S_old * x * x^T * S_old) / (1/beta + x^T * S_old * x)
            Sx = self.cov @ x_new
            denominator = 1.0 / beta_weighted + x_new.T @ Sx
            self.cov = self.cov - (Sx @ Sx.T) / denominator
            
            # 평균 업데이트: m_new = m_old + beta * S_new * x * (y - x^T * m_old)
            prediction_error = y_val - x_new.T @ self.mean
            self.mean = self.mean + beta_weighted * self.cov @ x_new.flatten() * prediction_error
    
    def reset_to_prior(self):
        """사전분포로 리셋"""
        self.mean = None
        self.cov = None
        self.fitted = False