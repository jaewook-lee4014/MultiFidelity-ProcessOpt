import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import train_test_split


class VariationalLinear(nn.Module):
    """
    Variational Linear Layer for Bayesian Neural Network
    Uses local reparameterization trick for efficient computation
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and rho for std)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weight means with Xavier normal
        nn.init.xavier_normal_(self.weight_mu)
        # Initialize rho parameters (will be converted to std via softplus)
        nn.init.constant_(self.weight_rho, -3.0)  # softplus(-3) ≈ 0.05
        
        # Initialize bias
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using local reparameterization trick"""
        # Convert rho to std using softplus
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)
        
        # Local reparameterization trick
        # For y = Wx + b where W ~ N(μ_w, σ_w²), x is deterministic
        # y ~ N(μ_w * x + μ_b, x^T * σ_w² * x + σ_b²)
        
        # Mean of output
        mu_out = F.linear(x, self.weight_mu, self.bias_mu)
        
        # Variance of output
        var_out = F.linear(x.pow(2), weight_std.pow(2), bias_std.pow(2))
        
        # Sample from the output distribution
        eps = torch.randn_like(mu_out, device=mu_out.device)
        return mu_out + torch.sqrt(torch.clamp(var_out, min=1e-8)) * eps
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)
        
        # KL for weights: KL(N(μ, σ²) || N(0, prior_std²))
        weight_kl = self._kl_gaussian(self.weight_mu, weight_std, 0.0, self.prior_std)
        bias_kl = self._kl_gaussian(self.bias_mu, bias_std, 0.0, self.prior_std)
        
        return weight_kl + bias_kl
    
    def _kl_gaussian(self, mu_q, std_q, mu_p, std_p):
        """KL divergence between two Gaussians"""
        kl = torch.log(std_p / std_q) + (std_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * std_p**2) - 0.5
        return kl.sum()


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network with Variational Inference
    Supports both homoscedastic and heteroscedastic noise modeling
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64], 
                 prior_std: float = 1.0, noise_type: str = 'homoscedastic',
                 pretrained_weights: Optional[Dict] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.prior_std = prior_std
        self.noise_type = noise_type
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(VariationalLinear(prev_dim, hidden_dim, prior_std))
            prev_dim = hidden_dim
        
        # Output layer
        if noise_type == 'homoscedastic':
            # Single output for mean
            self.output_layer = VariationalLinear(prev_dim, 1, prior_std)
            # Learnable global noise parameter
            self.log_noise_var = nn.Parameter(torch.log(torch.tensor(0.1)))
        else:
            # Two outputs: mean and log variance
            self.output_layer = VariationalLinear(prev_dim, 2, prior_std)
        
        # Initialize with pretrained weights if provided
        if pretrained_weights is not None:
            self._load_pretrained_weights(pretrained_weights)
            
        # MPS 최적화 설정
        self._setup_device_optimizations()
    
    def _load_pretrained_weights(self, pretrained_weights: Dict):
        """Load pretrained weights as prior means"""
        # This would be implemented based on the structure of pretrained_weights
        # For now, we'll use the default initialization
        pass
    
    def _setup_device_optimizations(self):
        """MPS 및 기타 디바이스 최적화 설정"""
        # MPS 사용 시 특별한 설정이 필요하면 여기에 추가
        pass
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and noise variance"""
        # Hidden layers with ReLU activation
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Output layer
        output = self.output_layer(x)
        
        if self.noise_type == 'homoscedastic':
            # Single global noise variance
            mean = output
            noise_var = torch.exp(self.log_noise_var).expand_as(mean)
        else:
            # Heteroscedastic: network predicts both mean and log variance
            mean = output[:, :1]
            log_noise_var = output[:, 1:]
            noise_var = torch.exp(log_noise_var)
        
        return mean, noise_var
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence for all layers"""
        kl_total = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for layer in self.layers:
            kl_total += layer.kl_divergence()
        
        kl_total += self.output_layer.kl_divergence()
        
        return kl_total
    
    def sample_predict(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty quantification using multiple samples"""
        self.train()  # Enable sampling
        
        predictions = []
        noise_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, noise_var = self.forward(x)
                predictions.append(mean)
                noise_vars.append(noise_var)
        
        predictions = torch.stack(predictions, dim=0)  # [n_samples, batch_size, 1]
        noise_vars = torch.stack(noise_vars, dim=0)    # [n_samples, batch_size, 1]
        
        # Compute epistemic uncertainty (uncertainty about the function)
        pred_mean = predictions.mean(dim=0)
        epistemic_var = predictions.var(dim=0)
        
        # Compute aleatoric uncertainty (noise in observations)
        aleatoric_var = noise_vars.mean(dim=0)
        
        # Total uncertainty
        total_var = epistemic_var + aleatoric_var
        
        return pred_mean, total_var


class TransferLearningBNN:
    """
    Transfer Learning with Bayesian Neural Networks
    Replaces DNGO in the optimization pipeline
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64], 
                 device: str = 'cpu', prior_std: float = 1.0, 
                 noise_type: str = 'homoscedastic', use_hyperparameter_bo: bool = False):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.prior_std = prior_std
        self.noise_type = noise_type
        
        # Training history
        self.pretrain_losses = []
        self.finetune_losses = []
        
        # Models
        self.feature_extractor = None
        self.bnn = None
        self.fitted = False
        self.use_direct_bnn = False  # Flag for direct BNN mode
        
        # 하이퍼파라미터 최적화 관련
        self.use_hyperparameter_bo = use_hyperparameter_bo
        self.pretrain_best_params = None
        self.finetune_best_params = None
        self.pretrain_bo_history = []
        self.finetune_bo_history = []
        
        # MPS 최적화
        self._setup_mps_optimizations()
    
    def _setup_mps_optimizations(self):
        """MPS 사용 시 메모리 및 성능 최적화"""
        if self.device == 'mps':
            # MPS 캐시 정리 함수 저장 (나중에 사용)
            self._clear_cache = self._clear_mps_cache
        else:
            self._clear_cache = lambda: None
    
    def _clear_mps_cache(self):
        """MPS 캐시 정리"""
        if hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()
        
    def _build_feature_extractor(self, hidden_dim: int = 64):
        """Build deterministic feature extractor (same as original)"""
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device).float()
    
    def pretrain(self, X_low: np.ndarray, y_low: np.ndarray, 
                 epochs: int = 200, lr: float = 1e-3, verbose: bool = False,
                 bo_trials: Optional[int] = None, data_size: str = 'small'):
        """Pretrain feature extractor on low-fidelity data with optional BO"""
        self.pretrain_losses = []
        X_low = np.asarray(X_low, dtype=np.float32)
        y_low = np.asarray(y_low, dtype=np.float32).flatten()
        
        if verbose:
            print(f"Pretrain BNN feature extractor with {len(X_low)} low-fidelity samples")
        
        # Hyperparameter BO for pretrain
        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            if verbose:
                print(f"🔍 Pretrain 하이퍼파라미터 최적화 시작 ({bo_trials} trials)...")
            
            # 검증 데이터 분할
            from sklearn.model_selection import train_test_split
            if len(X_low) >= 5:  # 최소 5개 샘플 필요
                X_train, X_val, y_train, y_val = train_test_split(
                    X_low, y_low, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train, X_val, y_val = X_low, y_low, X_low, y_low
            
            # Pretrain 전용 BO 실행
            from hyperparameter_optimization_bnn import optimize_bnn_hyperparameters
            best_params, best_performance, history = optimize_bnn_hyperparameters(
                X_train, y_train, X_val, y_val,
                input_dim=self.input_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                verbose=verbose,
                stage='pretrain'  # pretrain 전용 탐색 공간 사용
            )
            
            self.pretrain_best_params = best_params
            self.pretrain_bo_history = history
            
            if verbose:
                print(f"✅ 최적 Pretrain 파라미터: {best_params}")
                print(f"✅ 최적 성능: {best_performance:.4f}")
            
            # 최적 하이퍼파라미터 사용
            epochs = best_params.get('pretrain_epochs', epochs)
            lr = best_params.get('pretrain_lr', lr)
            # hidden_dims는 모델 전체에 영향을 주므로 여기서는 변경하지 않음
        
        # Build feature extractor
        self._build_feature_extractor()
        
        # Add output layer for pretraining
        feature_dim = self.hidden_dims[0] if self.hidden_dims else 64
        output_layer = nn.Linear(feature_dim, 1).to(self.device).float()
        pretrain_model = nn.Sequential(self.feature_extractor, output_layer)
        
        # Training
        X_tensor = torch.tensor(X_low, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_low, dtype=torch.float32).view(-1, 1).to(self.device)
        
        optimizer = optim.Adam(pretrain_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        pretrain_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = pretrain_model(X_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            self.pretrain_losses.append(loss.item())
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'[Pretrain] Epoch {epoch+1}/{epochs}: Loss {loss.item():.4f}')
    
    def train_bnn(self, X: np.ndarray, y: np.ndarray,
                  epochs: int = 100, lr: float = 1e-4, kl_weight: float = 1.0,
                  kl_warmup_epochs: int = 10, verbose: bool = False,
                  pretrained_model: Optional['TransferLearningBNN'] = None):
        """
        Train BNN directly (with optional transfer from pretrained model)
        
        Args:
            X, y: Training data
            epochs, lr, kl_weight, kl_warmup_epochs: Training parameters
            pretrained_model: Optional pretrained model to copy weights from
        """
        self.finetune_losses = []
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()
        
        if verbose:
            transfer_msg = " (with transfer learning)" if pretrained_model else ""
            print(f"Train BNN with {len(X)} samples{transfer_msg}")
        
        # Build BNN
        self.bnn = BayesianNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            prior_std=self.prior_std,
            noise_type=self.noise_type
        ).to(self.device)
        
        # Transfer weights from pretrained model if provided
        if pretrained_model is not None and pretrained_model.bnn is not None:
            self.bnn.load_state_dict(pretrained_model.bnn.state_dict())
            if verbose:
                print("✅ Loaded weights from pretrained model")
        
        # Training
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        optimizer = optim.Adam(self.bnn.parameters(), lr=lr)
        
        self.bnn.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            pred_mean, pred_var = self.bnn(X_tensor)
            
            # Negative log likelihood (NLL)
            nll = 0.5 * torch.log(2 * math.pi * pred_var) + 0.5 * (y_tensor - pred_mean).pow(2) / pred_var
            nll = nll.mean()
            
            # KL divergence with warm-up
            kl_div = self.bnn.kl_divergence()
            kl_weight_current = min(1.0, (epoch + 1) / kl_warmup_epochs) * kl_weight / len(X)
            
            # ELBO loss
            loss = nll + kl_weight_current * kl_div
            
            loss.backward()
            optimizer.step()
            self.finetune_losses.append(loss.item())
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'[BNN] Epoch {epoch+1}/{epochs}: Loss {loss.item():.4f}, NLL {nll.item():.4f}, KL {kl_div.item():.4f}')
        
        # 학습 완료 후 캐시 정리
        self._clear_cache()
        self.fitted = True
    
    def finetune(self, X_high: np.ndarray, y_high: np.ndarray, 
                 epochs: int = 100, lr: float = 1e-4, kl_weight: float = 1.0,
                 kl_warmup_epochs: int = 10, verbose: bool = False,
                 bo_trials: Optional[int] = None, data_size: str = 'small'):
        """Finetune with BNN on high-fidelity data"""
        self.finetune_losses = []
        X_high = np.asarray(X_high, dtype=np.float32)
        y_high = np.asarray(y_high, dtype=np.float32).flatten()
        
        if verbose:
            print(f"Finetune BNN with {len(X_high)} high-fidelity samples")
        
        # Extract features using pretrained feature extractor
        if self.feature_extractor is None:
            self._build_feature_extractor()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_high, dtype=torch.float32).to(self.device)
            features = self.feature_extractor(X_tensor)
            feature_dim = features.shape[1]
        
        # 하이퍼파라미터 베이지안 최적화
        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            if verbose:
                print(f"🔍 BNN 하이퍼파라미터 최적화 시작 ({bo_trials} trials)...")
            
            # 검증 데이터 분할
            from sklearn.model_selection import train_test_split
            if len(X_high) >= 3:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_high, y_high, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train, X_val, y_val = X_high, y_high, X_high, y_high
            
            # BNN 하이퍼파라미터 최적화 실행
            from hyperparameter_optimization_bnn import optimize_bnn_hyperparameters
            best_params, best_performance, history = optimize_bnn_hyperparameters(
                X_train, y_train, X_val, y_val,
                input_dim=feature_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                verbose=verbose
            )
            
            self.finetune_best_params = best_params
            self.finetune_bo_history = history
            
            if verbose:
                print(f"✅ 최적 BNN 파라미터: {best_params}")
                print(f"✅ 최적 성능: {best_performance:.4f}")
            
            # 최적 하이퍼파라미터로 모델 구성
            self.hidden_dims = best_params['hidden_dims']
            self.prior_std = best_params['prior_std']
            self.noise_type = best_params['noise_type']
            epochs = best_params['finetune_epochs']
            lr = best_params['finetune_lr']
            kl_weight = best_params['kl_weight']
            kl_warmup_epochs = best_params['kl_warmup_epochs']
        
        # Build BNN
        self.bnn = BayesianNeuralNetwork(
            input_dim=feature_dim,
            hidden_dims=self.hidden_dims,
            prior_std=self.prior_std,
            noise_type=self.noise_type
        ).to(self.device)
        
        # Training
        features_np = features.cpu().numpy()
        X_tensor = torch.tensor(features_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_high, dtype=torch.float32).view(-1, 1).to(self.device)
        
        optimizer = optim.Adam(self.bnn.parameters(), lr=lr)
        
        self.bnn.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            pred_mean, pred_var = self.bnn(X_tensor)
            
            # Negative log likelihood (NLL)
            nll = 0.5 * torch.log(2 * math.pi * pred_var) + 0.5 * (y_tensor - pred_mean).pow(2) / pred_var
            nll = nll.mean()
            
            # KL divergence with warm-up
            kl_div = self.bnn.kl_divergence()
            kl_weight_current = min(1.0, (epoch + 1) / kl_warmup_epochs) * kl_weight / len(X_high)
            
            # ELBO loss
            loss = nll + kl_weight_current * kl_div
            
            loss.backward()
            optimizer.step()
            self.finetune_losses.append(loss.item())
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'[Finetune] Epoch {epoch+1}/{epochs}: Loss {loss.item():.4f}, NLL {nll.item():.4f}, KL {kl_div.item():.4f}')
        
        # 학습 완료 후 캐시 정리
        self._clear_cache()
        self.fitted = True
    
    def predict(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification
        
        Returns:
            mean: Predicted mean values
            std: Predicted standard deviation (total uncertainty)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call train_bnn() first.")
        
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Direct BNN prediction (no feature extraction needed)
        pred_mean, pred_var = self.bnn.sample_predict(X_tensor, n_samples=n_samples)
        
        mean = pred_mean.cpu().numpy().flatten()
        std = torch.sqrt(pred_var).cpu().numpy().flatten()
        
        return mean, std
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features using the pretrained feature extractor"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not built. Call pretrain() first.")
        
        self.feature_extractor.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            features = self.feature_extractor(X_tensor)
            return features.cpu().numpy()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit method for compatibility with BLR interface"""
        # This method is for compatibility - the actual training happens in pretrain/finetune
        pass
    
    def predict_single(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict for a single point (compatibility with BLR interface)
        
        Returns:
            mu: Predicted mean
            var: Predicted variance
        """
        x = x.reshape(1, -1)
        mean, std = self.predict(x, n_samples=100)
        return float(mean[0]), float(std[0]**2)
    
    def get_hyperparameter_summary(self) -> Dict:
        """하이퍼파라미터 최적화 결과 요약"""
        summary = {
            'use_hyperparameter_bo': self.use_hyperparameter_bo,
            'pretrain_best_params': self.pretrain_best_params,
            'finetune_best_params': self.finetune_best_params,
            'pretrain_trials': len(self.pretrain_bo_history),
            'finetune_trials': len(self.finetune_bo_history)
        }
        return summary