import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import train_test_split


class ScaleMixturePrior:
    """
    Scale Mixture Prior from Blundell et al. 2015
    P(w) = π·N(0, σ₁²) + (1-π)·N(0, σ₂²)

    This creates a "spike-and-slab" like prior that encourages sparsity
    while allowing some weights to be large.
    """

    def __init__(self, pi: float = 0.5, sigma1: float = 1.0, sigma2: float = 0.002):
        """
        Args:
            pi: Mixture weight (default 0.5)
            sigma1: Std of the wider Gaussian (default 1.0, allows large weights)
            sigma2: Std of the narrow Gaussian (default 0.002, encourages sparsity)
        """
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """Compute log probability of weights under the mixture prior"""
        # Gaussian 1: wider distribution
        log_prob1 = -0.5 * math.log(2 * math.pi) - math.log(self.sigma1) - \
                    0.5 * (w / self.sigma1).pow(2)
        prob1 = self.pi * torch.exp(log_prob1)

        # Gaussian 2: narrow distribution (spike)
        log_prob2 = -0.5 * math.log(2 * math.pi) - math.log(self.sigma2) - \
                    0.5 * (w / self.sigma2).pow(2)
        prob2 = (1 - self.pi) * torch.exp(log_prob2)

        # Mixture: log(π·p1 + (1-π)·p2)
        return torch.log(prob1 + prob2 + 1e-10)


class VariationalLinear(nn.Module):
    """
    Variational Linear Layer for Bayesian Neural Network

    Implements Bayes by Backprop (Blundell et al. 2015) with:
    - Scale Mixture Prior for better sparsity
    - Local reparameterization trick for efficient computation
    - Proper initialization as per the paper
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_pi: float = 0.5, prior_sigma1: float = 1.0, prior_sigma2: float = 0.002):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale Mixture Prior (Blundell et al. 2015)
        self.prior = ScaleMixturePrior(pi=prior_pi, sigma1=prior_sigma1, sigma2=prior_sigma2)

        # Weight parameters (mean and rho for std)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # Initialize parameters following Blundell et al. 2015
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters as per Blundell et al. 2015:
        - mu: uniform(-0.2, 0.2)
        - rho: uniform(-5, -4) → softplus gives σ ≈ 0.007~0.018
        """
        # Weight initialization (Blundell et al. 2015 recommendations)
        nn.init.uniform_(self.weight_mu, -0.2, 0.2)
        nn.init.uniform_(self.weight_rho, -5.0, -4.0)  # softplus → small initial uncertainty

        # Bias initialization
        nn.init.uniform_(self.bias_mu, -0.2, 0.2)
        nn.init.uniform_(self.bias_rho, -5.0, -4.0)

    def _rho_to_std(self, rho: torch.Tensor) -> torch.Tensor:
        """Convert rho to std using softplus: σ = log(1 + exp(ρ))"""
        return F.softplus(rho)

    def _sample_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample weights using reparameterization trick: w = μ + σ·ε"""
        weight_std = self._rho_to_std(self.weight_rho)
        bias_std = self._rho_to_std(self.bias_rho)

        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps

        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using local reparameterization trick"""
        # Convert rho to std using softplus
        weight_std = self._rho_to_std(self.weight_rho)
        bias_std = self._rho_to_std(self.bias_rho)

        # Local reparameterization trick (more efficient than weight sampling)
        # For y = Wx + b where W ~ N(μ_w, σ_w²), x is deterministic
        # y ~ N(μ_w·x + μ_b, x²·σ_w² + σ_b²)

        # Mean of output
        mu_out = F.linear(x, self.weight_mu, self.bias_mu)

        # Variance of output
        var_out = F.linear(x.pow(2), weight_std.pow(2), bias_std.pow(2))

        # Sample from the output distribution
        eps = torch.randn_like(mu_out, device=mu_out.device)
        return mu_out + torch.sqrt(torch.clamp(var_out, min=1e-8)) * eps

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence using Monte Carlo estimation.

        KL(q(w|θ) || P(w)) = E_q[log q(w|θ) - log P(w)]

        For Scale Mixture Prior, we need MC estimation since no closed form exists.
        """
        # Sample weights
        weight, bias = self._sample_weights()

        # Log probability under variational posterior q(w|θ)
        weight_std = self._rho_to_std(self.weight_rho)
        bias_std = self._rho_to_std(self.bias_rho)

        log_q_weight = self._log_gaussian(weight, self.weight_mu, weight_std)
        log_q_bias = self._log_gaussian(bias, self.bias_mu, bias_std)
        log_q = log_q_weight + log_q_bias

        # Log probability under prior P(w) - Scale Mixture
        log_p_weight = self.prior.log_prob(weight).sum()
        log_p_bias = self.prior.log_prob(bias).sum()
        log_p = log_p_weight + log_p_bias

        # KL = E_q[log q - log p]
        return log_q - log_p

    def _log_gaussian(self, x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Log probability of x under Gaussian(mu, std²)"""
        return (-0.5 * math.log(2 * math.pi) - torch.log(std) - \
                0.5 * ((x - mu) / std).pow(2)).sum()


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network with Variational Inference

    Implements Bayes by Backprop (Blundell et al. 2015) with:
    - Scale Mixture Prior for weight regularization
    - Supports both homoscedastic and heteroscedastic noise modeling
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64],
                 prior_pi: float = 0.5, prior_sigma1: float = 1.0, prior_sigma2: float = 0.002,
                 noise_type: str = 'homoscedastic',
                 pretrained_weights: Optional[Dict] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.noise_type = noise_type

        # Build network layers with Scale Mixture Prior
        self.layers = nn.ModuleList()

        # Hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(VariationalLinear(
                prev_dim, hidden_dim,
                prior_pi=prior_pi, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2
            ))
            prev_dim = hidden_dim

        # Output layer
        if noise_type == 'homoscedastic':
            # Single output for mean
            self.output_layer = VariationalLinear(
                prev_dim, 1,
                prior_pi=prior_pi, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2
            )
            # Learnable global noise parameter (initialized to reasonable value)
            self.log_noise_var = nn.Parameter(torch.log(torch.tensor(0.01)))
        else:
            # Two outputs: mean and log variance
            self.output_layer = VariationalLinear(
                prev_dim, 2,
                prior_pi=prior_pi, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2
            )

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

    Supports two training modes to address pretrain-finetune consistency:

    1. 'consistent_bnn' (recommended): Full BNN for both pretrain and finetune
       - Pretrain: Train full BNN on LOFI data
       - Finetune: Continue training BNN on HIFI data (warm start)
       - Provides consistent Bayesian treatment throughout

    2. 'dngo_style': Deterministic feature extractor + BNN head
       - Pretrain: Train deterministic DNN on LOFI data
       - Finetune: Freeze features, train BNN head on HIFI data
       - Similar to DNGO but with BNN instead of BLR

    The 'consistent_bnn' mode is recommended as it avoids the
    deterministic-to-Bayesian transition problem.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64],
                 device: str = 'cpu',
                 prior_pi: float = 0.5, prior_sigma1: float = 1.0, prior_sigma2: float = 0.002,
                 noise_type: str = 'homoscedastic', use_hyperparameter_bo: bool = False,
                 kl_weight: float = 1.0, kl_warmup_epochs: int = 10,
                 transfer_mode: str = 'consistent_bnn'):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            device: Device to use ('cpu', 'cuda', 'mps')
            prior_pi: Scale mixture prior mixing weight (default 0.5)
            prior_sigma1: Scale mixture prior sigma1 (default 1.0)
            prior_sigma2: Scale mixture prior sigma2 (default 0.002)
            noise_type: 'homoscedastic' or 'heteroscedastic'
            use_hyperparameter_bo: Whether to use Bayesian optimization for hyperparameters
            kl_weight: Weight for KL divergence term
            kl_warmup_epochs: Number of epochs for KL warmup
            transfer_mode: 'consistent_bnn' or 'dngo_style'
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device

        # Scale Mixture Prior parameters (Blundell et al. 2015)
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

        self.noise_type = noise_type
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.transfer_mode = transfer_mode

        # Training history
        self.pretrain_losses = []
        self.finetune_losses = []

        # Models
        self.feature_extractor = None  # For dngo_style mode
        self.bnn = None  # Main BNN model
        self.fitted = False

        # 하이퍼파라미터 최적화 관련
        self.use_hyperparameter_bo = use_hyperparameter_bo
        self.pretrain_best_params = None
        self.finetune_best_params = None
        self.pretrain_bo_history = []
        self.finetune_bo_history = []

        # Incremental learning 관련 변수
        self.incremental_params = None
        self.data_buffer = {'X_low': [], 'y_low': [], 'X_high': [], 'y_high': []}
        self.update_counter = 0
        self.last_learning_rates = {'pretrain': 1e-3, 'finetune': 1e-4}
        self.previous_bnn_params = None

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
        
    def _build_feature_extractor(self, hidden_dim: int = 64, num_layers: int = 2):
        """Build deterministic feature extractor with ModuleList for layer-wise freeze control"""
        # Use ModuleList for layer-wise control (similar to DNGO)
        self.feature_layer_list = nn.ModuleList()

        prev_dim = self.input_dim
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            )
            self.feature_layer_list.append(layer)
            prev_dim = hidden_dim

        self.feature_layer_list = self.feature_layer_list.to(self.device).float()

        # Create sequential wrapper for compatibility
        self.feature_extractor = nn.Sequential(*self.feature_layer_list).to(self.device).float()

    def _apply_unfreeze_ratio_to_feature_extractor(self, unfreeze_ratio: float):
        """
        Apply unfreeze ratio to feature extractor layers.

        unfreeze_ratio = 0.0: Freeze all feature_extractor layers (only BNN learns)
        unfreeze_ratio = 1.0: Full fine-tuning (all layers learn)

        Freezes layers from the front (input side).
        """
        if not hasattr(self, 'feature_layer_list') or self.feature_layer_list is None:
            return

        n_layers = len(self.feature_layer_list)
        n_unfreeze = int(n_layers * unfreeze_ratio)
        n_unfreeze = max(0, min(n_layers, n_unfreeze))  # Clamp to valid range

        frozen_count = 0
        unfrozen_count = 0

        for i, layer in enumerate(self.feature_layer_list):
            # Freeze from the front: layers 0, 1, ... are frozen first
            freeze = (i < n_layers - n_unfreeze)
            for param in layer.parameters():
                param.requires_grad = not freeze
                if freeze:
                    frozen_count += 1
                else:
                    unfrozen_count += 1

        return frozen_count, unfrozen_count

    def _get_trainable_parameters_for_finetune(self, include_bnn: bool = True):
        """Get trainable parameters for finetuning (respecting freeze settings)"""
        params = []

        # Feature extractor parameters (may be partially frozen)
        if hasattr(self, 'feature_layer_list') and self.feature_layer_list is not None:
            for layer in self.feature_layer_list:
                for param in layer.parameters():
                    if param.requires_grad:
                        params.append(param)

        # BNN parameters (always trainable during finetune)
        if include_bnn and self.bnn is not None:
            for param in self.bnn.parameters():
                params.append(param)

        return params
    
    def pretrain(self, X_low: np.ndarray, y_low: np.ndarray,
                 epochs: int = 200, lr: float = 1e-3, batch_size: int = None,
                 verbose: bool = False,
                 bo_trials: Optional[int] = None, data_size: str = 'small',
                 use_loocv: bool = False, use_uncertainty_loss: bool = False,
                 uncertainty_weight: float = 0.3):
        """
        Pretrain on low-fidelity data.

        In 'consistent_bnn' mode: Train full BNN with ELBO loss
        In 'dngo_style' mode: Train deterministic feature extractor with MSE loss

        Args:
            batch_size: 미니배치 크기 (None이면 전체 배치)
        """
        self.pretrain_losses = []
        X_low = np.asarray(X_low, dtype=np.float32)
        y_low = np.asarray(y_low, dtype=np.float32).flatten()

        if verbose:
            print(f"Pretrain BNN with {len(X_low)} low-fidelity samples (mode: {self.transfer_mode})")

        # Hyperparameter BO for pretrain
        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            if verbose:
                print(f"  Running Pretrain BO with {bo_trials} trials...")

            from sklearn.model_selection import train_test_split
            if len(X_low) >= 5:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_low, y_low, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train, X_val, y_val = X_low, y_low, X_low, y_low

            from .hyperparameter_optimization_bnn_optuna import optimize_bnn_hyperparameters_optuna
            best_params, best_performance, history = optimize_bnn_hyperparameters_optuna(
                X_train, y_train, X_val, y_val,
                input_dim=self.input_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                verbose=verbose,
                stage='pretrain',
                use_loocv=False,
                use_uncertainty_loss=use_uncertainty_loss,
                uncertainty_weight=uncertainty_weight
            )

            self.pretrain_best_params = best_params
            self.pretrain_bo_history = history
            epochs = best_params.get('pretrain_epochs', epochs)
            lr = best_params.get('pretrain_lr', lr)
            batch_size = best_params.get('batch_size', batch_size)

        # Choose training mode
        if self.transfer_mode == 'consistent_bnn':
            self._pretrain_consistent_bnn(X_low, y_low, epochs, lr, batch_size, verbose)
        else:  # dngo_style
            self._pretrain_dngo_style(X_low, y_low, epochs, lr, batch_size, verbose)

    def _pretrain_consistent_bnn(self, X_low: np.ndarray, y_low: np.ndarray,
                                  epochs: int, lr: float, batch_size: int, verbose: bool):
        """
        Consistent BNN mode: Train full BNN on LOFI data.
        The same BNN will be fine-tuned on HIFI data later.

        Args:
            batch_size: 미니배치 크기 (None이면 전체 배치)
        """
        # Build full BNN if not exists
        if self.bnn is None:
            self.bnn = BayesianNeuralNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                prior_pi=self.prior_pi,
                prior_sigma1=self.prior_sigma1,
                prior_sigma2=self.prior_sigma2,
                noise_type=self.noise_type
            ).to(self.device)

        # Training data
        X_tensor = torch.tensor(X_low, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_low, dtype=torch.float32).view(-1, 1).to(self.device)

        optimizer = optim.Adam(self.bnn.parameters(), lr=lr)
        n_samples = len(X_low)
        actual_batch_size = batch_size if batch_size is not None else n_samples

        self.bnn.train()
        for epoch in range(epochs):
            # 미니배치 학습
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, actual_batch_size):
                end_idx = min(start_idx + actual_batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                optimizer.zero_grad()

                # Forward pass (미니배치)
                pred_mean, pred_var = self.bnn(X_tensor[batch_indices])

                # Negative log likelihood
                nll = 0.5 * torch.log(2 * math.pi * pred_var) + \
                      0.5 * (y_tensor[batch_indices] - pred_mean).pow(2) / pred_var
                nll = nll.mean()

                # KL divergence with warmup
                kl_div = self.bnn.kl_divergence()
                kl_weight_current = min(1.0, (epoch + 1) / self.kl_warmup_epochs) * self.kl_weight / len(batch_indices)

                # ELBO loss
                loss = nll + kl_weight_current * kl_div

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bnn.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.pretrain_losses.append(avg_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'  [Pretrain-BNN] Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}')

        self.fitted = True

    def _pretrain_dngo_style(self, X_low: np.ndarray, y_low: np.ndarray,
                              epochs: int, lr: float, batch_size: int, verbose: bool):
        """
        DNGO-style mode: Train deterministic feature extractor on LOFI data.

        Args:
            batch_size: 미니배치 크기 (None이면 전체 배치)
        """
        feature_dim = self.hidden_dims[0] if self.hidden_dims else 64
        self._build_feature_extractor(hidden_dim=feature_dim)

        # Add output layer for pretraining
        output_layer = nn.Linear(feature_dim, 1).to(self.device).float()
        pretrain_model = nn.Sequential(self.feature_extractor, output_layer)

        # Training
        X_tensor = torch.tensor(X_low, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_low, dtype=torch.float32).view(-1, 1).to(self.device)

        optimizer = optim.Adam(pretrain_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n_samples = len(X_low)
        actual_batch_size = batch_size if batch_size is not None else n_samples

        pretrain_model.train()
        for epoch in range(epochs):
            # 미니배치 학습
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, actual_batch_size):
                end_idx = min(start_idx + actual_batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                optimizer.zero_grad()
                pred = pretrain_model(X_tensor[batch_indices])
                loss = loss_fn(pred, y_tensor[batch_indices])
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.pretrain_losses.append(avg_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'  [Pretrain-DNN] Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}')
    
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
        
        # Build BNN (Scale Mixture Prior 사용)
        self.bnn = BayesianNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            prior_pi=self.prior_pi,
            prior_sigma1=self.prior_sigma1,
            prior_sigma2=self.prior_sigma2,
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
                 kl_warmup_epochs: int = 10, batch_size: int = None, verbose: bool = False,
                 bo_trials: Optional[int] = None, data_size: str = 'small',
                 use_loocv: bool = False, use_uncertainty_loss: bool = False,
                 uncertainty_weight: float = 0.3,
                 use_freeze: bool = False, unfreeze_ratio: float = 1.0):
        """
        Finetune on high-fidelity data.

        In 'consistent_bnn' mode: Continue training the same BNN (warm start)
        In 'dngo_style' mode: Train new BNN head on frozen features

        Args:
            batch_size: 미니배치 크기 (None이면 전체 배치)
        """
        self.finetune_losses = []
        X_high = np.asarray(X_high, dtype=np.float32)
        y_high = np.asarray(y_high, dtype=np.float32).flatten()

        if verbose:
            print(f"Finetune BNN with {len(X_high)} high-fidelity samples (mode: {self.transfer_mode})")

        # Hyperparameter BO
        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            if verbose:
                print(f"  Running Finetune BO with {bo_trials} trials...")

            from sklearn.model_selection import train_test_split
            if len(X_high) >= 3:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_high, y_high, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train, X_val, y_val = X_high, y_high, X_high, y_high

            fixed_structure = {
                'hidden_layers': len(self.hidden_dims),
                'hidden_dim': self.hidden_dims[0] if self.hidden_dims else 64
            }

            from .hyperparameter_optimization_bnn_optuna import optimize_bnn_hyperparameters_optuna
            best_params, best_performance, history = optimize_bnn_hyperparameters_optuna(
                X_train, y_train, X_val, y_val,
                input_dim=self.input_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                verbose=verbose,
                stage='finetune',
                fixed_structure=fixed_structure,
                use_loocv=use_loocv,
                use_uncertainty_loss=use_uncertainty_loss,
                uncertainty_weight=uncertainty_weight
            )

            self.finetune_best_params = best_params
            self.finetune_bo_history = history
            epochs = best_params.get('finetune_epochs', epochs)
            lr = best_params.get('finetune_lr', lr)
            kl_weight = best_params.get('kl_weight', kl_weight)
            batch_size = best_params.get('batch_size', batch_size)

        # Choose finetuning mode
        if self.transfer_mode == 'consistent_bnn':
            self._finetune_consistent_bnn(X_high, y_high, epochs, lr, kl_weight,
                                          kl_warmup_epochs, batch_size, use_freeze, unfreeze_ratio, verbose)
        else:  # dngo_style
            self._finetune_dngo_style(X_high, y_high, epochs, lr, kl_weight,
                                       kl_warmup_epochs, batch_size, use_freeze, unfreeze_ratio, verbose)

    def _finetune_consistent_bnn(self, X_high: np.ndarray, y_high: np.ndarray,
                                  epochs: int, lr: float, kl_weight: float,
                                  kl_warmup_epochs: int, batch_size: int, use_freeze: bool,
                                  unfreeze_ratio: float, verbose: bool):
        """
        Consistent BNN mode: Continue training the pretrained BNN on HIFI data.
        This is a warm start - the BNN already has learned representations from LOFI.

        Args:
            batch_size: 미니배치 크기 (None이면 전체 배치)
        """
        if self.bnn is None:
            raise ValueError("BNN not initialized. Call pretrain() first in consistent_bnn mode.")

        # Optionally freeze some layers
        if use_freeze and unfreeze_ratio < 1.0:
            n_layers = len(self.bnn.layers)
            n_freeze = int(n_layers * (1 - unfreeze_ratio))

            for i, layer in enumerate(self.bnn.layers):
                if i < n_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                    if verbose:
                        print(f"  Froze BNN layer {i}")

        # Training data
        X_tensor = torch.tensor(X_high, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_high, dtype=torch.float32).view(-1, 1).to(self.device)

        # Use lower learning rate for fine-tuning (warm start benefits from smaller steps)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.bnn.parameters()), lr=lr)
        n_samples = len(X_high)
        actual_batch_size = batch_size if batch_size is not None else n_samples

        self.bnn.train()
        for epoch in range(epochs):
            # 미니배치 학습
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, actual_batch_size):
                end_idx = min(start_idx + actual_batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                optimizer.zero_grad()

                # Forward pass (미니배치)
                pred_mean, pred_var = self.bnn(X_tensor[batch_indices])

                # Negative log likelihood
                nll = 0.5 * torch.log(2 * math.pi * pred_var) + \
                      0.5 * (y_tensor[batch_indices] - pred_mean).pow(2) / pred_var
                nll = nll.mean()

                # KL divergence with warmup (shorter warmup for fine-tuning)
                kl_div = self.bnn.kl_divergence()
                warmup_factor = min(1.0, (epoch + 1) / max(1, kl_warmup_epochs // 2))
                kl_weight_current = warmup_factor * kl_weight / len(batch_indices)

                # ELBO loss
                loss = nll + kl_weight_current * kl_div

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bnn.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.finetune_losses.append(avg_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'  [Finetune-BNN] Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}')

        # Restore requires_grad
        for layer in self.bnn.layers:
            for param in layer.parameters():
                param.requires_grad = True

        self._clear_cache()
        self.fitted = True

    def _finetune_dngo_style(self, X_high: np.ndarray, y_high: np.ndarray,
                              epochs: int, lr: float, kl_weight: float,
                              kl_warmup_epochs: int, batch_size: int, use_freeze: bool,
                              unfreeze_ratio: float, verbose: bool):
        """
        DNGO-style mode: Train BNN head on frozen feature extractor outputs.

        Args:
            batch_size: 미니배치 크기 (None이면 전체 배치)
        """
        # Build feature extractor if not exists
        if self.feature_extractor is None:
            self._build_feature_extractor()

        # Apply freeze to feature extractor
        if use_freeze:
            result = self._apply_unfreeze_ratio_to_feature_extractor(unfreeze_ratio)
            if result is not None and verbose:
                frozen_count, unfrozen_count = result
                n_layers = len(self.feature_layer_list) if hasattr(self, 'feature_layer_list') else 0
                n_unfreeze = int(n_layers * unfreeze_ratio)
                print(f"  Freeze: {n_layers - n_unfreeze}/{n_layers} layers frozen")

        # Get feature dimension
        with torch.no_grad():
            X_tensor = torch.tensor(X_high, dtype=torch.float32).to(self.device)
            features = self.feature_extractor(X_tensor)
            feature_dim = features.shape[1]

        # Build BNN head (smaller network on top of features)
        bnn_hidden_dims = [feature_dim // 2] if feature_dim > 32 else [32]
        self.bnn = BayesianNeuralNetwork(
            input_dim=feature_dim,
            hidden_dims=bnn_hidden_dims,
            prior_pi=self.prior_pi,
            prior_sigma1=self.prior_sigma1,
            prior_sigma2=self.prior_sigma2,
            noise_type=self.noise_type
        ).to(self.device)

        # Prepare data
        X_raw_tensor = torch.tensor(X_high, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_high, dtype=torch.float32).view(-1, 1).to(self.device)

        # Set up optimizer
        if use_freeze:
            trainable_params = self._get_trainable_parameters_for_finetune(include_bnn=True)
            optimizer = optim.Adam(trainable_params, lr=lr)
            self.feature_extractor.train()
        else:
            optimizer = optim.Adam(self.bnn.parameters(), lr=lr)

        n_samples = len(X_high)
        actual_batch_size = batch_size if batch_size is not None else n_samples
        self.bnn.train()

        for epoch in range(epochs):
            # 미니배치 학습
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, actual_batch_size):
                end_idx = min(start_idx + actual_batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                optimizer.zero_grad()

                # Forward pass through feature extractor (미니배치)
                if use_freeze:
                    features = self.feature_extractor(X_raw_tensor[batch_indices])
                else:
                    with torch.no_grad():
                        features = self.feature_extractor(X_raw_tensor[batch_indices])

                # Forward pass through BNN
                pred_mean, pred_var = self.bnn(features)

                # NLL
                nll = 0.5 * torch.log(2 * math.pi * pred_var) + \
                      0.5 * (y_tensor[batch_indices] - pred_mean).pow(2) / pred_var
                nll = nll.mean()

                # KL divergence with warmup
                kl_div = self.bnn.kl_divergence()
                kl_weight_current = min(1.0, (epoch + 1) / kl_warmup_epochs) * kl_weight / len(batch_indices)

                # ELBO loss
                loss = nll + kl_weight_current * kl_div

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bnn.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.finetune_losses.append(avg_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'  [Finetune-DNGO] Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}')

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
            raise ValueError("Model not fitted. Call pretrain() and finetune() first.")

        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.transfer_mode == 'consistent_bnn':
            # Direct BNN prediction
            pred_mean, pred_var = self.bnn.sample_predict(X_tensor, n_samples=n_samples)
        else:  # dngo_style
            # Extract features first, then BNN prediction
            features = self.extract_features(X)
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            pred_mean, pred_var = self.bnn.sample_predict(features_tensor, n_samples=n_samples)

        mean = pred_mean.cpu().numpy().flatten()
        std = torch.sqrt(pred_var).cpu().numpy().flatten()

        return mean, std

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features using the pretrained feature extractor (dngo_style mode only)"""
        if self.transfer_mode == 'consistent_bnn':
            # In consistent_bnn mode, features are internal to BNN
            # Return the input as-is or raise an error
            raise ValueError("extract_features() not applicable in consistent_bnn mode. "
                           "Features are internal to the BNN.")

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
    
    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str = 'high'):
        """
        BNN 점진적 업데이트 (불확실성 기반 + KL regularization)
        
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
            kl_weight = self.incremental_params.get('kl_weight', 0.1)
        else:
            # 기본값
            mode = 'incremental'
            lr_boost = 2.0
            inc_epochs = 10
            retrain_interval = 5
            replay_ratio = 0.3
            weight_decay = 0.95
            kl_weight = 0.1
        
        # 모드에 따른 업데이트 전략
        if mode == 'full':
            # 항상 전체 재학습
            self._full_retrain_bnn(X_new, y_new, fidelity)
        elif mode == 'hybrid' and self.update_counter % retrain_interval == 0:
            # 주기적 전체 재학습
            self._full_retrain_bnn(X_new, y_new, fidelity)
        else:
            # 점진적 업데이트
            self._incremental_train_bnn(X_new, y_new, fidelity, lr_boost, inc_epochs,
                                       replay_ratio, weight_decay, kl_weight)
        
        # 데이터 버퍼 업데이트
        self._update_buffer_bnn(X_new, y_new, fidelity)
    
    def _full_retrain_bnn(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str):
        """전체 재학습"""
        if fidelity == 'high':
            self.finetune(X_new, y_new,
                         epochs=self.finetune_best_params.get('epochs', 50) if self.finetune_best_params else 50,
                         lr=self.finetune_best_params.get('learning_rate', 1e-4) if self.finetune_best_params else 1e-4,
                         kl_weight=self.finetune_best_params.get('kl_weight', 1.0) if self.finetune_best_params else 1.0)
        else:
            self.pretrain(X_new, y_new,
                         epochs=self.pretrain_best_params.get('pretrain_epochs', 50) if self.pretrain_best_params else 50,
                         lr=self.pretrain_best_params.get('pretrain_lr', 1e-3) if self.pretrain_best_params else 1e-3)
    
    def _incremental_train_bnn(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str,
                              lr_boost: float, inc_epochs: int, replay_ratio: float, 
                              weight_decay: float, kl_weight: float):
        """BNN 점진적 학습 (KL regularization으로 catastrophic forgetting 방지)"""
        if not self.fitted or self.bnn is None:
            # BNN이 없으면 전체 학습
            self._full_retrain_bnn(X_new, y_new, fidelity)
            return
        
        # 이전 파라미터 저장 (KL regularization용)
        if self.previous_bnn_params is None:
            self.previous_bnn_params = {name: param.clone().detach() 
                                       for name, param in self.bnn.named_parameters()}
        
        # 불확실성 기반 가중치 계산
        importance_weights = self._compute_uncertainty_weights(X_new)
        
        # 기본 학습률에 부스트 적용
        base_lr = self.last_learning_rates.get('finetune' if fidelity == 'high' else 'pretrain', 1e-4)
        boosted_lr = base_lr * lr_boost
        
        # Experience Replay 데이터 준비
        X_combined, y_combined, sample_weights = self._prepare_replay_data_bnn(
            X_new, y_new, fidelity, replay_ratio, weight_decay, importance_weights
        )
        
        # 점진적 학습 (BNN은 feature extraction 사용)
        X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
        if self.feature_extractor is not None:
            with torch.no_grad():
                input_tensor = self.feature_extractor(X_tensor)
        else:
            input_tensor = X_tensor
        
        y_tensor = torch.tensor(y_combined, dtype=torch.float32).to(self.device)
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(self.device)
        
        optimizer = torch.optim.Adam(self.bnn.parameters(), lr=boosted_lr)
        
        self.bnn.train()
        for epoch in range(inc_epochs):
            optimizer.zero_grad()
            
            # BNN 예측
            pred_mean, pred_var = self.bnn.sample_predict(input_tensor, n_samples=10)
            
            # 데이터 손실 (가중치 적용)
            data_loss = torch.nn.functional.mse_loss(pred_mean, y_tensor.view(-1, 1), reduction='none')
            weighted_data_loss = (data_loss.squeeze() * weights_tensor).mean()
            
            # KL regularization (catastrophic forgetting 방지)
            kl_loss = self._compute_kl_divergence_from_previous()
            
            # 총 손실
            total_loss = weighted_data_loss + kl_weight * kl_loss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.bnn.parameters(), 1.0)
            optimizer.step()
        
        # 현재 파라미터를 이전 파라미터로 업데이트
        self.previous_bnn_params = {name: param.clone().detach() 
                                   for name, param in self.bnn.named_parameters()}
        
        # 학습률 기록
        self.last_learning_rates[fidelity] = boosted_lr
    
    def _compute_uncertainty_weights(self, X_new: np.ndarray) -> np.ndarray:
        """불확실성 기반 가중치 계산 (불확실한 영역일수록 높은 가중치)"""
        if not self.fitted:
            return np.ones(len(X_new))
        
        try:
            # 예측 불확실성 계산
            pred_mean, pred_var = self.predict(X_new, n_samples=30)
            uncertainties = np.sqrt(pred_var)
            
            # 불확실성을 가중치로 변환 (정규화)
            if np.std(uncertainties) > 1e-8:
                weights = 1.0 + (uncertainties - uncertainties.mean()) / uncertainties.std()
                weights = np.maximum(weights, 0.1)  # 최소 가중치 보장
            else:
                weights = np.ones(len(X_new))
            
            return weights
        except:
            return np.ones(len(X_new))
    
    def _prepare_replay_data_bnn(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str,
                                replay_ratio: float, weight_decay: float, importance_weights: np.ndarray):
        """BNN용 Experience Replay 데이터 준비"""
        buffer_key_x = f'X_{fidelity}'
        buffer_key_y = f'y_{fidelity}'
        
        # 새 데이터 (중요도 가중치 적용)
        X_combined = X_new.copy()
        y_combined = y_new.copy()
        sample_weights = importance_weights.copy()
        
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
    
    def _update_buffer_bnn(self, X_new: np.ndarray, y_new: np.ndarray, fidelity: str, max_buffer_size: int = 100):
        """BNN용 데이터 버퍼 업데이트"""
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
    
    def _compute_kl_divergence_from_previous(self) -> torch.Tensor:
        """이전 파라미터와의 KL divergence 계산 (catastrophic forgetting 방지)"""
        if self.previous_bnn_params is None:
            return torch.tensor(0.0).to(self.device)
        
        kl_loss = 0.0
        for name, current_param in self.bnn.named_parameters():
            if name in self.previous_bnn_params:
                previous_param = self.previous_bnn_params[name]
                # 간단한 L2 regularization으로 KL divergence 근사
                kl_loss += torch.nn.functional.mse_loss(current_param, previous_param)
        
        return kl_loss
    
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


class OnlineTransferLearningBNN(TransferLearningBNN):
    """
    Online Learning을 위한 Transfer Learning BNN
    DNGO-OL과 동일한 패턴으로 온라인 업데이트 지원

    개선 사항 (Scale Mixture Prior 적용):
    - prior_std 대신 Scale Mixture Prior (prior_pi, prior_sigma1, prior_sigma2) 사용
    - 기본 kl_weight를 0.5로 설정 (최적화된 값)
    - 기본 hidden_dims를 [64, 64, 64]로 설정 (3층 구조)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64, 64],
                 device: str = 'cpu',
                 prior_pi: float = 0.5, prior_sigma1: float = 1.0, prior_sigma2: float = 0.002,
                 noise_type: str = 'homoscedastic', use_hyperparameter_bo: bool = False,
                 kl_weight: float = 0.5, kl_warmup_epochs: int = 10,
                 replay_buffer_size: int = 100, online_batch_size: int = 16,
                 online_epochs: int = 5, transfer_mode: str = 'consistent_bnn'):
        """
        Args:
            input_dim: 입력 차원
            hidden_dims: 히든 레이어 차원들 (기본값: [64, 64, 64])
            device: 디바이스
            prior_pi: Scale Mixture Prior 혼합 가중치
            prior_sigma1: Scale Mixture Prior sigma1
            prior_sigma2: Scale Mixture Prior sigma2
            noise_type: 노이즈 타입
            use_hyperparameter_bo: 하이퍼파라미터 BO 사용 여부
            kl_weight: KL divergence 가중치 (기본값: 0.5)
            kl_warmup_epochs: KL warmup epochs
            replay_buffer_size: 리플레이 버퍼 크기
            online_batch_size: 온라인 학습 배치 크기
            online_epochs: 온라인 업데이트 시 epoch 수
            transfer_mode: 전이학습 모드 ('consistent_bnn' 또는 'dngo_style')
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            device=device,
            prior_pi=prior_pi,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            noise_type=noise_type,
            use_hyperparameter_bo=use_hyperparameter_bo,
            kl_weight=kl_weight,
            kl_warmup_epochs=kl_warmup_epochs,
            transfer_mode=transfer_mode
        )

        self.replay_buffer_size = replay_buffer_size
        self.online_batch_size = online_batch_size
        self.online_epochs = online_epochs

        # 리플레이 버퍼
        from collections import deque
        self.replay_buffer = {
            'low': deque(maxlen=replay_buffer_size),
            'high': deque(maxlen=replay_buffer_size)
        }

        # 온라인 학습 메트릭 추적
        self.online_training_history = {
            'losses': [],
            'losses_per_epoch': [],
            'update_counts': [],
            'fidelities': [],
            'buffer_sizes': [],
            'learning_rates': []
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

        if len(self.replay_buffer[fidelity]) >= self.online_batch_size and self.bnn is not None:
            optimizer = optim.Adam(self.bnn.parameters(), lr=lr)

            self.bnn.train()
            if self.feature_extractor is not None:
                self.feature_extractor.eval()

            for epoch in range(self.online_epochs):
                # 랜덤 샘플링
                indices = np.random.choice(
                    len(self.replay_buffer[fidelity]),
                    size=min(self.online_batch_size, len(self.replay_buffer[fidelity])),
                    replace=False
                )

                batch_data = [self.replay_buffer[fidelity][i] for i in indices]
                X_batch = np.array([x for x, _ in batch_data], dtype=np.float32)
                y_batch = np.array([y for _, y in batch_data], dtype=np.float32)

                # Tensor 변환
                X_tensor = torch.FloatTensor(X_batch).to(self.device)
                y_tensor = torch.FloatTensor(y_batch).view(-1, 1).to(self.device)

                # Feature extraction
                if self.feature_extractor is not None:
                    with torch.no_grad():
                        features = self.feature_extractor(X_tensor)
                else:
                    features = X_tensor

                # Forward pass
                optimizer.zero_grad()
                pred_mean, pred_var = self.bnn(features)

                # NLL loss
                nll = 0.5 * torch.log(2 * math.pi * pred_var) + 0.5 * (y_tensor - pred_mean).pow(2) / pred_var
                nll = nll.mean()

                # KL divergence
                kl_div = self.bnn.kl_divergence()
                kl_weight_current = self.kl_weight / len(X_batch)

                # Total loss
                loss = nll + kl_weight_current * kl_div

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bnn.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            final_loss = epoch_losses[-1] if epoch_losses else None

        # 메트릭 기록
        self.online_training_history['losses'].append(final_loss)
        self.online_training_history['losses_per_epoch'].append(epoch_losses)
        self.online_training_history['update_counts'].append(self.online_update_count)
        self.online_training_history['fidelities'].append(fidelity)
        self.online_training_history['buffer_sizes'].append(len(self.replay_buffer[fidelity]))
        self.online_training_history['learning_rates'].append(lr)