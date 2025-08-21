"""
완전히 독립적인 DNGO 테스트 - 모든 필요한 함수를 내장
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import os
import time
from pathlib import Path
from scipy.stats import norm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ================== 1. 데이터 로드 함수 ==================
def load_lookup_table():
    """lookup table 로드"""
    data_path = Path(__file__).parent.parent.parent / "0.Data" / "lookup_table.pkl"
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def load_label_maps():
    """라벨 맵 로드"""
    data_dir = Path(__file__).parent.parent.parent / "0.Data"
    
    with open(data_dir / "organics.json", 'r') as f:
        organics = json.load(f)
    with open(data_dir / "cations.json", 'r') as f:
        cations = json.load(f)
    with open(data_dir / "anions.json", 'r') as f:
        anions = json.load(f)
    
    return {
        'organic': organics,
        'cation': cations,
        'anion': anions
    }

# ================== 2. 데이터 유틸리티 함수 ==================
def sample_param_space(param_space: Dict, n_samples: int, random_state: int = 42) -> List:
    """파라미터 공간에서 랜덤 샘플링"""
    np.random.seed(random_state)
    samples = []
    for _ in range(n_samples):
        sample = [
            np.random.choice(list(range(1, len(param_space['organic']) + 1))),
            np.random.choice(list(range(1, len(param_space['cation']) + 1))),
            np.random.choice(list(range(1, len(param_space['anion']) + 1)))
        ]
        samples.append(sample)
    return samples

def assign_fidelities(n_samples: int, high_ratio: float, random_state: int = 42) -> List:
    """fidelity 할당"""
    np.random.seed(random_state)
    n_high = int(n_samples * high_ratio)
    fidelities = [1.0] * n_high + [0.1] * (n_samples - n_high)
    np.random.shuffle(fidelities)
    return fidelities

def measure_from_label(label: List, fidelity: float, label_maps: Dict, lookup: Dict) -> float:
    """라벨로부터 측정값 얻기"""
    key = f"{label[0]},{label[1]},{label[2]}"
    if fidelity == 1.0:
        return lookup[key]['bandgap_hse06']
    else:
        return lookup[key]['bandgap_gga']

def prepare_initial_data(samples: List, fidelities: List, label_maps: Dict, lookup: Dict):
    """초기 데이터 준비"""
    X_low, y_low = [], []
    X_high, y_high = [], []
    
    for sample, fidelity in zip(samples, fidelities):
        x = np.array(sample, dtype=np.float32)
        y = measure_from_label(sample, fidelity, label_maps, lookup)
        
        if fidelity == 1.0:
            X_high.append(x)
            y_high.append(y)
        else:
            X_low.append(x)
            y_low.append(y)
    
    return np.array(X_low), np.array(y_low), np.array(X_high), np.array(y_high)

# ================== 3. 간단한 DNN 모델 ==================
class SimpleDNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, device='cpu'):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
    def forward(self, x):
        return self.model(x)
    
    def train_model(self, X, y, epochs=50, lr=1e-3):
        """간단한 학습"""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Loss {loss.item():.4f}")
    
    def predict(self, X):
        """예측"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred = self(X_tensor)
            return pred.cpu().numpy().flatten()

# ================== 4. Expected Improvement ==================
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """EI 계산"""
    sigma = np.maximum(sigma, 1e-8)
    z = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

# ================== 5. 간단한 시각화 ==================
def plot_simple_result(iterations, best_values, costs, save_path='test_result.png'):
    """간단한 결과 플롯"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Best value curve
    ax1.plot(iterations, best_values, 'b-o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Value')
    ax1.set_title('Convergence')
    ax1.axhline(y=1.34, color='r', linestyle='--', label='Target')
    ax1.legend()
    ax1.grid(True)
    
    # Cost curve
    ax2.plot(iterations, costs, 'g-s')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Cost')
    ax2.set_title('Cost Accumulation')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Plot saved to {save_path}")

# ================== 6. 메인 테스트 함수 ==================
def test_dngo_simple():
    """간단한 DNGO 테스트"""
    print("🚀 Starting simplified DNGO test...")
    print("=" * 60)
    
    # 데이터 로드
    print("📊 Loading data...")
    lookup = load_lookup_table()
    label_maps = load_label_maps()
    param_space = {
        'organic': list(label_maps['organic'].keys()),
        'cation': list(label_maps['cation'].keys()),
        'anion': list(label_maps['anion'].keys())
    }
    print(f"✓ Loaded {len(lookup)} combinations")
    
    # 초기 샘플링
    print("\n🎲 Initial sampling...")
    samples = sample_param_space(param_space, 5, random_state=42)
    fidelities = assign_fidelities(5, 0.2, random_state=42)
    X_low, y_low, X_high, y_high = prepare_initial_data(samples, fidelities, label_maps, lookup)
    print(f"✓ Low-fidelity: {len(y_low)} samples")
    print(f"✓ High-fidelity: {len(y_high)} samples")
    
    # 모델 학습
    print("\n🧠 Training model...")
    model = SimpleDNN(input_dim=3, hidden_dim=32)
    
    # 모든 데이터로 학습
    if len(X_low) > 0 and len(X_high) > 0:
        X_all = np.vstack([X_low, X_high])
        y_all = np.concatenate([y_low, y_high])
    elif len(X_low) > 0:
        X_all = X_low
        y_all = y_low
    else:
        X_all = X_high
        y_all = y_high
    
    model.train_model(X_all, y_all, epochs=30)
    
    # 간단한 BO 루프 (5 iterations)
    print("\n🔄 Running optimization loop...")
    cost = sum(fidelities)
    best_value = np.min(y_high) if len(y_high) > 0 else np.inf
    
    iterations = []
    best_values = []
    costs = []
    
    for iter_ in range(5):
        print(f"\n  Iteration {iter_+1}:")
        
        # 전체 공간에서 예측
        import itertools
        all_combinations = list(itertools.product(
            range(1, len(param_space['organic']) + 1),
            range(1, len(param_space['cation']) + 1),
            range(1, len(param_space['anion']) + 1)
        ))
        X_grid = np.array(all_combinations, dtype=np.float32)
        
        # 예측
        y_pred = model.predict(X_grid)
        y_std = np.abs(y_pred * 0.1)  # 간단한 불확실성 추정
        
        # EI 계산
        ei = expected_improvement(y_pred, y_std, best_value)
        
        # 최대 EI 선택
        best_idx = np.argmax(ei)
        next_x = X_grid[best_idx]
        
        # fidelity 결정
        fidelity = 1.0 if (iter_ % 4 == 0) else 0.1
        
        # 측정
        next_y = measure_from_label(next_x.astype(int).tolist(), fidelity, label_maps, lookup)
        
        print(f"    Selected: {next_x.astype(int).tolist()}")
        print(f"    Fidelity: {fidelity}")
        print(f"    Value: {next_y:.4f}")
        
        # 데이터 업데이트
        if fidelity == 1.0:
            X_high = np.vstack([X_high, next_x.reshape(1, -1)])
            y_high = np.concatenate([y_high, [next_y]])
            best_value = min(best_value, next_y)
        else:
            X_low = np.vstack([X_low, next_x.reshape(1, -1)])
            y_low = np.concatenate([y_low, [next_y]])
        
        # 비용 업데이트
        cost += fidelity
        
        # 기록
        iterations.append(iter_+1)
        best_values.append(best_value)
        costs.append(cost)
        
        # 모델 재학습 (간단히)
        if len(X_low) > 0 and len(X_high) > 0:
            X_all = np.vstack([X_low, X_high])
            y_all = np.concatenate([y_low, y_high])
        elif len(X_low) > 0:
            X_all = X_low
            y_all = y_low
        else:
            X_all = X_high
            y_all = y_high
        
        model.train_model(X_all, y_all, epochs=10)
        
        # 예산 체크
        if cost >= 10:
            print(f"\n  💰 Budget exhausted at cost {cost:.1f}")
            break
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📈 Final Results:")
    print(f"  - Total cost: {cost:.1f}")
    print(f"  - Best value: {best_value:.4f}")
    print(f"  - High-fidelity samples: {len(y_high)}")
    print(f"  - Low-fidelity samples: {len(y_low)}")
    
    # 시각화
    print("\n🎨 Creating plot...")
    os.makedirs("test_results", exist_ok=True)
    plot_simple_result(iterations, best_values, costs, 'test_results/dngo_test.png')
    
    print("\n✅ Test completed successfully!")
    print("📁 Check 'test_results/' directory for results")

if __name__ == "__main__":
    test_dngo_simple()