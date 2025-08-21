"""
시각화 기능 테스트용 더미 데이터 생성 및 테스트 스크립트

DNGO 시각화 코드를 실제 최적화 없이 테스트할 수 있도록 더미 데이터를 생성합니다.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
import os
import sys
from pathlib import Path

# 현재 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent))

from DNGO.visualization import (
    plot_step_visualization, 
    plot_optimization_progress, 
    plot_multiple_runs_boxplot
)


class DummyDNN(nn.Module):
    """더미 DNN 모델"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 간단한 2층 네트워크
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(device)
        
        # 랜덤 웨이트로 초기화 (재현 가능하도록)
        torch.manual_seed(42)
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.1)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.feature_net(x)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Feature 추출"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            features = self.feature_net(X_tensor)
            return features.cpu().numpy()


class DummyBLR:
    """더미 Bayesian Linear Regression"""
    
    def __init__(self, feature_dim: int = 64):
        # 더미 파라미터 설정
        np.random.seed(42)
        self.mean = np.random.normal(0, 0.1, feature_dim)
        self.cov = np.eye(feature_dim) * 0.01  # 작은 분산
        
    def predict(self, phi: np.ndarray):
        """예측 (평균과 분산 반환)"""
        # 간단한 선형 예측 + 노이즈
        mean = np.dot(phi, self.mean) + np.random.normal(0, 0.02)
        var = max(0.001, phi @ self.cov @ phi.T + 0.01)
        return mean, var


def generate_dummy_optimization_data(n_iterations: int = 10, n_grid: int = 100) -> Dict:
    """
    더미 최적화 데이터 생성
    
    Args:
        n_iterations: 반복 횟수
        n_grid: 그리드 포인트 수
        
    Returns:
        시각화용 데이터 딕셔너리
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 더미 모델들 생성
    dummy_dnn = DummyDNN()
    dummy_blr = DummyBLR()
    
    # 그리드 생성 (3D 파라미터 공간)
    X_grid = np.random.uniform(1, 10, (n_grid, 3)).astype(np.float32)
    
    # 실제 함수 시뮬레이션 (Rosenbrock 함수 변형)
    def true_function(X):
        return 2.0 + 0.1 * ((X[:, 0] - 5)**2 + 10 * (X[:, 1] - X[:, 0]**2)**2) + 0.05 * X[:, 2]**2
    
    # 초기 측정 데이터
    n_initial_low = 8
    n_initial_high = 4
    
    X_low = X_grid[:n_initial_low].copy()
    X_high = X_grid[n_initial_low:n_initial_low+n_initial_high].copy()
    
    # 노이즈 추가 (low-fidelity는 더 많은 노이즈)
    y_low = true_function(X_low) + np.random.normal(0, 0.2, n_initial_low)
    y_high = true_function(X_high) + np.random.normal(0, 0.05, n_initial_high)
    
    # 시각화 데이터 생성
    visualization_data = []
    best_values_history = []
    cost_history = []
    fidelity_history = []
    ei_history = []
    
    current_best = np.min(y_high) if len(y_high) > 0 else np.inf
    cumulative_cost = n_initial_low * 0.1 + n_initial_high * 1.0
    
    for iter_ in range(1, n_iterations + 1):
        # 그리드 전체에 대한 예측
        features_grid = dummy_dnn.extract_features(X_grid)
        y_pred = []
        y_std = []
        
        for phi in features_grid:
            mu, var = dummy_blr.predict(phi)
            y_pred.append(mu)
            y_std.append(np.sqrt(var))
        
        y_pred = np.array(y_pred)
        y_std = np.array(y_std)
        
        # Expected Improvement 계산
        if current_best != np.inf:
            from scipy.stats import norm
            sigma = np.maximum(y_std, 1e-8)
            z = (current_best - y_pred - 0.01) / sigma
            ei = (current_best - y_pred - 0.01) * norm.cdf(z) + sigma * norm.pdf(z)
        else:
            ei = y_std  # 불확실성 기반
        
        # 최대 EI 점 선택
        best_idx = np.argmax(ei)
        
        # Fidelity 스케줄링 (8번 중 1번 high-fidelity)
        fidelity = 1.0 if (iter_ % 8 == 0) else 0.1
        
        # 새로운 측정값 시뮬레이션
        new_x = X_grid[best_idx:best_idx+1]
        noise_std = 0.05 if fidelity == 1.0 else 0.2
        new_y = true_function(new_x) + np.random.normal(0, noise_std, 1)
        
        # 데이터 업데이트
        if fidelity == 1.0:
            X_high = np.vstack([X_high, new_x])
            y_high = np.concatenate([y_high, new_y])
            current_best = min(current_best, new_y[0])
        else:
            X_low = np.vstack([X_low, new_x])
            y_low = np.concatenate([y_low, new_y])
        
        cumulative_cost += fidelity
        
        # 시각화 데이터 저장
        viz_data = {
            'iteration': iter_,
            'y_pred': y_pred.copy(),
            'y_std': y_std.copy(),
            'ei': ei.copy(),
            'best_idx': best_idx,
            'X_grid': X_grid.copy(),
            'fidelity': fidelity,
            'recommended_point': X_grid[best_idx].astype(int).tolist(),
            'model': dummy_dnn,
            'blr': dummy_blr,
            'X_low': X_low.copy(),
            'y_low': y_low.copy(),
            'X_high': X_high.copy(),
            'y_high': y_high.copy()
        }
        visualization_data.append(viz_data)
        
        # 기록 업데이트
        best_values_history.append(current_best)
        cost_history.append(cumulative_cost)
        fidelity_history.append(fidelity)
        ei_history.append(ei[best_idx])
    
    # 최종 결과 딕셔너리
    return {
        'total_cost': cumulative_cost,
        'best_so_far': current_best,
        'iterations': n_iterations,
        'visualization_data': visualization_data,
        'best_values_history': best_values_history,
        'cost_history': cost_history,
        'fidelity_history': fidelity_history,
        'ei_history': ei_history,
        'model_type': 'DNGO-Test'
    }


def generate_multiple_dummy_runs(n_runs: int = 20, n_iterations: int = 8) -> List[Dict]:
    """다중 실행 더미 데이터 생성"""
    all_results = []
    
    for run in range(n_runs):
        # 각 run마다 다른 seed 사용
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)
        
        result = generate_dummy_optimization_data(n_iterations)
        result['run_id'] = run + 1
        all_results.append(result)
    
    return all_results


def test_step_visualization():
    """단일 스텝 시각화 테스트"""
    print("🔍 Testing step visualization...")
    
    # 더미 데이터 생성
    result = generate_dummy_optimization_data(n_iterations=5)
    
    # 첫 번째 스텝 시각화
    first_step = result['visualization_data'][0]
    
    # 저장 디렉토리 생성
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # 시각화 테스트
    plot_step_visualization(first_step, save_path=f"{test_dir}/step_001_test.png")
    print(f"✅ Step visualization saved to {test_dir}/step_001_test.png")


def test_optimization_progress():
    """전체 최적화 진행 시각화 테스트"""
    print("🔍 Testing optimization progress visualization...")
    
    # 더미 데이터 생성
    result = generate_dummy_optimization_data(n_iterations=10)
    
    # 저장 디렉토리 생성
    test_dir = "test_images/progress"
    
    # 전체 진행 과정 시각화
    plot_optimization_progress(result, save_dir=test_dir)
    print(f"✅ Optimization progress saved to {test_dir}/")


def test_multiple_runs_boxplot():
    """다중 실행 박스플롯 테스트"""
    print("🔍 Testing multiple runs boxplot...")
    
    # 다중 실행 더미 데이터 생성
    all_results = generate_multiple_dummy_runs(n_runs=50, n_iterations=15)
    
    # 저장 디렉토리 생성
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # 박스플롯 시각화
    plot_multiple_runs_boxplot(all_results, save_path=f"{test_dir}/boxplot_test.png")
    print(f"✅ Multiple runs boxplot saved to {test_dir}/boxplot_test.png")


def run_all_tests():
    """모든 시각화 테스트 실행"""
    print("🚀 Starting visualization tests with dummy data...\n")
    
    try:
        # 1. 단일 스텝 시각화 테스트
        test_step_visualization()
        print()
        
        # 2. 전체 최적화 진행 시각화 테스트
        test_optimization_progress()
        print()
        
        # 3. 다중 실행 박스플롯 테스트
        test_multiple_runs_boxplot()
        print()
        
        print("🎉 All visualization tests completed successfully!")
        print("📁 Check the 'test_images/' directory for results.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()