#!/usr/bin/env python3
"""
다양한 Multi-Fidelity 접근법 비교
1. Direct HF: X -> HF (LF 없이 직접 HF 예측)
2. Residual: LF 학습 후, HF = LF + delta 학습
3. Stacked: [X, y_LF] -> HF (현재 방식)
4. Transfer Features: LF feature extractor 동결, HF head만 학습
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

# Local imports
from common.data_utils import load_lookup_table, create_label_maps, create_param_space
from common.config import DATA_PATHS


class FeatureNetwork(nn.Module):
    """공유 Feature Extractor"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_net(x)


class DirectHead(nn.Module):
    """직접 예측 Head"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        return self.out_layer(features)


class ResidualHead(nn.Module):
    """Residual 예측 Head: delta 학습"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.delta_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # features + y_lf
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, y_lf):
        combined = torch.cat([features, y_lf], dim=-1)
        delta = self.delta_layer(combined)
        return y_lf + delta  # HF = LF + delta


def load_mf_data():
    """Multi-fidelity 데이터 로드"""
    lookup = load_lookup_table(DATA_PATHS['lookup_table'])
    param_space = create_param_space(lookup)
    label_maps = create_label_maps(param_space)

    X_all = []
    y_lf_all = []
    y_hf_all = []

    reverse_maps = {
        'organic': {v: k for k, v in label_maps['organic'].items()},
        'cation': {v: k for k, v in label_maps['cation'].items()},
        'anion': {v: k for k, v in label_maps['anion'].items()}
    }

    for org_label, org_name in reverse_maps['organic'].items():
        for cat_label, cat_name in reverse_maps['cation'].items():
            for ani_label, ani_name in reverse_maps['anion'].items():
                try:
                    org_cap = org_name.capitalize()
                    y_lf = np.amin(lookup[org_cap][cat_name][ani_name]['bandgap_gga'])
                    y_hf = np.amin(lookup[org_cap][cat_name][ani_name]['bandgap_hse06'])

                    X_all.append([org_label, cat_label, ani_label])
                    y_lf_all.append(y_lf)
                    y_hf_all.append(y_hf)
                except (KeyError, ValueError):
                    continue

    X = np.array(X_all, dtype=np.float32)
    y_lf = np.array(y_lf_all, dtype=np.float32)
    y_hf = np.array(y_hf_all, dtype=np.float32)

    return X, y_lf, y_hf


def train_direct_hf(X_train, y_hf_train, X_test, y_hf_test, epochs=300, device='cpu'):
    """방법 1: Direct HF - LF 없이 직접 HF 예측"""
    input_dim = X_train.shape[1]

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_hf_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_t).cpu().numpy().flatten()
        test_pred = model(X_test_t).cpu().numpy().flatten()

    return r2_score(y_hf_train, train_pred), r2_score(y_hf_test, test_pred)


def train_transfer_features(X_train, y_lf_train, y_hf_train, X_test, y_lf_test, y_hf_test,
                            lf_epochs=200, hf_epochs=100, device='cpu'):
    """방법 2: Transfer Features - LF로 feature extractor 학습 후 HF head만 학습"""
    input_dim = X_train.shape[1]
    hidden_dim = 64

    feature_net = FeatureNetwork(input_dim, hidden_dim).to(device)
    lf_head = DirectHead(hidden_dim).to(device)
    hf_head = DirectHead(hidden_dim).to(device)

    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf_train, dtype=torch.float32).view(-1, 1).to(device)
    y_hf_t = torch.tensor(y_hf_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Stage 1: LF 학습 (feature_net + lf_head)
    optimizer = optim.Adam(list(feature_net.parameters()) + list(lf_head.parameters()), lr=1e-3)

    feature_net.train()
    lf_head.train()
    for epoch in range(lf_epochs):
        optimizer.zero_grad()
        features = feature_net(X_t)
        pred = lf_head(features)
        loss = loss_fn(pred, y_lf_t)
        loss.backward()
        optimizer.step()

    # Stage 2: HF 학습 (feature_net 동결, hf_head만 학습)
    for param in feature_net.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(hf_head.parameters(), lr=1e-3)
    hf_head.train()

    for epoch in range(hf_epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            features = feature_net(X_t)
        pred = hf_head(features)
        loss = loss_fn(pred, y_hf_t)
        loss.backward()
        optimizer.step()

    # 평가
    feature_net.eval()
    hf_head.eval()
    with torch.no_grad():
        train_features = feature_net(X_t)
        train_pred = hf_head(train_features).cpu().numpy().flatten()
        test_features = feature_net(X_test_t)
        test_pred = hf_head(test_features).cpu().numpy().flatten()

    return r2_score(y_hf_train, train_pred), r2_score(y_hf_test, test_pred)


def train_residual(X_train, y_lf_train, y_hf_train, X_test, y_lf_test, y_hf_test,
                   lf_epochs=200, hf_epochs=200, device='cpu'):
    """방법 3: Residual Learning - HF = LF + delta 학습"""
    input_dim = X_train.shape[1]
    hidden_dim = 64

    feature_net = FeatureNetwork(input_dim, hidden_dim).to(device)
    lf_head = DirectHead(hidden_dim).to(device)
    residual_head = ResidualHead(hidden_dim).to(device)

    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf_train, dtype=torch.float32).view(-1, 1).to(device)
    y_hf_t = torch.tensor(y_hf_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_lf_test_t = torch.tensor(y_lf_test, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: LF 학습 (feature_net + lf_head)
    optimizer = optim.Adam(list(feature_net.parameters()) + list(lf_head.parameters()), lr=1e-3)

    feature_net.train()
    lf_head.train()
    for epoch in range(lf_epochs):
        optimizer.zero_grad()
        features = feature_net(X_t)
        pred = lf_head(features)
        loss = loss_fn(pred, y_lf_t)
        loss.backward()
        optimizer.step()

    # Stage 2: Residual 학습 (feature_net 동결)
    for param in feature_net.parameters():
        param.requires_grad = False
    for param in lf_head.parameters():
        param.requires_grad = False

    # **중요: 학습 시 실제 y_lf 값 사용 (true LF values)**
    optimizer = optim.Adam(residual_head.parameters(), lr=1e-3)
    residual_head.train()

    for epoch in range(hf_epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            features = feature_net(X_t)
        # 실제 y_lf 값 사용
        pred = residual_head(features, y_lf_t)
        loss = loss_fn(pred, y_hf_t)
        loss.backward()
        optimizer.step()

    # 평가 (테스트 시에도 실제 y_lf 값 사용 가능 - 이것이 핵심!)
    feature_net.eval()
    residual_head.eval()
    with torch.no_grad():
        train_features = feature_net(X_t)
        train_pred = residual_head(train_features, y_lf_t).cpu().numpy().flatten()
        test_features = feature_net(X_test_t)
        test_pred = residual_head(test_features, y_lf_test_t).cpu().numpy().flatten()

    return r2_score(y_hf_train, train_pred), r2_score(y_hf_test, test_pred)


def train_residual_predicted_lf(X_train, y_lf_train, y_hf_train, X_test, y_lf_test, y_hf_test,
                                lf_epochs=200, hf_epochs=200, device='cpu'):
    """방법 4: Residual with Predicted LF - 예측된 LF 사용"""
    input_dim = X_train.shape[1]
    hidden_dim = 64

    feature_net = FeatureNetwork(input_dim, hidden_dim).to(device)
    lf_head = DirectHead(hidden_dim).to(device)
    residual_head = ResidualHead(hidden_dim).to(device)

    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf_train, dtype=torch.float32).view(-1, 1).to(device)
    y_hf_t = torch.tensor(y_hf_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Stage 1: LF 학습
    optimizer = optim.Adam(list(feature_net.parameters()) + list(lf_head.parameters()), lr=1e-3)

    feature_net.train()
    lf_head.train()
    for epoch in range(lf_epochs):
        optimizer.zero_grad()
        features = feature_net(X_t)
        pred = lf_head(features)
        loss = loss_fn(pred, y_lf_t)
        loss.backward()
        optimizer.step()

    # Stage 2: Residual 학습 (예측된 y_lf 사용)
    for param in feature_net.parameters():
        param.requires_grad = False
    for param in lf_head.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(residual_head.parameters(), lr=1e-3)
    residual_head.train()

    for epoch in range(hf_epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            features = feature_net(X_t)
            y_lf_pred = lf_head(features)
        # **예측된 y_lf 사용**
        pred = residual_head(features, y_lf_pred)
        loss = loss_fn(pred, y_hf_t)
        loss.backward()
        optimizer.step()

    # 평가
    feature_net.eval()
    lf_head.eval()
    residual_head.eval()
    with torch.no_grad():
        train_features = feature_net(X_t)
        train_lf_pred = lf_head(train_features)
        train_pred = residual_head(train_features, train_lf_pred).cpu().numpy().flatten()

        test_features = feature_net(X_test_t)
        test_lf_pred = lf_head(test_features)
        test_pred = residual_head(test_features, test_lf_pred).cpu().numpy().flatten()

    return r2_score(y_hf_train, train_pred), r2_score(y_hf_test, test_pred)


def main():
    print("="*60)
    print("Multi-Fidelity Approach Comparison")
    print("="*60)

    # 데이터 로드
    print("\nLoading data...")
    X, y_lf, y_hf = load_mf_data()
    print(f"Total samples: {len(X)}")
    print(f"LF range: [{y_lf.min():.4f}, {y_lf.max():.4f}]")
    print(f"HF range: [{y_hf.min():.4f}, {y_hf.max():.4f}]")

    # LF-HF 상관관계
    corr = np.corrcoef(y_lf, y_hf)[0, 1]
    print(f"LF-HF correlation: {corr:.4f}")

    # Delta (HF - LF) 통계
    delta = y_hf - y_lf
    print(f"Delta (HF-LF) range: [{delta.min():.4f}, {delta.max():.4f}]")
    print(f"Delta mean: {delta.mean():.4f}, std: {delta.std():.4f}")

    device = 'cpu'

    # K-Fold Cross Validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {
        'direct_hf': [],
        'transfer_features': [],
        'residual_true_lf': [],
        'residual_pred_lf': []
    }

    print(f"\n{n_folds}-Fold Cross Validation")
    print("="*60)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_lf_train, y_lf_test = y_lf[train_idx], y_lf[test_idx]
        y_hf_train, y_hf_test = y_hf[train_idx], y_hf[test_idx]

        # Method 1: Direct HF
        train_r2, test_r2 = train_direct_hf(X_train, y_hf_train, X_test, y_hf_test, device=device)
        results['direct_hf'].append(test_r2)
        print(f"  Direct HF:        Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

        # Method 2: Transfer Features
        train_r2, test_r2 = train_transfer_features(X_train, y_lf_train, y_hf_train,
                                                    X_test, y_lf_test, y_hf_test, device=device)
        results['transfer_features'].append(test_r2)
        print(f"  Transfer Features: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

        # Method 3: Residual (True LF)
        train_r2, test_r2 = train_residual(X_train, y_lf_train, y_hf_train,
                                           X_test, y_lf_test, y_hf_test, device=device)
        results['residual_true_lf'].append(test_r2)
        print(f"  Residual (True LF): Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

        # Method 4: Residual (Predicted LF)
        train_r2, test_r2 = train_residual_predicted_lf(X_train, y_lf_train, y_hf_train,
                                                        X_test, y_lf_test, y_hf_test, device=device)
        results['residual_pred_lf'].append(test_r2)
        print(f"  Residual (Pred LF): Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method, r2_list in results.items():
        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)
        print(f"{method:20s}: {mean_r2:.4f} ± {std_r2:.4f}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)


if __name__ == "__main__":
    main()
