#!/usr/bin/env python3
"""
간단한 Multi-Fidelity 학습 테스트
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Local imports
from common.data_utils import load_lookup_table, create_label_maps, create_param_space
from common.config import DATA_PATHS


class LFNetwork(nn.Module):
    """Low-Fidelity 네트워크"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        features = self.feature_net(x)
        return self.out_layer(features)


class HFNetwork(nn.Module):
    """High-Fidelity 네트워크"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        hf_input_dim = input_dim + 1
        layers = []
        in_dim = hf_input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        features = self.feature_net(combined)
        return self.out_layer(features)


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


def train_sequential(lf_net, hf_net, X_train, y_lf_train, y_hf_train,
                     lf_epochs=200, hf_epochs=100, lf_lr=1e-3, hf_lr=1e-4, device='cpu'):
    """Sequential 학습"""
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf_train, dtype=torch.float32).view(-1, 1).to(device)
    y_hf_t = torch.tensor(y_hf_train, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: LF 학습
    print("Stage 1: Training LF network...")
    lf_optimizer = optim.Adam(lf_net.parameters(), lr=lf_lr)
    lf_net.train()

    for epoch in range(lf_epochs):
        lf_optimizer.zero_grad()
        y_lf_pred = lf_net(X_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        lf_loss.backward()
        lf_optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{lf_epochs}, LF Loss: {lf_loss.item():.4f}")

    # LF 평가
    lf_net.eval()
    with torch.no_grad():
        y_lf_pred = lf_net(X_t).cpu().numpy().flatten()
    lf_r2 = r2_score(y_lf_train, y_lf_pred)
    print(f"  LF Train R²: {lf_r2:.4f}")

    # Stage 2: HF 학습 (LF 동결)
    print("\nStage 2: Training HF network (LF frozen)...")
    for param in lf_net.parameters():
        param.requires_grad = False

    hf_optimizer = optim.Adam(hf_net.parameters(), lr=hf_lr)
    hf_net.train()

    for epoch in range(hf_epochs):
        hf_optimizer.zero_grad()

        with torch.no_grad():
            y_lf_for_hf = lf_net(X_t)

        y_hf_pred = hf_net(X_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)
        hf_loss.backward()
        hf_optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{hf_epochs}, HF Loss: {hf_loss.item():.4f}")

    # HF 평가
    hf_net.eval()
    with torch.no_grad():
        y_lf_for_hf = lf_net(X_t)
        y_hf_pred = hf_net(X_t, y_lf_for_hf).cpu().numpy().flatten()
    hf_r2 = r2_score(y_hf_train, y_hf_pred)
    print(f"  HF Train R²: {hf_r2:.4f}")

    return lf_r2, hf_r2


def main():
    print("="*60)
    print("Multi-Fidelity Training Test")
    print("="*60)

    # 데이터 로드
    print("\nLoading data...")
    X, y_lf, y_hf = load_mf_data()
    print(f"Total samples: {len(X)}")
    print(f"LF range: [{y_lf.min():.4f}, {y_lf.max():.4f}]")
    print(f"HF range: [{y_hf.min():.4f}, {y_hf.max():.4f}]")

    # Train/Test 분할
    X_train, X_test, y_lf_train, y_lf_test, y_hf_train, y_hf_test = train_test_split(
        X, y_lf, y_hf, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    device = 'cpu'
    input_dim = X.shape[1]

    # 모델 생성
    lf_net = LFNetwork(input_dim, hidden_dim=64, num_layers=2).to(device)
    hf_net = HFNetwork(input_dim, hidden_dim=64, num_layers=2).to(device)

    # Sequential 학습
    print("\n" + "="*60)
    print("Sequential Training")
    print("="*60)
    train_sequential(lf_net, hf_net, X_train, y_lf_train, y_hf_train, device=device)

    # 테스트 평가
    print("\n" + "="*60)
    print("Test Evaluation")
    print("="*60)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    lf_net.eval()
    hf_net.eval()

    with torch.no_grad():
        y_lf_pred_test = lf_net(X_test_t).cpu().numpy().flatten()
        y_lf_for_hf_test = lf_net(X_test_t)
        y_hf_pred_test = hf_net(X_test_t, y_lf_for_hf_test).cpu().numpy().flatten()

    lf_test_r2 = r2_score(y_lf_test, y_lf_pred_test)
    hf_test_r2 = r2_score(y_hf_test, y_hf_pred_test)

    print(f"LF Test R²: {lf_test_r2:.4f}")
    print(f"HF Test R²: {hf_test_r2:.4f}")

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
