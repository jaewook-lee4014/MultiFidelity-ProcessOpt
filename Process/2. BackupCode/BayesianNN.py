#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:54:00 2025

@author: k23070952
"""

import torch
import torch.nn as nn
import torch.optim as optim

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.3):
        super(BayesianNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # MSP 예측

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)  # 확률적 예측값
