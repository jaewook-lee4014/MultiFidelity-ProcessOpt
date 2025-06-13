import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Dict, List


def plot_iteration_results(ori_data: pd.DataFrame, y_pred: np.ndarray, y_std: np.ndarray, 
                          ei: np.ndarray, best_idx: int, X_grid: np.ndarray, 
                          X_low: np.ndarray, X_high: np.ndarray, iter_: int) -> None:
    """
    반복별 결과 시각화 (정렬된 bandgap 기준)
    
    Args:
        ori_data: 원본 데이터 DataFrame
        y_pred: 예측값
        y_std: 예측 표준편차
        ei: Expected Improvement 값
        best_idx: 최적 인덱스
        X_grid: 전체 조합 그리드
        X_low: low-fidelity 데이터
        X_high: high-fidelity 데이터
        iter_: 현재 반복 횟수
    """
    # 데이터를 bandgap_hse06 기준으로 정렬
    sorted_data = ori_data.sort_values('bandgap_hse06').copy()
    sorted_data['y_pred'] = y_pred[sorted_data.index]
    sorted_data['y_std'] = y_std[sorted_data.index]
    
    # 정렬된 인덱스에 맞춰 ei도 재정렬
    ei_sorted = ei[sorted_data.index]
    
    # 학습에 사용된 조합 set 만들기
    train_combo_set = set(tuple(map(int, row)) for row in np.vstack([X_low, X_high]))

    # 전체 조합 중 학습에 쓰인 인덱스 찾기 (정렬된 인덱스 기준)
    train_indices_low = [i for i, combo in enumerate(X_grid[sorted_data.index].astype(int)) 
                        if tuple(combo) in set(tuple(map(int, row)) for row in X_low)]
    train_indices_high = [i for i, combo in enumerate(X_grid[sorted_data.index].astype(int)) 
                         if tuple(combo) in set(tuple(map(int, row)) for row in X_high)]

    fig, ax1 = plt.subplots(figsize=(18, 7))
    x_idx = range(len(sorted_data))

    # True / 예측 / Uncertainty
    ax1.scatter(x_idx, sorted_data['bandgap_hse06'], s=40, label='True bandgap', color='royalblue')
    ax1.scatter(x_idx, sorted_data['y_pred'], s=40, label='BLR prediction', color='orange', alpha=0.7)
    ax1.fill_between(
        x_idx,
        sorted_data['y_pred'] - sorted_data['y_std'],
        sorted_data['y_pred'] + sorted_data['y_std'],
        color='orange', alpha=0.2, label='Pred. std. dev.'
    )

    # 학습 포인트 표시
    ax1.scatter(
        train_indices_low, sorted_data['bandgap_hse06'].iloc[train_indices_low],
        s=110, color='black', label='Training (low, s=0.1)', zorder=10, marker='^'
    )
    ax1.scatter(
        train_indices_high, sorted_data['bandgap_hse06'].iloc[train_indices_high],
        s=110, color='crimson', label='Training (high, s=1.0)', zorder=10, marker='^'
    )

    # Global optimal 별표
    optimal_combo = '12,2,4'
    optimal_idx = sorted_data.index[sorted_data['combo'] == optimal_combo].tolist()[0]
    optimal_idx_in_sorted = sorted_data.index.get_loc(optimal_idx)
    optimal_bandgap = sorted_data.loc[optimal_idx, 'bandgap_hse06']
    ax1.scatter(
        optimal_idx_in_sorted, optimal_bandgap,
        marker='*', color='purple', s=250, edgecolor='black',
        label='Global optimum', zorder=20
    )

    ax1.set_ylabel('Bandgap (hse06)', color='navy')
    ax1.set_xlabel('Combinations (organic, cation, anion)')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(sorted_data['combo'], rotation=90, fontsize=7)

    # 제목 강조
    if (iter_ % 8 == 0):
        ax1.set_title(f'True Bandgap (sorted), Prediction, Uncertainty, and EI\niter: {iter_}',
                      color='crimson', fontsize=18, fontweight='bold', backgroundcolor='#ffe6e6')
    else:
        ax1.set_title(f'True Bandgap (sorted), Prediction, Uncertainty, and EI\niter: {iter_}')
    ax1.tick_params(axis='y', labelcolor='navy')

    # EI 오른쪽축
    ax2 = ax1.twinx()
    ax2.plot(x_idx, ei_sorted, marker='o', color='forestgreen', label='EI', linewidth=2)
    # best_idx를 정렬된 인덱스에 맞춰 변환
    best_idx_in_sorted = sorted_data.index.get_loc(best_idx)
    ax2.scatter(best_idx_in_sorted, ei_sorted[best_idx], color='red', s=120, zorder=15, label='Recommended (max EI)')
    ax2.set_ylabel('Expected Improvement (EI)', color='forestgreen')
    ax2.tick_params(axis='y', labelcolor='forestgreen')

    # 범례
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right')

    plt.xlim(-1, len(sorted_data))
    plt.tight_layout()
    plt.show()


def plot_prediction_scatter(ori_data: pd.DataFrame, X_grid: np.ndarray, X_low: np.ndarray, X_high: np.ndarray) -> None:
    """
    실제값 vs 예측값 산점도
    
    Args:
        ori_data: 원본 데이터 (y_pred, y_std 포함)
        X_grid: 전체 조합 그리드
        X_low: low-fidelity 데이터
        X_high: high-fidelity 데이터
    """
    y_true = ori_data['bandgap_hse06'].values
    y_pred = ori_data['y_pred'].values

    # R², MAE 계산
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"R² score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # 학습 데이터 인덱스
    train_combo_set = set(tuple(map(int, row)) for row in np.vstack([X_low, X_high]))
    train_indices = set([i for i, combo in enumerate(X_grid.astype(int)) if tuple(combo) in train_combo_set])
    all_indices = set(range(len(y_true)))
    non_train_indices = all_indices - train_indices

    # 산점도
    plt.figure(figsize=(8, 8))
    
    # 비학습 데이터 (연한색)
    plt.scatter(
        y_true[list(non_train_indices)], y_pred[list(non_train_indices)],
        alpha=0.4, s=40, color='grey', label='Unmeasured (candidates)'
    )
    
    # 학습 데이터 (진한색)
    plt.scatter(
        y_true[list(train_indices)], y_pred[list(train_indices)],
        alpha=0.9, s=80, color='black', label='Training points', edgecolor='w'
    )
    
    # 기준선
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal: y=x')

    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.title(f'Actual vs. Predicted\nR²: {r2:.3f}, MAE: {mae:.3f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_convergence_curve(best_so_far_curve: List, title: str = "Convergence Curve") -> None:
    """
    수렴 곡선 시각화
    
    Args:
        best_so_far_curve: best-so-far 기록 리스트
        title: 그래프 제목
    """
    iterations = [x[1] for x in best_so_far_curve]
    best_values = [x[3] for x in best_so_far_curve]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_values, 'b-', linewidth=2, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Best-so-far value')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cost_analysis(cost_data: List, title: str = "Cost Analysis") -> None:
    """
    비용 분석 시각화
    
    Args:
        cost_data: 비용 데이터 리스트
        title: 그래프 제목
    """
    iterations = [x[1] for x in cost_data]
    cumulative_costs = [x[2] for x in cost_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cumulative_costs, 'g-', linewidth=2, marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Cost')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(model, title: str = "Learning Curves") -> None:
    """
    학습 곡선 시각화
    
    Args:
        model: TransferLearningDNN 모델
        title: 그래프 제목
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pretrain loss
    if model.pretrain_losses:
        ax1.plot(model.pretrain_losses, 'b-', linewidth=2)
        ax1.set_title('Pretrain Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.grid(True)
    
    # Finetune loss
    if model.finetune_losses:
        ax2.plot(model.finetune_losses, 'r-', linewidth=2)
        ax2.set_title('Finetune Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_multiple_runs_summary(results_df: pd.DataFrame) -> None:
    """
    여러 실행 결과 요약 시각화
    
    Args:
        results_df: 결과 DataFrame (run, total_cost 컬럼 포함)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 히스토그램
    ax1.hist(results_df['total_cost'], bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(results_df['total_cost'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["total_cost"].mean():.2f}')
    ax1.axvline(results_df['total_cost'].median(), color='green', linestyle='--', 
                label=f'Median: {results_df["total_cost"].median():.2f}')
    ax1.set_xlabel('Total Cost')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Total Costs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 실행별 비용
    ax2.plot(results_df['run'], results_df['total_cost'], 'bo-', alpha=0.6)
    ax2.axhline(results_df['total_cost'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["total_cost"].mean():.2f}')
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Total Cost')
    ax2.set_title('Total Cost per Run')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 통계 요약 출력
    print(f"Total runs: {len(results_df)}")
    print(f"Mean cost: {results_df['total_cost'].mean():.2f}")
    print(f"Median cost: {results_df['total_cost'].median():.2f}")
    print(f"Std cost: {results_df['total_cost'].std():.2f}")
    print(f"Min cost: {results_df['total_cost'].min():.2f}")
    print(f"Max cost: {results_df['total_cost'].max():.2f}") 