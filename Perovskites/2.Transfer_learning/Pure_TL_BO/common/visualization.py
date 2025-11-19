import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
from pathlib import Path


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

    # 학습 포인트 표시 - 회색 계열
    ax1.scatter(
        train_indices_low, sorted_data['bandgap_hse06'].iloc[train_indices_low],
        s=100, color='#7F8C8D', label='Training (low, s=0.1)', zorder=10, marker='^',
        edgecolor='#34495E', linewidth=1  # 회색 삼각형
    )
    ax1.scatter(
        train_indices_high, sorted_data['bandgap_hse06'].iloc[train_indices_high],
        s=120, color='#E74C3C', label='Training (high, s=1.0)', zorder=10, marker='^',
        edgecolor='#922B21', linewidth=1  # 빨간색 삼각형
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

    ax1.set_ylabel('Bandgap (eV)', color='#2C3E50')
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


def plot_bnn_iteration_results(iteration_data: Dict, original_data: pd.DataFrame, 
                              X_low: np.ndarray, X_high: np.ndarray, show: bool = True) -> None:
    """
    BNN 반복별 결과 시각화 (Dual Surrogate 지원)
    
    Args:
        iteration_data: 단일 iteration 데이터 딕셔너리
        original_data: 원본 데이터 DataFrame
        X_low: low-fidelity 데이터
        X_high: high-fidelity 데이터
        show: plt.show() 호출 여부 (기본 True)
    """
    iter_ = iteration_data['iteration']
    y_pred = iteration_data['y_pred']
    y_std = iteration_data['y_std']
    ei = iteration_data['ei']
    best_idx = iteration_data['best_idx']
    X_grid = iteration_data['X_grid']
    fidelity = iteration_data['fidelity']
    recommended_point = iteration_data['recommended_point']
    
    # 데이터를 bandgap_hse06 기준으로 정렬
    sorted_data = original_data.sort_values('bandgap_hse06').copy()
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

    # True bandgaps
    ax1.scatter(x_idx, sorted_data['bandgap_hse06'], s=35, label='True bandgap (HSE06)', 
                color='navy', marker='o', alpha=0.6)
    if 'bandgap_gga' in sorted_data.columns:
        ax1.scatter(x_idx, sorted_data['bandgap_gga'], s=25, label='True bandgap (GGA)', 
                    color='lightblue', alpha=0.4, marker='s')
    
    # 두 모델의 예측값 모두 표시
    if 'y_pred_L' in iteration_data and 'y_pred_H' in iteration_data:
        # Surrogate_L 예측 (Low-fidelity 모델) - 주황색
        sorted_data['y_pred_L'] = iteration_data['y_pred_L'][sorted_data.index]
        sorted_data['y_std_L'] = iteration_data['y_std_L'][sorted_data.index]
        ax1.plot(x_idx, sorted_data['y_pred_L'], color='orange', alpha=0.7, linewidth=1.2, 
                 label='Surrogate_L pred', linestyle='--')
        ax1.fill_between(
            x_idx,
            sorted_data['y_pred_L'] - sorted_data['y_std_L'],
            sorted_data['y_pred_L'] + sorted_data['y_std_L'],
            color='orange', alpha=0.15
        )
        
        # Surrogate_H 예측 (High-fidelity 모델) - 빨간색
        sorted_data['y_pred_H'] = iteration_data['y_pred_H'][sorted_data.index]
        sorted_data['y_std_H'] = iteration_data['y_std_H'][sorted_data.index]
        ax1.plot(x_idx, sorted_data['y_pred_H'], color='red', alpha=0.9, linewidth=1.5, 
                 label='Surrogate_H pred', linestyle='-')
        ax1.fill_between(
            x_idx,
            sorted_data['y_pred_H'] - sorted_data['y_std_H'],
            sorted_data['y_pred_H'] + sorted_data['y_std_H'],
            color='red', alpha=0.15
        )
    else:
        # 기존 방식 (호환성 유지)
        ax1.scatter(x_idx, sorted_data['y_pred'], s=40, label='BNN prediction', color='orange', alpha=0.7)
        ax1.fill_between(
            x_idx,
            sorted_data['y_pred'] - sorted_data['y_std'],
            sorted_data['y_pred'] + sorted_data['y_std'],
            color='orange', alpha=0.2, label='Pred. std. dev.'
        )

    # 학습 포인트 표시
    ax1.scatter(
        train_indices_low, sorted_data['bandgap_hse06'].iloc[train_indices_low],
        s=100, color='grey', label='Training (low, s=0.1)', zorder=10, marker='^',
        edgecolor='black', linewidth=1
    )
    ax1.scatter(
        train_indices_high, sorted_data['bandgap_hse06'].iloc[train_indices_high],
        s=120, color='red', label='Training (high, s=1.0)', zorder=10, marker='^',
        edgecolor='darkred', linewidth=1
    )

    # Global optimal 별표
    optimal_combo = '12,2,4'
    if optimal_combo in sorted_data['combo'].values:
        optimal_idx = sorted_data.index[sorted_data['combo'] == optimal_combo].tolist()[0]
        optimal_idx_in_sorted = sorted_data.index.get_loc(optimal_idx)
        optimal_bandgap = sorted_data.loc[optimal_idx, 'bandgap_hse06']
        ax1.scatter(
            optimal_idx_in_sorted, optimal_bandgap,
            marker='*', color='gold', s=300, edgecolor='darkorange',
            linewidth=2, label='Global optimum', zorder=20
        )

    ax1.set_ylabel('Bandgap (eV)', color='navy')
    ax1.set_xlabel('Combinations (organic, cation, anion)')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(sorted_data['combo'], rotation=90, fontsize=7)

    # 제목 강조 - Dual Surrogate 정보 추가
    fidelity_name = 'HIGH (Surrogate_H)' if fidelity == 1.0 else 'LOW (Surrogate_L)'
    if (iter_ % 8 == 0):
        ax1.set_title(f'BNN Dual Surrogate: {fidelity_name}\niter: {iter_}, recommended: {recommended_point}',
                      color='crimson', fontsize=16, fontweight='bold', backgroundcolor='#ffe6e6')
    else:
        ax1.set_title(f'BNN Dual Surrogate: {fidelity_name}\niter: {iter_}, recommended: {recommended_point}',
                      fontsize=14)
    ax1.tick_params(axis='y', labelcolor='navy')

    # EI 오른쪽축 - 검은색
    ax2 = ax1.twinx()
    ax2.plot(x_idx, ei_sorted, marker='o', color='black', label='EI', linewidth=1.5, 
             markersize=2, alpha=0.7)
    # best_idx를 정렬된 인덱스에 맞춰 변환
    best_idx_in_sorted = sorted_data.index.get_loc(best_idx)
    
    # 추천점을 실제 밴드갭 값 위치에 표시 (EI가 아닌 실제 값)
    recommended_bandgap_hse = sorted_data.loc[best_idx, 'bandgap_hse06']
    ax1.scatter(best_idx_in_sorted, recommended_bandgap_hse, 
                color='magenta', s=200, zorder=25, marker='D', 
                edgecolor='purple', linewidth=2,
                label='Recommended point')
    
    # EI 값도 표시 (작게)
    ax2.scatter(best_idx_in_sorted, ei_sorted[best_idx], 
                color='darkgreen', s=80, zorder=15, marker='^',
                label='Max EI', edgecolor='green', linewidth=1)
    ax2.set_ylabel('Expected Improvement (EI)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 범례
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right')

    plt.xlim(-1, len(sorted_data))
    plt.tight_layout()
    
    if show:
        plt.show()


def plot_bnn_optimization_summary(result: Dict) -> None:
    """
    BNN 최적화 결과 종합 시각화
    
    Args:
        result: BNN 최적화 결과 딕셔너리
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Convergence curve
    if 'best_so_far_curve' in result:
        best_curve = result['best_so_far_curve']
        iterations = [x[1] for x in best_curve]
        best_values = [x[3] for x in best_curve]
        axes[0, 0].plot(iterations, best_values, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Best-so-far Value')
        axes[0, 0].set_title('BNN Convergence')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1.5249, color='red', linestyle='--', alpha=0.7, label='Target')
        axes[0, 0].legend()
    
    # 2. Cost accumulation
    if 'cost_data' in result:
        cost_data = result['cost_data']
        iterations = [x[1] for x in cost_data]
        costs = [x[2] for x in cost_data]
        axes[0, 1].plot(iterations, costs, 'g-', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cumulative Cost')
        axes[0, 1].set_title('Cost Accumulation')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Fidelity selection pattern
    if 'visualization_data' in result:
        fidelities = [data['fidelity'] for data in result['visualization_data']]
        low_count = fidelities.count(0.1)
        high_count = fidelities.count(1.0)
        
        axes[1, 0].bar(['Low (0.1)', 'High (1.0)'], [low_count, high_count], 
                      color=['skyblue', 'salmon'], edgecolor='black')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Fidelity Selection Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 비율 표시
        total = low_count + high_count
        axes[1, 0].text(0, low_count + 1, f'{low_count/total:.1%}', ha='center')
        axes[1, 0].text(1, high_count + 1, f'{high_count/total:.1%}', ha='center')
    
    # 4. 하이퍼파라미터 최적화 결과
    if result.get('use_hyperparameter_bo') and 'hyperparameter_history' in result:
        hp_history = result['hyperparameter_history']
        if hp_history:
            iterations = [h['iteration'] for h in hp_history]
            axes[1, 1].scatter(iterations, [1]*len(iterations), s=100, 
                              color='purple', alpha=0.7, marker='*')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Hyperparameter BO Events')
            axes[1, 1].set_title('Hyperparameter Optimization')
            axes[1, 1].set_ylim(0.5, 1.5)
            axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Hyperparameter\nOptimization', 
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=12, style='italic')
        axes[1, 1].set_title('Hyperparameter Optimization')
    
    # 전체 제목
    model_type = result.get('model_type', 'BNN')
    final_best = result.get('best_so_far', 'N/A')
    total_cost = result.get('total_cost', 'N/A')
    fig.suptitle(f'{model_type} Optimization Results\nFinal Best: {final_best:.4f}, Total Cost: {total_cost:.2f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_optimization_results(result: Dict, save_path: Optional[str] = None) -> None:
    """
    최적화 결과를 종합적으로 시각화 (기존 DNGO 방식 유지)
    
    Args:
        result: 최적화 결과 딕셔너리
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    model_type = result.get('model_type', 'DNGO')
    
    # 1. Convergence plot
    if 'best_values_history' in result:
        axes[0, 0].plot(result['best_values_history'], 'b-', linewidth=2, marker='o')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Best Value')
        axes[0, 0].set_title(f'{model_type} Convergence')
        axes[0, 0].grid(True, alpha=0.3)
        # 목표선 추가
        axes[0, 0].axhline(y=1.5249, color='red', linestyle='--', alpha=0.7, label='Target')
        axes[0, 0].legend()
    
    # 2. Cost accumulation  
    if 'cost_history' in result:
        cumulative_cost = np.cumsum(result['cost_history'])
        axes[0, 1].plot(cumulative_cost, 'g-', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cumulative Cost')
        axes[0, 1].set_title('Cost Accumulation')
        axes[0, 1].grid(True, alpha=0.3)
    elif 'cost_data' in result:
        # BNN 방식의 cost_data 사용
        iterations = [x[1] for x in result['cost_data']]
        costs = [x[2] for x in result['cost_data']]
        axes[0, 1].plot(iterations, costs, 'g-', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cumulative Cost')
        axes[0, 1].set_title('Cost Accumulation')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Fidelity selection
    if 'fidelity_history' in result:
        fidelities = result['fidelity_history']
        unique_fidelities = list(set(fidelities))
        counts = [fidelities.count(f) for f in unique_fidelities]
        
        axes[1, 0].bar([f'Fidelity {f}' for f in unique_fidelities], counts, 
                      color=['skyblue', 'salmon'], edgecolor='black')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Fidelity Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 비율 표시
        total = sum(counts)
        for i, (f, count) in enumerate(zip(unique_fidelities, counts)):
            axes[1, 0].text(i, count + max(counts)*0.05, f'{count/total:.1%}', ha='center')
    
    # 4. EI history
    if 'ei_history' in result:
        axes[1, 1].plot(result['ei_history'], 'purple', linewidth=2, marker='d')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Expected Improvement')
        axes[1, 1].set_title('Acquisition Function Value')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 전체 제목
    final_best = result.get('best_so_far', 'N/A')
    total_cost = result.get('total_cost', 'N/A')
    fig.suptitle(f'{model_type} Optimization Results\nFinal Best: {final_best:.4f}, Total Cost: {total_cost:.2f}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_bnn_full_analysis(model, blr, ori_data: pd.DataFrame, X_grid: np.ndarray, 
                          X_train: np.ndarray, y_train: np.ndarray, 
                          train_indices: np.ndarray, config_params: Dict,
                          ei_values: np.ndarray = None, best_idx: int = None,
                          save_dir: str = None) -> None:
    """
    BNN을 활용한 전체 분석 시각화 (Transfer Learning DNN + BLR 방식과 동일)
    
    Args:
        model: BNN 모델 (또는 TransferLearningDNN)
        blr: Bayesian Linear Regression 모델
        ori_data: 원본 데이터 DataFrame (combo, bandgap 컬럼 포함)
        X_grid: 전체 조합 그리드 (192, 3)
        X_train: 학습 데이터 조합
        y_train: 학습 데이터 타겟값
        train_indices: 학습에 사용된 인덱스
        config_params: 설정 파라미터 딕셔너리
        ei_values: Expected Improvement 값들 (선택사항)
        best_idx: 추천 포인트 인덱스 (선택사항)
        save_dir: 이미지 저장 디렉토리 (None이면 저장하지 않음)
    """
    # 모델 특성 추출 (BNN 또는 DNN)
    if hasattr(model, 'extract_features'):
        features_grid = model.extract_features(X_grid)
    else:
        # BNN의 경우 predict 메서드 사용
        y_pred_mean, y_pred_std = model.predict(X_grid)
        features_grid = X_grid  # 또는 적절한 feature extraction
    
    # BLR 예측
    y_pred = []
    y_std = []
    for phi in features_grid:
        mu, var = blr.predict(phi)
        y_pred.append(mu)
        y_std.append(np.sqrt(var))
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)
    
    # 원본 데이터에 예측값 추가
    ori_data_copy = ori_data.copy()
    ori_data_copy['y_pred'] = y_pred
    ori_data_copy['y_std'] = y_std
    
    # 시각화
    fig, ax1 = plt.subplots(figsize=(18, 7))
    x_idx = range(len(ori_data_copy))
    
    # True bandgap - HSE06과 GGA 둘 다 표시
    ax1.scatter(x_idx, ori_data_copy['bandgap'], s=40, label='True bandgap (HSE06)', color='royalblue')
    # GGA 밴드갭도 표시 (존재하는 경우)
    if 'bandgap_gga' in ori_data_copy.columns:
        ax1.scatter(x_idx, ori_data_copy['bandgap_gga'], s=30, label='True bandgap (GGA)', color='lightblue', alpha=0.6, marker='s')
    # 예측값
    ax1.scatter(x_idx, ori_data_copy['y_pred'], s=40, label='BNN+BLR prediction', color='orange', alpha=0.7)
    # 불확실성 범위
    ax1.fill_between(
        x_idx,
        ori_data_copy['y_pred'] - ori_data_copy['y_std'],
        ori_data_copy['y_pred'] + ori_data_copy['y_std'],
        color='orange', alpha=0.2, label='Pred. std. dev.'
    )
    
    # 학습에 사용된 포인트 인덱스 (검정색)
    train_indices_sorted = sorted(train_indices)
    ax1.scatter(
        train_indices_sorted, ori_data_copy['bandgap'].iloc[train_indices_sorted],
        s=110, color='black', label=f'Training points ({len(train_indices)})', zorder=10, marker='o'
    )
    
    # Global optimal 별표
    optimal_combo = '12,2,4'
    if optimal_combo in ori_data_copy['combo'].values:
        optimal_idx = ori_data_copy.index[ori_data_copy['combo'] == optimal_combo].tolist()[0]
        optimal_bandgap = ori_data_copy.loc[optimal_idx, 'bandgap']
        ax1.scatter(
            optimal_idx, optimal_bandgap,
            marker='*', color='purple', s=250, edgecolor='black',
            label='Global optimum', zorder=20
        )
    
    ax1.set_ylabel('Bandgap (HSE06)', color='navy')
    ax1.set_xlabel('Combinations (organic, cation, anion)')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(ori_data_copy['combo'], rotation=90, fontsize=7)
    ax1.tick_params(axis='y', labelcolor='navy')
    
    # EI와 추천점 표시 (제공된 경우)
    if ei_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(x_idx, ei_values, marker='o', color='forestgreen', label='EI', linewidth=2)
        ax2.set_ylabel('Expected Improvement (EI)', color='forestgreen')
        ax2.tick_params(axis='y', labelcolor='forestgreen')
        
        # 추천 포인트 강조
        if best_idx is not None:
            ax2.scatter(best_idx, ei_values[best_idx], color='red', s=120, zorder=15, 
                       label='Recommended (max EI)')
            
        # 범례 통합
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    # 제목 설정
    title_parts = [f'BNN Analysis: True Bandgap, Prediction, Uncertainty ({len(train_indices)} training points)']
    if ei_values is not None:
        title_parts.append('with Expected Improvement')
    if best_idx is not None:
        recommended_combo = ori_data_copy['combo'].iloc[best_idx]
        title_parts.append(f'Recommended: {recommended_combo}')
    
    ax1.set_title(' - '.join(title_parts))
    ax1.set_xlim(-1, len(ori_data_copy))
    plt.tight_layout()
    
    # 이미지 저장
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config_params.get('model_type', 'BNN')
        num_train = len(train_indices)
        epochs = config_params.get('epochs', 'unknown')
        
        folder_name = f"{model_name}_train{num_train}_epochs{epochs}_{timestamp}"
        full_save_dir = os.path.join(save_dir, folder_name)
        os.makedirs(full_save_dir, exist_ok=True)
        
        filename = f"bnn_full_analysis_{timestamp}.png"
        plt.savefig(os.path.join(full_save_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Image saved to: {os.path.join(full_save_dir, filename)}")
    
    plt.show()
    
    # 성능 메트릭 계산 및 출력
    y_true = ori_data_copy['bandgap'].values
    y_pred_vals = ori_data_copy['y_pred'].values
    
    r2 = r2_score(y_true, y_pred_vals)
    mae = mean_absolute_error(y_true, y_pred_vals)
    
    print(f"BNN+BLR Performance:")
    print(f"R² score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Training samples: {len(train_indices)}")
    
    # 성능 메트릭도 저장
    if save_dir:
        metrics_file = os.path.join(full_save_dir, f"metrics_{timestamp}.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"BNN+BLR Analysis Results\n")
            f.write(f"========================\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Training samples: {len(train_indices)}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"R² score: {r2:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"Timestamp: {timestamp}\n")
            
            # 설정 파라미터도 저장
            f.write(f"\nConfiguration:\n")
            for key, value in config_params.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Metrics saved to: {metrics_file}")
    
    return full_save_dir if save_dir else None 