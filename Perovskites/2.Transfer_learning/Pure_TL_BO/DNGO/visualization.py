"""
DNGO 최적화 과정 시각화 모듈

매 스텝별 선택 과정, 모델 피팅 결과, 예측값 vs 실제값 등을 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score


def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    베이지안 모델 성능 지표 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        
    Returns:
        성능 지표 딕셔너리 (MSE, RMSE, R^2)
    """
    # NaN 값 제거
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'r2': np.nan}
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    return {'mse': mse, 'rmse': rmse, 'r2': r2}


def plot_prediction_vs_actual(model, blr, X_grid: np.ndarray, X_measured: np.ndarray, 
                              y_measured: np.ndarray, title: str = "Prediction vs Actual",
                              ax: Optional[plt.Axes] = None, y_actual_all: Optional[np.ndarray] = None) -> plt.Axes:
    """
    예측값 vs 실제값 산점도 플롯
    
    Args:
        model: 학습된 DNN 모델
        blr: 학습된 BLR 모델
        X_grid: 전체 후보 그리드
        X_measured: 측정된 점들의 입력
        y_measured: 측정된 점들의 출력
        title: 플롯 제목
        ax: matplotlib axes (없으면 새로 생성)
    
    Returns:
        플롯이 그려진 axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # 전체 그리드에 대한 예측
    features_grid = model.extract_features(X_grid)
    y_pred_grid = []
    for phi in features_grid:
        mu, _ = blr.predict(phi)
        y_pred_grid.append(mu)
    y_pred_grid = np.array(y_pred_grid)
    
    # 측정된 점들에 대한 예측
    if len(X_measured) > 0:
        features_measured = model.extract_features(X_measured)
        y_pred_measured = []
        for phi in features_measured:
            mu, _ = blr.predict(phi)
            y_pred_measured.append(mu)
        y_pred_measured = np.array(y_pred_measured)
    else:
        y_pred_measured = np.array([])
    
    # 실제값 범위 계산 (그리드의 예측값 사용)
    y_min = min(y_pred_grid.min(), y_measured.min() if len(y_measured) > 0 else y_pred_grid.min())
    y_max = max(y_pred_grid.max(), y_measured.max() if len(y_measured) > 0 else y_pred_grid.max())
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    # Unmeasured candidates (연한 점)
    # 측정된 점들의 인덱스 찾기
    measured_indices = set()
    for x_m in X_measured:
        for i, x_g in enumerate(X_grid):
            if np.allclose(x_m, x_g):
                measured_indices.add(i)
                break
    
    unmeasured_indices = [i for i in range(len(X_grid)) if i not in measured_indices]
    
    # Unmeasured candidates - 실제값이 있으면 사용, 없으면 스킵
    if unmeasured_indices and y_actual_all is not None:
        # 실제값을 x축에, 예측값을 y축에 표시
        ax.scatter(y_actual_all[unmeasured_indices], y_pred_grid[unmeasured_indices], 
                  alpha=0.3, s=20, color='gray', label='Unmeasured candidates')
    
    # Measured points (진한 점)
    r2_measured_str = ""
    r2_unmeasured_str = ""
    
    if len(y_pred_measured) > 0:
        ax.scatter(y_measured, y_pred_measured, 
                  alpha=0.8, s=50, color='blue', label='Measured points', zorder=5)
        
        # Measured points에 대한 성능 지표 계산
        metrics_measured = calculate_performance_metrics(y_measured, y_pred_measured)
        r2_measured_str = f"R²(train) = {metrics_measured['r2']:.3f}" if not np.isnan(metrics_measured['r2']) else "R²(train) = N/A"
    
    # Unmeasured candidates에 대한 R² 계산 (실제값이 있는 경우)
    if unmeasured_indices and y_actual_all is not None:
        # Unmeasured candidates의 실제값과 예측값
        y_actual_unmeasured = y_actual_all[unmeasured_indices]
        y_pred_unmeasured = y_pred_grid[unmeasured_indices]
        
        # NaN이 아닌 값들만 사용
        mask = ~(np.isnan(y_actual_unmeasured) | np.isnan(y_pred_unmeasured))
        if np.sum(mask) > 0:
            metrics_unmeasured = calculate_performance_metrics(
                y_actual_unmeasured[mask], 
                y_pred_unmeasured[mask]
            )
            r2_unmeasured_str = f"R²(test) = {metrics_unmeasured['r2']:.3f}" if not np.isnan(metrics_unmeasured['r2']) else "R²(test) = N/A"
    
    # 타이틀 구성
    r2_parts = [part for part in [r2_measured_str, r2_unmeasured_str] if part]
    if r2_parts:
        r2_combined = ", ".join(r2_parts)
        title_with_r2 = f"{title} ({r2_combined})"
    else:
        title_with_r2 = title
    
    # y=x 직선 (빨간색)
    ax.plot([y_min, y_max], [y_min, y_max], 'r-', alpha=0.5, label='y=x')
    
    # 축 설정
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Actual Value', fontsize=10)
    ax.set_ylabel('Predicted Value', fontsize=10)
    ax.set_title(title_with_r2, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_step_visualization(viz_data: Dict, save_path: Optional[str] = None,
                           ori_data: Optional[Dict] = None):
    """
    매 스텝의 선택 과정과 모델 피팅 결과를 시각화
    
    상단 좌우: Low-fidelity와 High-fidelity 예측 vs 실제값
    하단: 기존 plot_iteration_results 스타일의 시각화
    
    Args:
        viz_data: 시각화 데이터 (single step)
        save_path: 저장 경로 (None이면 저장하지 않음)
        ori_data: 원본 데이터 DataFrame (combo, bandgap 정보 포함)
    """
    # Figure 생성 (3개 subplot)
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], width_ratios=[1, 1])
    
    # 전체 실제값 가져오기 (있는 경우)
    y_actual_all = viz_data.get('y_actual', None)
    
    # Subplot 1: Low-fidelity 예측 vs 실제값 (상단 왼쪽)
    ax1 = fig.add_subplot(gs[0, 0])
    blr_low = viz_data.get('blr_low', viz_data.get('blr'))  # LOFI 전용 모델 사용
    if len(viz_data['X_low']) > 0 and blr_low is not None:
        plot_prediction_vs_actual(
            viz_data['model'], blr_low,
            viz_data['X_grid'], viz_data['X_low'], viz_data['y_low'],
            title=f"Low-fidelity Model (n={len(viz_data['y_low'])})",
            ax=ax1,
            y_actual_all=y_actual_all
        )
    else:
        ax1.text(0.5, 0.5, 'No low-fidelity data yet', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Low-fidelity Model', fontsize=12, fontweight='bold')
    
    # Subplot 2: High-fidelity 예측 vs 실제값 (상단 오른쪽)
    ax2 = fig.add_subplot(gs[0, 1])
    blr_high = viz_data.get('blr_high', viz_data.get('blr'))  # HIFI 전용 모델 사용
    if len(viz_data['X_high']) > 0 and blr_high is not None:
        plot_prediction_vs_actual(
            viz_data['model'], blr_high,
            viz_data['X_grid'], viz_data['X_high'], viz_data['y_high'],
            title=f"High-fidelity Model (n={len(viz_data['y_high'])})",
            ax=ax2,
            y_actual_all=y_actual_all
        )
    else:
        ax2.text(0.5, 0.5, 'No high-fidelity data yet', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('High-fidelity Model', fontsize=12, fontweight='bold')
    
    # Subplot 3: 기존 plot_iteration_results 스타일 (하단 전체)
    ax3 = fig.add_subplot(gs[1, :])
    
    # 실제값 기준으로 정렬
    if 'y_actual' in viz_data:
        y_actual = viz_data['y_actual']
        # 실제값이 큰 것부터 작은 것 순으로 정렬 (내림차순)
        sorted_indices = np.argsort(-y_actual)  # 음수를 취해서 내림차순 정렬
    else:
        # 실제값이 없으면 예측값 기준으로 정렬
        sorted_indices = np.argsort(-viz_data['y_pred'])
    
    # 정렬된 순서로 데이터 재배열 - 모든 데이터가 같은 순서로 정렬됨을 보장
    x_idx = range(len(viz_data['X_grid']))
    y_pred_original = viz_data['y_pred']
    y_std_original = viz_data['y_std'] 
    ei_original = viz_data['ei']
    
    y_pred = y_pred_original[sorted_indices]
    y_std = y_std_original[sorted_indices] 
    ei = ei_original[sorted_indices]
    X_grid_sorted = viz_data['X_grid'][sorted_indices]
    
    # 실제값도 정렬
    if 'y_actual' in viz_data:
        y_actual_sorted = y_actual[sorted_indices]
        # 실제값 표시 (파란 점)
        ax3.scatter(x_idx, y_actual_sorted, s=30, label='Actual', color='blue', alpha=0.6, zorder=5)
    
    # 예측값과 불확실성
    ax3.scatter(x_idx, y_pred, s=40, label='Prediction', color='orange', alpha=0.7)
    ax3.fill_between(
        x_idx,
        y_pred - y_std,
        y_pred + y_std,
        color='orange', alpha=0.2, label='Pred. std. dev.'
    )
    
    # 학습 포인트 표시 - 정렬된 인덱스 기준으로 찾기
    train_indices_low = []
    train_indices_high = []
    
    # 원본 X_grid에서 훈련 포인트들의 인덱스를 먼저 찾음
    original_train_low = []
    original_train_high = []
    
    for i, combo in enumerate(viz_data['X_grid']):
        combo_int = combo.astype(int)
        # Low-fidelity 포인트 확인
        for x_low in viz_data['X_low']:
            if np.allclose(combo_int, x_low.astype(int)):
                original_train_low.append(i)
                break
        # High-fidelity 포인트 확인  
        for x_high in viz_data['X_high']:
            if np.allclose(combo_int, x_high.astype(int)):
                original_train_high.append(i)
                break
    
    # 정렬된 인덱스에서 해당 훈련 포인트들의 새로운 위치 찾기
    for old_idx in original_train_low:
        new_idx = np.where(sorted_indices == old_idx)[0]
        if len(new_idx) > 0:
            train_indices_low.append(new_idx[0])
    
    for old_idx in original_train_high:
        new_idx = np.where(sorted_indices == old_idx)[0]  
        if len(new_idx) > 0:
            train_indices_high.append(new_idx[0])
    
    # Low-fidelity 학습점 (회색 삼각형) - 정렬된 실제값 사용
    if train_indices_low:
        if 'y_actual' in viz_data:
            y_values_low = [y_actual_sorted[i] for i in train_indices_low]
        else:
            y_values_low = [y_pred[i] for i in train_indices_low]
        ax3.scatter(
            train_indices_low, y_values_low,
            s=100, color='#7F8C8D', label='Training (low, s=0.1)', zorder=10, marker='^',
            edgecolor='#34495E', linewidth=1
        )
    
    # High-fidelity 학습점 (빨간색 삼각형) - 정렬된 실제값 사용
    if train_indices_high:
        if 'y_actual' in viz_data:
            y_values_high = [y_actual_sorted[i] for i in train_indices_high]
        else:
            y_values_high = [y_pred[i] for i in train_indices_high]
        ax3.scatter(
            train_indices_high, y_values_high,
            s=120, color='#E74C3C', label='Training (high, s=1.0)', zorder=10, marker='^',
            edgecolor='#922B21', linewidth=1
        )
    
    # 추천점 표시 (다이아몬드) - 정렬된 인덱스에서 찾기
    best_idx_original = viz_data['best_idx']
    best_idx_sorted_array = np.where(sorted_indices == best_idx_original)[0]
    
    if len(best_idx_sorted_array) > 0:
        best_idx_sorted = best_idx_sorted_array[0]
        
        # 실제값이 있으면 실제값 사용, 없으면 예측값
        if 'y_actual' in viz_data:
            y_value_best = y_actual_sorted[best_idx_sorted]
        else:
            y_value_best = y_pred[best_idx_sorted]
        
        ax3.scatter(
            best_idx_sorted, y_value_best,
            marker='D', color='magenta', s=200, edgecolor='purple',
            linewidth=2, label='Recommended point', zorder=20
        )
    
    ax3.set_ylabel('Bandgap (eV)', color='#2C3E50')
    ax3.set_xlabel('Candidate Index')
    
    # 제목 강조
    iter_ = viz_data['iteration']
    if (iter_ % 8 == 0):
        ax3.set_title(f'Prediction, Uncertainty, and EI\niter: {iter_} (HIGH FIDELITY)',
                      color='crimson', fontsize=14, fontweight='bold', backgroundcolor='#ffe6e6')
    else:
        ax3.set_title(f'Prediction, Uncertainty, and EI\niter: {iter_} (low fidelity)',
                     fontsize=14)
    ax3.tick_params(axis='y', labelcolor='navy')
    ax3.grid(True, alpha=0.3)
    
    # EI 오른쪽축 (정렬된 데이터 사용)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x_idx, ei, marker='o', color='forestgreen', 
                 label='EI', linewidth=2, markersize=3, alpha=0.7)
    
    # 최대 EI 점 표시 (추천점과 일치해야 함)
    if len(best_idx_sorted_array) > 0:
        ax3_twin.scatter(best_idx_sorted, ei[best_idx_sorted], color='red', s=120, 
                        zorder=15, label='Max EI')
        recommended_ei = ei[best_idx_sorted]
    else:
        recommended_ei = 0
        
    ax3_twin.set_ylabel('Expected Improvement (EI)', color='forestgreen')
    ax3_twin.tick_params(axis='y', labelcolor='forestgreen')
    
    # 범례 통합
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(h1+h2, l1+l2, loc='upper right', fontsize=9)
    
    # 추천점 정보 박스 (정렬된 인덱스 사용)
    recommended_str = f"Recommended: {viz_data['recommended_point']}\nEI: {recommended_ei:.6f}"
    ax3.text(0.02, 0.95, recommended_str, transform=ax3.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # x축 범위 설정
    ax3.set_xlim(-1, len(viz_data['X_grid']))
    
    plt.suptitle(f"DNGO Optimization Step {viz_data['iteration']}", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 저장
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_optimization_progress(results: Dict, save_dir: Optional[str] = None):
    """
    전체 최적화 진행 과정 시각화
    
    Args:
        results: 최적화 결과 딕셔너리
        save_dir: 이미지 저장 디렉토리 (None이면 저장하지 않음)
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 각 iteration의 시각화
    if 'visualization_data' in results:
        for viz_data in results['visualization_data']:
            if save_dir:
                save_path = os.path.join(save_dir, f"step_{viz_data['iteration']:03d}.png")
            else:
                save_path = None
            plot_step_visualization(viz_data, save_path)
    
    # 최종 요약 플롯
    if save_dir:
        plot_final_summary(results, os.path.join(save_dir, 'summary.png'))
    else:
        plot_final_summary(results)


def plot_final_summary(results: Dict, save_path: Optional[str] = None):
    """
    최적화 결과 요약 플롯
    
    Args:
        results: 최적화 결과 딕셔너리
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Best so far curve
    ax = axes[0, 0]
    if 'cost_history' in results and 'best_values_history' in results:
        ax.plot(results['cost_history'], results['best_values_history'], 
               'b-', linewidth=2, label='Best value')
        ax.axhline(y=1.34, color='r', linestyle='--', alpha=0.5, label='Target (1.34)')
        ax.set_xlabel('Cumulative Cost')
        ax.set_ylabel('Best Value Found')
        ax.set_title('Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Fidelity scheduling
    ax = axes[0, 1]
    if 'fidelity_history' in results:
        iterations = np.arange(len(results['fidelity_history']))
        colors = ['red' if f == 1.0 else 'blue' for f in results['fidelity_history']]
        ax.scatter(iterations, results['fidelity_history'], c=colors, alpha=0.6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity Scheduling')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    # 3. EI history
    ax = axes[1, 0]
    if 'ei_history' in results:
        ax.plot(results['ei_history'], 'g-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max Expected Improvement')
        ax.set_title('Acquisition Function Evolution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # 4. Final prediction accuracy
    ax = axes[1, 1]
    if 'visualization_data' in results and len(results['visualization_data']) > 0:
        last_viz = results['visualization_data'][-1]
        if len(last_viz['X_high']) > 0:
            plot_prediction_vs_actual(
                last_viz['model'], last_viz['blr'],
                last_viz['X_grid'], last_viz['X_high'], last_viz['y_high'],
                title='Final Model Performance',
                ax=ax
            )
    
    plt.suptitle(f"DNGO Optimization Summary (Cost: {results.get('total_cost', 0):.1f})", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multiple_runs_boxplot(all_results: List[Dict], model_type: str = 'DNGO', 
                               save_path: Optional[str] = None):
    """
    다중 실행 결과를 박스플롯으로 시각화 - cumulative cost 기준
    
    Args:
        all_results: 여러 실행의 결과 리스트  
        model_type: 모델 타입 (x축 레이블용)
        save_path: 저장 경로
    """
    # 각 실행의 최종 누적 비용 수집
    final_costs = []
    for result in all_results:
        if 'total_cost' in result:
            final_costs.append(result['total_cost'])
    
    if not final_costs:
        print("No cost data found in results")
        return
    
    # 박스플롯 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 단일 모델 타입에 대한 박스플롯
    bp = ax.boxplot([final_costs], positions=[1], widths=0.6, 
                   patch_artist=True, showfliers=True)
    
    # 박스 색상 설정
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # 통계 정보 계산
    final_costs_array = np.array(final_costs)
    mean_cost = np.mean(final_costs_array)
    median_cost = np.median(final_costs_array)
    
    # 박스플롯의 whisker 값 계산 (1.5*IQR 방식)
    q1 = np.percentile(final_costs_array, 25)
    q3 = np.percentile(final_costs_array, 75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    
    # 실제 데이터 범위 내에서 whisker 조정
    lower_whisker = max(lower_whisker, np.min(final_costs_array))
    upper_whisker = min(upper_whisker, np.max(final_costs_array))
    
    variance = np.var(final_costs_array)
    
    # 축 설정
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Cumulative Cost (au)', fontsize=12) 
    ax.set_title(f'{model_type} Performance over {len(all_results)} Runs', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels([model_type])
    
    # y축 범위를 0-50으로 고정, 10 간격
    ax.set_ylim(0, 50)
    ax.set_yticks(np.arange(0, 51, 10))  # 0, 10, 20, 30, 40, 50
    
    ax.grid(True, alpha=0.3)
    
    # 통계 정보를 범례로 추가
    stats_legend = [
        f'Mean: {mean_cost:.2f}',
        f'Median: {median_cost:.2f}', 
        f'Variance: {variance:.2f}',
        f'Lower Whisker: {lower_whisker:.2f}',
        f'Upper Whisker: {upper_whisker:.2f}',
        f'1.5×IQR: {1.5*iqr:.2f}'
    ]
    
    # 범례 추가 (오른쪽 상단)
    ax.text(0.98, 0.98, '\n'.join(stats_legend), transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()