"""
분리된 LOFI/HIFI 시각화 모듈

LOFI와 HIFI 모델을 완전히 독립적으로 시각화합니다.
각 모델은 자체 예측, 불확실성, EI를 가집니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional
import os


def plot_model_visualization(viz_data: Dict, model_type: str = 'LOFI', save_path: Optional[str] = None):
    """
    단일 모델 (LOFI 또는 HIFI)의 시각화
    
    Args:
        viz_data: 시각화 데이터 딕셔너리
        model_type: 'LOFI' 또는 'HIFI'
        save_path: 저장 경로
    """
    # 모델 타입별 데이터 선택
    if model_type == 'LOFI':
        blr_model = viz_data.get('blr_low')
        X_train = viz_data.get('X_low', np.array([]))
        y_train = viz_data.get('y_low', np.array([]))
        y_pred_key = 'y_pred_low'
        y_std_key = 'y_std_low'
        ei_key = 'ei_low'
        model_title = 'Low-Fidelity (LOFI) Model'
        title_color = '#3498db'  # 파란색
        plot_color = '#5dade2'
    else:  # HIFI
        blr_model = viz_data.get('blr_high')
        X_train = viz_data.get('X_high', np.array([]))
        y_train = viz_data.get('y_high', np.array([]))
        y_pred_key = 'y_pred_high'
        y_std_key = 'y_std_high'
        ei_key = 'ei_high'
        model_title = 'High-Fidelity (HIFI) Model'
        title_color = '#e74c3c'  # 빨간색
        plot_color = '#ec7063'
    
    # Figure 생성
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
    
    # 실제값 가져오기
    y_actual = viz_data.get('y_actual')
    X_grid = viz_data.get('X_grid', np.array([]))
    
    # Subplot 1: Prediction vs Actual (좌상단)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if blr_model is not None and viz_data.get(y_pred_key) is not None:
        y_pred_model = viz_data[y_pred_key]
        y_std_model = viz_data[y_std_key]
        
        # 실제값과 예측값 산점도
        if y_actual is not None:
            # 전체 점들 (연한 색)
            ax1.scatter(y_actual, y_pred_model, alpha=0.3, s=20, color='gray', label='All candidates')
            
            # 훈련 데이터 점들 (진한 색)
            if len(X_train) > 0:
                train_indices = []
                for x_train in X_train:
                    for i, x_grid in enumerate(X_grid):
                        if np.allclose(x_train, x_grid):
                            train_indices.append(i)
                            break
                
                if train_indices:
                    ax1.scatter(y_actual[train_indices], y_pred_model[train_indices], 
                              alpha=0.8, s=60, color=plot_color, label=f'Training points (n={len(train_indices)})',
                              edgecolor='black', linewidth=0.5)
            
            # 대각선
            min_val = min(y_actual.min(), y_pred_model.min())
            max_val = max(y_actual.max(), y_pred_model.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect prediction')
            
            ax1.set_xlabel('Actual Bandgap (eV)')
            ax1.set_ylabel('Predicted Bandgap (eV)')
            ax1.set_title(f'{model_title} - Prediction Accuracy', fontweight='bold', color=title_color)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, f'No {model_type} model trained yet', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f'{model_title} - Not Available', fontweight='bold', color='gray')
    
    # Subplot 2: Training Data Distribution (우상단)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if len(y_train) > 0:
        # 히스토그램과 박스플롯
        ax2.hist(y_train, bins=20, alpha=0.7, color=plot_color, edgecolor='black')
        ax2.set_xlabel('Bandgap (eV)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{model_title} - Training Data Distribution (n={len(y_train)})', 
                     fontweight='bold', color=title_color)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 통계 정보 추가
        stats_text = f'Mean: {np.mean(y_train):.3f}\nStd: {np.std(y_train):.3f}\nMin: {np.min(y_train):.3f}\nMax: {np.max(y_train):.3f}'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, f'No {model_type} training data yet', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'{model_title} - No Training Data', fontweight='bold', color='gray')
    
    # Subplot 3: Prediction, Uncertainty, and EI (하단 전체)
    ax3 = fig.add_subplot(gs[1, :])
    
    if viz_data.get(y_pred_key) is not None:
        y_pred = viz_data[y_pred_key]
        y_std = viz_data[y_std_key]
        ei = viz_data[ei_key]
        
        # 정렬 (EI 기준으로 내림차순)
        sorted_indices = np.argsort(-ei)
        x_idx = range(len(X_grid))
        
        y_pred_sorted = y_pred[sorted_indices]
        y_std_sorted = y_std[sorted_indices]
        ei_sorted = ei[sorted_indices]
        
        if y_actual is not None:
            y_actual_sorted = y_actual[sorted_indices]
            # 실제값 표시
            ax3.scatter(x_idx, y_actual_sorted, s=30, label='Actual', color='blue', alpha=0.6, zorder=5)
        
        # 예측값과 불확실성
        ax3.scatter(x_idx, y_pred_sorted, s=40, label='Prediction', color=plot_color, alpha=0.7)
        ax3.fill_between(
            x_idx,
            y_pred_sorted - y_std_sorted,
            y_pred_sorted + y_std_sorted,
            color=plot_color, alpha=0.2, label='Uncertainty (±1σ)'
        )
        
        # 훈련 데이터 표시
        if len(X_train) > 0:
            train_indices_sorted = []
            for x_train in X_train:
                for i, x_grid in enumerate(X_grid):
                    if np.allclose(x_train, x_grid):
                        # 정렬된 인덱스에서 위치 찾기
                        sorted_pos = np.where(sorted_indices == i)[0]
                        if len(sorted_pos) > 0:
                            train_indices_sorted.append(sorted_pos[0])
                        break
            
            if train_indices_sorted:
                train_y = [y_actual_sorted[i] if y_actual is not None else y_pred_sorted[i] 
                          for i in train_indices_sorted]
                ax3.scatter(train_indices_sorted, train_y, s=100, 
                          color=title_color, marker='^', label=f'{model_type} Training Points',
                          edgecolor='black', linewidth=1, zorder=10)
        
        # 최대 EI 점 표시
        max_ei_idx = 0  # 이미 EI로 정렬했으므로 첫 번째가 최대
        if y_actual is not None:
            max_ei_y = y_actual_sorted[max_ei_idx]
        else:
            max_ei_y = y_pred_sorted[max_ei_idx]
        
        ax3.scatter(max_ei_idx, max_ei_y, marker='D', color='magenta', s=200,
                   edgecolor='purple', linewidth=2, label='Max EI Point', zorder=20)
        
        ax3.set_xlabel('Candidate Index (sorted by EI)')
        ax3.set_ylabel('Bandgap (eV)', color='navy')
        ax3.set_title(f'{model_title} - Predictions and Uncertainty', fontweight='bold', color=title_color)
        ax3.tick_params(axis='y', labelcolor='navy')
        ax3.grid(True, alpha=0.3)
        
        # EI를 오른쪽 축에 표시
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x_idx, ei_sorted, marker='o', color='forestgreen',
                     label='Expected Improvement', linewidth=2, markersize=3, alpha=0.7)
        ax3_twin.scatter(max_ei_idx, ei_sorted[max_ei_idx], color='red', s=120,
                        zorder=15, label='Max EI')
        ax3_twin.set_ylabel('Expected Improvement (EI)', color='forestgreen')
        ax3_twin.tick_params(axis='y', labelcolor='forestgreen')
        ax3_twin.set_yscale('log' if ei_sorted.max() > 0 and ei_sorted.max() / (ei_sorted[ei_sorted > 0].min() + 1e-10) > 100 else 'linear')
        
        # 범례 통합
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(h1+h2, l1+l2, loc='upper right', fontsize=9, ncol=2)
        
        # 정보 박스
        info_text = f"Max EI: {ei_sorted[max_ei_idx]:.6f}\n"
        info_text += f"Iteration: {viz_data.get('iteration', 'N/A')}\n"
        info_text += f"Fidelity: {'High' if viz_data.get('fidelity', 0) == 1.0 else 'Low'}"
        ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax3.text(0.5, 0.5, f'No {model_type} predictions available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title(f'{model_title} - No Predictions', fontweight='bold', color='gray')
    
    # 전체 제목
    iteration = viz_data.get('iteration', 0)
    fidelity_str = "HIGH" if viz_data.get('fidelity', 0) == 1.0 else "LOW"
    plt.suptitle(f'{model_title} - Iteration {iteration} ({fidelity_str} fidelity selected)',
                fontsize=16, fontweight='bold', color=title_color)
    plt.tight_layout()
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_step_visualization_separated(viz_data: Dict, save_path: Optional[str] = None):
    """
    LOFI와 HIFI 모델을 완전히 분리하여 시각화
    
    Args:
        viz_data: 시각화 데이터 딕셔너리
        save_path: 저장 경로 (기본 경로, _LOFI와 _HIFI가 추가됨)
    """
    # LOFI 시각화
    if save_path:
        base_path, ext = os.path.splitext(save_path)
        lofi_path = f"{base_path}_LOFI{ext}"
    else:
        lofi_path = None
    
    plot_model_visualization(viz_data, model_type='LOFI', save_path=lofi_path)
    
    # HIFI 시각화
    if save_path:
        base_path, ext = os.path.splitext(save_path)
        hifi_path = f"{base_path}_HIFI{ext}"
    else:
        hifi_path = None
    
    plot_model_visualization(viz_data, model_type='HIFI', save_path=hifi_path)


def plot_optimization_progress_separated(results: Dict, save_dir: Optional[str] = None):
    """
    전체 최적화 진행 과정을 LOFI/HIFI 분리하여 시각화
    
    Args:
        results: 최적화 결과 딕셔너리
        save_dir: 이미지 저장 디렉토리
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 각 iteration의 시각화 (LOFI/HIFI 분리)
    if 'visualization_data' in results:
        for viz_data in results['visualization_data']:
            if save_dir:
                save_path = os.path.join(save_dir, f"step_{viz_data['iteration']:03d}.png")
            else:
                save_path = None
            plot_step_visualization_separated(viz_data, save_path)