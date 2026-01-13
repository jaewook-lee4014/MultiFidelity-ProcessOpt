#!/usr/bin/env python3
"""
Test R² Comparison Visualization

6가지 Multi-Fidelity DNN 방법론 + MFGP 비교
10-Fold CV 결과 시각화

Author: Claude Code
Date: 2025-12-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Style settings
try:
    plt.style.use('seaborn-whitegrid')
except:
    pass
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150


def load_results():
    """결과 데이터 로드"""
    # 가장 최신 strict 결과 사용
    results_path = Path(__file__).parent / 'visualizations/20251211_210923_residual_uq_strict/results_strict.csv'

    if not results_path.exists():
        # 대안 경로
        results_path = Path(__file__).parent / 'visualizations/20251211_163454_all_6methods/results_summary.csv'

    df = pd.read_csv(results_path)
    print(f"Loaded: {results_path}")
    print(f"Shape: {df.shape}")
    return df


def create_bar_chart(df, output_dir):
    """1. 메인 Bar Chart: Test R² 비교"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential',
               'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint', 'Grad.\nScaling', 'Sequential',
              'Progressive', 'Curriculum', 'Two-Stage\nJoint']

    # R² 컬럼 추출
    r2_data = {}
    for method in methods:
        col = f'{method}_r2'
        if col in df.columns:
            r2_data[method] = df[col].values

    # 평균/표준편차 계산
    means = [np.mean(r2_data[m]) for m in methods]
    stds = [np.std(r2_data[m]) for m in methods]

    # 정렬 (내림차순)
    sorted_idx = np.argsort(means)[::-1]
    sorted_methods = [methods[i] for i in sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_means = [means[i] for i in sorted_idx]
    sorted_stds = [stds[i] for i in sorted_idx]

    # 색상 설정 (MFGP는 다른 색)
    colors = []
    for m in sorted_methods:
        if m == 'mfgp':
            colors.append('#e74c3c')  # 빨간색
        else:
            colors.append('#3498db')  # 파란색

    # 플롯
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(sorted_methods))
    bars = ax.bar(x_pos, sorted_means, yerr=sorted_stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # 값 표시
    for i, (bar, mean, std) in enumerate(zip(bars, sorted_means, sorted_stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_labels, fontsize=10)
    ax.set_ylabel('Test R² Score', fontsize=12)
    ax.set_title('Multi-Fidelity Model Comparison: Test R² (10-Fold CV)\n'
                 'Data: 72 LF (GGA) + 9 HF (HSE06) samples', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='R²=0.8')
    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, label='R²=0.7')

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='DNN Methods'),
                       Patch(facecolor='#e74c3c', label='MFGP (Baseline)')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / '1_bar_chart_r2_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '1_bar_chart_r2_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("Created: 1_bar_chart_r2_comparison.png")
    return sorted_methods, sorted_labels


def create_box_plot(df, output_dir):
    """2. Box Plot: Fold별 R² 분포"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential',
               'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint', 'Grad.Scaling', 'Sequential',
              'Progressive', 'Curriculum', 'Two-Stage']

    # 데이터 준비
    data_list = []
    for method, label in zip(methods, labels):
        col = f'{method}_r2'
        if col in df.columns:
            for val in df[col].values:
                data_list.append({'Method': label, 'R²': val})

    plot_df = pd.DataFrame(data_list)

    # 평균으로 정렬
    method_order = plot_df.groupby('Method')['R²'].mean().sort_values(ascending=False).index.tolist()

    # 플롯
    fig, ax = plt.subplots(figsize=(12, 6))

    # 색상 설정
    palette = {label: '#e74c3c' if label == 'MFGP' else '#3498db' for label in labels}

    sns.boxplot(x='Method', y='R²', data=plot_df, order=method_order,
                palette=palette, ax=ax, width=0.6)
    sns.stripplot(x='Method', y='R²', data=plot_df, order=method_order,
                  color='black', alpha=0.5, size=6, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('Test R² Score', fontsize=12)
    ax.set_title('R² Distribution Across 10 Folds\n'
                 'Box: IQR, Whiskers: 1.5×IQR, Points: Individual Folds', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / '2_box_plot_r2_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '2_box_plot_r2_distribution.pdf', bbox_inches='tight')
    plt.close()

    print("Created: 2_box_plot_r2_distribution.png")


def create_line_plot(df, output_dir):
    """3. Line Plot: Fold별 상세 비교"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential',
               'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint', 'Grad.Scaling', 'Sequential',
              'Progressive', 'Curriculum', 'Two-Stage']

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
              '#f39c12', '#1abc9c', '#e67e22']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']

    fig, ax = plt.subplots(figsize=(14, 7))

    folds = df['fold'].values

    for method, label, color, marker in zip(methods, labels, colors, markers):
        col = f'{method}_r2'
        if col in df.columns:
            r2_values = df[col].values
            linewidth = 2.5 if method == 'mfgp' else 1.5
            ax.plot(folds, r2_values, marker=marker, label=label,
                    color=color, linewidth=linewidth, markersize=8, alpha=0.8)

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Test R² Score', fontsize=12)
    ax.set_title('R² Score by Fold: All Methods Comparison\n'
                 'Each fold uses different random seed for data split', fontsize=13, fontweight='bold')
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {i}\n(seed={s})' for i, s in zip(df['fold'], df['seed'])], fontsize=9)
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '3_line_plot_fold_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '3_line_plot_fold_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("Created: 3_line_plot_fold_comparison.png")


def create_heatmap(df, output_dir):
    """4. Heatmap: 모델 × Fold R² 매트릭스"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential',
               'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint', 'Grad.Scaling', 'Sequential',
              'Progressive', 'Curriculum', 'Two-Stage']

    # 매트릭스 생성
    matrix = []
    for method in methods:
        col = f'{method}_r2'
        if col in df.columns:
            matrix.append(df[col].values)

    matrix = np.array(matrix)

    # 평균으로 정렬
    means = matrix.mean(axis=1)
    sorted_idx = np.argsort(means)[::-1]
    matrix = matrix[sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]

    # 플롯
    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=0.9)

    # 값 표시
    for i in range(len(sorted_labels)):
        for j in range(len(df)):
            val = matrix[i, j]
            color = 'white' if val < 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([f'Fold {i}' for i in df['fold']], fontsize=10)
    ax.set_yticks(np.arange(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=11)

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('R² Heatmap: Method × Fold\n'
                 'Green: High R², Red: Low R²', fontsize=13, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('R² Score', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / '4_heatmap_r2_matrix.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '4_heatmap_r2_matrix.pdf', bbox_inches='tight')
    plt.close()

    print("Created: 4_heatmap_r2_matrix.png")


def create_summary_table(df, output_dir):
    """5. Summary Table 생성"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential',
               'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint', 'Gradient Scaling', 'Sequential',
              'Progressive', 'Curriculum', 'Two-Stage Joint']

    summary_data = []
    for method, label in zip(methods, labels):
        col = f'{method}_r2'
        if col in df.columns:
            values = df[col].values
            summary_data.append({
                'Method': label,
                'Mean R²': np.mean(values),
                'Std R²': np.std(values),
                'Min R²': np.min(values),
                'Max R²': np.max(values),
                'Median R²': np.median(values)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean R²', ascending=False)

    # CSV 저장
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)

    # 텍스트 테이블 생성
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # 테이블 데이터 포맷팅
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Method'],
            f"{row['Mean R²']:.4f}",
            f"{row['Std R²']:.4f}",
            f"{row['Min R²']:.4f}",
            f"{row['Max R²']:.4f}",
            f"{row['Median R²']:.4f}"
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Method', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#3498db']*6)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # 헤더 스타일
    for i in range(6):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Summary Statistics: Test R² (10-Fold CV)\n'
                 'Data: 72 LF + 9 HF, Sorted by Mean R²',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / '5_summary_table.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '5_summary_table.pdf', bbox_inches='tight')
    plt.close()

    print("Created: 5_summary_table.png")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    return summary_df


def create_combined_figure(df, output_dir):
    """6. Combined Figure: 모든 차트 통합"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential',
               'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint', 'Grad.Scaling', 'Sequential',
              'Progressive', 'Curriculum', 'Two-Stage']

    fig = plt.figure(figsize=(16, 12))

    # 1. Bar Chart (상단 좌측)
    ax1 = fig.add_subplot(2, 2, 1)

    r2_data = {m: df[f'{m}_r2'].values for m in methods if f'{m}_r2' in df.columns}
    means = [np.mean(r2_data[m]) for m in methods]
    stds = [np.std(r2_data[m]) for m in methods]

    sorted_idx = np.argsort(means)[::-1]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_means = [means[i] for i in sorted_idx]
    sorted_stds = [stds[i] for i in sorted_idx]
    sorted_methods = [methods[i] for i in sorted_idx]

    colors = ['#e74c3c' if m == 'mfgp' else '#3498db' for m in sorted_methods]

    bars = ax1.bar(range(len(sorted_methods)), sorted_means, yerr=sorted_stds,
                   capsize=4, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(sorted_methods)))
    ax1.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('R²')
    ax1.set_title('(A) Mean R² Comparison', fontweight='bold')
    ax1.set_ylim(0, 1.0)

    # 2. Box Plot (상단 우측)
    ax2 = fig.add_subplot(2, 2, 2)

    data_list = []
    for method, label in zip(methods, labels):
        col = f'{method}_r2'
        if col in df.columns:
            for val in df[col].values:
                data_list.append({'Method': label, 'R²': val})

    plot_df = pd.DataFrame(data_list)
    method_order = plot_df.groupby('Method')['R²'].mean().sort_values(ascending=False).index.tolist()
    palette = {label: '#e74c3c' if label == 'MFGP' else '#3498db' for label in labels}

    sns.boxplot(x='Method', y='R²', data=plot_df, order=method_order,
                palette=palette, ax=ax2, width=0.6)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('')
    ax2.set_title('(B) R² Distribution', fontweight='bold')
    ax2.set_ylim(-0.5, 1.0)

    # 3. Line Plot (하단 좌측)
    ax3 = fig.add_subplot(2, 2, 3)

    colors_line = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
                   '#f39c12', '#1abc9c', '#e67e22']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']

    for method, label, color, marker in zip(methods, labels, colors_line, markers):
        col = f'{method}_r2'
        if col in df.columns:
            ax3.plot(df['fold'].values, df[col].values, marker=marker,
                     label=label, color=color, linewidth=1.5, markersize=6, alpha=0.8)

    ax3.set_xlabel('Fold')
    ax3.set_ylabel('R²')
    ax3.set_title('(C) R² by Fold', fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8, ncol=2)
    ax3.set_ylim(-0.5, 1.0)
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap (하단 우측)
    ax4 = fig.add_subplot(2, 2, 4)

    matrix = np.array([df[f'{m}_r2'].values for m in methods if f'{m}_r2' in df.columns])
    means_hm = matrix.mean(axis=1)
    sorted_idx_hm = np.argsort(means_hm)[::-1]
    matrix = matrix[sorted_idx_hm]
    sorted_labels_hm = [labels[i] for i in sorted_idx_hm]

    im = ax4.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=0.9)
    ax4.set_xticks(np.arange(len(df)))
    ax4.set_xticklabels([f'F{i}' for i in df['fold']], fontsize=9)
    ax4.set_yticks(np.arange(len(sorted_labels_hm)))
    ax4.set_yticklabels(sorted_labels_hm, fontsize=9)
    ax4.set_xlabel('Fold')
    ax4.set_title('(D) R² Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8)

    plt.suptitle('Multi-Fidelity Model Comparison: Test R² (10-Fold CV)\n'
                 'Data: 72 LF (GGA) + 9 HF (HSE06)', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / '0_combined_figure.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / '0_combined_figure.pdf', bbox_inches='tight')
    plt.close()

    print("Created: 0_combined_figure.png")


def main():
    print("="*60)
    print("Test R² Comparison Visualization")
    print("="*60)

    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_test_r2_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # 데이터 로드
    df = load_results()

    # 시각화 생성
    print("\nGenerating visualizations...")

    create_combined_figure(df, output_dir)
    create_bar_chart(df, output_dir)
    create_box_plot(df, output_dir)
    create_line_plot(df, output_dir)
    create_heatmap(df, output_dir)
    summary_df = create_summary_table(df, output_dir)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nAll visualizations saved to:\n{output_dir}")

    return summary_df


if __name__ == '__main__':
    main()
