"""
실제 DNGO 테스트 스크립트 - cost 10으로 빠른 테스트
시각화 기능이 제대로 작동하는지 확인
(import 문제 회피 버전)
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
import os

# 경로 설정
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "0.Data"))

# 직접 필요한 것만 import
from DNGO.optimization_base import single_optimization_run
from DNGO.visualization import plot_optimization_progress, plot_step_visualization

def load_lookup_table():
    """lookup table 로드"""
    data_path = Path(__file__).parent.parent.parent / "0.Data" / "lookup_table.pkl"
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def load_label_maps():
    """라벨 맵 로드"""
    import json
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

def test_dngo_with_visualization():
    """DNGO를 실제 데이터로 테스트 (시각화 포함)"""
    
    print("🚀 Starting DNGO test with real data...")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("📊 Loading data...")
    lookup = load_lookup_table()
    label_maps = load_label_maps()
    
    # 파라미터 공간 정의
    param_space = {
        'organic': list(label_maps['organic'].keys()),
        'cation': list(label_maps['cation'].keys()),
        'anion': list(label_maps['anion'].keys())
    }
    
    print(f"✓ Loaded {len(lookup)} combinations")
    print(f"✓ Parameter space: {len(param_space['organic'])}x{len(param_space['cation'])}x{len(param_space['anion'])}")
    
    # 2. DNGO 설정
    config = {
        'input_dim': 3,
        'hidden_dim': 32,  # 작은 네트워크로 빠른 테스트
        'pretrain_epochs': 50,  # 적은 epoch
        'finetune_epochs': 30,
        'pretrain_lr': 1e-3,
        'finetune_lr': 1e-4,
        'device': 'cpu'
    }
    
    # 3. 단일 최적화 실행 (cost 10으로 빠른 테스트)
    print("\n🔧 Running DNGO optimization...")
    print(f"Settings:")
    print(f"  - Cost budget: 10.0")
    print(f"  - Initial samples: 5")
    print(f"  - High-fidelity ratio: 0.2")
    print(f"  - Model config: {config}")
    print("-" * 60)
    
    result = single_optimization_run(
        param_space=param_space,
        label_maps=label_maps,
        lookup=lookup,
        cost_budget=10.0,  # 작은 budget으로 빠른 테스트
        num_init_design=5,  # 초기 샘플 5개
        high_fidelity_ratio=0.2,
        min_target=1.34,  # 목표값
        random_state=42,
        verbose=True,  # 상세 출력
        model_config=config,
        use_hyperparameter_bo=False,  # 하이퍼파라미터 BO 비활성화 (빠른 테스트)
        data_size='small'
    )
    
    print("\n" + "=" * 60)
    print("📈 Optimization Results:")
    print(f"  - Total cost: {result['total_cost']:.2f}")
    print(f"  - Best value found: {result['best_so_far']:.4f}")
    print(f"  - Iterations: {result['iterations']}")
    print(f"  - Final high-fidelity samples: {len(result['final_y_high'])}")
    print(f"  - Final low-fidelity samples: {len(result['final_y_low'])}")
    
    # 4. 시각화
    print("\n🎨 Creating visualizations...")
    
    # 이미지 저장 디렉토리
    save_dir = "test_dngo_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 전체 최적화 과정 시각화
    print(f"  - Saving to {save_dir}/")
    plot_optimization_progress(result, save_dir=save_dir)
    
    # 마지막 스텝 상세 시각화
    if result.get('visualization_data'):
        last_step = result['visualization_data'][-1]
        print(f"  - Creating detailed view of final step (iteration {last_step['iteration']})")
        plot_step_visualization(last_step, save_path=f"{save_dir}/final_step.png")
    
    print("\n✅ Test completed successfully!")
    print(f"📁 Check '{save_dir}/' directory for visualization results")
    
    # 5. 간단한 통계 출력
    if result.get('visualization_data'):
        print("\n📊 Optimization Statistics:")
        fidelities = [d['fidelity'] for d in result['visualization_data']]
        low_count = fidelities.count(0.1)
        high_count = fidelities.count(1.0)
        print(f"  - Low-fidelity evaluations: {low_count}")
        print(f"  - High-fidelity evaluations: {high_count}")
        print(f"  - Ratio: {low_count}:{high_count}")
        
        # EI 변화
        ei_values = [d['ei'][d['best_idx']] for d in result['visualization_data']]
        print(f"  - Initial max EI: {ei_values[0]:.6f}")
        print(f"  - Final max EI: {ei_values[-1]:.6f}")
    
    return result


if __name__ == "__main__":
    # 테스트 실행
    result = test_dngo_with_visualization()
    
    # 결과를 pickle로 저장 (나중에 분석용)
    import pickle
    with open('test_dngo_results/test_result.pkl', 'wb') as f:
        pickle.dump(result, f)
    print("\n💾 Result saved to 'test_dngo_results/test_result.pkl'")