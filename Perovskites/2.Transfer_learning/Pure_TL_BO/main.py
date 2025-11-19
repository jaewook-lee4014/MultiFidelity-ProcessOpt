#!/usr/bin/env python3
"""
Transfer Learning Bayesian Optimization with Hyperparameter Optimization
메인 실행 스크립트 - 하이퍼파라미터 베이지안 최적화 지원
"""

import time
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
import argparse
import sys
import os
from pathlib import Path

# 로컬 모듈 import
from common.config import *
from common.data_utils import (
    load_lookup_table, create_label_maps, sample_param_space, 
    assign_fidelities, prepare_initial_data, create_all_combinations_data,
    create_param_space
)
# Universal optimization using optimization_base (supports both DNGO and BNN)
from DNGO.optimization_base import single_optimization_run, multiple_optimization_runs
# Legacy imports for backward compatibility with DNGO-OL
from DNGO.optimization import single_optimization_run_dngo_ol, multiple_optimization_runs_dngo_ol
from common.device_utils import setup_device_for_bnn, clear_mps_cache
from common.visualization import (
    plot_iteration_results, plot_prediction_scatter, 
    plot_multiple_runs_summary, plot_learning_curves,
    plot_optimization_results, plot_bnn_iteration_results
)
from DNGO.visualization import plot_optimization_progress, plot_multiple_runs_boxplot
from DNGO.visualization_separated import plot_optimization_progress_separated

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


def setup_device():
    """디바이스 설정 (GPU/CPU) - 자동 감지"""
    try:
        from common.config import DEVICE_CONFIG
        prefer_gpu = DEVICE_CONFIG.get('use_gpu', True)
    except:
        prefer_gpu = True
    
    device = setup_device_for_bnn(prefer_gpu=prefer_gpu, verbose=True)
    return device


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='Transfer Learning Bayesian Optimization with Hyperparameter BO'
    )
    
    # 기본 실행 옵션
    parser.add_argument('--model-type', choices=['dngo', 'bnn', 'dngo-ol'], default='dngo',
                       help='모델 타입: DNGO, BNN, 또는 DNGO-OL (Online Learning)')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='실행 횟수 (1: single run, >1: multiple runs)')
    parser.add_argument('--verbose', action='store_true',
                       help='상세 출력 활성화')
    
    # 최적화 파라미터
    parser.add_argument('--cost-budget', type=float, default=50.0,
                       help='총 비용 예산')
    parser.add_argument('--num-init-design', type=int, default=10,
                       help='초기 설계점 개수')
    parser.add_argument('--high-fidelity-ratio', type=float, default=0.2,
                       help='초기 설계에서 high-fidelity 비율')
    parser.add_argument('--min-target', type=float, default=1.5249,
                       help='목표 최솟값 (조기 종료 조건)')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='기본 hidden layer 차원')
    parser.add_argument('--pretrain-epochs', type=int, default=200,
                       help='기본 pretrain epochs')
    parser.add_argument('--finetune-epochs', type=int, default=100,
                       help='기본 finetune epochs')
    parser.add_argument('--device', default='auto',
                       help='PyTorch 디바이스 (auto/cpu/cuda/mps)')
    
    # 하이퍼파라미터 BO 옵션
    parser.add_argument('--use-hyperparameter-bo', action='store_true',
                       help='하이퍼파라미터 베이지안 최적화 사용')
    parser.add_argument('--pretrain-bo-trials', type=int, default=5,
                       help='Pretrain 하이퍼파라미터 BO 시행 횟수')
    parser.add_argument('--finetune-bo-trials', type=int, default=5,
                       help='Finetune 하이퍼파라미터 BO 시행 횟수')
    parser.add_argument('--data-size', choices=['small', 'medium', 'large'], default='small',
                       help='데이터 크기 (하이퍼파라미터 탐색 공간 결정)')
    
    # 점진적 학습 옵션
    parser.add_argument('--use-incremental-learning', action='store_true',
                       help='점진적 학습 사용 (새 데이터에 점진적 업데이트)')
    parser.add_argument('--incremental-mode', choices=['full', 'incremental', 'hybrid'], default='incremental',
                       help='점진적 학습 모드 (full: 항상 전체 재학습, incremental: 항상 점진적, hybrid: 주기적 전체)')
    parser.add_argument('--lr-boost-factor', type=float, default=2.0,
                       help='점진적 학습 시 학습률 부스트 계수')
    parser.add_argument('--incremental-epochs', type=int, default=10,
                       help='점진적 학습 시 epoch 수')
    parser.add_argument('--replay-ratio', type=float, default=0.2,
                       help='점진적 학습 시 경험 재생 비율 (0.0~0.5)')
    parser.add_argument('--weight-decay-factor', type=float, default=0.9,
                       help='점진적 학습 시 가중치 감쇠 계수')
    parser.add_argument('--full-retrain-interval', type=int, default=5,
                       help='hybrid 모드에서 전체 재학습 주기')
    parser.add_argument('--kl-reg-weight', type=float, default=0.1,
                       help='BNN 점진적 학습 시 KL 정규화 가중치 (catastrophic forgetting 방지)')
    
    # 시각화 옵션
    parser.add_argument('--save-images', action='store_true',
                       help='각 iteration마다 시각화 이미지 저장')
    parser.add_argument('--images-dir', default='images',
                       help='이미지 저장 디렉토리')
    
    # BNN 전용 옵션
    parser.add_argument('--bnn-hidden-dims', nargs='+', type=int, default=[64, 64],
                       help='BNN hidden layer 차원들 (예: --bnn-hidden-dims 64 64)')
    parser.add_argument('--kl-weight', type=float, default=1.0,
                       help='BNN KL divergence 가중치')
    parser.add_argument('--kl-warmup-epochs', type=int, default=10,
                       help='BNN KL warm-up epochs')
    parser.add_argument('--prior-std', type=float, default=1.0,
                       help='BNN 사전분포 표준편차')
    parser.add_argument('--noise-type', choices=['homoscedastic', 'heteroscedastic'], default='homoscedastic',
                       help='BNN 노이즈 모델링 타입')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='BNN 예측 시 샘플링 횟수')
    
    # DNGO-OL 전용 옵션
    parser.add_argument('--online-lr', type=float, default=1e-5,
                       help='DNGO-OL 온라인 학습률')
    parser.add_argument('--forgetting-factor', type=float, default=0.99,
                       help='DNGO-OL 망각 계수 (0~1, 클수록 이전 데이터 영향 증가)')
    parser.add_argument('--memory-size', type=int, default=100,
                       help='DNGO-OL BLR 메모리 버퍼 크기')
    parser.add_argument('--replay-buffer-size', type=int, default=100,
                       help='DNGO-OL DNN 리플레이 버퍼 크기')
    parser.add_argument('--online-batch-size', type=int, default=16,
                       help='DNGO-OL 온라인 학습 배치 크기')
    parser.add_argument('--online-epochs', type=int, default=5,
                       help='DNGO-OL 온라인 업데이트 시 epoch 수')
    
    # 결과 저장
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='결과를 CSV 파일로 저장')
    parser.add_argument('--results-filename', default='tl_bo_results.csv',
                       help='결과 파일명')
    parser.add_argument('--plot-results', action='store_true',
                       help='결과 시각화')
    parser.add_argument('--separated-viz', action='store_true',
                       help='LOFI와 HIFI 모델을 분리하여 시각화')
    
    return parser.parse_args()


def setup_incremental_params(args):
    """점진적 학습 파라미터 설정"""
    if not args.use_incremental_learning:
        return None
    
    return {
        'mode': args.incremental_mode,
        'lr_boost_factor': args.lr_boost_factor,
        'incremental_epochs': args.incremental_epochs,
        'replay_ratio': args.replay_ratio,
        'weight_decay_factor': args.weight_decay_factor,
        'full_retrain_interval': args.full_retrain_interval,
        'kl_reg_weight': args.kl_reg_weight  # BNN 전용
    }


def setup_model_config(args):
    """모델 설정 딕셔너리 생성"""
    # 디바이스 설정
    if args.device == 'auto':
        device = setup_device()
    else:
        device = args.device
        print(f"🎯 사용자 지정 디바이스: {device.upper()}")
    
    if args.model_type == 'dngo':
        return {
            'input_dim': 3,  # organic, cation, anion
            'hidden_dim': args.hidden_dim,
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'device': device
        }
    elif args.model_type == 'dngo-ol':
        return {
            'input_dim': 3,  # organic, cation, anion
            'hidden_dim': args.hidden_dim,
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'pretrain_lr': 1e-3,
            'finetune_lr': 1e-4,
            'online_lr': args.online_lr,
            'forgetting_factor': args.forgetting_factor,
            'memory_size': args.memory_size,
            'replay_buffer_size': args.replay_buffer_size,
            'online_batch_size': args.online_batch_size,
            'online_epochs': args.online_epochs,
            'device': device
        }
    else:  # BNN
        return {
            'input_dim': 3,  # organic, cation, anion
            'hidden_dims': args.bnn_hidden_dims,
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'pretrain_lr': 1e-3,
            'finetune_lr': 1e-4,
            'kl_weight': args.kl_weight,
            'kl_warmup_epochs': args.kl_warmup_epochs,
            'prior_std': args.prior_std,
            'noise_type': args.noise_type,
            'n_samples': args.n_samples,
            'device': device
        }


def print_configuration(args, param_space, model_config, incremental_params=None):
    """설정 정보 출력"""
    print("=" * 60)
    print("Transfer Learning Bayesian Optimization Configuration")
    print("=" * 60)
    execution_mode = "Single Run" if args.num_runs == 1 else f"Multiple Runs ({args.num_runs})"
    print(f"Execution Mode: {execution_mode}")
    print(f"Model Type: {args.model_type.upper()}")
    
    print(f"\nOptimization Parameters:")
    print(f"  Cost budget: {args.cost_budget}")
    print(f"  Initial design points: {args.num_init_design}")
    print(f"  High-fidelity ratio: {args.high_fidelity_ratio}")
    print(f"  Target minimum: {args.min_target}")
    
    print(f"\nParameter Space:")
    print(f"  Organic materials: {len(param_space['organic'])}")
    print(f"  Cations: {len(param_space['cation'])}")
    print(f"  Anions: {len(param_space['anion'])}")
    print(f"  Total combinations: {len(param_space['organic']) * len(param_space['cation']) * len(param_space['anion'])}")
    
    print(f"\nModel Configuration:")
    print(f"  Input dimension: {model_config['input_dim']}")
    if args.model_type == 'dngo':
        print(f"  Hidden dimension: {model_config['hidden_dim']}")
    elif args.model_type == 'dngo-ol':
        print(f"  Hidden dimension: {model_config['hidden_dim']}")
        print(f"  Online learning rate: {model_config['online_lr']}")
        print(f"  Forgetting factor: {model_config['forgetting_factor']}")
        print(f"  Memory size: {model_config['memory_size']}")
        print(f"  Replay buffer size: {model_config['replay_buffer_size']}")
        print(f"  Online batch size: {model_config['online_batch_size']}")
        print(f"  Online epochs: {model_config['online_epochs']}")
    else:  # BNN
        print(f"  Hidden dimensions: {model_config['hidden_dims']}")
        print(f"  KL weight: {model_config['kl_weight']}")
        print(f"  KL warmup epochs: {model_config['kl_warmup_epochs']}")
        print(f"  Prior std: {model_config['prior_std']}")
        print(f"  Noise type: {model_config['noise_type']}")
        print(f"  Prediction samples: {model_config['n_samples']}")
    print(f"  Pretrain epochs: {model_config['pretrain_epochs']}")
    print(f"  Finetune epochs: {model_config['finetune_epochs']}")
    print(f"  Device: {model_config['device']}")
    
    # 하이퍼파라미터 BO 설정
    if args.use_hyperparameter_bo:
        print(f"\n🔧 Hyperparameter Bayesian Optimization:")
        print(f"  Enabled: Yes")
        print(f"  Data size category: {args.data_size}")
        print(f"  Pretrain BO trials: {args.pretrain_bo_trials}")
        print(f"  Finetune BO trials: {args.finetune_bo_trials}")
        print(f"  ⚠️  Note: This will significantly increase computation time!")
    else:
        print(f"\n🔧 Hyperparameter Bayesian Optimization:")
        print(f"  Enabled: No (using fixed hyperparameters)")
    
    # 점진적 학습 설정
    if args.use_incremental_learning and incremental_params:
        print(f"\n🔄 Incremental Learning:")
        print(f"  Enabled: Yes")
        print(f"  Mode: {incremental_params['mode']}")
        print(f"  Learning rate boost factor: {incremental_params['lr_boost_factor']}")
        print(f"  Incremental epochs: {incremental_params['incremental_epochs']}")
        print(f"  Replay ratio: {incremental_params['replay_ratio']}")
        print(f"  Weight decay factor: {incremental_params['weight_decay_factor']}")
        print(f"  Full retrain interval: {incremental_params['full_retrain_interval']}")
        if args.model_type == 'bnn':
            print(f"  KL regularization weight: {incremental_params['kl_reg_weight']}")
        print(f"  ⚡ Note: Models will update incrementally on new data!")
    else:
        print(f"\n🔄 Incremental Learning:")
        print(f"  Enabled: No (using full retraining)")
    
    print("=" * 60)


def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 데이터 로드
    print("Loading data...")
    lookup = load_lookup_table(DATA_PATHS['lookup_table'])
    label_maps = create_label_maps(PARAM_SPACE)
    param_space = create_param_space(lookup)
    
    # 모델 설정
    model_config = setup_model_config(args)
    
    # 점진적 학습 파라미터 설정
    incremental_params = setup_incremental_params(args)
    
    # 설정 출력
    print_configuration(args, param_space, model_config, incremental_params)
    
    # 하이퍼파라미터 BO 사용 시 경고
    if args.use_hyperparameter_bo:
        total_bo_trials = args.pretrain_bo_trials + args.finetune_bo_trials
        estimated_time_per_iter = total_bo_trials * 2  # 대략적인 추정
        print(f"\n⚠️  WARNING: Hyperparameter BO is enabled!")
        print(f"   Each iteration may take ~{estimated_time_per_iter} times longer.")
        print(f"   Consider using smaller trial numbers for testing.")
        
        if args.num_runs > 5:
            print(f"   Multiple runs with BO may take very long time!")
            response = input("   Continue? (y/N): ")
            if response.lower() != 'y':
                print("   Aborted.")
                return
    
    try:
        if args.num_runs == 1:
            # 단일 실행
            print(f"\nStarting single optimization run...")
            
            if args.model_type == 'dngo':
                result = single_optimization_run(
                    param_space=param_space,
                    label_maps=label_maps,
                    lookup=lookup,
                    cost_budget=args.cost_budget,
                    num_init_design=args.num_init_design,
                    high_fidelity_ratio=args.high_fidelity_ratio,
                    min_target=args.min_target,
                    random_state=42,
                    verbose=args.verbose,
                    model_config=model_config,
                    use_hyperparameter_bo=args.use_hyperparameter_bo,
                    pretrain_bo_trials=args.pretrain_bo_trials,
                    finetune_bo_trials=args.finetune_bo_trials,
                    data_size=args.data_size,
                    model_type='DNGO',
                    use_incremental_learning=args.use_incremental_learning,
                    incremental_params=incremental_params
                )
            elif args.model_type == 'dngo-ol':
                result = single_optimization_run_dngo_ol(
                    param_space=param_space,
                    label_maps=label_maps,
                    lookup=lookup,
                    cost_budget=args.cost_budget,
                    num_init_design=args.num_init_design,
                    high_fidelity_ratio=args.high_fidelity_ratio,
                    min_target=args.min_target,
                    random_state=42,
                    verbose=args.verbose,
                    model_config=model_config,
                    save_images=args.save_images,
                    images_dir=args.images_dir
                )
            else:  # BNN
                result = single_optimization_run(
                    param_space=param_space,
                    label_maps=label_maps,
                    lookup=lookup,
                    cost_budget=args.cost_budget,
                    num_init_design=args.num_init_design,
                    high_fidelity_ratio=args.high_fidelity_ratio,
                    min_target=args.min_target,
                    random_state=42,
                    verbose=args.verbose,
                    model_config=model_config,
                    use_hyperparameter_bo=args.use_hyperparameter_bo,
                    pretrain_bo_trials=args.pretrain_bo_trials,
                    finetune_bo_trials=args.finetune_bo_trials,
                    data_size=args.data_size,
                    model_type='BNN',
                    use_incremental_learning=args.use_incremental_learning,
                    incremental_params=incremental_params
                )
            
            # 결과 출력
            print(f"\n=== Single Run Results ===")
            print(f"Total cost: {result['total_cost']:.2f}")
            print(f"Best value found: {result['best_so_far']:.4f}")
            print(f"Total iterations: {result['iterations']}")
            print(f"Target achieved: {'Yes' if result['best_so_far'] <= args.min_target else 'No'}")
            
            # 하이퍼파라미터 기록 출력
            if args.use_hyperparameter_bo and result.get('hyperparameter_history'):
                print(f"\n🔧 Hyperparameter Optimization Summary:")
                print(f"   Total BO iterations: {len(result['hyperparameter_history'])}")
                
                # 마지막 최적 하이퍼파라미터 출력
                last_hp = result['hyperparameter_history'][-1]
                if args.model_type == 'dngo':
                    if last_hp.get('pretrain_params'):
                        print(f"   Final pretrain params: {last_hp['pretrain_params']}")
                    if last_hp.get('finetune_params'):
                        print(f"   Final finetune params: {last_hp['finetune_params']}")
                else:  # BNN
                    if last_hp.get('finetune_params'):
                        print(f"   Final BNN params: {last_hp['finetune_params']}")
            
            # 시각화 및 결과 저장
            if args.plot_results:
                print("Generating plots...")
                if args.model_type in ['dngo', 'dngo-ol'] and 'visualization_data' in result:
                    # DNGO 시각화 (3개 subplot 포함)
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                    images_dir = f"images/{args.model_type}_{timestamp}_run01"
                    print(f"📸 Creating DNGO visualizations in {images_dir}/")
                    
                    # 분리된 시각화 사용 여부
                    if args.separated_viz:
                        print("  Using separated LOFI/HIFI visualization...")
                        plot_optimization_progress_separated(result, save_dir=images_dir)
                    else:
                        plot_optimization_progress(result, save_dir=images_dir)
                    
                    print(f"✅ DNGO visualization completed: {images_dir}/")
                    
                    # 베이지안 최적화 결과 저장
                    from common.result_saver import save_optimization_results
                    save_optimization_results(result, images_dir, args.model_type)
                else:
                    # BNN 시각화 (파일로 저장)
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                    
                    # BNN용 시각화 디렉토리 생성
                    images_dir = f"images/bnn_{timestamp}_run01"
                    os.makedirs(images_dir, exist_ok=True)
                    
                    # plot_optimization_results를 파일로 저장
                    summary_path = os.path.join(images_dir, 'summary.png')
                    print(f"📸 Creating BNN visualization: {summary_path}")
                    plot_optimization_results(result, save_path=summary_path)
                    print(f"✅ BNN visualization completed: {summary_path}")
                
            # BNN + MPS 사용 시 메모리 정리
            if args.model_type == 'bnn' and model_config['device'] == 'mps':
                clear_mps_cache()
                print("🧹 MPS 캐시 정리 완료")
        
        else:
            # 다중 실행
            print(f"\nStarting {args.num_runs} optimization runs...")
            
            if args.model_type == 'dngo':
                results = multiple_optimization_runs(
                    param_space=param_space,
                    label_maps=label_maps,
                    lookup=lookup,
                    num_runs=args.num_runs,
                    cost_budget=args.cost_budget,
                    num_init_design=args.num_init_design,
                    high_fidelity_ratio=args.high_fidelity_ratio,
                    min_target=args.min_target,
                    model_config=model_config,
                    save_results=args.save_results,
                    results_filename=args.results_filename,
                    use_hyperparameter_bo=args.use_hyperparameter_bo,
                    pretrain_bo_trials=args.pretrain_bo_trials,
                    finetune_bo_trials=args.finetune_bo_trials,
                    data_size=args.data_size,
                    save_visualizations=args.plot_results,  # 시각화 저장 여부
                    visualizations_dir='images',  # 시각화 저장 디렉토리
                    model_type='DNGO',
                    use_incremental_learning=args.use_incremental_learning,
                    incremental_params=incremental_params
                )
            elif args.model_type == 'dngo-ol':
                results = multiple_optimization_runs_dngo_ol(
                    param_space=param_space,
                    label_maps=label_maps,
                    lookup=lookup,
                    num_runs=args.num_runs,
                    cost_budget=args.cost_budget,
                    num_init_design=args.num_init_design,
                    high_fidelity_ratio=args.high_fidelity_ratio,
                    min_target=args.min_target,
                    model_config=model_config,
                    save_results=args.save_results,
                    results_filename='dngo_ol_results.csv'
                )
            else:  # BNN
                results = multiple_optimization_runs(
                    param_space=param_space,
                    label_maps=label_maps,
                    lookup=lookup,
                    num_runs=args.num_runs,
                    cost_budget=args.cost_budget,
                    num_init_design=args.num_init_design,
                    high_fidelity_ratio=args.high_fidelity_ratio,
                    min_target=args.min_target,
                    model_config=model_config,
                    save_results=args.save_results,
                    results_filename=args.results_filename,
                    use_hyperparameter_bo=args.use_hyperparameter_bo,
                    pretrain_bo_trials=args.pretrain_bo_trials,
                    finetune_bo_trials=args.finetune_bo_trials,
                    data_size=args.data_size,
                    save_visualizations=args.plot_results,
                    visualizations_dir='images',
                    model_type='BNN',
                    use_incremental_learning=args.use_incremental_learning,
                    incremental_params=incremental_params
                )
            
            # 시각화
            if args.plot_results:
                print("Generating plots...")
                # 모든 모델 타입에 대해 동일한 박스플롯 시각화 사용
                from datetime import datetime
                timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                boxplot_path = f"images/{args.model_type}_{timestamp}_boxplot_{args.num_runs}runs.png"
                print(f"📊 Creating {args.model_type.upper()} boxplot: {boxplot_path}")
                
                # images 디렉토리 생성
                import os
                os.makedirs('images', exist_ok=True)
                
                plot_multiple_runs_boxplot(results, model_type=args.model_type.upper(), save_path=boxplot_path)
                print(f"✅ {args.model_type.upper()} boxplot completed: {boxplot_path}")
                
            # BNN + MPS 사용 시 메모리 정리
            if args.model_type == 'bnn' and model_config['device'] == 'mps':
                clear_mps_cache()
                print("🧹 MPS 캐시 정리 완료")
    
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nOptimization completed!")


if __name__ == "__main__":
    main() 