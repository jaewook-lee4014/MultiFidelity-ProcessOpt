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
from config import *
from data_utils import (
    load_lookup_table, create_label_maps, sample_param_space, 
    assign_fidelities, prepare_initial_data, create_all_combinations_data,
    create_param_space
)
from optimization import single_optimization_run, multiple_optimization_runs
from visualization import (
    plot_iteration_results, plot_prediction_scatter, 
    plot_multiple_runs_summary, plot_learning_curves,
    plot_optimization_results
)

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


def setup_device():
    """디바이스 설정 (GPU/CPU)"""
    if DEVICE_CONFIG['use_gpu'] and torch.cuda.is_available():
        device = 'cuda'
        print("CUDA 사용 가능합니다.")
    elif DEVICE_CONFIG['use_gpu'] and torch.backends.mps.is_available():
        device = 'mps'
        print("MPS 사용 가능합니다.")
    else:
        device = 'cpu'
        print("CPU를 사용합니다.")
    return device


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='Transfer Learning Bayesian Optimization with Hyperparameter BO'
    )
    
    # 기본 실행 옵션
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single',
                       help='실행 모드: single run 또는 multiple runs')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Multiple runs 모드에서 실행 횟수')
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
    parser.add_argument('--device', default='cpu',
                       help='PyTorch 디바이스 (cpu/cuda)')
    
    # 하이퍼파라미터 BO 옵션
    parser.add_argument('--use-hyperparameter-bo', action='store_true',
                       help='하이퍼파라미터 베이지안 최적화 사용')
    parser.add_argument('--pretrain-bo-trials', type=int, default=5,
                       help='Pretrain 하이퍼파라미터 BO 시행 횟수')
    parser.add_argument('--finetune-bo-trials', type=int, default=5,
                       help='Finetune 하이퍼파라미터 BO 시행 횟수')
    parser.add_argument('--data-size', choices=['small', 'medium', 'large'], default='small',
                       help='데이터 크기 (하이퍼파라미터 탐색 공간 결정)')
    
    # 결과 저장
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='결과를 CSV 파일로 저장')
    parser.add_argument('--results-filename', default='tl_bo_results.csv',
                       help='결과 파일명')
    parser.add_argument('--plot-results', action='store_true',
                       help='결과 시각화')
    
    return parser.parse_args()


def setup_model_config(args):
    """모델 설정 딕셔너리 생성"""
    return {
        'input_dim': 3,  # organic, cation, anion
        'hidden_dim': args.hidden_dim,
        'pretrain_epochs': args.pretrain_epochs,
        'finetune_epochs': args.finetune_epochs,
        'device': args.device
    }


def print_configuration(args, param_space, model_config):
    """설정 정보 출력"""
    print("=" * 60)
    print("Transfer Learning Bayesian Optimization Configuration")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    if args.mode == 'multiple':
        print(f"Number of runs: {args.num_runs}")
    
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
    print(f"  Hidden dimension: {model_config['hidden_dim']}")
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
    
    # 설정 출력
    print_configuration(args, param_space, model_config)
    
    # 하이퍼파라미터 BO 사용 시 경고
    if args.use_hyperparameter_bo:
        total_bo_trials = args.pretrain_bo_trials + args.finetune_bo_trials
        estimated_time_per_iter = total_bo_trials * 2  # 대략적인 추정
        print(f"\n⚠️  WARNING: Hyperparameter BO is enabled!")
        print(f"   Each iteration may take ~{estimated_time_per_iter} times longer.")
        print(f"   Consider using smaller trial numbers for testing.")
        
        if args.mode == 'multiple' and args.num_runs > 5:
            print(f"   Multiple runs with BO may take very long time!")
            response = input("   Continue? (y/N): ")
            if response.lower() != 'y':
                print("   Aborted.")
                return
    
    try:
        if args.mode == 'single':
            # 단일 실행
            print(f"\nStarting single optimization run...")
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
                data_size=args.data_size
            )
            
            # 결과 출력
            print(f"\n=== Single Run Results ===")
            print(f"Total cost: {result['total_cost']:.2f}")
            print(f"Best value found: {result['best_so_far']:.4f}")
            print(f"Total iterations: {result['iterations']}")
            print(f"Target achieved: {'Yes' if result['best_so_far'] <= args.min_target else 'No'}")
            
            # 하이퍼파라미터 기록 출력
            if args.use_hyperparameter_bo and result['hyperparameter_history']:
                print(f"\n🔧 Hyperparameter Optimization Summary:")
                print(f"   Total BO iterations: {len(result['hyperparameter_history'])}")
                
                # 마지막 최적 하이퍼파라미터 출력
                last_hp = result['hyperparameter_history'][-1]
                if last_hp['pretrain_params']:
                    print(f"   Final pretrain params: {last_hp['pretrain_params']}")
                if last_hp['finetune_params']:
                    print(f"   Final finetune params: {last_hp['finetune_params']}")
            
            # 시각화
            if args.plot_results:
                print("Generating plots...")
                plot_optimization_results([result], save_path='single_run_results.png')
        
        else:
            # 다중 실행
            print(f"\nStarting {args.num_runs} optimization runs...")
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
                data_size=args.data_size
            )
            
            # 시각화
            if args.plot_results:
                print("Generating plots...")
                plot_optimization_results(results, save_path='multiple_runs_results.png')
    
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nOptimization completed!")


if __name__ == "__main__":
    main() 