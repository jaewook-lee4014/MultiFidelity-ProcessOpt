"""
Experiment Runner for Model Comparison
재현 가능한 실험 실행을 위한 유틸리티
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import random
import hashlib

class ExperimentRunner:
    """실험 실행 및 관리 클래스"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.configs_dir = self.base_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)
        
    def set_seed(self, seed: int):
        """모든 랜덤 시드 설정 (재현성 보장)"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def create_experiment_id(self, config: Dict) -> str:
        """설정 기반 고유 실험 ID 생성"""
        config_str = json.dumps(config, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{hash_obj.hexdigest()[:8]}"
    
    def save_config(self, exp_id: str, config: Dict):
        """실험 설정 저장"""
        config_path = self.configs_dir / f"{exp_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return config_path
    
    def save_results(self, exp_id: str, results: Dict):
        """실험 결과 저장"""
        # Pickle로 전체 결과 저장
        pickle_path = self.results_dir / f"{exp_id}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # CSV로 주요 메트릭 저장
        if 'metrics' in results:
            csv_path = self.results_dir / f"{exp_id}_metrics.csv"
            df = pd.DataFrame([results['metrics']])
            df.to_csv(csv_path, index=False)
        
        return pickle_path
    
    def load_results(self, exp_id: str) -> Dict:
        """저장된 결과 로드"""
        pickle_path = self.results_dir / f"{exp_id}_results.pkl"
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    def run_experiment(self, model, config: Dict, verbose: bool = True) -> Dict:
        """단일 실험 실행"""
        # 시드 설정
        self.set_seed(config.get('seed', 42))
        
        # 실험 ID 생성
        exp_id = self.create_experiment_id(config)
        
        # 설정 저장
        self.save_config(exp_id, config)
        
        if verbose:
            print(f"Starting experiment: {exp_id}")
            print(f"Model: {config.get('model_type', 'unknown')}")
            print(f"Seed: {config.get('seed', 42)}")
        
        # 실험 실행 (여기에 실제 BO 로직 연결)
        from optimization import single_optimization_run
        results = single_optimization_run(
            model=model,
            budget=config.get('budget', 50),
            target_value=config.get('target_value', 1.34),
            verbose=verbose
        )
        
        # 메타데이터 추가
        results['experiment_id'] = exp_id
        results['config'] = config
        results['timestamp'] = datetime.now().isoformat()
        
        # 결과 저장
        self.save_results(exp_id, results)
        
        if verbose:
            print(f"Experiment completed: {exp_id}")
            print(f"Best value: {results.get('best_value', 'N/A')}")
            print(f"Total cost: {results.get('total_cost', 'N/A')}")
        
        return results
    
    def run_comparison(self, models: Dict[str, Any], base_config: Dict, 
                      n_seeds: int = 5, verbose: bool = True) -> pd.DataFrame:
        """여러 모델 비교 실험"""
        all_results = []
        
        for model_name, model_obj in models.items():
            for seed in range(n_seeds):
                config = base_config.copy()
                config['model_type'] = model_name
                config['seed'] = seed
                
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"Running {model_name} with seed {seed}")
                    print('='*50)
                
                results = self.run_experiment(model_obj, config, verbose)
                
                # 결과 요약
                summary = {
                    'model': model_name,
                    'seed': seed,
                    'best_value': results.get('best_value'),
                    'total_cost': results.get('total_cost'),
                    'n_iterations': results.get('n_iterations'),
                    'experiment_id': results['experiment_id']
                }
                all_results.append(summary)
        
        # DataFrame으로 변환
        results_df = pd.DataFrame(all_results)
        
        # 통계 요약
        if verbose:
            print("\n" + "="*60)
            print("COMPARISON SUMMARY")
            print("="*60)
            summary_stats = results_df.groupby('model').agg({
                'best_value': ['mean', 'std', 'min'],
                'total_cost': ['mean', 'std'],
                'n_iterations': ['mean', 'std']
            })
            print(summary_stats)
        
        # 결과 저장
        comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"comparison_{comparison_id}.csv"
        results_df.to_csv(csv_path, index=False)
        
        return results_df


class ModelRegistry:
    """모델 레지스트리 (쉬운 모델 교체를 위함)"""
    
    def __init__(self):
        self._models = {}
        self._configs = {}
    
    def register(self, name: str, model_class, default_config: Dict = None):
        """모델 등록"""
        self._models[name] = model_class
        if default_config:
            self._configs[name] = default_config
    
    def create(self, name: str, **kwargs) -> Any:
        """모델 인스턴스 생성"""
        if name not in self._models:
            raise ValueError(f"Model {name} not registered. Available: {list(self._models.keys())}")
        
        # 기본 설정 로드
        config = self._configs.get(name, {}).copy()
        config.update(kwargs)
        
        # 모델 생성
        return self._models[name](**config)
    
    def list_models(self) -> List[str]:
        """등록된 모델 목록"""
        return list(self._models.keys())
    
    def get_config(self, name: str) -> Dict:
        """모델 기본 설정 반환"""
        return self._configs.get(name, {})


# 전역 레지스트리 인스턴스
model_registry = ModelRegistry()


def register_default_models():
    """기본 모델들 등록"""
    from models import TransferLearningDNN, TransferLearningDNNWithHyperOpt
    
    # Transfer Learning DNN
    model_registry.register(
        'tl_dnn',
        TransferLearningDNN,
        {
            'input_dim': 3,
            'hidden_dim': 128,
            'output_dim': 1,
            'n_layers': 3,
            'dropout_rate': 0.1
        }
    )
    
    # Transfer Learning DNN with Hyperparameter Optimization
    model_registry.register(
        'tl_dnn_hyperopt',
        TransferLearningDNNWithHyperOpt,
        {
            'input_dim': 3,
            'output_dim': 1,
            'use_hyperparameter_bo': True
        }
    )
    
    # 추가 모델들을 여기에 등록
    # model_registry.register('gp', GaussianProcessModel, {...})
    # model_registry.register('rf', RandomForestModel, {...})


# 초기화 시 기본 모델 등록
register_default_models()