"""
베이지안 최적화 결과 저장 모듈

선택된 하이퍼파라미터, 추천점, 성능 지표 등을 JSON 형태로 저장합니다.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


def save_optimization_results(results: Dict, save_dir: str, model_type: str = 'dngo'):
    """
    베이지안 최적화 결과를 저장
    
    Args:
        results: 최적화 결과 딕셔너리
        save_dir: 저장 디렉토리 (이미지와 동일한 폴더명)
        model_type: 모델 타입
    """
    # results 디렉토리 생성
    results_dir = save_dir.replace('images/', 'results/')
    os.makedirs(results_dir, exist_ok=True)
    
    # 전체 실행 요약 저장
    save_run_summary(results, results_dir)
    
    # 각 iteration별 상세 정보 저장
    if 'visualization_data' in results:
        save_iteration_details(results['visualization_data'], results_dir)
    
    # 하이퍼파라미터 최적화 기록 저장
    if results.get('use_hyperparameter_bo', False) and 'hyperparameter_history' in results:
        save_hyperparameter_history(results['hyperparameter_history'], results_dir)
    
    print(f"📋 Results saved to {results_dir}/")


def save_run_summary(results: Dict, results_dir: str):
    """실행 요약 정보 저장"""
    summary = {
        'run_info': {
            'total_cost': results.get('total_cost', 0),
            'best_so_far': results.get('best_so_far', float('inf')),
            'iterations': results.get('iterations', 0),
            'target_achieved': results.get('best_so_far', float('inf')) <= 1.5249,
            'timestamp': datetime.now().isoformat(),
            'use_hyperparameter_bo': results.get('use_hyperparameter_bo', False)
        },
        'optimization_config': results.get('config', {}),
        'final_performance': {
            'cost_history': results.get('cost_history', []),
            'best_values_history': results.get('best_values_history', []),
            'fidelity_history': results.get('fidelity_history', []),
            'ei_history': results.get('ei_history', [])
        }
    }
    
    # NumPy 배열을 리스트로 변환
    summary = convert_numpy_to_list(summary)
    
    with open(os.path.join(results_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_iteration_details(visualization_data: List[Dict], results_dir: str):
    """각 iteration별 상세 정보 저장"""
    iterations_dir = os.path.join(results_dir, 'iterations')
    os.makedirs(iterations_dir, exist_ok=True)
    
    iteration_summaries = []
    
    for viz_data in visualization_data:
        iter_num = viz_data['iteration']
        
        # 추천점 정보
        recommended_point = viz_data['recommended_point']
        best_idx = viz_data['best_idx']
        fidelity = viz_data['fidelity']
        
        # 성능 지표 계산
        y_pred = viz_data['y_pred']
        y_actual = viz_data.get('y_actual', None)
        ei = viz_data['ei']
        
        iteration_info = {
            'iteration': iter_num,
            'recommended_point': {
                'combination': recommended_point,
                'index': int(best_idx),
                'fidelity': fidelity,
                'expected_improvement': float(ei[best_idx]),
                'predicted_value': float(y_pred[best_idx])
            },
            'model_performance': calculate_model_performance(viz_data),
            'data_info': {
                'num_low_fidelity': len(viz_data['X_low']),
                'num_high_fidelity': len(viz_data['X_high']),
                'total_candidates': len(viz_data['X_grid'])
            }
        }
        
        # 하이퍼파라미터 정보 추가
        if 'hyperparameters' in viz_data:
            iteration_info['hyperparameters'] = viz_data['hyperparameters']
        
        # 실제값 정보 추가 (있는 경우)
        if y_actual is not None:
            iteration_info['recommended_point']['actual_value'] = float(y_actual[best_idx])
        
        iteration_summaries.append(iteration_info)
        
        # 개별 iteration 파일로도 저장
        with open(os.path.join(iterations_dir, f'iteration_{iter_num:03d}.json'), 'w') as f:
            json.dump(convert_numpy_to_list(iteration_info), f, indent=2, ensure_ascii=False)
    
    # 전체 iteration 요약
    with open(os.path.join(results_dir, 'iterations_summary.json'), 'w') as f:
        json.dump(convert_numpy_to_list(iteration_summaries), f, indent=2, ensure_ascii=False)


def save_hyperparameter_history(hyperparameter_history: List[Dict], results_dir: str):
    """하이퍼파라미터 최적화 기록 저장"""
    hp_dir = os.path.join(results_dir, 'hyperparameters')
    os.makedirs(hp_dir, exist_ok=True)
    
    # 하이퍼파라미터 기록을 DataFrame으로 변환하여 CSV로 저장
    hp_records = []
    for record in hyperparameter_history:
        flat_record = {
            'iteration': record['iteration']
        }
        
        # Pretrain 파라미터 평탄화
        if record.get('pretrain_params'):
            for key, value in record['pretrain_params'].items():
                flat_record[f'pretrain_{key}'] = value
        
        # Finetune 파라미터 평탄화
        if record.get('finetune_params'):
            for key, value in record['finetune_params'].items():
                flat_record[f'finetune_{key}'] = value
        
        hp_records.append(flat_record)
    
    # CSV로 저장
    df = pd.DataFrame(hp_records)
    df.to_csv(os.path.join(hp_dir, 'hyperparameter_history.csv'), index=False)
    
    # JSON으로도 저장
    with open(os.path.join(hp_dir, 'hyperparameter_history.json'), 'w') as f:
        json.dump(convert_numpy_to_list(hyperparameter_history), f, indent=2, ensure_ascii=False)


def calculate_model_performance(viz_data: Dict) -> Dict:
    """모델 성능 지표 계산"""
    from sklearn.metrics import mean_squared_error, r2_score
    
    performance = {}
    
    # High-fidelity 성능
    if len(viz_data['X_high']) > 0:
        # High-fidelity 훈련 데이터에 대한 예측
        X_high = viz_data['X_high']
        y_high = viz_data['y_high']
        
        # 예측값 계산 (모델과 BLR 사용)
        model = viz_data['model']
        blr = viz_data['blr']
        
        features_high = model.extract_features(X_high)
        y_pred_high = []
        for phi in features_high:
            mu, _ = blr.predict(phi)
            y_pred_high.append(mu)
        y_pred_high = np.array(y_pred_high)
        
        # 성능 지표 계산
        mse_high = mean_squared_error(y_high, y_pred_high)
        rmse_high = np.sqrt(mse_high)
        r2_high = r2_score(y_high, y_pred_high)
        
        performance['high_fidelity'] = {
            'mse': float(mse_high),
            'rmse': float(rmse_high),
            'r2': float(r2_high),
            'num_samples': len(y_high)
        }
    
    # Low-fidelity 성능
    if len(viz_data['X_low']) > 0:
        X_low = viz_data['X_low']
        y_low = viz_data['y_low']
        
        model = viz_data['model']
        blr = viz_data['blr']
        
        features_low = model.extract_features(X_low)
        y_pred_low = []
        for phi in features_low:
            mu, _ = blr.predict(phi)
            y_pred_low.append(mu)
        y_pred_low = np.array(y_pred_low)
        
        mse_low = mean_squared_error(y_low, y_pred_low)
        rmse_low = np.sqrt(mse_low)
        r2_low = r2_score(y_low, y_pred_low)
        
        performance['low_fidelity'] = {
            'mse': float(mse_low),
            'rmse': float(rmse_low),
            'r2': float(r2_low),
            'num_samples': len(y_low)
        }
    
    return performance


def convert_numpy_to_list(obj: Any) -> Any:
    """NumPy 배열과 스칼라를 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_list(item) for item in obj)
    else:
        return obj