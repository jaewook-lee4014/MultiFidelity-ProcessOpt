import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any


def load_lookup_table(file_path: str) -> Dict:
    """Lookup table 로드"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_label_maps(param_space: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    파라미터 공간에서 라벨 매핑 생성
    
    Args:
        param_space: 파라미터 공간 정의
        
    Returns:
        라벨 매핑 딕셔너리
    """
    label_maps = {
        key: {val: idx for idx, val in enumerate(vals, 1)}
        for key, vals in param_space.items()
    }
    return label_maps


def sample_param_space(param_space: Dict[str, List[str]], n_samples: int, random_state: int = None) -> List[Dict]:
    """
    파라미터 공간에서 랜덤 샘플링
    
    Args:
        param_space: 파라미터 공간
        n_samples: 샘플 개수
        random_state: 랜덤 시드
        
    Returns:
        샘플링된 파라미터 조합 리스트
    """
    rng = np.random.default_rng(random_state)
    samples = []
    
    for _ in range(n_samples):
        sample = {}
        for key, opts in param_space.items():
            if isinstance(opts, (list, tuple)) and len(opts) > 0:
                if isinstance(opts[0], (str, int)):  # 범주형
                    sample[key] = rng.choice(opts)
                elif isinstance(opts, tuple) and len(opts) == 2:  # 연속형
                    low, high = opts
                    sample[key] = float(rng.uniform(low, high))
            else:
                raise ValueError(f"Unknown parameter type for {key}: {opts}")
        samples.append(sample)
    
    return samples


def assign_fidelities(n_samples: int, high_ratio: float = 0.1, random_state: int = None) -> List[float]:
    """
    샘플에 대한 fidelity 할당
    
    Args:
        n_samples: 전체 샘플 개수
        high_ratio: high-fidelity 비율
        random_state: 랜덤 시드
        
    Returns:
        fidelity 값 리스트 (0.1 또는 1.0)
    """
    rng = np.random.default_rng(random_state)
    n_high = max(1, int(round(n_samples * high_ratio)))
    n_low = n_samples - n_high
    fids = [1.0] * n_high + [0.1] * n_low
    rng.shuffle(fids)
    return fids


def measure_from_label(label_arr: List[int], s: float, label_maps: Dict, lookup: Dict) -> float:
    """
    라벨 배열에서 측정값 계산
    
    Args:
        label_arr: [organic_label, cation_label, anion_label]
        s: fidelity (0.1 또는 1.0)
        label_maps: 라벨 매핑
        lookup: lookup table
        
    Returns:
        측정값
    """
    # 라벨 역변환
    reverse_maps = {
        "organic": {v: k for k, v in label_maps["organic"].items()},
        "cation": {v: k for k, v in label_maps["cation"].items()},
        "anion": {v: k for k, v in label_maps["anion"].items()},
    }
    
    organic = reverse_maps["organic"][int(label_arr[0])]
    cation = reverse_maps["cation"][int(label_arr[1])]
    anion = reverse_maps["anion"][int(label_arr[2])]

    # 측정값 계산
    if s == 1.0:
        measurement = np.amin(
            lookup[organic.capitalize()][cation][anion]['bandgap_hse06']
        )
    elif s == 0.1:
        measurement = np.amin(
            lookup[organic.capitalize()][cation][anion]['bandgap_gga']
        )
    else:
        raise ValueError("fidelity는 0.1 또는 1.0만 가능합니다.")
    
    return measurement


def append_measurement_to_data(existing_X_low: np.ndarray, existing_y_low: np.ndarray,
                               existing_X_high: np.ndarray, existing_y_high: np.ndarray,
                               label_arr: List[int], s: float, label_maps: Dict, lookup: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    기존 데이터에 새로운 측정값 추가
    
    Args:
        existing_X_low: 기존 low-fidelity X 데이터
        existing_y_low: 기존 low-fidelity y 데이터
        existing_X_high: 기존 high-fidelity X 데이터
        existing_y_high: 기존 high-fidelity y 데이터
        label_arr: 새로운 측정 라벨
        s: fidelity
        label_maps: 라벨 매핑
        lookup: lookup table
        
    Returns:
        업데이트된 데이터 (X_low, y_low, X_high, y_high)
    """
    # 측정값 계산
    measurement = measure_from_label(label_arr, s, label_maps, lookup)
    
    # 데이터 추가
    label_arr = np.array(label_arr, dtype=np.float32).reshape(1, -1)
    measurement = np.array([measurement], dtype=np.float32)

    if s == 0.1:
        existing_X_low = np.vstack([existing_X_low, label_arr]) if existing_X_low.size else label_arr
        existing_y_low = np.concatenate([existing_y_low, measurement]) if existing_y_low.size else measurement
    elif s == 1.0:
        existing_X_high = np.vstack([existing_X_high, label_arr]) if existing_X_high.size else label_arr
        existing_y_high = np.concatenate([existing_y_high, measurement]) if existing_y_high.size else measurement
    else:
        raise ValueError("fidelity는 0.1 또는 1.0만 가능합니다.")

    return existing_X_low, existing_y_low, existing_X_high, existing_y_high


def create_all_combinations_data(param_ranges: List[range], lookup: Dict, 
                                organic_options: List[str], cation_options: List[str], 
                                anion_options: List[str]) -> pd.DataFrame:
    """
    모든 가능한 조합에 대한 데이터 생성
    
    Args:
        param_ranges: 파라미터 범위 리스트
        lookup: lookup table
        organic_options: organic 옵션 리스트
        cation_options: cation 옵션 리스트
        anion_options: anion 옵션 리스트
        
    Returns:
        모든 조합에 대한 DataFrame
    """
    all_results = []
    
    for i, organic in enumerate(organic_options, 1):
        for j, cation in enumerate(cation_options, 1):
            for k, anion in enumerate(anion_options, 1):
                try:
                    bandgap = np.amin(
                        lookup[organic.capitalize()][cation][anion]['bandgap_hse06']
                    )
                    combo_label = f"{i},{j},{k}"
                    all_results.append({
                        'combo': combo_label,
                        'bandgap_hse06': bandgap
                    })
                except Exception as e:
                    print(f"Skip: {organic}-{cation}-{anion} ({e})")
                    continue
    
    return pd.DataFrame(all_results)


def prepare_initial_data(init_samples: List[Dict], init_fids: List[float], 
                        label_maps: Dict, lookup: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    초기 데이터 준비
    
    Args:
        init_samples: 초기 샘플 리스트
        init_fids: 초기 fidelity 리스트
        label_maps: 라벨 매핑
        lookup: lookup table
        
    Returns:
        초기 데이터 (X_low, y_low, X_high, y_high)
    """
    measurements = []
    for params, s in zip(init_samples, init_fids):
        measurement = measure_from_label(
            [label_maps['organic'][params['organic']],
             label_maps['cation'][params['cation']],
             label_maps['anion'][params['anion']]],
            s, label_maps, lookup
        )
        measurements.append({"params": params, "s": s, "measurement": measurement})

    # 데이터프레임 변환
    df = pd.DataFrame([
        {**obs['params'], 's': obs['s'], 'measurement': obs['measurement']}
        for obs in measurements
    ])
    
    for col in ['organic', 'cation', 'anion']:
        df[col + '_label'] = df[col].map(label_maps[col])

    ini_X = df[['organic_label', 'cation_label', 'anion_label', 's']].values
    ini_y = df['measurement'].values

    # fidelity별 분할
    ini_X_low = ini_X[df['s'] == 0.1][:, :3]
    ini_y_low = ini_y[df['s'] == 0.1]
    ini_X_high = ini_X[df['s'] == 1.0][:, :3]
    ini_y_high = ini_y[df['s'] == 1.0]

    return ini_X_low, ini_y_low, ini_X_high, ini_y_high 