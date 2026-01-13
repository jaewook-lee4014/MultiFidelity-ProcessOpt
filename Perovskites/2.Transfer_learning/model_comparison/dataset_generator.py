"""
Dataset Generator for Model Comparison Experiments

8:1 ratio (lofi:hifi) dataset generation:
- Cost 15: 72 lofi + 9 hifi (total cost = 72*0.125 + 9*1 = 9 + 9 = 18, approx 15)
- Cost 30: 136 lofi + 17 hifi (total cost = 136*0.125 + 17*1 = 17 + 17 = 34, approx 30)

Note: Using cost ratio lofi=0.125, hifi=1.0 (8:1 ratio)
"""

import numpy as np
import pickle
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR.parent / 'Pure_TL_BO'))

from common.data_utils import (
    load_lookup_table, create_param_space, create_label_maps,
    create_all_combinations_data
)


def generate_random_dataset(
    lookup: Dict,
    param_space: Dict,
    label_maps: Dict,
    n_lofi: int,
    n_hifi: int,
    random_state: int = None
) -> Dict:
    """
    Generate a random dataset with specified lofi and hifi counts

    Args:
        lookup: Lookup table with bandgap data
        param_space: Parameter space definition
        label_maps: Label mappings
        n_lofi: Number of low-fidelity samples
        n_hifi: Number of high-fidelity samples
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with X_low, y_low, X_high, y_high arrays
    """
    rng = np.random.default_rng(random_state)

    # Get all possible combinations
    organics = param_space['organic']
    cations = param_space['cation']
    anions = param_space['anion']

    # Generate all combinations as label arrays
    all_combinations = []
    for i, org in enumerate(organics, 1):
        for j, cat in enumerate(cations, 1):
            for k, ani in enumerate(anions, 1):
                all_combinations.append([i, j, k])

    all_combinations = np.array(all_combinations)
    n_total = len(all_combinations)

    # Randomly select indices for lofi and hifi (can overlap for multi-fidelity)
    lofi_indices = rng.choice(n_total, size=n_lofi, replace=False)
    hifi_indices = rng.choice(n_total, size=n_hifi, replace=False)

    # Get X arrays
    X_low = all_combinations[lofi_indices].astype(np.float32)
    X_high = all_combinations[hifi_indices].astype(np.float32)

    # Get y values from lookup table
    def get_bandgap(label_arr, fidelity):
        reverse_maps = {
            "organic": {v: k for k, v in label_maps["organic"].items()},
            "cation": {v: k for k, v in label_maps["cation"].items()},
            "anion": {v: k for k, v in label_maps["anion"].items()},
        }

        organic = reverse_maps["organic"][int(label_arr[0])]
        cation = reverse_maps["cation"][int(label_arr[1])]
        anion = reverse_maps["anion"][int(label_arr[2])]

        if fidelity == 'low':
            return np.amin(lookup[organic.capitalize()][cation][anion]['bandgap_gga'])
        else:  # high
            return np.amin(lookup[organic.capitalize()][cation][anion]['bandgap_hse06'])

    y_low = np.array([get_bandgap(x, 'low') for x in X_low], dtype=np.float32)
    y_high = np.array([get_bandgap(x, 'high') for x in X_high], dtype=np.float32)

    # Calculate actual cost
    actual_cost = n_lofi * 0.125 + n_hifi * 1.0

    return {
        'X_low': X_low,
        'y_low': y_low,
        'X_high': X_high,
        'y_high': y_high,
        'n_lofi': n_lofi,
        'n_hifi': n_hifi,
        'actual_cost': actual_cost,
        'random_state': random_state
    }


def generate_all_datasets(output_dir: str, n_sets_per_config: int = 5):
    """
    Generate all datasets for model comparison experiments

    Configurations:
    - cost_15: 72 lofi + 9 hifi (cost ~= 18)
    - cost_30: 136 lofi + 17 hifi (cost ~= 34)

    Args:
        output_dir: Directory to save datasets
        n_sets_per_config: Number of random sets per configuration
    """
    # Load lookup table
    lookup_path = SCRIPT_DIR.parent.parent / '0.Data' / 'lookup_table.pkl'
    lookup = load_lookup_table(str(lookup_path))

    # Create parameter space and label maps
    param_space = create_param_space(lookup)
    label_maps = create_label_maps(param_space)

    # Save param_space and label_maps for later use
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'param_space.json'), 'w') as f:
        json.dump(param_space, f, indent=2)

    with open(os.path.join(output_dir, 'label_maps.json'), 'w') as f:
        json.dump(label_maps, f, indent=2)

    # Dataset configurations
    configs = {
        'cost_15': {'n_lofi': 72, 'n_hifi': 9},    # cost = 72*0.125 + 9 = 18
        'cost_30': {'n_lofi': 136, 'n_hifi': 17},  # cost = 136*0.125 + 17 = 34
    }

    # Generate datasets
    all_datasets = {}

    for config_name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Generating {config_name} datasets ({n_sets_per_config} sets)")
        print(f"  n_lofi: {config['n_lofi']}, n_hifi: {config['n_hifi']}")
        print(f"{'='*50}")

        config_datasets = []

        for set_idx in range(n_sets_per_config):
            # Use different random seed for each set
            random_state = set_idx * 1000 + hash(config_name) % 1000

            dataset = generate_random_dataset(
                lookup=lookup,
                param_space=param_space,
                label_maps=label_maps,
                n_lofi=config['n_lofi'],
                n_hifi=config['n_hifi'],
                random_state=random_state
            )

            config_datasets.append(dataset)

            print(f"  Set {set_idx + 1}/{n_sets_per_config}: "
                  f"lofi={len(dataset['X_low'])}, hifi={len(dataset['X_high'])}, "
                  f"cost={dataset['actual_cost']:.1f}")

        all_datasets[config_name] = config_datasets

        # Save config datasets
        config_path = os.path.join(output_dir, f'{config_name}_datasets.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config_datasets, f)
        print(f"  Saved to: {config_path}")

    # Save all datasets together
    all_path = os.path.join(output_dir, 'all_datasets.pkl')
    with open(all_path, 'wb') as f:
        pickle.dump(all_datasets, f)
    print(f"\nAll datasets saved to: {all_path}")

    # Generate summary
    summary = {
        'configs': configs,
        'n_sets_per_config': n_sets_per_config,
        'total_datasets': len(configs) * n_sets_per_config,
        'param_space_size': {
            'organics': len(param_space['organic']),
            'cations': len(param_space['cation']),
            'anions': len(param_space['anion']),
            'total_combinations': len(param_space['organic']) * len(param_space['cation']) * len(param_space['anion'])
        }
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("Dataset Generation Summary")
    print(f"{'='*50}")
    print(f"Total combinations in dataset: {summary['param_space_size']['total_combinations']}")
    print(f"Total datasets generated: {summary['total_datasets']}")
    for config_name, config in configs.items():
        expected_cost = config['n_lofi'] * 0.125 + config['n_hifi'] * 1.0
        print(f"  {config_name}: {config['n_lofi']} lofi + {config['n_hifi']} hifi (cost={expected_cost:.1f})")

    return all_datasets


def load_datasets(datasets_dir: str) -> Dict:
    """Load all datasets from directory"""
    all_path = os.path.join(datasets_dir, 'all_datasets.pkl')
    with open(all_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # Generate datasets
    output_dir = SCRIPT_DIR / 'datasets'
    generate_all_datasets(str(output_dir), n_sets_per_config=5)
