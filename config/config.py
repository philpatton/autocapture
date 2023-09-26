from typing import Optional

import os
import yaml
import logging
import argparse
import numpy as np
import pandas as pd

def parse():
    parser = argparse.ArgumentParser(description="Writing config files")
    parser.add_argument('-s', "--scenario")
    return parser.parse_args()

class Config(dict):
    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            return super().__getattr__(key)
        if isinstance(val, dict):
            return Config(val)
        return val

def load_config(path: str, default_path: Optional[str]) -> Config:
    with open(path) as f:
        cfg = Config(yaml.full_load(f))
    if default_path is not None:
        # set keys not included in `path` by default
        with open(default_path) as f:
            default_cfg = Config(yaml.full_load(f))
        for key, val in default_cfg.items():
            if key not in cfg:
                logging.debug(f"used default config {key}: {val}")
                cfg[key] = val
    return cfg

def scale_p(mean, p_max, p_min):

    scaled = ((p_max - p_min) * (mean - min(mean)) / 
            (max(mean) - min(mean))) + p_min
    
    return scaled

def write_configs():

    args = parse()
    SCENARIO = args.scenario

    config_dir = 'config/catalogs'
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)

    # hyperparameters for configs 
    MARK_CHANGE_RATE = 0.5
    GHOST_RATE = 1 - MARK_CHANGE_RATE
    B0 = 0.35
    T = 10
    PHI = 0.9
    RELEASED_COUNT = 25

    # entrance probabilities 
    b = np.zeros(T)
    b[0] = B0
    b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 
    b = b.tolist()

    # directories
    path = f'input/rates.csv'
    rates = pd.read_csv(path)

    cats = rates.catalog_id.unique()
    for cat in cats:
        
        cat_rates = rates.loc[rates.catalog_id== cat]
        matches_checked = cat_rates.matches_checked
        matches_checked = matches_checked.replace(1, 0)
        
        # error rates from each catalog
        false_reject_rate = cat_rates['FR']
        alpha = (false_reject_rate * GHOST_RATE).to_list()
        beta = (false_reject_rate * MARK_CHANGE_RATE).to_list()
        gamma = cat_rates['FA'].to_list()
        
        # translate number of training images per id to scale (P_MIN, P_MAX)
        p = cat_rates['p'].unique()[0].item()
        N = cat_rates['N'].unique()[0].item()

        keys = [f'check_{mc}' for mc in matches_checked]
        mini_dicts = [{'alpha':alpha[i], 'beta':beta[i], 'gamma':gamma[i]} 
                    for i in range(len(matches_checked))]
        
        catalog_dict = {
            'N': N,
            'p': p,
            'phi': PHI,
            'b': b,
            'T': T,
            'released_count': RELEASED_COUNT,
            keys[0]: mini_dicts[0],
            keys[1]: mini_dicts[1],
            keys[2]: mini_dicts[2],
            keys[3]: mini_dicts[3],
            keys[4]: mini_dicts[4],
            keys[5]: mini_dicts[5],
            'test': {'alpha':0.0, 'beta':0.0, 'gamma':0.0}
        }

        if SCENARIO == 'test':
            catalog_dict['tune'] = 1000
            catalog_dict['draws'] = 1000
            catalog_dict['trial_count'] = 30

        yaml_path = f'{config_dir}/{cat}.yaml'

        with open(yaml_path, 'w') as outfile:
            yaml.dump(catalog_dict, outfile, default_flow_style=False)

    debug_dict = {
        'N': 200,
        'T': 5,
        'p': 0.5,
        'phi': 0.5,
        'b': [0.2, 0.2, 0.2, 0.2, 0.2],
        'trial_count': 2,
        'debug': {
            'alpha': 0.,
            'beta': 0.,
            'gamma': 0.,
        }
    }
    yaml_path = f'{config_dir}/debug.yaml'
    with open(yaml_path, 'w') as outfile:
        yaml.dump(debug_dict, outfile, default_flow_style=False)

    return None    

if __name__ == '__main__':
    write_configs()
