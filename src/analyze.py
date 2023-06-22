import numpy as np
import arviz as az

import argparse
import os
import json

from src.config import load_config, Config
from src.cjs import CJS
from src.popan import POPAN

def parse():
    parser = argparse.ArgumentParser(description="Analyzing results")
    parser.add_argument('-s', "--scenario", default="test")
    parser.add_argument('-e', "--estimator", default="popan")
    return parser.parse_args()

def summarize_json():

    args = parse()

    catalog_ids = np.load('input/catalog_ids.npy', allow_pickle=True)

    for catalog in catalog_ids:

        results_dir = f'results/{args.scenario}/{catalog}'
        data_dir = f'sim_data/{args.scenario}/{catalog}'

        config_path = f'config/{args.scenario}/{catalog}.yaml'
        cfg = load_config(config_path, "config/default.yaml")
        trial_count = cfg.trial_count

        summary_dir = f'results/{args.scenario}/summaries/{catalog}/'
        if os.path.isdir(summary_dir):
            continue
        else:
            os.makedirs(summary_dir)

        i = 0
        for trial in range(trial_count):

            path = f'{results_dir}/trial_{trial}.json'
            if os.path.exists(path):
                idata = az.from_json(path)
            else:
                print(f'{path} is missing')
                i += 1
                continue
            
            # calculate summary statistics 
            summary = az.summary(idata)

            # add the p value to the summary, after selecting  method
            if args.estimator == 'popan':
                method = POPAN()
            elif args.estimator == 'cjs':
                method = CJS()

            # load in data 
            data_path = f'{data_dir}/trial_{trial}.json'
            with open(data_path, 'r') as f:
                data = json.loads(json.load(f))
            
            # perform check 
            ch = np.array(data['capture_history'])
            check_results = method.check(idata, ch)

            # calculate p value
            ft_obs = check_results['freeman_tukey_observed']
            ft_new = check_results['freeman_tukey_new']
            p_val = (ft_new > ft_obs).mean()

            summary['p_val'] = p_val
            
            # export file 
            out_file = f'{summary_dir}/trial_{trial}.csv'
            summary.to_csv(out_file)

        if i != 0:
            print(f'{catalog} had {i} missing json files')

    return None

if __name__ == '__main__':
    summarize_json()
