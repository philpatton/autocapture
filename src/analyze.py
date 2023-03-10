import numpy as np
import arviz as az

import argparse
import os

from src.config import load_config, Config

def parse():
    parser = argparse.ArgumentParser(description="Analyzing results")
    parser.add_argument("--scenario", default="test")
    return parser.parse_args()

def summarize_json():

    args = parse()

    catalog_ids = np.load('input/catalog_ids.npy', allow_pickle=True)

    for catalog in catalog_ids:

        results_dir = f'results/{args.scenario}/{catalog}'

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

            summary = az.summary(idata)
            out_file = f'{summary_dir}/trial_{trial}.csv'
            summary.to_csv(out_file)

        if i != 0:
            print(f'{catalog} had {i} missing json files')

    return None

if __name__ == '__main__':
    summarize_json()
