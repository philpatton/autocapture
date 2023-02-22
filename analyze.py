import numpy as np
import arviz as az

import argparse
import os

from config import load_config, Config

def parse():
    parser = argparse.ArgumentParser(description="Analyzing results")
    parser.add_argument("--scenario", default="test")
    return parser.parse_args()

def summarize_json():

    args = parse()

    catalog_ids = np.load('input/catalog_ids.npy', allow_pickle=True)

    for catalog in catalog_ids:

        config_path = f'config/{args.scenario}/{catalog}.yaml'
        cfg = load_config(config_path, "config/default.yaml")
        trial_count = cfg.trial_count

        summary_dir = f'/results/{args.scenario}/{catalog}/summary'
        if not os.path.isdir(summary_dir):
            os.makedirs(summary_dir)

        for trial in trial_count:
            path = f'/results/{args.scenario}/{catalog}/trial_{trial}.json'
            idata = az.from_json(path)  

            summary = az.summary(idata)
            out_file = f'{summary_dir}/trial_{trial}.csv'
            summary.to_csv(out_file)

    return None

if __name__ == '__main__':
    summarize_json()
