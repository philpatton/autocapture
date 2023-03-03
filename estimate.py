import numpy as np 
import pymc as pm
import pymc.sampling_jax

import time
import json 
import argparse
import os 
import logging

from config import load_config, Config
from popan import POPANEstimator

def parse():
    parser = argparse.ArgumentParser(description="Estimating Jolly-Seber")
    parser.add_argument("--scenario", default="debug")
    parser.add_argument('--no_jax', action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def main():

    args = parse()

    # TODO: Figure out logging 
    logging.basicConfig(filename=f'results/{args.scenario}.log')

    id_path = ('input/catalog_ids.npy')
    catalog_ids = np.load(id_path, allow_pickle=True)

    if args.scenario == 'debug':

        results_dir =  'results/debug/debug'
        os.makedirs(results_dir, exist_ok=True)
        analyze_catalog('debug', 'debug', args.no_jax)

    else:
        for catalog in catalog_ids:
            # pass on catalog if we've already analyzed it 
            results_dir =  f'results/{args.scenario}/{catalog}'
            if os.path.isdir(results_dir):
                continue
            else:
                os.makedirs(results_dir)

            analyze_catalog(args.scenario, catalog, args.no_jax)

    return None

def analyze_catalog(scenario, catalog, no_jax=False):

    logging.info(f'Analyzing {catalog}...')
    print(f'Analyzing {catalog}...')

    config_path = f'config/{scenario}/{catalog}.yaml'
    cfg = load_config(config_path, "config/default.yaml")

    # check to see theres a json file for each trial in trial_count
    data_dir = f'sim_data/{scenario}/{catalog}'
    files = [f'{data_dir}/trial_{t}.json' for t in range(cfg.trial_count)]
    file_existence = [os.path.isfile(f) for f in files]
    if not all(file_existence):
        e = f'{data_dir} missing data for each trial in {cfg.trial_count}'
        raise OSError(e)

    draws = cfg.draws
    tune = cfg.tune

    if no_jax:
        SAMPLE_KWARGS = {
            'draws': draws,
            'tune': tune,
            'progressbar': False
        }
    else: 
         SAMPLE_KWARGS = {
            'draws': draws,
            'tune': tune,
            'progress_bar': False
        }     

    logging.info(f'Sample kwargs:\n{SAMPLE_KWARGS}')
    print(f'Sample kwargs:\n{SAMPLE_KWARGS}')

    for trial in range(cfg.trial_count):

        print(f'Sampling for trial {trial} of {scenario}...')
        start = time.time()

        # load in capture history
        trial_path = f'{data_dir}/trial_{trial}.json'
        with open(trial_path, 'r') as f:
            trial_results = json.loads(json.load(f))

        # summarize history for js model
        capture_history = np.asarray(trial_results["capture_history"])

        # estimate N, p, phi, and b from capture history 
        pe = POPANEstimator(capture_history)
        popan = pe.compile()
        idata = sample_model(popan, SAMPLE_KWARGS, no_jax=no_jax)

        # dump results to json
        out_file = f'results/{scenario}/{catalog}/trial_{trial}.json'
        idata.to_json(out_file)

        stop = time.time()
        duration = stop-start
        logging.info(f'Trial {trial} lasted {duration}')
        print(f'Trial {trial} lasted {duration}')

    return None

def sample_model(model, SAMPLE_KWARGS, no_jax=False):

    with model:

        if no_jax:
            idata = pm.sample(**SAMPLE_KWARGS)
        else:
            idata = pm.sampling_jax.sample_numpyro_nuts(**SAMPLE_KWARGS)
    
    return idata

if __name__ == '__main__':
    main()