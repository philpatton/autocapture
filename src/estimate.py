import numpy as np 
import pymc as pm
# import pymc.sampling_jax

import time
import json 
import argparse
import os 
import logging

from src.config import load_config, Config
from src.popan import POPAN
from src.cjs import CJS
from src.utils import create_full_array

def parse():
    parser = argparse.ArgumentParser(description="Estimating Jolly-Seber")
    parser.add_argument('-s', "--scenario", default="debug")
    parser.add_argument('-e', "--estimator", default="popan")
    parser.add_argument('--no_jax', action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def main():

    args = parse()

    if not os.path.isdir('results'):
        os.mkdir('results')

    # TODO: Figure out logging 
    logging.basicConfig(filename=f'results/{args.scenario}.log')

    id_path = ('input/catalog_ids.npy')
    catalog_ids = np.load(id_path, allow_pickle=True)

    if args.scenario == 'debug':

        results_dir =  'results/debug/debug'
        os.makedirs(results_dir, exist_ok=True)
        analyze_catalog('debug', 'debug', args.estimator, args.no_jax)

    else:
        for catalog in catalog_ids:

            # if catalog == 'beluga-1':
            #     break

            # create dir for results if not there
            results_dir =  f'results/{args.scenario}/{catalog}'
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            analyze_catalog(args.scenario, catalog, args.estimator, args.no_jax)

    return None

def analyze_catalog(scenario, catalog, estimator, no_jax=True):

    MAX_ATTEMPTS = 5

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

    if no_jax:
        SAMPLE_KWARGS = {
            'draws': cfg.draws,
            'tune': cfg.tune,
            'progressbar': True
        }
    else: 
         SAMPLE_KWARGS = {
            'draws': cfg.draws,
            'tune': cfg.tune,
            'progressbar': True
        }     

    # find paths to trials that have already been completed 
    results_dir = f'results/{scenario}/{catalog}/'
    completed_paths = [i for i in os.listdir(results_dir) 
                       if i.endswith('.json')]    

    # run all trials if there are no completed paths
    if not completed_paths:
        remaining_trials = range(cfg.trial_count)

    # otherwise, run trials 
    else: 
        completed_trials = [extract_trial_number(p) for p in completed_paths]
        all_trials = range(cfg.trial_count)
        remaining_trials = [t for t in all_trials if t not in completed_trials]

        print(f'Remaing trials for {catalog} are {remaining_trials}')

    if not remaining_trials:
        return print(f'All trials for {catalog} already completed.')

    for trial in remaining_trials:

        print(f'Sampling for trial {trial} of {scenario}...')
        start = time.time()

        # load in capture history
        trial_path = f'{data_dir}/trial_{trial}.json'
        with open(trial_path, 'r') as f:
            trial_results = json.loads(json.load(f))

        # summarize history for js model
        capture_history = np.asarray(trial_results["capture_history"])

        # estimate N, p, phi, and b from capture history 
        if estimator == 'popan':
            popan = POPAN()
            model = popan.compile_pymc_model(capture_history)
        elif estimator == 'cjs':
            cjs = CJS()
            model = cjs.compile_pymc_model(capture_history)
        else:
            raise ValueError('estimator must be "popan" or "cjs"')

        # sample from model
        with model:

            # PyMC sometimes fails to sample. In these cases, it often works
            # if tried again, perhaps for the stochastic nature of the sampler.
            # Below, we work in that 'retry' code, noting when a trial has a 
            # failure. 

            atts = []
            for attempt in range(MAX_ATTEMPTS):
                try:
                    idata = pm.sample(**SAMPLE_KWARGS)
                except ValueError:
                    atts.append(attempt)
                else:
                    if len(atts):
                        m = f'{catalog}: trial {trial} had {len(atts)} failures'
                        print(m)
                    break
            else:
                m = f'{catalog}: trial {trial} failed {MAX_ATTEMPTS} times, exceeding MAX_ATTEMPTS'
                raise ValueError(m)

        # dump results to json
        path = f'{results_dir}/trial_{trial}.json'
        idata.to_json(path)

        stop = time.time()
        duration = stop-start
        logging.info(f'Trial {trial} lasted {duration}')
        print(f'Trial {trial} of {catalog} lasted {duration:.0f} seconds')

    return None

def sample_model(model, SAMPLE_KWARGS, no_jax=True):

    with model:
        idata = pm.sample(**SAMPLE_KWARGS)

        # if no_jax:
        #     idata = pm.sample(**SAMPLE_KWARGS)
        # else:
        #     idata = pm.sampling_jax.sample_numpyro_nuts(**SAMPLE_KWARGS)
    
    return idata


def extract_trial_number(path):
    """Extracts trial integer from 'resuts/test/beluga-0/trial_17.json'"""
    number_extension = path.split('trial_')[1]
    number = int(number_extension.split('.')[0])
    return number


if __name__ == '__main__':
    main()