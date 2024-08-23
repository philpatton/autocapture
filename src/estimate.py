"""Run MCMC for every trial and dataset in the scenario.

Estimate parameters using PyMC, for all datasets under a given strategy. The 
actual PyMC code is written in the model.py model. This script then runs the
MCMC sampler, and saves each trial's results to a json file. The stript relies
on the multiprocessing module to parallelize the estimation process. As such,
each trial is run in parallel, while the chains are ran sequentially. PyMC
defaults to parallelizing the chains but, since n_trials > n_cores > n_chains,
I decided to parallelize the trials instead.

The script is called from the command line with the following arguments:
    -s: strategy name (default: debug)
    -m: model (default: popan)

Typical usage example:
    $ python -m src.estimate --strategy check_0
"""

from multiprocessing import Pool, cpu_count

import json
import argparse
import os
import logging

import numpy as np
import pymc as pm
import pandas as pd

from config.config import load_config
from src.model import POPAN, CJS

def parse():
    '''Parse arguments from the command line.'''
    parser = argparse.ArgumentParser(description="Estimating Jolly-Seber")
    parser.add_argument('-s', "--strategy", default="test")
    parser.add_argument('-e', "--model", default="popan")
    return parser.parse_args()

def main():
    '''Estimate all the datasets under a given strategy.'''
    args = parse()

    # add a directory for the output 
    if not os.path.isdir('results'):
        os.mkdir('results')

    # TODO: Improve logging 
    logging.basicConfig(filename=f'results/{args.strategy}.log')

    # read in the dataset information
    rates = pd.read_csv('input/rates.csv')
    dataset_ids = rates.catalog_id.unique()

    # there are some quirks about the debug strategy, 
    # making it helpful for quick runs 
    if args.strategy == 'debug':

        dataset = Dataset('debug', strategy='debug', model=args.model)
        results_dir =  'results/debug/debug'
        os.makedirs(results_dir, exist_ok=True)
        dataset.estimate()

    # estimate parameters for each catalgog
    else:
        for dataset in dataset_ids:

            # create dir for results if not there
            results_dir =  f'results/{args.strategy}/{dataset}'
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            # Estimate the paramaters for the dataset
            dataset = Dataset(dataset, strategy=args.strategy, model=args.model)
            dataset.estimate()

            # break

class Dataset:
    '''Convenience class for estimating parameters '''
    def __init__(self, dataset, strategy, model) -> None:
        self.model = model
        self.strategy = strategy
        self.dataset = dataset
        self.data_dir = f'sim_data/{strategy}/{dataset}'
        self.results_dir = f'results/{strategy}/{dataset}/'
        self.config_path  = f'config/datasets/{dataset}.yaml'
    
    def estimate(self): 
        '''Use PyMC to estimate parameters for every trial for the dataset.'''

        logging.info(f'Estimating {self.dataset}...')
        print(f'Estimating {self.dataset}...')

        cfg = load_config(self.config_path, "config/default.yaml")

        # check to see theres a json file for each trial in trial_count
        files = [f'{self.data_dir}/trial_{t}.json' for t in range(cfg.trial_count)]
        file_existence = [os.path.isfile(f) for f in files]
        if not all(file_existence):
            e = f'{self.data_dir} missing data for each trial in {cfg.trial_count}'
            raise OSError(e) 

        # find paths to trials that have already been completed 
        completed_paths = [i for i in os.listdir(self.results_dir) 
                           if i.endswith('.json')]    

        # run all trials if there are no completed paths
        if not completed_paths:
            remaining_trials = [t for t in range(cfg.trial_count)]

        # otherwise, run remaining trials 
        else: 
            completed_trials = [extract_trial_number(p) for p in completed_paths]
            all_trials = range(cfg.trial_count)
            remaining_trials = [t for t in all_trials if t not in completed_trials]

            print(f'Remaing trials for {self.dataset} are {remaining_trials}')

        if not remaining_trials:
            return print(f'All trials for {self.dataset} already completed.')  

        # arguments for mcmc sampler
        self.sample_kwargs = {
            'draws': cfg.draws,
            'tune': cfg.tune,
            'progressbar': True,
            'chains': 4,
            'cores': 1
        } 

        counts = cpu_count() - 2
        with Pool(counts) as p:
            p.map(self.run_trial, remaining_trials)

    def run_trial(self, trial):
        '''Use PyMC to estimate parameters for the trial.'''
        print(f'Sampling for trial {trial} of {self.strategy}...')

        # load in capture history
        trial_path = f'{self.data_dir}/trial_{trial}.json'
        with open(trial_path, 'r') as f:
            trial_results = json.loads(json.load(f))

        # summarize history for js model
        capture_history = np.asarray(trial_results["capture_history"])

        # mcmc sampling
        idata = sample_model(self.model, capture_history, self.sample_kwargs)

        # dump results to json
        path = f'{self.results_dir}/trial_{trial}.json'
        idata.to_json(path)
    
def sample_model(model, capture_history, SAMPLE_KWARGS):
    '''Wrapper for sampling a model.'''
    # estimate N, p, phi, and b from capture history 
    if model == 'popan':
        popan = POPAN()
        model = popan.compile_pymc_model(capture_history)
    elif model == 'cjs':
        cjs = CJS()
        model = cjs.compile_pymc_model(capture_history)
    else:
        raise ValueError('model must be "popan" or "cjs"')

    # sample from model
    with model:
        idata = pm.sample(**SAMPLE_KWARGS)
    
    return idata

def extract_trial_number(path):
    """Extracts trial integer from 'resuts/test/beluga-0/trial_17.json'"""
    number_extension = path.split('trial_')[1]
    number = int(number_extension.split('.')[0])
    return number

if __name__ == '__main__':
    main()
