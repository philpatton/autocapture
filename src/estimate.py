import numpy as np 
import pymc as pm

import json 
import argparse
import os 
import logging
from multiprocessing import Pool, cpu_count

from config.config import load_config, Config
from src.popan import POPAN
from src.cjs import CJS

def parse():
    parser = argparse.ArgumentParser(description="Estimating Jolly-Seber")
    parser.add_argument('-s', "--scenario", default="debug")
    parser.add_argument('-e', "--estimator", default="popan")
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

        catalog = Catalog('debug', scenario='debug', estimator=args.estimator)
        results_dir =  'results/debug/debug'
        os.makedirs(results_dir, exist_ok=True)
        catalog.analyze()

    else:
        for catalog in catalog_ids:

            # create dir for results if not there
            results_dir =  f'results/{args.scenario}/{catalog}'
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            cat = Catalog(catalog, scenario=args.scenario, 
                          estimator=args.estimator)
            cat.analyze()

            # if catalog == 'beluga-1':
            #     break

    return None

class Catalog:

    def __init__(self, catalog, scenario, estimator) -> None:
        self.estimator = estimator
        self.scenario = scenario
        self.catalog = catalog
        self.data_dir = f'sim_data/{scenario}/{catalog}'
        self.results_dir = f'results/{scenario}/{catalog}/'
        self.config_path  = f'config/catalogs/{catalog}.yaml'
    
    def analyze(self): 

        logging.info(f'Analyzing {self.catalog}...')
        print(f'Analyzing {self.catalog}...')

        cfg = load_config(self.config_path, "config/default.yaml")

        # check to see theres a json file for each trial in trial_count
        files = [f'{self.data_dir}/trial_{t}.json' for t in range(cfg.trial_count)]
        file_existence = [os.path.isfile(f) for f in files]
        if not all(file_existence):
            e = f'{self.ata_dir} missing data for each trial in {cfg.trial_count}'
            raise OSError(e) 

        # find paths to trials that have already been completed 
        completed_paths = [i for i in os.listdir(self.results_dir) 
                           if i.endswith('.json')]    

        # run all trials if there are no completed paths
        if not completed_paths:
            remaining_trials = range(cfg.trial_count)

        # otherwise, run remaining trials 
        else: 
            completed_trials = [extract_trial_number(p) for p in completed_paths]
            all_trials = range(cfg.trial_count)
            remaining_trials = [t for t in all_trials if t not in completed_trials]

            print(f'Remaing trials for {self.catalog} are {remaining_trials}')

        if not remaining_trials:
            return print(f'All trials for {self.atalog} already completed.')  

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
            p.map(self.run_trial, range(remaining_trials))

        return None

    def run_trial(self, trial):
        
        print(f'Sampling for trial {trial} of {self.scenario}...')

        # load in capture history
        trial_path = f'{self.data_dir}/trial_{trial}.json'
        with open(trial_path, 'r') as f:
            trial_results = json.loads(json.load(f))

        # summarize history for js model
        capture_history = np.asarray(trial_results["capture_history"])

        # mcmc sampling 
        idata = sample_model(self.estimator, capture_history, self.sample_kwargs)

        # dump results to json
        path = f'{self.results_dir}/trial_{trial}.json'
        idata.to_json(path)
    
def sample_model(estimator, capture_history, SAMPLE_KWARGS):

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
        idata = pm.sample(**SAMPLE_KWARGS)
    
    return idata


def extract_trial_number(path):
    """Extracts trial integer from 'resuts/test/beluga-0/trial_17.json'"""
    number_extension = path.split('trial_')[1]
    number = int(number_extension.split('.')[0])
    return number


if __name__ == '__main__':
    main()