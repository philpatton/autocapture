"""Simulates trial_count datasets for 39 catalogs for a strategy

Various functions that wrap around the actual simulation code, which can be 
found in cjs.py or popan.py. 

"""
import argparse
import json
import os
import logging
import numpy as np
import pandas as pd

from src.model import POPAN, CJS
from src.miss_id import MissID
from config.config import load_config

def parse():
    '''Parses arguments from the command line'''
    parser = argparse.ArgumentParser(description="Simulating strategy")
    parser.add_argument('-s', "--strategy", default="debug")
    parser.add_argument('-e', '--estimator', default='popan')
    return parser.parse_args()

class NumpyEncoder(json.JSONEncoder):
    '''Easy conversion between numpy and json.'''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    '''Simulate trial_count datasets for each of the 39 catalogs.'''
    args = parse()

    # names of the catalogs
    id_path = ('input/catalog_ids.npy')
    catalog_ids = np.load(id_path, allow_pickle=True)

    rates = pd.read_csv('input/rates.csv')
    catalog_ids = rates.catalog_id.unique()

    # only simulate debug catalog in debug strategy
    if args.strategy == 'debug':
        simulate_catalog(args.estimator, args.strategy, catalog='debug')

    # otherwise, simulate all the other catalogs
    else:
        for catalog in catalog_ids:
            print(f'Simulating {catalog}')
            simulate_catalog(args.estimator, args.strategy, catalog)

    return None

def simulate_catalog(estimator, strategy, catalog):
    """Simulate 100 datasets for a given catalog and strategy."""

    # load the cfg for the strategy/catalog
    config_path = f'config/catalogs/{catalog}.yaml'
    cfg = load_config(config_path, "config/default.yaml")

    # don't overwrite, unless we're writing the debug strategy 
    catalog_dir =  f'sim_data/{strategy}/{catalog}'
    if os.path.isdir(catalog_dir):
        if catalog != 'debug':
            raise NameError(f'Directory: {catalog_dir} already exists.')
    else:
        os.makedirs(catalog_dir)

    # TODO: improve logging
    logging.basicConfig(filename=f'{catalog_dir}/test.log', level=logging.DEBUG)
    logging.debug(f'Simulating data for catalog: {catalog}')

    # entrance probabilities - constant except for b0 
    b = np.array(cfg.b)

    alpha = cfg[strategy]['alpha']
    beta = cfg[strategy]['beta']
    gamma = cfg[strategy]['gamma']

    # init the misidentification class 
    mi = MissID(alpha=alpha, beta=beta, gamma=gamma)

    for trial in range(cfg.trial_count):

        # conduct simulate one trial 
        if estimator == 'popan':
            popan = POPAN()
            sim_results = popan.simulate(N=cfg.N, T=cfg.T, phi=cfg.phi, 
                                         p=cfg.p, b=b)
        elif estimator == 'cjs':
            cjs = CJS()
            sim_results = cjs.simulate(T=cfg.T, phi=cfg.phi, p=cfg.p,
                                       released_count=cfg.released_count)

        # corrupt the true history with errors 
        true_history = sim_results['capture_history']
        capture_history = mi.simulate_capture_history(true_history)

        results = {
            'true_history':true_history,
            'capture_history':capture_history
        }

        # bundle the results in json format
        dumped = json.dumps(results, cls=NumpyEncoder)        

        # save the file as json
        path = f'{catalog_dir}/trial_{trial}.json'
        with open(path, 'w') as f:
            json.dump(dumped, f)

    # save the settings as well (perhaps redundant with config)
    settings = {'N': cfg.N, 'b': b, 'phi': cfg.phi, 'p': cfg.p, 
                'alpha': alpha, 'beta': beta, 'gamma': gamma}       
    dumped = json.dumps(settings, cls=NumpyEncoder)
    path = f'sim_data/{strategy}/{catalog}/catalog_settings.json'
    with open(path, 'w') as f:
        json.dump(dumped, f)

    logging.debug('catalog complete.')

if __name__ == '__main__':
    main()