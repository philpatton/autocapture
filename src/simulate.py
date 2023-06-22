"""Simulates trial_count datasets for 39 catalogs for a scenario

Various functions that wrap around the actual simulation code, which can be 
found in cjs.py or popan.py. 

"""
import argparse
import json
import os
import logging
import numpy as np
import pandas as pd

from src.popan import POPAN
from src.cjs import CJS
from src.miss_id import MissID
from src.config import Config, load_config

def parse():
    parser = argparse.ArgumentParser(description="Simulating scenario")
    # parser.add_argument("--sim_data_dir", default="sim_data")
    parser.add_argument('-s', "--scenario", default="debug")
    parser.add_argument('-e', '--estimator', default='popan')
    # parser.add_argument("--config_path", default="config/debug.yaml")
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

    # only simulate debug catalog in debug scenario
    if args.scenario == 'debug':
        simulate_catalog(args.scenario, catalog='debug')

    # otherwise, simulate all the other catalogs
    else:
        for catalog in catalog_ids:
            simulate_catalog(args.estimator, args.scenario, catalog)

    return None

def simulate_catalog(estimator, scenario, catalog):
    """Simulate 100 datasets for a given catalog and scenario."""

    # load the cfg for the scenario/catalog
    config_path = f'config/{scenario}/{catalog}.yaml'
    cfg = load_config(config_path, "config/default.yaml")

    # don't overwrite, unless we're writing the debug scenario 
    catalog_dir =  f'sim_data/{scenario}/{catalog}'
    if os.path.isdir(catalog_dir):
        if catalog != 'debug':
            raise NameError(f'Directory: {catalog_dir} already exists.')
    else:
        os.makedirs(catalog_dir)

    # TODO: Figure out better logging
    logging.basicConfig(filename=f'{catalog_dir}/test.log', level=logging.DEBUG)
    logging.debug(f'Simulating data for catalog: {catalog}')

    # entrance probabilities - constant except for b0 
    b = np.zeros(cfg.T)
    b[0] = cfg.b0
    b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 

    # init the misidentification class 
    mi = MissID(alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma)

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
                'alpha': cfg.alpha, 'beta': cfg.beta, 'gamma': cfg.gamma}       
    dumped = json.dumps(settings, cls=NumpyEncoder)
    path = f'sim_data/{scenario}/{catalog}/catalog_settings.json'
    with open(path, 'w') as f:
        json.dump(dumped, f)

    logging.debug('catalog complete.')

if __name__ == '__main__':
    main()