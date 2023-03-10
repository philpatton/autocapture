
''' Simulate data for a Jolly-Seber model. 

This code was adapted from Kery and Schaub (2011) BPA, Chapter 10.
'''

import argparse
import json
import os
import logging
import numpy as np
import pandas as pd

from popan import POPANSimulator
from miss_id import MissID
from config import Config, load_config

def parse():
    parser = argparse.ArgumentParser(description="Simulating scenario")
    # parser.add_argument("--sim_data_dir", default="sim_data")
    parser.add_argument("--scenario", default="debug")
    # parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():

    args = parse()

    id_path = ('input/catalog_ids.npy')
    catalog_ids = np.load(id_path, allow_pickle=True)

    if args.scenario == 'debug':
        catalog = 'debug'        
        simulate_catalog(args.scenario, catalog)

    else:
        for catalog in catalog_ids:
            simulate_catalog(args.scenario, catalog)

    return None

def simulate_catalog(scenario, catalog):
    """Runs the simulation study for a given scenario."""

    config_path = f'config/{scenario}/{catalog}.yaml'
    cfg = load_config(config_path, "config/default.yaml")

    catalog_dir =  f'sim_data/{scenario}/{catalog}'
    # don't overwrite, unless we're writing to tmp 
    if os.path.isdir(catalog_dir):
        if catalog != 'debug':
            raise NameError(f'Directory: {catalog_dir} already exists.')
    else:
        os.makedirs(catalog_dir)

    logging.basicConfig(filename=f'{catalog_dir}/test.log', 
                        level=logging.DEBUG)

    # entrance probabilities 
    b = np.zeros(cfg.T)
    b[0] = cfg.b0
    b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 

    # false reject, false accept, mark change rates; superpopulation size 
    alpha = cfg.alpha 
    beta = cfg.beta 
    gamma = cfg.gamma  

    ps = POPANSimulator(N=cfg.N, T=cfg.T, phi=cfg.phi, p=cfg.p, b=b)
    mi = MissID(alpha=alpha, beta=beta, gamma=gamma)

    logging.debug(f'Simulating data for catalog: {catalog}')

    for trial in range(cfg.trial_count):

        # conduct the simulation
        sim_results = ps.simulate() 
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

    settings = {'N': cfg.N, 'b': b, 'phi': cfg.phi, 'p': cfg.p, 
                'alpha': alpha, 'beta':beta, 'gamma':gamma}
                
    dumped = json.dumps(settings, cls=NumpyEncoder)
    
    path = f'sim_data/{scenario}/{catalog}/catalog_settings.json'
    with open(path, 'w') as f:
        json.dump(dumped, f)

    logging.debug('catalog complete.')

if __name__ == '__main__':
    main()