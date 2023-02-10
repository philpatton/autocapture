
''' Simulate data for a Jolly-Seber model. 

This code was adapted from Kery and Schaub (2011) BPA, Chapter 10.
'''

import numpy as np
import argparse
import json
import os

from tqdm import tqdm
from jolly_seber import JollySeber
from config import Config, load_config

def parse():
    parser = argparse.ArgumentParser(description="Simulating Jolly-Seber")
    parser.add_argument("--sim_data_dir", default="sim_data")
    parser.add_argument("--experiment_name", default="tmp")
    parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    """Runs the simulation study for a given scenario."""
    args = parse()
    cfg = load_config(args.config_path, "config/default.yaml")

    experiment_dir =  f'{args.base_output_dir}/{args.experiment_name}'

    # don't overwrite, unless we're writing to tmp 
    if os.path.isdir(experiment_dir):
        if args.experiment_name != 'tmp':
            raise NameError(f'Directory: {experiment_dir} already exists.')
    else:
        os.mkdir(experiment_dir)

    # survival probabilities 
    phi_shape = (cfg.N, cfg.T - 1)
    PHI = np.full(phi_shape, cfg.phi)

    # capture probabilities 
    p_shape = (cfg.N, cfg.T)
    P = np.full(p_shape, cfg.p)

    # entrance probabilities 
    b = np.zeros(cfg.T)
    b[0] = cfg.b0
    b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 

    js = JollySeber(N=cfg.N, PHI=PHI, P=P, b=b)

    print(f'Starting experiment: {args.experiment_name}')

    for trial in tqdm(range(cfg.trial_count)):

        # conduct the simulation
        results = js.simulate_data() 

        # bundle the results in json format
        dumped = json.dumps(results, cls=NumpyEncoder)        

        # save the file as json
        path = f'{experiment_dir}/trial_{trial}.json'
        with open(path, 'w') as f:
            json.dump(dumped, f)

    settings = {'N': cfg.N, 'b': b, 'phi': cfg.phi, 'p': cfg.p}
    dumped = json.dumps(settings, cls=NumpyEncoder)
    
    path = (
        f'{args.base_output_dir}/{args.experiment_name}/experiment_settings.json'
    )
    with open(path, 'w') as f:
        json.dump(dumped, f)

    print('Experiment complete.')

if __name__ == '__main__':
    main()