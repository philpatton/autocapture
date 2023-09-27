import numpy as np
import arviz as az
import pandas as pd

import argparse
import os
import json
import time 

from multiprocessing import Pool, cpu_count

from config.config import load_config, Config
from src.cjs import CJS
from src.popan import POPAN

def parse():
    parser = argparse.ArgumentParser(description="Analyzing results")
    parser.add_argument('-s', "--scenario", default="test")
    parser.add_argument('-e', "--estimator", default="popan")
    return parser.parse_args()

def analyze_scenario():

    args = parse()

    rates = pd.read_csv('input/rates.csv')
    catalog_ids = rates.catalog_id.unique()
    
    catalog_list = []
    for catalog in catalog_ids:

        cat = Catalog(catalog, scenario=args.scenario, estimator=args.estimator)
        
        catalog_results = cat.analyze()
        catalog_list.append(catalog_results)

    scenario_results = pd.concat(catalog_list)

    # out_file = f'{summary_dir}/trial_{trial}.csv'
    scenario_results.to_csv(f'results/{args.scenario}/summary.csv')

    return None

class Catalog:

    def __init__(self, catalog, scenario, estimator) -> None:
        self.estimator = estimator
        self.scenario = scenario
        self.catalog = catalog
        self.data_dir = f'sim_data/{scenario}/{catalog}'
        self.results_dir = f'results/{scenario}/{catalog}'
        self.config_path  = f'config/catalogs/{catalog}.yaml'
    
    def analyze(self):

        print(f'Analyzing {self.catalog}...')

        cfg = load_config(self.config_path, "config/default.yaml")
        trials = [t for t in range(cfg.trial_count)]
            
        counts = cpu_count() - 2
        with Pool(counts) as p:
            trial_summary_list = p.map(self.analyze_trial, trials)

        # concatenate results and return truth
        catalog_results = pd.concat(trial_summary_list)
        truth = get_truth(cfg)
        catalog_results = catalog_results.merge(truth)

        return catalog_results

    def analyze_trial(self, trial):

        print(f'Summarizing trial {trial} for {self.catalog}')

        path = f'{self.results_dir}/trial_{trial}.json'
        idata = az.from_json(path)
        
        # calculate summary statistics 
        summary = az.summary(idata).reset_index(names='parameter')

        # report number of divergent transitions
        divergences = idata.sample_stats.diverging.to_numpy().sum()
        summary['divergences'] = divergences

        # add the p value to the summary, after selecting  method
        if self.estimator == 'popan':
            method = POPAN()
        elif self.estimator == 'cjs':
            method = CJS()

        # load in data 
        data_path = f'{self.data_dir}/trial_{trial}.json'
        with open(data_path, 'r') as f:
            data = json.loads(json.load(f))
                
        # perform check 
        ch = np.array(data['capture_history'])
        check_results = method.check(idata, ch)

        # calculate p value
        ft_obs = check_results['freeman_tukey_observed']
        ft_new = check_results['freeman_tukey_new']
        p_val = (ft_new > ft_obs).mean()

        summary['p_val'] = p_val

        # add additional information
        summary['trial'] = trial
        summary['catalog'] = self.catalog

        return summary
    
def get_truth(config):
    """Returns a dataframe with true values of p, phi, b0, N from config."""
    # get true parameter values from config
    parms = ['p', 'phi', 'N']
    vals = [config[k] for k in parms]
    
    # insert b0 values 
    b = config['b']
    vals.insert(2, b[0])
    parms.insert(2, 'b0')

    truth = pd.DataFrame({'parameter': parms, 'truth': vals})

    return truth

if __name__ == '__main__':
    analyze_scenario()
