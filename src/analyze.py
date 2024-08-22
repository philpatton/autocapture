"""Module for summarizing the output of the simlulation.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

from multiprocessing import Pool, cpu_count

import argparse
import json

import numpy as np
import arviz as az
import pandas as pd

from config.config import load_config
from src.model import CJS, POPAN

def parse():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description="Analyzing results")
    parser.add_argument('-s', "--strategy", default="test")
    parser.add_argument('-e', "--estimator", default="popan")
    return parser.parse_args()

def analyze_strategy():
    '''Summarize the results of a strategy.'''
    args = parse()

    # read in the dataset metadata 
    rates = pd.read_csv('input/rates.csv')
    dataset_ids = rates.catalog_id.unique()

    # for every dataset in the strategy
    dataset_list = []
    for dataset in dataset_ids:

        cat = Dataset(dataset, strategy=args.strategy, estimator=args.estimator)

        # analyze the output       
        dataset_results = cat.analyze()
        dataset_list.append(dataset_results)

    # combine datasets into strategy level results
    strategy_results = pd.concat(dataset_list)
    strategy_results['strategy'] = args.strategy

    # save to output
    out_path = f'results/{args.strategy}/{args.strategy}-summary.csv'
    strategy_results.to_csv(out_path)

class Dataset:
    '''Analyze a dataset.
    
    Attributes:
        dataset: string representing the dataset being analyzed
        strategy: string representing the strategy being evaluated
        estimator: string representing which model (CJS or POPAN) being used
        data_dir: path to the directory with the simulated data
        results_dir: path to the simulation output
        config_path: path to the config for the dataset
    '''
    def __init__(self, dataset, strategy, estimator) -> None:
        '''Initializes the instance based on dataset, strategy, and model.'''
        self.estimator = estimator
        self.strategy = strategy
        self.dataset = dataset
        self.data_dir = f'sim_data/{strategy}/{dataset}'
        self.results_dir = f'results/{strategy}/{dataset}'
        self.config_path  = f'config/datasets/{dataset}.yaml'
    
    def analyze(self):
        '''Analyze the dataset.'''
        print(f'Analyzing {self.dataset}...')

        cfg = load_config(self.config_path, "config/default.yaml")
        trials = [t for t in range(cfg.trial_count)]
            
        counts = cpu_count() - 2
        with Pool(counts) as p:
            trial_summary_list = p.map(self.analyze_trial, trials)

        # concatenate results and return truth
        dataset_results = pd.concat(trial_summary_list)
        truth = get_truth(cfg)
        dataset_results = dataset_results.merge(truth)

        return dataset_results

    def analyze_trial(self, trial):
        '''Analyze the individual trial from dataset and dataset.'''
        print(f'Summarizing trial {trial} for {self.dataset}')

        # import the inference data object
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

        # add to the summary 
        summary['p_val'] = p_val

        # add additional information
        summary['trial'] = trial
        summary['dataset'] = self.dataset

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
    analyze_strategy()
