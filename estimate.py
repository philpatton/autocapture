import numpy as np 
import scipy as sp
import arviz as az
import pymc as pm
import pymc.sampling_jax

import time
import json 
import argparse
import os 
import logging
import pandas as pd

from tqdm import tqdm
from config import load_config, Config
from utils import summarize_individual_history
from pymc.distributions.dist_math import factln
from pytensor import tensor as pt

def parse():
    parser = argparse.ArgumentParser(description="Estimating Jolly-Seber")
    # parser.add_argument("--sim_data_dir", default="sim_data")
    # parser.add_argument("--results_dir", default="results")
    # parser.add_argument("--experiment_name", default="tmp")
    # parser.add_argument("--config_path", default="config/debug.yaml")
    parser.add_argument("--scenario", default="debug")
    parser.add_argument('--posterior_summary', 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--no_jax', action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def main():

    args = parse()

    # TODO: Figure out logging 
    logger = logging.getLogger('pymc')
    # logger.setLevel(logging.ERROR)
    logging.basicConfig(filename=f'results/{args.scenario}.log')

    id_path = ('input/catalog_ids.npy')
    catalog_ids = np.load(id_path, allow_pickle=True)

    # parse arguments 
    summary = True if args.posterior_summary else False
    no_jax = True if args.no_jax else False

    if args.scenario == 'debug':

        results_dir =  'results/debug/debug'
        os.makedirs(results_dir, exist_ok=True)

        analyze_catalog('debug', 'debug', summary, no_jax)

    else:
        for catalog in catalog_ids:

            # pass on catalog if we've already analyzed it 
            results_dir =  f'results/{args.scenario}/{catalog}'
            if os.path.isdir(results_dir):
                continue
            else:
                os.makedirs(results_dir)

            analyze_catalog(args.scenario, catalog, summary, no_jax)

    return None

def analyze_catalog(scenario, catalog, posterior_summary=False, no_jax=False):

    logging.log(f'Analyzing {catalog}...')

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

    logging.log(f'Sample kwargs:\n{SAMPLE_KWARGS}')

    for trial in tqdm(range(cfg.trial_count)):

        start = time.time()
        #do some stuff

        # load in capture history
        trial_path = f'{data_dir}/trial_{trial}.json'
        with open(trial_path, 'r') as f:
            trial_results = json.loads(json.load(f))

        # summarize history for js model
        capture_history = np.asarray(trial_results["capture_history"])
        capture_summary = summarize_individual_history(capture_history)

        # estimate N, p, phi, and b from capture history 
        js_model = build_model(capture_summary)
        idata = sample_model(js_model, SAMPLE_KWARGS, no_jax=no_jax)

        if posterior_summary:

            trial_results = az.summary(idata, round_to=4)
            out_file = f'results/{scenario}/{catalog}/trial_{trial}.csv'
            trial_results.to_csv(out_file)

        else:

            out_file = f'results/{scenario}/{catalog}/trial_{trial}.json'
            idata.to_json(out_file)

        stop = time.time()
        duration = stop-start
        logging.log(f'Trial {trial} lasted {duration}')

    return None

# logp of the dist for unmarked animals {u1, ...} ~ Mult(N; psi1 * p, ...)
def logp(x, n, p):
    
    x_last = n - x.sum()
    
    # calculate thwe logp for the observations
    res = factln(n) + pt.sum(x * pt.log(p) - factln(x)) \
            + x_last * pt.log(1 - p.sum()) - factln(x_last)
    
    # ensure that the good conditions are met.
    good_conditions = pt.all(x >= 0) & pt.all(x <= n) & (pt.sum(x) <= n) & \
                        (n >= 0)
    res = pm.math.switch(good_conditions, res, -np.inf)

    return res

def fill_lower_diag_ones(x):
    return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)

def build_model(capture_summary):

    # number released at each occasion (assuming no losses on capture)
    R = capture_summary['number_released']

    # number of intervals (NOT occasions)
    occasion_count = len(R)

    # index for generating sequences like [[0], [0,1], [0,1,2]]
    alive_yet_unmarked_index = sp.linalg.circulant(np.arange(occasion_count))

    # M array 
    M = capture_summary['m_array']
    M = np.insert(M, 0, 0, axis=1)

    # number of unmarked animals captured at each occasion
    u = np.concatenate(([R[0]], R[1:] - M[:, 1:].sum(axis=0)))
    
    # ditch the last R for the CJS portion
    R = R[:-1]
    interval_count = len(R)
    
    # number of animals that were never recaptured
    never_recaptured_counts = R - M.sum(axis=1)
    
    # convenience vectors for recapture model
    i = np.arange(interval_count)[:, np.newaxis]
    j = np.arange(occasion_count)[np.newaxis]
    not_cap_visits = np.clip(j - i - 1, 0, np.inf)[:, 1:]

    with pm.Model() as js_sim:
        # priors for detection, survival, and pent
        p = pm.Uniform('p', 0., 1.)
        phi = pm.Uniform('phi', 0., 1.)
        beta = pm.Dirichlet(
            'beta', 
            np.ones(occasion_count), 
            shape=(occasion_count)
        )

        # improper flat prior for N
        flat_dist = pm.Flat.dist()
        N = pm.Truncated("N", flat_dist, lower=u.sum())

        # add [1] to ensure the addition of the raw beta_0
        PHI = np.repeat(phi, interval_count)
        p_alive_yet_unmarked = pt.concatenate(
            ([1], pt.cumprod((1 - p) * PHI))
        )

        # lower triangle of the index produces the [[0], [0,1], [0,1,2]] patterns,
        # building in the recursion
        psi = pt.tril(
            beta * p_alive_yet_unmarked[alive_yet_unmarked_index]
        ).sum(axis=1)
    
        # distribution for the unmarked animals, L'''1 in schwarz arnason
        unmarked = pm.CustomDist('unmarked', N, psi * p, logp=logp, observed=u)

        # matrix of survival probabilities
        p_alive = pt.triu(
            pt.cumprod(
                fill_lower_diag_ones(pt.ones_like(M[:, 1:]) * PHI),
                axis=1
            )
        )

        # define nu, the probabilities of each cell in the m array 
        p_not_cap = pt.triu((1 - p) ** not_cap_visits)
        nu = p_alive * p_not_cap * p

        # vectorize the counts and probabilities of recaptures 
        upper_triangle_indices = np.triu_indices_from(M[:, 1:])
        recapture_counts = M[:, 1:][upper_triangle_indices]
        recapture_probabilities = nu[upper_triangle_indices]

        # distribution for the recaptures 
        recaptured = pm.Binomial(
            'recaptured', 
            n=recapture_counts, 
            p=recapture_probabilities,
            observed=recapture_counts
        )

        # distribution for the observed animals who were never recaptured
        chi = 1 - nu.sum(axis=1)
        never_recaptured_rv = pm.Binomial(
            'never_recaptured', 
            n=never_recaptured_counts, 
            p=chi, 
            observed=never_recaptured_counts
        )
    
    return js_sim

def sample_model(model, SAMPLE_KWARGS, no_jax=False):

    with model:

        if no_jax:
            idata = pm.sample(**SAMPLE_KWARGS)
        else:
            idata = pm.sampling_jax.sample_numpyro_nuts(**SAMPLE_KWARGS)
    
    return idata

if __name__ == '__main__':
    main()