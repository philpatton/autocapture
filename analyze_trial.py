import numpy as np 
import scipy as sp
import arviz as az
import pymc as pm
import pymc.sampling_jax

import json 
import argparse
import os 

from config import load_config, Config
from capture_history_utils import summarize_individual_history
from pymc.distributions.dist_math import factln
from pytensor import tensor as pt

def parse():
    parser = argparse.ArgumentParser(description="Simulating Jolly-Seber")
    parser.add_argument("--sim_data_dir", default="sim_data")
    parser.add_argument("--experiment_name", default="tmp")
    parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()

def main():

    args = parse()
    cfg = load_config(args.config_path, "config/default.yaml")

    base_output_dir = 'sim_data'
    experiment_name = 'tmp'
    experiment_dir = f'{base_output_dir}/{experiment_name}'
    trial_path = f'{experiment_dir}/trial_2.json'

    with open(trial_path, 'r') as f:
        trial_results = json.loads(json.load(f))

    capture_history = np.asarray(trial_results["N"])

    capture_summary = summarize_individual_history(capture_history)

    SAMPLE_KWARGS = {
        'draws': cfg.draws,
        'tune': cfg.tune,
        'progress_bar': False
    }

    js_model = build_model(capture_summary)

    idata = sample_model(js_model, SAMPLE_KWARGS)

    summary_stats = az.summary(idata).round(2)

    # don't overwrite, unless we're writing to tmp 
    if os.path.isdir(experiment_dir):
        if args.experiment_name != 'tmp':
            raise NameError(f'Directory: {experiment_dir} already exists.')
    else:
        os.mkdir(experiment_dir)

    out_file = f'sim_results/{experiment_name}/trial_002_results.csv'

    summary_stats.to_csv(out_file)

    return None

# logp of the dist for unmarked animals {u1, ...} ~ Mult(N; psi1 * p, ...)
def logp(x, n, p):
    
    x_last = n - x.sum()
    
    # calculate thwe logp for the observations
    res = factln(n) + pt.sum(x * pt.log(p) - factln(x)) \
            + x_last * pt.log(1 - p.sum()) - factln(x_last)
    
    # ensure that the good conditions are met.
    good_conditions = pt.all(x >= 0) & pt.all(x <= n) & (pt.sum(x) <= n) & (n >= 0)
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

def sample_model(model, SAMPLE_KWARGS):

    with model:
        idata = pm.sampling_jax.sample_numpyro_nuts(**SAMPLE_KWARGS)
        # idata = pm.sample(**SAMPLE_KWARGS)
    
    return idata