import numpy as np 
import pymc as pm
import arviz as az

from src.cjs import Simulator, BayesEstimator

from src.utils import summarize_individual_history

debug_kwargs = {
    'marked': 25,
    'T': 10,
    'phi': 0.9,
    'p': 0.5,
    'seed': 17
}

sample_kwargs = {
    'draws' : 2000,
    'tune': 2000
}

def test_bayes_estimator():

    cs = Simulator(**debug_kwargs)   
    results = cs.simulate()

    be = BayesEstimator(results['capture_history'])
    model = be.compile()

    with  model:
        idata = pm.sample()

    summary = az.summary(idata, var_names='phi')

    phi_true = debug_kwargs['phi']

    # check coverage
    phi_low = summary['hdi_3%']
    phi_high = summary['hdi_97%']
    is_between = (phi_true > phi_low) & (phi_true < phi_high)
    assert all(is_between)

    # check bias
    abs_bias = (phi_true - summary['mean']).abs()

    assert all(abs_bias < 0.1)

    # compute tukey-statistic

    # expected values of cells 
    T = debug_kwargs['T']
    occasion_count = T
    interval_count = T - 1

    intervals = np.arange(interval_count)
    occasions = np.arange(occasion_count)

    i = np.reshape(intervals, (interval_count, 1))
    j = np.reshape(occasions, (1, occasion_count))
    not_cap_visits = np.clip(j - i - 1, 0, np.inf)[:, 1:]

    # p_not_cap = np.triu((1 - p) ** not_cap_visits)

    # with model:
    #     pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # idata.posterior_predictive

    stacked = az.extract(idata)

    sum = summarize_individual_history(results['capture_history'])

    # M array (add zero row to for compatability below)
    M = sum['m_array']
    M = np.insert(M, 0, 0, axis=1)

    # vectorize the recapture counts and probabilities 
    upper_triangle_indices = np.triu_indices_from(M[:, 1:])
    recapture_counts = M[:, 1:][upper_triangle_indices]
    # recapture_probabilities = nu[upper_triangle_indices]

    p = stacked.p.values
    phi = stacked.phi.values

