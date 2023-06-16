import numpy as np 
import pymc as pm
import arviz as az

from src.cjs import Simulator, BayesEstimator, MLE

from src.utils import summarize_individual_history, expit

debug_kwargs = {
    'marked': 25,
    'T': 10,
    'phi': 0.9,
    'p': 0.25,
    'seed': 17
}

sample_kwargs = {
    'draws' : 1000,
    'tune': 1000
}

def test_simulator():

    cs = Simulator(**debug_kwargs)
    results = cs.simulate()
    ch = results['capture_history']

    assert type(ch) is np.ndarray
    assert ch.shape[0] == debug_kwargs['marked'] * (debug_kwargs['T'] - 1)

    print(ch.shape)

def test_mle():

    data = np.array(
        [[11, 2 , 0  ,0  ,0  ,0 , 9 ],
        [0 , 24, 1  ,0  ,0  ,0 , 35],
        [0 , 0 , 34 ,2  ,0  ,0 , 42],
        [0 , 0 , 0  ,45 ,1  ,2 , 32],
        [0 , 0 , 0  ,0  ,51 ,0 , 37],
        [0 , 0 , 0  ,0  ,0  ,52, 46]]
    )

    mle = MLE(data)

    results = mle.estimate()

    reals = expit(results.est_logit.values)

    assert np.allclose(reals, [0.9, 0.56], rtol=0.1)

    standard_errors = results.se.values

    assert np.allclose(standard_errors, [0.325, 0.102], rtol=0.01)

def test_bayes_estimator():

    # cs = Simulator(**debug_kwargs)   
    # results = cs.simulate()

    data = np.array(
        [[11, 2 , 0  ,0  ,0  ,0 , 9 ],
        [0 , 24, 1  ,0  ,0  ,0 , 35],
        [0 , 0 , 34 ,2  ,0  ,0 , 42],
        [0 , 0 , 0  ,45 ,1  ,2 , 32],
        [0 , 0 , 0  ,0  ,51 ,0 , 37],
        [0 , 0 , 0  ,0  ,0  ,52, 46]])    

    be = BayesEstimator(data)
    model = be.compile()

    with  model:
        idata = pm.sample(**sample_kwargs)

    summary = az.summary(idata)

    phi_summary = summary.loc[summary.index.str.contains('phi')]

    phi_mle = np.array([0.626, 0.454, 0.478, 0.624, 0.608, 0.583])
    phi_posterior_mean = phi_summary['mean'].values
    assert np.allclose(phi_mle, phi_posterior_mean, rtol=0.05)

    # check coverage
    phi_low = phi_summary['hdi_3%']
    phi_high = phi_summary['hdi_97%']
    is_between = (phi_mle > phi_low) & (phi_mle < phi_high)
    assert all(is_between)

    # check bias
    # abs_bias = (phi_true - summary['mean']).abs()

    # assert all(abs_bias < 0.2)

# #     # compute tukey-statistic

# #     # expected values of cells 
# #     T = debug_kwargs['T']
# #     occasion_count = T
# #     interval_count = T - 1

# #     intervals = np.arange(interval_count)
# #     occasions = np.arange(occasion_count)

# #     i = np.reshape(intervals, (interval_count, 1))
# #     j = np.reshape(occasions, (1, occasion_count))
# #     not_cap_visits = np.clip(j - i - 1, 0, np.inf)[:, 1:]

# #     # p_not_cap = np.triu((1 - p) ** not_cap_visits)

# #     # with model:
# #     #     pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# #     # idata.posterior_predictive

# #     stacked = az.extract(idata)

# #     sum = summarize_individual_history(results['capture_history'])

# #     # M array (add zero row to for compatability below)
# #     M = sum['m_array']
# #     M = np.insert(M, 0, 0, axis=1)

# #     # vectorize the recapture counts and probabilities 
# #     upper_triangle_indices = np.triu_indices_from(M[:, 1:])
# #     recapture_counts = M[:, 1:][upper_triangle_indices]
# #     # recapture_probabilities = nu[upper_triangle_indices]

# #     p_samples = stacked.p.values
# #     phi_samples = stacked.phi.values

# #     p = np.repeat(p_samples[0], interval_count)
# #     phi = phi_samples[:,0]
# #     pbar = 1 - p
    
# #     ni = interval_count
# #     nj = interval_count

# #     # q defines the multinomial cell probabilities
# #     q = np.zeros((ni, nj + 1))

# #     # fill the diagonal elemens 
# #     diag_indices = np.diag_indices(ni)
# #     q[diag_indices] = phi * p
    
# #     # and the off diagonal elements
# #     for i in range(ni - 1):
# #         for j in range(i + 1, nj):
# #             q[i, j] = np.prod(phi[i:j+1]) * np.prod(pbar[i+1:j+1]) * p[j + 1]

# #     # Calculate the disappearing animal probabilities
# #     for i in range(ni):
# #         q[i, nj] = 1 - np.sum(q[i, i:nj])

# #     rng = np.random.Generator()
# #     r = M.sum(axis=1)

# #     for i in range(ni):
# #         m_new = rng.multinomial(r[i], q[i,:])

# #     print(m_new)
