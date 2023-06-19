import numpy as np 
import pymc as pm
import arviz as az

from src.cjs import Simulator, BayesEstimator, MLE

from src.utils import expit

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

    data = np.array([
        [11, 2 , 0  ,0  ,0  ,0 , 9 ],
        [0 , 24, 1  ,0  ,0  ,0 , 35],
        [0 , 0 , 34 ,2  ,0  ,0 , 42],
        [0 , 0 , 0  ,45 ,1  ,2 , 32],
        [0 , 0 , 0  ,0  ,51 ,0 , 37],
        [0 , 0 , 0  ,0  ,0  ,52, 46]
    ])    

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