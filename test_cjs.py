import numpy as np 
import pymc as pm
import arviz as az

from src.cjs import BayesEstimator
from src.popan import POPANSimulator

T = 10
b0 = 0.4
bb = np.repeat((1 - b0) / (T - 1), T - 1)
b = np.insert(bb, 0, b0)

debug_kwargs = {
    'N': 1000,
    'T': T,
    'phi': 0.5,
    'p': 0.8,
    'b': b,
    'seed': 17
}

def test_bayes_estimator():

    ps = POPANSimulator(**debug_kwargs)   
    results = ps.simulate()

    e = BayesEstimator(results['capture_history'])
    model = e.compile()

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