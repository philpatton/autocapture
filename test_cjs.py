import numpy as np 
import pymc as pm
import arviz as az

from src.cjs import CJSEstimator
from src.popan import POPANSimulator

T = 10
b0 = 0.4
bb = np.repeat((1 - b0) / (T - 1), T - 1)
b = np.concatenate((b0, bb))

debug_kwargs = {
    'N': 500,
    'T': T,
    'phi': 0.5,
    'p': 0.8,
    'b': b,
    'seed': 17
}

def test_cjs_estimator():

    ps = POPANSimulator(**debug_kwargs)   
    results = ps.simulate()

    e = CJSEstimator(results['capture_history'])
    model = e.compile()

    with  model:
        idata = pm.sample()

    summary = az.summary(idata, var_names='phi')

    phi_hat = summary.mean