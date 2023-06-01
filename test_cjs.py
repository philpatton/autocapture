import numpy as np 
import pymc as pm
import arviz as az

from src.cjs import CJSEstimator
from src.popan import POPANSimulator

debug_kwargs = {
    'N': 50,
    'T': 5,
    'phi': 0.5,
    'p': 0.8,
    'b': np.array([0.6, 0.1, 0.1, 0.1, 0.1]),
    'seed': 17
}

def test_cjs_estimator():

    ps = POPANSimulator(**debug_kwargs)   
    results = ps.simulate()

    e = CJSEstimator(results['capture_history'])
    model = e.compile()

    with  model:
        idata = pm.sample(e)

    summary = az.summarize(idata, var_names='phi')

    phi_hat = summary.mean

if __name__ == '__main__':
    test_cjs_estimator()
