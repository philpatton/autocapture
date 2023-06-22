import numpy as np 
import pymc as pm
import arviz as az
import pandas as pd

from src.cjs import CJS

from src.utils import expit

debug_kwargs = {
    'released_count': 25,
    'T': 10,
    'phi': 0.9,
    'p': 0.25,
    'seed': 17
}

sample_kwargs = {
    'draws' : 1000,
    'tune': 1000
}

class TestCJS:

    cjs = CJS()

    dipper = pd.read_csv('input/dipper.csv').values

    model = cjs.compile_pymc_model(dipper)
    with model:
        idata = pm.sample()

    def test_estimate_mle(self):

        results = self.cjs.estimate_mle(self.dipper)

        reals = expit(results.est_logit.values)

        assert np.allclose(reals, [0.9, 0.56], rtol=0.1)

        standard_errors = results.se.values

        assert np.allclose(standard_errors, [0.325, 0.102], rtol=0.01)

    def test_estimate_bayes(self):

        summary = az.summary(self.idata)

        phi_summary = summary.loc[summary.index.str.contains('phi')]

        phi_mle = np.array([0.626, 0.454, 0.478, 0.624, 0.608, 0.583])
        phi_posterior_mean = phi_summary['mean'].values
        assert np.allclose(phi_mle, phi_posterior_mean, rtol=0.05)

        # check coverage
        phi_low = phi_summary['hdi_3%']
        phi_high = phi_summary['hdi_97%']
        is_between = (phi_mle > phi_low) & (phi_mle < phi_high)
        assert all(is_between)
    
    def test_check(self):

        ppc_results = self.cjs.check(self.idata, self.dipper)

        d_new = ppc_results['freeman_tukey_new']
        d_obs = ppc_results['freeman_tukey_observed']

        p_val = np.mean(d_new > d_obs)
        should_be = 0.09

        assert np.isclose(should_be, p_val, atol=0.025)

    def test_simulate(self):

        results = self.cjs.simulate(**debug_kwargs)
        ch = results['capture_history']
        interval_count = (debug_kwargs['T'] - 1)

        assert type(ch) is np.ndarray
        assert ch.shape[0] == debug_kwargs['released_count'] * interval_count 