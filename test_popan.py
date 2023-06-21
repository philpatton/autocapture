import numpy as np
import pandas as pd 
import pymc as pm
import arviz as az

from src.popan import POPAN

class TestSimulator:

    N = 2
    T = 2

    phi = 0.5
    PHI = np.full((N, T - 1), phi)

    p = 0.5
    P = np.full((N, T), p)  

    b = np.array([0.5, 0.5])
    seed = 42

    popan = POPAN(seed=seed)

    def test_simulate(self):
        
        results = self.popan.simulate(N=self.N, T=self.T, phi=self.phi, 
                                      b=self.b, p=self.p)

        ch_should_be = np.array([[0, 1], [0, 1]])
        B_should_be = np.array([1, 1])
        N_should_be = np.array([1, 2])

        assert np.array_equal(results['capture_history'], ch_should_be)
        assert np.array_equal(results['B'], B_should_be)
        assert np.array_equal(results['N'], N_should_be)

    def test_simulate_capture(self):

        capture = self.popan.simulate_capture(P=self.P, N=self.N)

        should_be = np.array([[0, 0], [0, 1]])
        assert np.array_equal(capture, should_be)

    def test_simulate_entry(self):
        entry_occasions = self.popan.simulate_entry(b=self.b, N=self.N)
        should_be = np.array([0, 0])
        assert np.array_equal(entry_occasions, should_be)

        self.entry_occasions = entry_occasions

    def test_simulate_z(self):
        entry_occasions = self.popan.simulate_entry(b=self.b, N=self.N)
        Z = self.popan.simulate_z(PHI=self.PHI, N=self.N,
                                  entry_occasions=entry_occasions)
        should_be = np.array([[0, 1], [0, 1]])

        assert np.array_equal(Z, should_be)

class TestEstimator:

    popan = POPAN()

    dipper = pd.read_csv('input/dipper.csv').values

    model = popan.compile_pymc_model(dipper)
    with model:
        idata = pm.sample()

    def test_popan(self):

        summary = az.summary(self.idata)
        reals = summary.loc[['p','phi', 'b0'], 'mean'].values

        mles = np.array([0.90, 0.56, 0.08])

        assert np.allclose(reals, mles, atol=0.01)

        bayes_n = summary.loc['N', 'mean']
        mle_n = 310

        assert np.isclose(bayes_n, mle_n, rtol=5)

    def test_check(self):

        ppc_results = self.popan.check(self.idata, self.dipper)

        d_new = ppc_results['freeman_tukey_new']
        d_obs = ppc_results['freeman_tukey_observed']

        p_val = np.mean(d_new > d_obs)
        should_be = 0.06

        assert np.isclose(p_val, should_be, atol=0.025)