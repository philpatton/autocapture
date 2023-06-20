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

    def test_popan(self):

        model = self.popan.compile_pymc_model(self.dipper)
        with model:
            idata = pm.sample()

        summary = az.summary(idata)
        reals = summary.loc[['p','phi'], 'mean'].values

        mles = [0.90, 0.56]

        assert np.allclose(reals, mles, rtol=0.02)

        bayes_n = summary.loc['N', 'mean']
        mle_n = 310

        assert np.isclose(bayes_n, mle_n, rtol=5)

        b_reals = summary.loc[summary.index.str.contains('beta'), 'mean'].values
        mle_b = np.array([0.08, 0.17, 0.18, 0.15, 0.14, 0.16, 0.13])

        assert np.allclose(b_reals, mle_b, rtol=0.03)

# debug_kwargs = {
#     'N': 2,
#     'T': 2,
#     'phi': 0.5,
#     'p': 0.5,
#     'b': np.array([0.5, 0.5]),
#     'seed': 42
# }

# def test_simulator(): 

#     ps = SimulatorPOPAN(**debug_kwargs)

#     assert ps.N == debug_kwargs['N'] 

# def test_simulate_capture():

#     ps = SimulatorPOPAN(**debug_kwargs)

#     capture = ps.simulate_capture()
#     should_be = np.array([[1, 0], [1, 1]])

#     assert np.array_equal(capture, should_be)

# def test_simulate_z():

#     ps = SimulatorPOPAN(**debug_kwargs)

#     entry_occasions = ps.simulate_entry()

#     Z = ps.simulate_z(entry_occasions)
#     should_be = np.array([[1, 1], [0, 1]])

#     assert np.array_equal(Z, should_be)

# def test_simulate_entry():

#     ps = SimulatorPOPAN(**debug_kwargs)

#     entry_occasions = ps.simulate_entry()
#     should_be = np.array([0, 1])

#     assert np.array_equal(entry_occasions, should_be)

# def test_simulate():

#     ps = SimulatorPOPAN(**debug_kwargs)   

#     results = ps.simulate()

#     ch_should_be = np.array([[0, 1], [0, 1]])
#     B_should_be = np.array([1, 1])
#     N_should_be = np.array([1, 2])

#     assert np.array_equal(results['capture_history'], ch_should_be)
#     assert np.array_equal(results['B'], B_should_be)
#     assert np.array_equal(results['N'], N_should_be)

# def test_main():

#     N = 1000
#     T = 10
#     phi = 0.9
#     p = 0.4 
#     b0 = 0.35

#     alpha = 0.1
#     beta = 0.1
#     gamma = 0.1

#     b = np.zeros(T)
#     b[0] = b0
#     b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 

#     seed = 42

#     ps = SimulatorPOPAN(N=N, T=T, phi=phi, p=p, b=b, seed=seed)

#     sim = ps.simulate()
