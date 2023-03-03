import numpy as np
from popan import POPANSimulator

debug_kwargs = {
    'N': 2,
    'T': 2,
    'phi': 0.5,
    'p': 0.5,
    'b': np.array([0.5, 0.5]),
    'seed': 42
}

def test_POPANSimulator(): 

    ps = POPANSimulator(**debug_kwargs)

    assert ps.N == debug_kwargs['N'] 

def test_simulate_capture():

    ps = POPANSimulator(**debug_kwargs)

    capture = ps.simulate_capture()
    should_be = np.array([[1, 0], [1, 1]])

    assert np.array_equal(capture, should_be)

def test_simulate_z():

    ps = POPANSimulator(**debug_kwargs)

    entry_occasions = ps.simulate_entry()

    Z = ps.simulate_z(entry_occasions)
    should_be = np.array([[1, 1], [0, 1]])

    assert np.array_equal(Z, should_be)

def test_simulate_entry():

    ps = POPANSimulator(**debug_kwargs)

    entry_occasions = ps.simulate_entry()
    should_be = np.array([0, 1])

    assert np.array_equal(entry_occasions, should_be)

def test_simulate():

    ps = POPANSimulator(**debug_kwargs)   

    results = ps.simulate()

    ch_should_be = np.array([[0, 1], [0, 1]])
    B_should_be = np.array([1, 1])
    N_should_be = np.array([1, 2])

    assert np.array_equal(results['capture_history'], ch_should_be)
    assert np.array_equal(results['B'], B_should_be)
    assert np.array_equal(results['N'], N_should_be)

def test_main():

    N = 1000
    T = 10
    phi = 0.9
    p = 0.4 
    b0 = 0.35

    alpha = 0.1
    beta = 0.1
    gamma = 0.1

    b = np.zeros(T)
    b[0] = b0
    b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 

    seed = 42

    ps = POPANSimulator(N=N, T=T, phi=phi, p=p, b=b, seed=seed)

    sim = ps.simulate()

    pass