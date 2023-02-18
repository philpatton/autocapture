import numpy as np
from jolly_seber import JollySeber

debug_kwargs = {
    'N': 2,
    'T': 2,
    'PHI': np.array([[0.5], [0.5]]),
    'P': np.array([[0.5, 0.5], [0.5, 0.5]]),
    'b': np.array([0.5, 0.5]),
    'seed': 42
}

def test_JollySeber(): 

    N = 2
    alpha = 0.1
    beta = 0.1
    gamma = 0.1

    js = JollySeber(alpha=alpha, beta=beta, gamma=gamma, **debug_kwargs)

    assert js.N == N 

def test_simulate_capture():

    js = JollySeber(**debug_kwargs)

    capture = js.simulate_capture()
    should_be = np.array([[1, 0], [1, 1]])

    assert np.array_equal(capture, should_be)

def test_simulate_z():

    js = JollySeber(**debug_kwargs)

    entry_occasions = js.simulate_entry()

    Z = js.simulate_z(entry_occasions)
    should_be = np.array([[1, 1], [0, 1]])

    assert np.array_equal(Z, should_be)

def test_simulate_entry():

    js = JollySeber(**debug_kwargs)

    entry_occasions = js.simulate_entry()
    should_be = np.array([0, 1])

    assert np.array_equal(entry_occasions, should_be)

def test_simulate():

    js = JollySeber(**debug_kwargs)   

    results = js.simulate()

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

    # survival probabilities 
    PHI = np.full((N, T - 1), phi)

    # capture probabilities 
    P = np.full((N, T), p)

    seed = 42

    js = JollySeber(N=N, T=T, PHI=PHI, P=P, b=b, seed=seed)

    sim = js.simulate()

    pass