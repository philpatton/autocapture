import numpy as np
from jolly_seber import JollySeber


def test_JollySeber(): 

    N = 2
    PHI = np.array([[0.5], [0.5]])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    b = np.array([0.5, 0.5])

    js = JollySeber(N=N, PHI=PHI, P=P, b=b, seed=42)

    assert js.N == N 

def test_simulate_capture():

    N = 2
    PHI = np.array([[0.5], [0.5]])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    b = np.array([0.5, 0.5])

    js = JollySeber(N=N, PHI=PHI, P=P, b=b, seed=42)

    capture = js.simulate_capture()
    should_be = np.array([[1, 0], [1, 1]])

    assert np.array_equal(capture, should_be)

def test_simulate_z():

    N = 2
    PHI = np.array([[0.5], [0.5]])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    b = np.array([0.5, 0.5])

    js = JollySeber(N=N, PHI=PHI, P=P, b=b, seed=42)

    entry_occasions = js.simulate_entry()

    Z = js.simulate_z(entry_occasions)
    should_be = np.array([[1, 1], [0, 1]])

    assert np.array_equal(Z, should_be)

def test_simulate_entry():

    N = 2
    PHI = np.array([[0.5], [0.5]])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    b = np.array([0.5, 0.5])

    js = JollySeber(N=N, PHI=PHI, P=P, b=b, seed=42)

    entry_occasions = js.simulate_entry()
    should_be = np.array([0, 1])

    assert np.array_equal(entry_occasions, should_be)

def test_simulate_data():

    N = 2
    PHI = np.array([[0.5], [0.5]])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    b = np.array([0.5, 0.5])

    js = JollySeber(N=N, PHI=PHI, P=P, b=b, seed=42)   

    results = js.simulate_data()

    ch_should_be = np.array([[0, 1], [0, 1]])
    B_should_be = np.array([1, 1])
    N_should_be = np.array([1, 2])

    assert np.array_equal(results['capture_history'], ch_should_be)
    assert np.array_equal(results['B'], B_should_be)
    assert np.array_equal(results['N'], N_should_be)
   