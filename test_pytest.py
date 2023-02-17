import numpy as np
from jolly_seber import JollySeber

debug_kwargs = {
    'N': 2,
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

def test_create_recapture_history():

    js = JollySeber(**debug_kwargs)   

    capture_history = np.array([[1, 1], [0, 1]])
    recapture_history = js.create_recapture_history(capture_history)

    rh_should_be = np.array([[0,1],[0,0]])

    assert np.array_equal(recapture_history, rh_should_be)

def test_flag_errors():

    N = 20
    T = 5
    alpha = 0.3
    beta = 0.1
    gamma = 0.1

    js = JollySeber(**debug_kwargs, alpha=alpha, beta=beta, gamma=gamma)

    ch = np.reshape(js.rng.binomial(1, 0.5, N * T), (N, T))
    rh = js.create_recapture_history(ch)

    ef = js.flag_errors(rh)

    assert ef['ghost'].sum() == 10
    assert ef['mark_change'].sum() == 1
    assert ef['false_accept'].sum() == 3

def test_get_error_indices():

    N = 20
    T = 5
    alpha = 0.3
    beta = 0.1
    gamma = 0.1

    js = JollySeber(**debug_kwargs, alpha=alpha, beta=beta, gamma=gamma)

    ch = np.reshape(js.rng.binomial(1, 0.5, N * T), (N, T))
    rh = js.create_recapture_history(ch)
    ef = js.flag_errors(rh)

    ei_ghost_should_be = np.array([1, 3, 4, 4, 1, 2, 2, 4, 4, 4])
    ei_mark_should_be = np.array([9])
    ei_false_should_be = np.array([2, 3, 2])

    ei_ghost = js.get_error_indices(rh, ef['ghost'])
    ei_mark = js.get_error_indices(rh, ef['mark_change'])
    ei_false = js.get_error_indices(rh, ef['false_accept'])

    assert np.array_equal(ei_ghost[1], ei_ghost_should_be)
    assert np.array_equal(ei_mark[0], ei_mark_should_be)
    assert np.array_equal(ei_false[1], ei_false_should_be)

def test_create_ghost_history():

    js = JollySeber(**debug_kwargs)

    ghost_indices = (np.array([1]), np.array([1]))
    gh = js.create_ghost_history(ghost_indices)
    gh_should_be = np.array([[0, 1]])

    assert np.array_equal(gh, gh_should_be)

def test_pick_wrong_animals():

    js = JollySeber(**debug_kwargs)

    ch = np.array([[1,1,0,1],[0,0,1,0],[0,1,0,1]])
    false_accept_indices = (np.array([1]), np.array([2]))

    wrong_animals = js.pick_wrong_animals(false_accept_indices, ch)

    assert wrong_animals == 2

def test_copy_recaptures_to_changed_animal():

    kwargs = {
        'N': 3,
        'PHI': np.array([[0.5], [0.5]]),
        'P': np.array([[0.5, 0.5], [0.5, 0.5]]),
        'b': np.array([0.25, 0.25, 0.25, 0.25]),
        'seed': 42
    }

    js = JollySeber(**kwargs)

    mark_change_indices = (np.array([0]), np.array([1]))
    mark_change_history = js.create_ghost_history(mark_change_indices)

    capture_history = np.array([[1,1,0,1],[0,0,1,0],[0,1,0,1]])
    recapture_history = js.create_recapture_history(capture_history)

    mark_change_history = js.copy_recaptures_to_changed_animal(
        mark_change_indices, 
        mark_change_history, 
        recapture_history
    )

    # zero out recapture and subsequent history for changed animal 
    mark_change_animals, mark_change_occasions = mark_change_indices
    for animal, occasion in zip(mark_change_animals, mark_change_occasions):
        capture_history[animal, occasion:] = 0

    mch_should_be = np.array([[0,1,0,1]])
    ch_should_be =  np.array([1,0,0,0])

    assert np.array_equal(mark_change_history, mch_should_be)
    assert np.array_equal(capture_history[0], ch_should_be)

def test_simulate_similarity():

    js = JollySeber(**debug_kwargs)

    sim = js.simulate_similarity(10)

    assert all(sim.diagonal() == 0)
    assert sim.shape[0] == 10
    assert sim.shape[1] == 10

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
    phi_shape = (N, T - 1)
    PHI = np.full(phi_shape, phi)

    # capture probabilities 
    p_shape = (N, T)
    P = np.full(p_shape, p)

    seed = 42

    js = JollySeber(N=N, PHI=PHI, P=P, b=b, alpha=alpha, beta=beta, gamma=gamma,
                    seed=seed)

    sim = js.simulate()

    pass