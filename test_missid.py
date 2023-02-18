import numpy as np
from miss_id import MissID

debug_kwargs = {
    'alpha': 0.2,
    'beta': 0.2,
    'gamma': 0.2,
    'seed': 42
}

def test_create_recapture_history():

    mi = MissID(**debug_kwargs)   

    capture_history = np.array([[1, 1], [0, 1]])
    recapture_history = mi.create_recapture_history(capture_history)

    rh_should_be = np.array([[0,1],[0,0]])

    assert np.array_equal(recapture_history, rh_should_be)

def test_flag_errors():

    N = 20
    T = 5

    mi = MissID(**debug_kwargs)

    ch = np.reshape(mi.rng.binomial(1, 0.5, N * T), (N, T))
    rh = mi.create_recapture_history(ch)

    ef = mi.flag_errors(rh)

    assert ef['ghost'].sum() == 7
    assert ef['mark_change'].sum() == 5
    assert ef['false_accept'].sum() == 4

def test_get_error_indices():

    N = 20
    T = 5

    mi = MissID(**debug_kwargs)

    ch = np.reshape(mi.rng.binomial(1, 0.5, N * T), (N, T))
    rh = mi.create_recapture_history(ch)
    ef = mi.flag_errors(rh)

    ei_ghost = mi.get_error_indices(rh, ef['ghost'])
    ei_mark = mi.get_error_indices(rh, ef['mark_change'])
    ei_false = mi.get_error_indices(rh, ef['false_accept'])

    assert np.array_equal(ei_ghost[0], np.array([ 3,  4,  8, 12, 12, 15, 17]))
    assert np.array_equal(ei_mark[0], np.array([ 1,  3,  6,  8, 19]))
    assert np.array_equal(ei_false[0], np.array([ 4,  9, 15, 16]))

def test_create_ghost_history():

    mi = MissID(**debug_kwargs)

    ghost_indices = (np.array([1]), np.array([1]))
    gh = mi.create_ghost_history(ghost_indices, T=2)
    gh_should_be = np.array([[0, 1]])

    assert np.array_equal(gh, gh_should_be)

def test_pick_wrong_animals():

    mi = MissID(**debug_kwargs)

    ch = np.array([[1,1,0,1],[0,0,1,0],[0,1,0,1]])
    false_accept_indices = (np.array([1]), np.array([2]))

    wrong_animals = mi.pick_wrong_animals(false_accept_indices, ch)

    assert wrong_animals == 2

def test_copy_recaptures_to_changed_animal():

    mi = MissID(**debug_kwargs)

    ch = np.array([[1,1,0,1],[0,0,1,0],[0,1,0,1]])
    mark_change_indices = (np.array([0]), np.array([1]))
    mark_change_history = mi.create_ghost_history(
        mark_change_indices, 
        T=ch.shape[1]
    )

    recapture_history = mi.create_recapture_history(ch)

    mark_change_history = mi.copy_recaptures_to_changed_animal(
        mark_change_indices, 
        mark_change_history, 
        recapture_history
    )

    # zero out recapture and subsequent history for changed animal 
    mark_change_animals, mark_change_occasions = mark_change_indices
    for animal, occasion in zip(mark_change_animals, mark_change_occasions):
        ch[animal, occasion:] = 0

    mch_should_be = np.array([[0,1,0,1]])
    ch_should_be =  np.array([1,0,0,0])

    assert np.array_equal(mark_change_history, mch_should_be)
    assert np.array_equal(ch[0], ch_should_be)

def test_simulate_similarity():

    mi = MissID(**debug_kwargs)

    sim = mi.simulate_similarity(10)

    assert all(sim.diagonal() == 0)
    assert sim.shape[0] == 10
    assert sim.shape[1] == 10

def test_simulate_capture_history():

    kwargs={
        'alpha':0,
        'beta':0,
        'gamma':0,
        'seed':24
    }

    mi = MissID(**kwargs)

    true_history = np.array([[1,1,0,1],[0,0,1,0],[0,1,0,1]])

    capture_history = mi.simulate_capture_history(true_history)

    # when alpha, beta, gamma all zero, capture_history should == true_history
    assert np.array_equal(capture_history, true_history)

    mi_error = MissID(**debug_kwargs)

    error_history = mi_error.simulate_capture_history(true_history)

    # assert np.array_equal(error_dict)
    assert error_history.shape == (4, 4)

class TestErrorProcesses():

    ch = np.array([
        [1,1,1,1,1],
        [0,1,0,1,1],
        [1,0,1,0,1],
        [0,0,0,1,1]
    ])

    mi = MissID(**debug_kwargs)

    rh = mi.create_recapture_history(ch)

    recapture_count = rh.sum()

    flag_dict = {
        'ghost':np.full(recapture_count, False),
        'mark_change':np.full(recapture_count, False),
        'false_accept':np.full(recapture_count, False),
    }

    # mark change happens for animal 0 at occasion 2
    flag_dict['mark_change'][1] = True

    # mark change happens for animal 1 at occasion 3
    flag_dict['ghost'][4] = True

    # mark change happens for animal 2 at occasion 2
    flag_dict['false_accept'][6] = True

    def test_mark_change_process(self):

        ch2 = self.mi.mark_change_process(self.rh, self.flag_dict, self.ch)

        assert ch2.shape == (self.ch.shape[0] + 1, self.ch.shape[1])
        assert np.array_equal(ch2[0], np.array([1,1,0,0,0]))
        assert np.array_equal(ch2[4], np.array([0,0,1,1,1]))
        assert np.array_equal(self.ch[1:4], ch2[1:4])

    def test_ghost_process(self):

        ch2 = self.mi.ghost_process(self.rh, self.flag_dict, self.ch)

        assert ch2.shape == (self.ch.shape[0] + 1, self.ch.shape[1])
        assert np.array_equal(ch2[1], np.array([0,1,0,0,1]))
        assert np.array_equal(ch2[4], np.array([0,0,0,1,0]))
        assert np.array_equal(self.ch[0], ch2[0])

    def test_false_accept_process(self):

        ch2 = self.mi.false_accept_process(self.rh, self.flag_dict, self.ch)

        assert ch2.shape == self.ch.shape
        assert np.array_equal(ch2[0], self.ch[0])
        assert np.array_equal(ch2[2], np.array([1,0,0,0,1]))