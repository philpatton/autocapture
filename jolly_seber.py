
""" Simulate data for a Jolly-Seber model. 

This code was adapted from Kery and Schaub (2011) BPA, Chapter 10.

Typical usage example:

    js = JollySeber(N=N, PHI=PHI, P=P, b=b)
    results = js.simulate_data()
    print(results['capture_history'][:5])
"""

import numpy as np

def first_nonzero(arr, axis=1, invalid_val=-1):
    """Finds the first nonzero value along an axis."""
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def main():

    N = 1000
    T = 10
    phi = 0.9
    p = 0.4 
    b0 = 0.35
    frr = 0.05 
    mark_change_rate = 0.2

    gamma = (1 - mark_change_rate)
    alpha = (1 - frr)

    b = np.zeros(T)
    b[0] = b0
    b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 

    # survival probabilities 
    phi_shape = (N, T - 1)
    PHI = np.full(phi_shape, phi)

    # capture probabilities 
    p_shape = (N, T)
    P = np.full(p_shape, p)

    js = JollySeber(N=N, PHI=PHI, P=P, b=b, alpha=alpha, gamma=gamma)

    results = js.simulate_data()

    th = results['true_history']

    print(th)

class JollySeber:
    """Data simulator for jolly seber models.
    
    Attributes:
        N: An integer count of the superpopulation
        PHI: N by T-1 matrix of survival probabilities between occassions
        P: N by T matrix of capture probabilities
        b: T by 1 vector of entrance probabilities 
        rng: Random number generator used by the model 
    """

    def __init__(self, N: int, PHI: np.ndarray, P: np.ndarray,
                 b: np.ndarray, alpha: float = None, beta: float = None, 
                 gamma: float = None, seed: int = None):
        """Init the data genertor with hyperparameters and init the rng"""
        self.N = N
        self.PHI = PHI
        self.P = P
        self.b = b
        self.T = len(b)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
 
    def simulate_data(self):
        """Simulates the Jolly Seber model from the hyperparameters.
        
        Returns:
            out_dict: dictionary containing the capture history, number of 
            entrants, and the true N at each occasion. 
        """

        # each animal has some entry (birth, imm.) time; 0 if already entered
        entry_occasions = self.simulate_entry()
        _, B = np.unique(entry_occasions, return_counts=True)
        
        # Z in (0,1) of shape (N_super, T) indicating alive and entered
        Z = self.simulate_z(entry_occasions)
        N_true = Z.sum(axis=0)
        
        # matrix of coin flips (p=p) of size Z indicating a potential capture 
        captures = self.simulate_capture()

        # captured AND available for capture, i.e., has entered and is alive
        true_history = captures * Z
        
        # create observed history containing errors
        capture_history = true_history.copy()

        # adjust capture history with false rejects
        if self.alpha is not None:

            false_reject_indices = self.flag_false_rejects(capture_history)
            ghost_history = self.create_ghost_history(false_reject_indices,
                                                      capture_history)

            # copy recaptures to ghost histories if mark changes 
            if self.gamma is not None:
                ghost_history = self.copy_recaptures_to_ghost(
                    false_reject_indices,
                    ghost_history,
                    capture_history
                )
    
                # zero out all subsequent recaptures in true capture history
                for idx in false_reject_indices:
                    capture_history[idx[0], idx[1]:] = 0
            
            # if there are no mark changes 
            else:
                # zero out individual recapture in true capture history 
                capture_history[false_reject_indices] = 0

            # TODO: False positives in capture history 
            capture_history = np.vstack((capture_history, ghost_history))

        # filter all zero histories
        was_seen = (capture_history != 0).any(axis=1)
        capture_history = capture_history[was_seen]

        # filter all zero histories
        was_seen = (true_history != 0).any(axis=1)
        true_history = true_history[was_seen]
        
        out_dict = {'capture_history':capture_history, 'B':B, 'N':N_true,
                    'true_history':true_history}
        
        return out_dict

    def simulate_entry(self):
        """Simulate occasion for animal's entry into population."""

        # matrix where one indicates entry 
        entry_matrix = self.rng.multinomial(n=1, pvals=self.b, size=self.N)

        # index of the first nonzero value (entry)
        entry_occasions = entry_matrix.nonzero()[1]

        return entry_occasions

    def simulate_z(self, entry_occasions: np.ndarray):
        """Simulate discrete latent state, alive and entered, for jolly-seber

        Args: 
            entry_occasions: A 1D array with length N indicating the time of 
            entry. 

        Returns:
            N by T matrix indicating that the animal is alive and entered
        """
        
        # simulate survival between occasions
        life_matrix = [
            self.rng.binomial(n=1, p=self.PHI[i]) 
            for i in range(self.N)
        ]
        life_matrix = np.stack(life_matrix, axis=0)

        # add column such that survival between t and t+1 implies alive at t+1 
        life_matrix = np.insert(life_matrix, 0, np.ones(self.N), axis=1)

        # matrix where 1 will indicate that animal has entered
        entry_matrix = np.zeros(life_matrix.shape).astype(int)

        for i in range(self.N):

            # fill matrix with one after entry (non-zero value in entry_matrix)
            entry_matrix[i, entry_occasions[i]:] = 1    

            # ensure no death before or during entry occasion
            life_matrix[i, :(entry_occasions[i] + 1)] = 1

            # find first zero in the row of the life matrix 
            death_occasion = (life_matrix[i] == 0).argmax() 
            if death_occasion != 0: # argmax returns 0 if animal never dies
                life_matrix[i, death_occasion:] = 0    

        # N by T matrix indicating that animal is alive and entered
        Z = entry_matrix * life_matrix
            
        return Z

    def simulate_capture(self):
        """Generate a binomial matrix indicating capture."""
        capture = [
            self.rng.binomial(n=1, p=self.P[i]) 
            for i in range(self.N)
        ]
        capture = np.stack(capture, axis=0)

        return capture

    def flag_false_rejects(self, capture_history):
        """Flag captures as false rejects.
        
        This iteration only flags the recaptures as false rejects. 
        
        Return:
            tuple with (animal, occassion) for false rejects.
        """
        first_capture_occasion = first_nonzero(capture_history)

        # zero out the first capture occasion to get the recapture history 
        recapture_history = capture_history.copy()
        capture_count = capture_history.shape[0]
        recapture_history[np.arange(capture_count), first_capture_occasion] = 0
          
        recapture_count = recapture_history.sum()
        dummy_randoms = self.rng.uniform(size=recapture_count)
        is_false_reject = dummy_randoms > self.alpha

        # find the (animal, occassion) for each false reject
        recapture_idx = recapture_history.nonzero()
        false_reject_animal = recapture_idx[0][is_false_reject]
        false_reject_occasion = recapture_idx[1][is_false_reject]

        return false_reject_animal, false_reject_occasion

    def flag_mark_changes(self, false_reject_indices):

        false_reject_animal, false_reject_occasion = false_reject_indices

        # randomly classify some of the false rejects as mark change  
        total_false_rejects = len(false_reject_animal)      
        dummy_randoms = self.rng.uniform(size=total_false_rejects)
        is_mark_change = dummy_randoms > self.gamma

        mark_change_animal = false_reject_animal[is_mark_change]
        mark_change_occasion = false_reject_occasion[is_mark_change]

        return mark_change_animal, mark_change_occasion

    def copy_recaptures_to_ghost(
        self, 
        false_reject_indices, 
        ghost_history, 
        capture_history
    ):
        """Moves recaptures from true capture history to ghosts."""
        false_reject_animal, false_reject_occasion = false_reject_indices

        # randomly classify some of the false rejects as mark change  
        total_false_rejects = len(false_reject_animal)      
        dummy_randoms = self.rng.uniform(size=total_false_rejects)
        is_mark_change = dummy_randoms > self.gamma

        mark_change_animal = false_reject_animal[is_mark_change]
        mark_change_occasion = false_reject_occasion[is_mark_change]

        # filling in the ghost history after each change
        mark_change_idx = np.where(is_mark_change)
        for i in range(len(mark_change_idx)):

            # capture history after the mark change
            post_change = capture_history[mark_change_animal[i], 
                                            (mark_change_occasion[i] + 1):]

            # add this history to the ghost history
            occ = (mark_change_occasion[i] + 1)
            ghost_history[mark_change_idx[i], occ:] = post_change

        return ghost_history


    def create_ghost_history(self, false_reject_indices, capture_history):
        """Create capture histories for false rejects.
        
        If there are mark changes, recaptures will be allocated to the ghost 
        history. Otherwise, there will only be one capture for each ghost 
        history. 
        """
        _, false_reject_occasion = false_reject_indices
        
        # create ghost histories
        total_false_rejects = len(false_reject_occasion)
        ghost_history = np.zeros((total_false_rejects, self.T), dtype=int)

        # 'indices' ensures we select each false_reject_occasion in turn 
        indices = np.arange(total_false_rejects)
        ghost_history[indices, false_reject_occasion] = 1
            
        return ghost_history

if __name__ == '__main__':
    main()