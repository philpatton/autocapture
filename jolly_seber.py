
""" Simulate data for a Jolly-Seber model. 

This code was adapted from Kery and Schaub (2011) BPA, Chapter 10.

Typical usage example:

    js = JollySeber(N=N, PHI=PHI, P=P, b=b)
    results = js.simulate_data()
    print(results['capture_history'][:5])
"""

from utils import softmax, first_nonzero

import numpy as np

class JollySeber:
    """Data simulator for jolly seber models.
    
    Attributes:
        N: An integer count of the superpopulation
        PHI: N by T-1 matrix of survival probabilities between occassions
        P: N by T matrix of capture probabilities
        b: T by 1 vector of entrance probabilities 
        rng: np.random.Generator used by the model 
        seed: integer seed for the rng
        alpha: the proportion of recaptures resulting in ghosts 
        beta: the proportion of recatpures resulting in mark changes
        gamma: the proportion of recaptures resulting in false accepts 
        A: alpha parameter in beta distribution of similarity scores
        B: beta parameter in beta distribution of similarity scores
    """

    def __init__(self, N: int, PHI: np.ndarray, P: np.ndarray,
                 b: np.ndarray, alpha: float = 0, beta: float = 0, 
                 gamma: float = 0, seed: int = 0, A: float = 2, B: float = 5):
        """Init the data generator with hyperparameters and init the rng"""
        self.N = N
        self.PHI = PHI
        self.P = P
        self.b = b
        self.T = len(b)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.A = A
        self.B = B
 
    def simulate(self):
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
        
        # create observed capture_history, which may contain misid errors
        capture_history = true_history.copy()

        # both types of misids only occur on recaptures
        recapture_history = self.create_recapture_history(capture_history)

        # flag the recaptures for each error type (false accept or reject)
        flag_dict = self.flag_errors(recapture_history)

        if any(flag_dict['false_accept']):

            # copy recaptures to        
            false_accept_indices = self.get_error_indices(
                recapture_history,
                flag_dict['false_accept']
            )
            wrong_animals = self.pick_wrong_animals(
                false_accept_indices,
                capture_history
            )
                        
            # copy the recaptures to the misidentified animal 
            capture_history[wrong_animals, false_accept_indices[1]] = 1
                    
            # zero out falsely accepted animals
            capture_history[false_accept_indices] = 0

        if any(flag_dict['mark_change']):

            # create ghost histories for every changed animal
            mark_change_indices = self.get_error_indices(
                recapture_history,
                flag_dict['mark_change']
            )
            mark_change_history = self.create_ghost_history(mark_change_indices)

            # copy recaptures from the animals original history to the new one
            mark_change_history = self.copy_recaptures_to_changed_animal(
                mark_change_indices,
                mark_change_history,
                recapture_history
            )

            # zero out recapture and subsequent history for changed animal 
            mc_animals, mc_occasions = mark_change_indices
            for animal, occasion in zip(mc_animals, mc_occasions):
                capture_history[animal, occasion:] = 0

            capture_history = np.vstack((capture_history, mark_change_history))

        if any(flag_dict['ghost']):

            # create ghost histories for non-mark-changes
            ghost_indices = self.get_error_indices(
                recapture_history,
                flag_dict['ghost']
            )
            ghost_history = self.create_ghost_history(ghost_indices)
        
            # for ghosts, zero out recapture in original capture history 
            capture_history[ghost_indices] = 0

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

    def create_recapture_history(self, capture_history):

        # zero out the first capture occasion to get the recapture history 
        recapture_history = capture_history.copy()
        capture_count = capture_history.shape[0]
        first_capture_occasion = first_nonzero(capture_history)
        recapture_history[np.arange(capture_count), first_capture_occasion] = 0

        return recapture_history

    def flag_errors(self, recapture_history):
        """Flag recaptures as ghosts, mark changes, false accepts."""

        # draw random uniform for classifying each recapture
        recapture_count = recapture_history.sum()
        dummy_randoms = self.rng.uniform(size=recapture_count)

        # cutoffs for errors 
        abg = self.alpha + self.beta + self.gamma
        bg = self.beta + self.gamma 
        
        # classify errors
        ghost_flag = (abg > dummy_randoms) & (dummy_randoms > bg)
        mark_change_flag = (bg > dummy_randoms) & (dummy_randoms > self.beta)
        false_accept_flag = dummy_randoms < self.beta
        
        flag_dict = {
            'ghost':ghost_flag,
            'mark_change':mark_change_flag,
            'false_accept':false_accept_flag
        }

        return flag_dict

    def get_error_indices(self, recapture_history, error_flag):

        # find the (animal, occassion) for each false reject
        recapture_idx = recapture_history.nonzero()
        error_animal = recapture_idx[0][error_flag]
        error_occasion = recapture_idx[1][error_flag]

        return error_animal, error_occasion

    def create_ghost_history(self, ghost_indices):
        """Create capture histories for false rejects.
        
        If there are mark changes, recaptures will be allocated to the ghost 
        history. Otherwise, there will only be one capture for each ghost 
        history. 
        """
        _, ghost_occasion = ghost_indices
        
        # create ghost histories
        total_false_rejects = len(ghost_occasion)
        ghost_history = np.zeros((total_false_rejects, self.T), dtype=int)

        # 'indices' ensures we select each ghost_occasion in turn 
        indices = np.arange(total_false_rejects)
        ghost_history[indices, ghost_occasion] = 1
            
        return ghost_history

    def copy_recaptures_to_changed_animal(
            self, 
            mark_change_indices, 
            mark_change_history, 
            recapture_history
        ):
        """Copies recaptures from true capture history to ghosts."""
        mark_change_animal, mark_change_occasion = mark_change_indices

        # filling in the ghost history after each change
        for i in range(len(mark_change_animal)):

            # capture history after the mark change
            next_occasion = (mark_change_occasion[i] + 1)
            post_change = recapture_history[mark_change_animal[i], 
                                            next_occasion:]

            # add this history to the ghost history
            mark_change_history[i, next_occasion:] = post_change

        return mark_change_history

    def pick_wrong_animals(self, false_accept_indices, capture_history):
        
        false_accept_animal, false_accept_occasion = false_accept_indices

        animal_count = capture_history.shape[0]
        similarity = self.simulate_similarity(animal_count)

        def pick_wrong_animal(a, t):
            """Selects animal to be confused with"""

            # animals with higher similarity are more likely to be picked 
            pi = similarity[a].copy()

            # animals not captured cant be mistaken
            not_yet_captured = capture_history[:,:t].max(axis=1) == 0
            pi[not_yet_captured] = 0
            pi = softmax(pi)
            wrong_animal = np.argmax(self.rng.multinomial(1, pi))

            return wrong_animal

        # choose mididentified animals based on similarity 
        wrong_animal = [pick_wrong_animal(a, t) for a, t 
                        in zip(false_accept_animal, false_accept_occasion)]
        wrong_animal = np.asarray(wrong_animal)
        
        return wrong_animal

    def simulate_similarity(self, animal_count):

        # similutate similarity 
        similarity = self.rng.beta(self.A, self.B, (animal_count, animal_count))

        # ensure sim[i,j] == sim[j,i], and sim[i,i] = 0
        similarity = np.triu(similarity, 1)
        i_lower = np.tril_indices(animal_count, -1)
        similarity[i_lower] = similarity.T[i_lower]

        return similarity