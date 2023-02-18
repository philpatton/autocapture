
"""Simulate data for a Jolly-Seber model. 

This code was adapted from Kery and Schaub (2011) BPA, Chapter 10. This is the 
POPAN or JSSA (Jolly-Seber-Schwarz-Arnason) version

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

    def __init__(self, N: int, T: int, phi: np.ndarray, p: np.ndarray,
                 b: np.ndarray, alpha: float = 0, beta: float = 0, 
                 gamma: float = 0, seed: int = None, A: float = 2, B: float = 5):
        """Init the data generator with hyperparameters and init the rng"""
        self.N = N
        self.T = T
        self.PHI = np.full((N, T - 1), phi)
        self.P = np.full((N, T), p)
        self.b = b
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if len(b) != T:
            raise ValueError('b must have length T')
 
    def simulate(self):
        """Simulates the Jolly-Seber model from the hyperparameters.

        This is is the POPAN or JSSA version.
        
        Returns:
            out_dict: dictionary containing the true history, number of 
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
        capture_history = captures * Z

        # filter all zero histories
        was_seen = (capture_history != 0).any(axis=1)
        capture_history = capture_history[was_seen]
        
        out_dict = {'capture_history':capture_history, 'B':B, 'N':N_true}
        
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