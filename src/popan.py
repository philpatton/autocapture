
"""Simulating data and estimating parameters for a Jolly-Seber model.

The simulation code was adapted from Kery and Schaub (2011) BPA, Chapter 10. The
estimation code was adapted from Austin Rochford:
https://austinrochford.com/posts/2018-01-31-capture-recapture.html. Both classes  
represent POPAN, aka the JSSA (Jolly-Seber-Schwarz-Arnason), parameterization.

Typical usage example:

    N = 100; T = 4; phi = 0.9; p = 0.5
    b = np.array([0.4, 0.2, 0.2, 0.2])

    ps = POPANSimulator(N=N, T=T, phi=phi, p=p, b=b)
    results = ps.simulate_data()

    pe = POPANEstimator(results['capture_history'])
    popan = pe.compile()
    with popan:
        idata = pm.sample()
"""

import numpy as np
import scipy as sp
import pymc as pm
from pytensor import tensor as pt

from pymc.distributions.dist_math import factln
from src.utils import summarize_individual_history, create_m_array, create_full_array

class POPAN:

    def __init__(self, seed: int = None) -> None:
        self.rng = np.random.default_rng(seed)

    def compile_pymc_model(self, capture_history: np.ndarray) -> pm.Model:

        # summarize the data 
        full_array = create_full_array(capture_history)
        m_array = full_array[:,:-1]
        r = capture_history.sum(axis=0)

        # utility vectors for creating arrays and array indices
        interval_count, occasion_count = full_array.shape
        intervals = np.arange(interval_count)
        
        # generate indices for the m_array  
        row_indices = np.reshape(intervals, (interval_count, 1))
        col_indices = np.reshape(intervals, (1, interval_count))
        
        # matrix indicating the number of intervals between sampling occassions
        intervals_between = np.clip(col_indices - row_indices, 0, np.inf)

        # # number of unmarked animals captured at each occasion
        dummy_zeros = np.zeros((m_array.shape[0], 1))
        mm = np.hstack((dummy_zeros, m_array))

        u = r[1:] - mm[:, 1:].sum(axis=0)
        u = np.insert(u, 0, r[0])

        # index for generating sequences like [[0], [0,1], [0,1,2]]
        alive_yet_unmarked_index = sp.linalg.circulant(
            np.arange(occasion_count)
        )
         
        with pm.Model() as popan:
            # priors for detection, survival, and pent
            p = pm.Uniform('p', 0., 1.)
            phi = pm.Uniform('phi', 0., 1.)
            # beta = pm.Dirichlet(
            #     'beta', 
            #     np.ones(occasion_count), 
            #     shape=(occasion_count)
            # )

            b0 = pm.Uniform('b0', 0., 1.)
            b_other = (1 - b0) / (interval_count)
            beta = pt.concatenate(
                ([b0], pt.repeat(b_other, interval_count))
            )
            # improper flat prior for N
            flat_dist = pm.Flat.dist()
            N = pm.Truncated("N", flat_dist, lower=u.sum())

            # # add [1] to ensure the addition of the raw beta_0
            PHI = np.repeat(phi, interval_count)
            p_alive_yet_unmarked = pt.concatenate(
                ([1], pt.cumprod((1 - p) * PHI))
            )

            # tril produces the [[0], [0,1], [0,1,2]] patterns for the recursion
            psi = pt.tril(
                beta * p_alive_yet_unmarked[alive_yet_unmarked_index]
            ).sum(axis=1)
        
            # distribution for the unmarked animals, L'''1 in schwarz arnason
            unmarked = pm.CustomDist(
                'unmarked', 
                N, 
                psi * p, 
                logp=self.unmarked_dist_logp, 
                observed=u
            )

            # p_alive: probability of surviving between i and j in the m-array 
            phi_mat = pt.ones_like(m_array) * phi
            p_alive = pt.triu(
                pt.cumprod(fill_lower_diag_ones(phi_mat), axis=1)
            )

            # p_not_cap: probability of not being captured between i and j
            p_not_cap = pt.triu((1 - p) ** intervals_between)

            # nu: probabilities associated with each cell in the m-array
            nu = p_alive * p_not_cap * p

            # probability for the animals that were never recaptured
            chi = 1 - nu.sum(axis=1)

            # combine the probabilities into a matrix
            chi = pt.reshape(chi, (interval_count, 1))
            marr_probs = pt.horizontal_stack(nu, chi)

            # distribution of the m-array 
            marr = pm.Multinomial(
                'marr',
                n=r[:-1], # last count irrelevant for CJS
                p=marr_probs,
                observed=full_array
            )

        return popan
    
    # logp of the dist for unmarked animals {u1, ...} ~ Mult(N; psi1 * p, ...)
    def unmarked_dist_logp(self, x, n, p):
        
        x_last = n - x.sum()
        
        # calculate thwe logp for the observations
        res = factln(n) + pt.sum(x * pt.log(p) - factln(x)) \
                + x_last * pt.log(1 - p.sum()) - factln(x_last)
        
        # ensure that the good conditions are met.
        good_conditions = pt.all(x >= 0) & pt.all(x <= n) & (pt.sum(x) <= n) & \
                            (n >= 0)
        res = pm.math.switch(good_conditions, res, -np.inf)

        return res

    def fill_lower_diag_ones(self, x):
        return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)
    
    def simulate(self, N: int, T: int, phi: float, p: float, 
                 b: np.ndarray) -> dict:
        
        # each animal has some entry (birth, imm.) time; 0 if already entered
        entry_occasions = self.simulate_entry(b, N)
        _, B = np.unique(entry_occasions, return_counts=True)

        PHI = np.full((N, T - 1), phi)
        P = np.full((N, T), p)        

        # Z in (0,1) of shape (N_super, T) indicating alive and entered
        Z = self.simulate_z(PHI=PHI, N=N, entry_occasions=entry_occasions)
        N_true = Z.sum(axis=0)
        
        # matrix of coin flips (p=p) of size Z indicating a potential capture 
        captures = self.simulate_capture(P=P, N=N)

        # captured AND available for capture, i.e., has entered and is alive
        capture_history = captures * Z

        # filter all zero histories
        was_seen = (capture_history != 0).any(axis=1)
        capture_history = capture_history[was_seen]
        
        out_dict = {'capture_history':capture_history, 'B':B, 'N':N_true}
        
        return out_dict

    def simulate_entry(self, b, N):
        """Simulate occasion for animal's entry into population."""

        # matrix where one indicates entry 
        entry_matrix = self.rng.multinomial(n=1, pvals=b, size=N)

        # index of the first nonzero value (entry)
        entry_occasions = entry_matrix.nonzero()[1]

        return entry_occasions

    def simulate_z(self, PHI: np.ndarray, N: int, 
                   entry_occasions: np.ndarray) -> np.ndarray:
        """Simulate discrete latent state, alive and entered, for jolly-seber

        Args: 
            entry_occasions: A 1D array with length N indicating the time of 
            entry. 

        Returns:
            N by T matrix indicating that the animal is alive and entered
        """
        
        # simulate survival between occasions
        life_matrix = [
            self.rng.binomial(n=1, p=PHI[i]) 
            for i in range(N)
        ]
        life_matrix = np.stack(life_matrix, axis=0)

        # add column such that survival between t and t+1 implies alive at t+1 
        life_matrix = np.insert(life_matrix, 0, np.ones(N), axis=1)

        # matrix where 1 will indicate that animal has entered
        entry_matrix = np.zeros(life_matrix.shape).astype(int)

        for i in range(N):

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

    def simulate_capture(self, P: np.ndarray, N: np.ndarray) -> np.ndarray:
        """Generate a binomial matrix indicating capture."""
        capture = [
            self.rng.binomial(n=1, p=P[i]) 
            for i in range(N)
        ]
        capture = np.stack(capture, axis=0)

        return capture

class SimulatorPOPAN:
    """Data simulator for Jolly-Seber models.
    
    Attributes:
        N: An integer count of the superpopulation
        PHI: N by T-1 matrix of survival probabilities between occassions
        P: N by T matrix of capture probabilities
        b: T by 1 vector of entrance probabilities 
        rng: np.random.Generator used by the model 
        seed: integer seed for the rng
    """

    def __init__(self, N: int, T: int, phi: float, p: float, b: np.ndarray, 
                 seed: int = None):
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

    def simulate_z(self, entry_occasions: np.ndarray) -> np.ndarray:
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

class BayesPOPAN:
    """PyMC model for esimating parameters in a Jolly-Seber model
    
    Attributes:
        capture_history: detected animals count by occasion count np.ndarray 
          indicating capture (1) or otherwise (0)
    """
    def __init__(self, capture_history: np.ndarray) -> None:
        self.capture_history = capture_history

    def compile(self) -> pm.Model:

        capture_summary = summarize_individual_history(self.capture_history)

        # number released at each occasion (assuming no losses on capture)
        R = capture_summary['number_released']

        # number of intervals (NOT occasions)
        occasion_count = len(R)

        # index for generating sequences like [[0], [0,1], [0,1,2]]
        alive_yet_unmarked_index = sp.linalg.circulant(
            np.arange(occasion_count)
        )

        # M array 
        M = capture_summary['m_array']
        M = np.insert(M, 0, 0, axis=1)

        # number of unmarked animals captured at each occasion
        u = np.concatenate(([R[0]], R[1:] - M[:, 1:].sum(axis=0)))
        
        # ditch the last R for the CJS portion
        R = R[:-1]
        interval_count = len(R)
        
        # number of animals that were never recaptured
        never_recaptured_counts = R - M.sum(axis=1)
        
        # convenience vectors for recapture model
        i = np.arange(interval_count)[:, np.newaxis]
        j = np.arange(occasion_count)[np.newaxis]
        not_cap_visits = np.clip(j - i - 1, 0, np.inf)[:, 1:]

        with pm.Model() as popan:
            # priors for detection, survival, and pent
            p = pm.Uniform('p', 0., 1.)
            phi = pm.Uniform('phi', 0., 1.)
            beta = pm.Dirichlet(
                'beta', 
                np.ones(occasion_count), 
                shape=(occasion_count)
            )

            # improper flat prior for N
            flat_dist = pm.Flat.dist()
            N = pm.Truncated("N", flat_dist, lower=u.sum())

            # add [1] to ensure the addition of the raw beta_0
            PHI = np.repeat(phi, interval_count)
            p_alive_yet_unmarked = pt.concatenate(
                ([1], pt.cumprod((1 - p) * PHI))
            )

            # tril produces the [[0], [0,1], [0,1,2]] patterns for the recursion
            psi = pt.tril(
                beta * p_alive_yet_unmarked[alive_yet_unmarked_index]
            ).sum(axis=1)
        
            # distribution for the unmarked animals, L'''1 in schwarz arnason
            unmarked = pm.CustomDist('unmarked', N, psi * p, 
                                     logp=unmarked_dist_logp, observed=u)

            # matrix of survival probabilities
            p_alive = pt.triu(
                pt.cumprod(
                    fill_lower_diag_ones(pt.ones_like(M[:, 1:]) * PHI),
                    axis=1
                )
            )

            # define nu, the probabilities of each cell in the m array 
            p_not_cap = pt.triu((1 - p) ** not_cap_visits)
            nu = p_alive * p_not_cap * p

            # vectorize the counts and probabilities of recaptures 
            upper_triangle_indices = np.triu_indices_from(M[:, 1:])
            recapture_counts = M[:, 1:][upper_triangle_indices]
            recapture_probabilities = nu[upper_triangle_indices]

            # distribution for the recaptures 
            recaptured = pm.Binomial(
                'recaptured', 
                n=recapture_counts, 
                p=recapture_probabilities,
                observed=recapture_counts
            )

            # distribution for the observed animals who were never recaptured
            chi = 1 - nu.sum(axis=1)
            never_recaptured_rv = pm.Binomial(
                'never_recaptured', 
                n=never_recaptured_counts, 
                p=chi, 
                observed=never_recaptured_counts
            )
        
        return popan

# logp of the dist for unmarked animals {u1, ...} ~ Mult(N; psi1 * p, ...)
def unmarked_dist_logp(x, n, p):
    
    x_last = n - x.sum()
    
    # calculate thwe logp for the observations
    res = factln(n) + pt.sum(x * pt.log(p) - factln(x)) \
            + x_last * pt.log(1 - p.sum()) - factln(x_last)
    
    # ensure that the good conditions are met.
    good_conditions = pt.all(x >= 0) & pt.all(x <= n) & (pt.sum(x) <= n) & \
                        (n >= 0)
    res = pm.math.switch(good_conditions, res, -np.inf)

    return res

def fill_lower_diag_ones(x):
    return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)