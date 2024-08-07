
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
import arviz as az
from pytensor import tensor as pt

from pymc.distributions.dist_math import factln
from src.utils import fill_lower_diag_ones, freeman_tukey, create_full_array

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

            # only estimate first beta, others constant
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
                pt.cumprod(self.fill_lower_diag_ones(phi_mat), axis=1)
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

    def check(self, idata: az.InferenceData, capture_history: np.ndarray, 
                seed=None) -> dict:
        '''Conduct a posterior predictive check for CJS or POPAN model.
        
        The test statistic is the Freeman-Tukey statistic, measuring the 
        discrepancy between the observed (or predicted) and expected counts in 
        the m-array.

        Currently takes idata object but alternatively could be constructed with 
        samples of phi and p. 

        Args:
            idata: inference data object from PyMC sampling 
            data: summarized capture history consisting of m-array, where the 
              last column is the number of never recaptured animals 
        
        '''
        # M array (add zero row to for compatability below)
        full_array = create_full_array(capture_history)
        m_array = full_array[:,:-1]
        r = capture_history.sum(axis=0)[:-1]

        # utility vectors for creating arrays and array indices
        interval_count, _ = m_array.shape
        intervals = np.arange(interval_count)

        # generate indices for the m_array  
        row_indices = intervals[..., None]
        col_indices = intervals[None, ...]

        # matrix indicating the number of intervals between sampling occassions
        intervals_between = np.clip(col_indices - row_indices, 0, np.inf)

        # extract numpy arrays from the idata object
        stacked = az.extract(idata)
        p_samples = stacked.p.values
        phi_samples = stacked.phi.values

        # rng for drawing from the posterior predictive distribution
        rng = np.random.default_rng(seed=seed)

        # freeman-tukey statistics for each draw 
        freeman_tukey_observed = []
        freeman_tukey_new = []

        # loop over samples
        # TODO: vectorize 
        for i in range(len(p_samples)):

            p = np.repeat(p_samples[i], interval_count)
            phi = np.repeat(phi_samples[i], interval_count)

            # p_alive: probability of surviving between i and j in the m-array 
            phi_mat = np.ones_like(m_array) * phi
            p_alive = np.triu(
                np.cumprod(fill_lower_diag_ones(phi_mat), axis=1)
            )

            # p_not_cap: probability of not being captured between i and j
            p_not_cap = np.triu((1 - p) ** intervals_between)

            # nu: probabilities associated with each cell in the m-array
            nu = p_alive * p_not_cap * p

            # probability for the animals that were never recaptured
            chi = 1 - nu.sum(axis=1)

            # combine the probabilities into a matrix
            chi = np.reshape(chi, (interval_count, 1))
            full_array_probs = np.hstack((nu, chi))
            
            # expected values for the m_array 
            expected_counts = (full_array_probs.T * r).T

            # freeman-tukey statistic for the observed data
            D_obs = freeman_tukey(full_array, expected_counts)
            freeman_tukey_observed.append(D_obs)
            
            m_new = rng.multinomial(r, full_array_probs)
            D_new = freeman_tukey(m_new, expected_counts)  
            freeman_tukey_new.append(D_new)

        freeman_tukey_observed = np.array(freeman_tukey_observed)
        freeman_tukey_new = np.array(freeman_tukey_new)

        return {'freeman_tukey_observed': freeman_tukey_observed, 
                'freeman_tukey_new': freeman_tukey_new}

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