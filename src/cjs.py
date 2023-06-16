
import numpy as np
import pymc as pm
import pandas as pd

from pytensor import tensor as pt
from scipy.optimize import minimize

from src.utils import summarize_individual_history, expit

class Simulator:
    """Simulator for a CJS model.
    
    Code was adapted from Kery and Schaub (2011) Ch. 7.

    Attributes:
        T: integer indicating the number of marking occasions
        marked: integer for the count of released animals on each occasion
        phi: float indicating the probability of apparent survival
        p: float indicating the probability of recapture
        seed: integer for the rng
        rng: np.random.Generator
    """
    def __init__(self, T: int, marked: int, phi, p: float, seed: int) -> None:
        self.T = T
        self.marked = np.repeat(marked, T - 1)
        self.phi = np.repeat(phi, T - 1)
        self.p = np.repeat(p, T - 1)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate(self) -> dict:
        """Simulates a capture history from a CJS model."""

        # capture_history is a binary matrix indicating capture
        total_marked = self.marked.sum()
        capture_history = np.zeros((total_marked, self.T))

        # vector indicating the occasion of capture for each animal 
        mark_occasion = np.repeat(np.arange(self.T - 1), self.marked)

        # initialize and fill the capture history 
        for animal in range(total_marked):
            capture_history[animal, mark_occasion[animal]] = 1 

            # animals, of course, survive and are captured on first occasion
            if mark_occasion[animal] == self.T: 
                continue

            # capture and survival process for previously marked animals
            for t in range(mark_occasion[animal] + 1, self.T):
            
                death_probability = 1 - self.phi[t - 1]
                died = self.rng.binomial(1, death_probability, 1)

                # dead animals are no longer recaptured
                if died: 
                    break 

                # Bernoulli trial: is individual recaptured?
                recaptured = self.rng.binomial(1, self.p[t - 1], 1)
                if recaptured: 
                    capture_history[animal, t] = 1

        return {'capture_history': capture_history}

class BayesEstimator:
    """Bayesian formulation of the CJS model.
    
    For speed, this is formulated in aggregated counts, rather than the state
    space version (Royle and Dorazio, 2008).
    
    Following McCrea and Morgan (2014), the likelihood is formulated in terms 
    of nu and chi.  Chi is a recursively defined probability for individuals 
    that are never recaptured after a certain timestep. Nu defines the 
    probability for individuals that are recaptured later on.

    Attributes:
        capture_history: A np.ndarray where 1 indicates capture.
    """

    def __init__(self, capture_history: np.ndarray) -> None:
        self.capture_history = capture_history
        """Initializes the estimator.
        
        Args:
          capture_history: 1 indicates individual i was captured on occassion t
        
        """

    def compile(self) -> pm.Model:
        """Creates the pymc model object that can be sampled from."""
        
        capture_summary = summarize_individual_history(self.capture_history)

        # number released at each occasion (last count is irrelevant for CJS)
        R = capture_summary['number_released']
        R = R[:-1]

        # M array (add zero row to for compatability below)
        M = capture_summary['m_array']
        M = np.insert(M, 0, 0, axis=1)

        # number of animals that were never recaptured
        never_recaptured_counts = R - M.sum(axis=1)

        # sequences defining the intervals and occassions
        interval_count, occasion_count = M.shape
        intervals = np.arange(interval_count)
        occasions = np.arange(occasion_count)

        with pm.Model() as cjs:

            # priors for catchability and survival 
            p = pm.Uniform('p', 0., 1.)
            phi = pm.Uniform('phi', 0., 1., shape=interval_count)

            # fill irrelevant portion of m-array with ones
            def fill_lower_diag_ones(x):
                return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)

            # p_alive: probability of surviving between i and j in the m-array 
            phi_mat = pt.ones_like(M[:, 1:]) * phi
            p_alive = pt.triu(
                pt.cumprod(fill_lower_diag_ones(phi_mat), axis=1)
            )
   
            # create upper triangle matrix indicating time steps between cells 
            i = np.reshape(intervals, (interval_count, 1))
            j = np.reshape(occasions, (1, occasion_count))
            not_cap_visits = np.clip(j - i - 1, 0, np.inf)[:, 1:]

            # p_not_cap: probability of not being captured between i and j
            p_not_cap = pt.triu((1 - p) ** not_cap_visits)

            # probability of being recaptured at a later time step 
            nu = p_alive * p_not_cap * p

            # vectorize the recapture counts and probabilities 
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

        return cjs

class MLE:
    """Maximum likelihood esimator for a CJS model.
    
    Translated from the  online supplement for McCrea and Morgan (2014), Ch. 4
    https://www.capturerecapture.co.uk/capturerecapture.txt

    Attributes:
        data: np.array representing the m-array
        T: int for the number of occassions
        ni: int for number of release occasions
        nj: int for the number of recovery occasions
    
    """
    def __init__(self, data: np.array) -> None:
        self.data = data

    def loglik(self, theta):

        # extract the data 
        m_array = self.data[:,:-1]
        never_recaptured = self.data[:,-1]

        # utility vectors for creating arrays and array indices
        interval_count, _ = m_array.shape
        intervals = np.arange(interval_count)
        
        # generate indices for the m_array  
        row_indices = np.reshape(intervals, (interval_count, 1))
        col_indices = np.reshape(intervals, (1, interval_count))
        
        # matrix indicating the number of intervals between sampling occassions
        intervals_between = np.clip(col_indices - row_indices, 0, np.inf)
        
        # initialize the survival and catchability vectors
        phi = np.zeros(interval_count)
        p = np.zeros(interval_count)

        # tranform theta to real valued parameters 
        p[:] = expit(theta[0])
        phi[:] = expit(theta[1])

        # fill irrelevant portion of m-array with ones
        def fill_lower_diag_ones(x):
            return np.triu(x) + np.tril(np.ones_like(x), k=-1)

        # p_alive: probability of surviving between i and j in the m-array 
        phi_mat = np.ones_like(m_array) * phi
        p_alive = np.triu(
            np.cumprod(fill_lower_diag_ones(phi_mat), axis=1)
        )

        # probability the animal hasn't been captured until now 
        p_not_cap = np.triu((1 - p) ** intervals_between)

        # probabilities for each cell in the m-array 
        nu = p_alive * p_not_cap * p

        # convert the m_array to a vector
        upper_triangle_indices = np.triu_indices_from(m_array)
        m_vector = m_array[upper_triangle_indices]    
        
        # associated probabilities
        m_vector_probs = nu[upper_triangle_indices]

        # probability for the never recaptured animals 
        chi = 1 - nu.sum(axis=1)
        
        # calculate the multinomial likelihood of nu, chi given the data
        likhood = (m_vector * np.log(m_vector_probs)).sum()
        likhood += (never_recaptured * np.log(chi)).sum()
        
        return -likhood 
    
    def estimate(self):
        '''Estimates the likelihood.'''
        
        # theta_start = np.repeat(0.5, dipper.nj + 1)
        theta_start = np.repeat(0.5, 2)

        res = minimize(self.loglik, theta_start, method='BFGS')
        se = np.sqrt(np.diag(res.hess_inv))

        # put results in a dataframe
        results = pd.DataFrame({'est_logit':res['x'],'se':se})

        return results