
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

            # survival and capture process irrelevant on first occasion
            if mark_occasion[animal] == self.T: 
                continue

            # capture and survival process for previously marked animals
            for t in range(mark_occasion[animal] + 1, self.T):
            
                death_probability = 1 - self.phi[t - 1]
                died = self.rng.binomial(1, death_probability, 1)

                # dead animals are no longer recaptured
                if died: 
                    break 

                # alive animals are recaptured with probability p 
                recaptured = self.rng.binomial(1, self.p[t - 1], 1)
                if recaptured: 
                    capture_history[animal, t] = 1

        return {'capture_history': capture_history}

class BayesEstimator:
    """Bayesian formulation of the CJS model.
    
    For speed, this is formulated in aggregated counts, i.e., the m-array 
    version, rather than the state space version.
    
    Following McCrea and Morgan (2014), the likelihood is formulated in terms 
    of nu and chi. Nu represents the probabilities of the cells in the m-array,
    i.e., the number of animals captured at occasion i, not seen again until 
    occasion j. Chi represents the probabilities for animals captured at 
    occasion i, and never seen again.

    This code is adapted from Austin Rochford. The major differences are 
    updates related to converting from PyMC3 to PyMC, as well as simplifying 
    the likelihood, i.e., joint modeling of chi and nu via a Multinomial.
    https://austinrochford.com/posts/2018-01-31-capture-recapture.html

    Attributes:
        data: A np.ndarray with shape (n_intervals, n_occasions). The last 
          column is the number of animals released on occasion i that were never
          seen again. The other columns correspond to the m-array.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        """Initializes the estimator.
        
        Args:
            data: Dataset 
          
        """

    def compile(self) -> pm.Model:
        """Creates the pymc model object that can be sampled from."""

        # extract the data 
        m_array = self.data[:,:-1]
        r = self.data.sum(axis=1)

        # utility vectors for creating arrays and array indices
        interval_count, _ = m_array.shape
        intervals = np.arange(interval_count)
        
        # generate indices for the m_array  
        row_indices = np.reshape(intervals, (interval_count, 1))
        col_indices = np.reshape(intervals, (1, interval_count))
        
        # matrix indicating the number of intervals between sampling occassions
        intervals_between = np.clip(col_indices - row_indices, 0, np.inf)

        with pm.Model() as cjs:

            # priors for catchability and survival 
            p = pm.Uniform('p', 0., 1.)
            phi = pm.Uniform('phi', 0., 1., shape=interval_count)

            # fill irrelevant portion of m-array with ones
            def fill_lower_diag_ones(x):
                return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)

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
                n=r,
                p=marr_probs,
                observed=self.data
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
        likhood = sum(m_vector * np.log(m_vector_probs))
        likhood += sum(never_recaptured * np.log(chi))
        
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