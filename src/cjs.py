
import numpy as np
import scipy as sp
import pymc as pm
from pytensor import tensor as pt

from src.utils import summarize_individual_history

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

def fill_lower_diag_ones(x):
    return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)