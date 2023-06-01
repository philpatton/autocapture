
import numpy as np
import scipy as sp
import pymc as pm
from pytensor import tensor as pt

from src.utils import summarize_individual_history

class CJSEstimator:

    def __init__(self, capture_history: np.ndarray) -> None:
        self.capture_history = capture_history

    def compile(self) -> pm.Model:

        capture_summary = summarize_individual_history(self.capture_history)

        # number released at each occasion (assuming no losses on capture)
        R = capture_summary['number_released']

        # number of intervals (NOT occasions)
        occasion_count = len(R)

        # M array 
        M = capture_summary['m_array']
        M = np.insert(M, 0, 0, axis=1)

        # ditch the last R for the CJS portion
        R = R[:-1]
        interval_count = len(R)

        # number of animals that were never recaptured
        never_recaptured_counts = R - M.sum(axis=1)

        # convenience vectors for recapture model
        i = np.arange(interval_count)[:, np.newaxis]
        j = np.arange(occasion_count)[np.newaxis]
        not_cap_visits = np.clip(j - i - 1, 0, np.inf)[:, 1:]

        with pm.Model() as cjs:
            p = pm.Uniform('p', 0., 1.)
            phi = pm.Uniform('phi', 0., 1., shape=interval_count)

            p_alive = pt.triu(
                pt.cumprod(
                    fill_lower_diag_ones(pt.ones_like(M[:, 1:]) * phi),
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

        return cjs

def fill_lower_diag_ones(x):
    return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)