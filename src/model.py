"""Module for summarizing the output of the simlulation.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

from pymc.distributions.dist_math import factln
from scipy.optimize import minimize

import numpy as np
import pandas as pd
import scipy as sp
import pymc as pm
import arviz as az
from pytensor import tensor as pt

from src.utils import fill_lower_diag_ones, freeman_tukey, create_full_array
from src.utils import expit

class POPAN:
    """Simulating data and estimating parameters for a Jolly-Seber model.

    The simulation code was adapted from Kery and Schaub (2011) BPA, Chapter 10. 
    The estimation code was adapted from Austin Rochford:
    https://austinrochford.com/posts/2018-01-31-capture-recapture.html. 
    Both classes  represent POPAN, aka the JSSA (Jolly-Seber-Schwarz-Arnason), 
    parameterization.

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

    def __init__(self, seed: int = None) -> None:
        self.rng = np.random.default_rng(seed)

    def compile_pymc_model(self, capture_history: np.ndarray) -> pm.Model:
        '''Generate a the POPAN model in PyMC.'''

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
            pm.CustomDist('unmarked', N, psi * p, logp=self.unmarked_dist_logp, 
                          observed=u)

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
            pm.Multinomial(
                'marr',
                n=r[:-1], # last count irrelevant for CJS
                p=marr_probs,
                observed=full_array
            )

        return popan
    
    def unmarked_dist_logp(self, x, n, p):
        '''logp for unmarked animals, {u1, ...} ~ Mult(N; psi1 * p, ...).'''
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
        '''Fill the lower diag of a matrix with ones.'''
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
        for _, i in enumerate(p_samples):

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
        '''Simulate data under a POPAN model.'''
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

class CJS:
    '''Cormack Jolly Seber model.'''
    def __init__(self) -> None:
        pass

    def estimate_mle(self, capture_history: np.ndarray) -> pd.DataFrame:
        """Esimate the MLE for a CJS model.

        Args:
            data: np.array representing the m-array whose final column contains
              the number of animals never recaptured after occasion i     
        Returns:
            pd.DataFrame containing the logit and standard errors for the
              estimates
        """
        # theta_start = np.repeat(0.5, dipper.nj + 1)
        theta_start = np.repeat(0.5, 2)

        full_array = create_full_array(capture_history)

        res = minimize(self.loglik, theta_start, method='BFGS', args=full_array)
        se = np.sqrt(np.diag(res.hess_inv))

        # put results in a dataframe
        results = pd.DataFrame({'est_logit':res['x'],'se':se})

        return results

    def loglik(self, theta: np.ndarray, full_array: np.ndarray):
        """Likelihood function for a CJS Model
               
        Translated from the  online supplement for McCrea and Morgan (2014), 
        Ch. 4, https://www.capturerecapture.co.uk/capturerecapture.txt

        Args:
            theta: np.ndarray for the logit scale parameters
            full_array: m-array with the never recaptured counts
        Returns:
            float representing the negative log likelihood of the model.
        """
        m_array = full_array[:,:-1]
        never_recaptured = full_array[:,-1]

        # utility vectors for creating arrays and array indices
        interval_count, _ = full_array.shape
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
        def fill_lower_diag_ones_numpy(x):
            return np.triu(x) + np.tril(np.ones_like(x), k=-1)

        # p_alive: probability of surviving between i and j in the m-array 
        phi_mat = np.ones_like(m_array) * phi
        p_alive = np.triu(
            np.cumprod(fill_lower_diag_ones_numpy(phi_mat), axis=1)
        )

        # probability the animal hasn't been captured until now 
        p_not_cap = np.triu((1 - p) ** intervals_between)

        # probabilities for each cell in the m-array 
        nu = p_alive * p_not_cap * p

        # probability for the never recaptured animals 
        chi = 1 - nu.sum(axis=1)
        
        # combine the probabilities into a matrix
        chi = np.reshape(chi, (interval_count, 1))

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

    def compile_pymc_model(self, capture_history: np.ndarray=None) -> pm.Model:
        """Bayesian formulation of the CJS model.
        
        For speed, this is formulated in aggregated counts, i.e., the m-array 
        version, rather than the state space version.
        
        Following McCrea and Morgan (2014), the likelihood is formulated in 
        terms of nu and chi. Nu represents the probabilities of the cells in the 
        m-array, i.e., the number of animals captured at occasion i, not seen 
        again until occasion j. Chi represents the probabilities for animals 
        captured at occasion i, and never seen again.

        This code is adapted from Austin Rochford. The major differences are 
        updates related to converting from PyMC3 to PyMC, as well as simplifying 
        the likelihood, i.e., joint modeling of chi and nu via a Multinomial.
        https://austinrochford.com/posts/2018-01-31-capture-recapture.html

        Attributes:
            data: A np.ndarray with shape (n_intervals, n_occasions). The last 
                column is the number of animals released on occasion i that were 
                never seen again. The other columns correspond to the m-array.
        """

        # summarize the data 
        full_array = create_full_array(capture_history)
        m_array = full_array[:,:-1]
        r = full_array.sum(axis=1)

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

            # phi ~ t
            phi = pm.Uniform('phi', 0., 1., shape=interval_count)

            # phi ~ 1
            # phi = pm.Uniform('phi', 0., 1.)
            
            # logit(phi) = beta_int + beta_year * year 
            # beta_int = pm.Normal('beta_int', 0., 2.25)
            # beta_year = pm.Normal('beta_year', 0., 2.25)
            # logit_phi = beta_int + beta_year * intervals
            # phi = expit(logit_phi)

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
                n=r,
                p=marr_probs,
                observed=full_array
            )

        return cjs

    # fill irrelevant portion of m-array with ones
    def fill_lower_diag_ones(self, x):
        return pt.triu(x) + pt.tril(pt.ones_like(x), k=-1)


    def check(self, idata: az.InferenceData, capture_history: np.ndarray, 
              model=None, seed=None) -> dict:
        '''Conduct a posterior predictive check.
        
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
        r = full_array.sum(axis=1)

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

        if model:
            with model:
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)
                stacked = az.extract(idata.posterior_predictive)
                m_samples = stacked.marr.values

        # freeman-tukey statistics for each draw 
        freeman_tukey_observed = []
        freeman_tukey_new = []

        # loop over samples
        # TODO: vectorize 
        for _, i in enumerate(p_samples):

            p = np.repeat(p_samples[i], interval_count)
            phi = phi_samples[:, i]
            # phi = np.repeat(phi_samples[i], interval_count)

            def fill_lower_diag_ones(x):
                return np.triu(x) + np.tril(np.ones_like(x), k=-1)

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
            # the two .T allow for proper broadcasting
            expected_counts = (full_array_probs.T * r).T

            # freeman-tukey statistic for the observed data
            D_obs = freeman_tukey(full_array, expected_counts)
            freeman_tukey_observed.append(D_obs)
            
            if not model:
                m_new = rng.multinomial(r, full_array_probs)
            else:
                m_new = m_samples[:,:,i]

            D_new = freeman_tukey(m_new, expected_counts)  
            freeman_tukey_new.append(D_new)

        freeman_tukey_observed = np.array(freeman_tukey_observed)
        freeman_tukey_new = np.array(freeman_tukey_new)

        return {'freeman_tukey_observed': freeman_tukey_observed, 
                'freeman_tukey_new': freeman_tukey_new}

    def simulate(self, released_count: np.ndarray, T: int, 
                 phi: np.ndarray, p: np.ndarray, seed: int = None):
        """Simulator for a CJS model.
        
        Code was adapted from Kery and Schaub (2011) Ch. 7.

        Args:
            T: integer indicating the number of marking occasions
            marked: integer for the count of released animals on each occasion
            phi: float indicating the probability of apparent survival
            p: float indicating the probability of recapture
            seed: integer for the rng
            rng: np.random.Generator
        """        
        rng = np.random.default_rng(seed)

        # convert to vector if scaler is provided
        if isinstance(released_count, int):
            released_count = np.repeat(released_count, T - 1)
        if isinstance(phi, float):
            phi = np.repeat(phi, T - 1)
        if isinstance(p, float): 
            p = np.repeat(p, T - 1)

        # capture_history is a binary matrix indicating capture
        total_marked = released_count.sum()
        capture_history = np.zeros((total_marked, T), dtype=int)

        # vector indicating the occasion of capture for each animal 
        mark_occasion = np.repeat(np.arange(T - 1), released_count)

        # initialize and fill the capture history 
        for animal in range(total_marked):
            capture_history[animal, mark_occasion[animal]] = 1 

            # survival and capture process irrelevant on first occasion
            if mark_occasion[animal] == T: 
                continue

            # capture and survival process for previously marked animals
            for t in range(mark_occasion[animal] + 1, T):
            
                death_probability = 1 - phi[t - 1]
                died = rng.binomial(1, death_probability, 1)

                # dead animals are no longer recaptured
                if died: 
                    break 

                # alive animals are recaptured with probability p 
                recaptured = rng.binomial(1, p[t - 1], 1)
                if recaptured: 
                    capture_history[animal, t] = 1

        return {'capture_history': capture_history}