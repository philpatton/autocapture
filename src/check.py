# import numpy as np
# import arviz as az

# def posterior_predictive_check(idata: az.InferenceData, data: np.ndarray, 
#                                seed=None) -> dict:
#     '''Conduct a posterior predictive check for CJS or POPAN model.
    
#     The test statistic is the Freeman-Tukey statistic, measuring the discrepancy
#     between the observed (or predicted) and expected counts in the m-array.

#     Currently takes idata object but alternatively could be constructed with 
#     samples of phi and p. 

#     Args:
#         idata: inference data object from PyMC sampling of POPAN or CJS model
#         data: summarized capture history consisting of m-array, where the last
#           column is the number of never recaptured animals 
    
#     '''
#     # M array (add zero row to for compatability below)
#     m_array = data[:,:-1]
#     r = data.sum(axis=1)

#     # utility vectors for creating arrays and array indices
#     interval_count, _ = m_array.shape
#     intervals = np.arange(interval_count)

#     # generate indices for the m_array  
#     row_indices = intervals[..., None]
#     col_indices = intervals[None, ...]

#     # matrix indicating the number of intervals between sampling occassions
#     intervals_between = np.clip(col_indices - row_indices, 0, np.inf)

#     # extract numpy arrays from the idata object
#     stacked = az.extract(idata)
#     p_samples = stacked.p.values
#     phi_samples = stacked.phi.values

#     # rng for drawing from the posterior predictive distribution
#     rng = np.random.default_rng(seed=seed)

#     # freeman-tukey statistics for each draw 
#     freeman_tukey_observed = []
#     freeman_tukey_new = []

#     # loop over samples
#     # TODO: vectorize 
#     for i in range(len(p_samples)):

#         p = np.repeat(p_samples[i], interval_count)
#         phi = phi_samples[:, i]

#         # p_alive: probability of surviving between i and j in the m-array 
#         phi_mat = np.ones_like(m_array) * phi
#         p_alive = np.triu(
#             np.cumprod(fill_lower_diag_ones(phi_mat), axis=1)
#         )

#         # p_not_cap: probability of not being captured between i and j
#         p_not_cap = np.triu((1 - p) ** intervals_between)

#         # nu: probabilities associated with each cell in the m-array
#         nu = p_alive * p_not_cap * p

#         # probability for the animals that were never recaptured
#         chi = 1 - nu.sum(axis=1)

#         # combine the probabilities into a matrix
#         chi = np.reshape(chi, (interval_count, 1))
#         full_array_probs = np.hstack((nu, chi))
        
#         # expected values for the m_array 
#         expected_counts = (full_array_probs.T * r).T

#         # freeman-tukey statistic for the observed data
#         D_obs = freeman_tukey(data, expected_counts)
#         freeman_tukey_observed.append(D_obs)
        
#         m_new = rng.multinomial(r, full_array_probs)
#         D_new = freeman_tukey(m_new, expected_counts)  
#         freeman_tukey_new.append(D_new)

#     freeman_tukey_observed = np.array(freeman_tukey_observed)
#     freeman_tukey_new = np.array(freeman_tukey_new)

#     return {'freeman_tukey_observed': freeman_tukey_observed, 
#             'freeman_tukey_new': freeman_tukey_new}

# def freeman_tukey(observed, expected) -> float:
#     '''Calculate the Freeman '''
#     D = np.power(np.sqrt(observed) - np.sqrt(expected), 2).sum()
#     return D

# def expit(x):
#     '''Inverse logit of an array like'''
#     return 1 / (1 + np.exp(-x))

# def fill_lower_diag_ones(x: np.ndarray) -> np.ndarray:
#     '''Utility function to set the lower diag to one'''
#     return np.triu(x) + np.tril(np.ones_like(x), k=-1)

# def bayesian_p_value(replicate, observed) -> float:
#     return (replicate >= observed).mean()