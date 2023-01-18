# adapted from Kery and Schaub (2011) BPA, Chapter 10
# https://www.vogelwarte.ch/assets/files/publications/BPA/BPA%20with%20JAGS.txt

import numpy as np

# # Import the library
# import argparse
# # Create the parser
# parser = argparse.ArgumentParser()
# # Add an argument
# parser.add_argument('--sampling_occasions', type=int, required=True)
# # Parse the argument
# args = parser.parse_args()
# Print "Hello" + the user input argument
# print('Hello,', args.name)

# Define parameter values
# Number of capture occasions
n_occasions = 7 

# Superpopulation size
N = 400                                

# Survival probabilities
phi = 0.7

# Entry probabilities 
b = np.zeros(n_occasions)
b[0] = 0.34
b[1:] = 0.11

# Capture probabilities
p = 0.5        

phi_shape = (N, n_occasions - 1)
PHI = np.full(phi_shape, 0.7)

p_shape = (N, n_occasions)
P = np.full(p_shape, p)

# Function to simulate capture-recapture data under the JS model
def simul_js(PHI, P, b, N):
    
    rng = np.random.default_rng()

    # Generate number of entering ind. per occasion
    B = rng.multinomial(n=N, pvals=b)
    
    n_occasions = PHI.shape[1] + 1
    CH_sur = np.zeros((N, n_occasions))
    CH_p = np.zeros((N, n_occasions))
    
    # Define a vector with the occasion of entering the population
    occasion_index = np.arange(n_occasions)
    ent_occ = np.repeat(occasion_index, B)
    
    # Simulating survival
    for animal in range(N):
        
        # Write 1 when animal enters the population
        entrance_step = ent_occ[animal]
        CH_sur[animal, entrance_step] = 1 

        # don't need to simulate survival   
        if (entrance_step + 1) == n_occasions: 
            continue
        
        # first surival step on the occasion after entry
        start_point = ent_occ[animal] + 1
        end_point = n_occasions
        
        for t in range(start_point, end_point):
            
            # Bernoulli trial: has individual survived occasion?
            sur = rng.binomial(n=1, p=PHI[animal, t-1], size=1)

            if sur == 1: 
                CH_sur[animal, t] = 1 
            else: 
                break
                
    # Simulating capture
    for animal in range(N):
        CH_p[animal] = rng.binomial(n=1, p=P[animal], size=n_occasions)
    #i
    
    # Full capture-recapture matrix
    CH = CH_sur * CH_p

    # Remove individuals never captured
    detected = CH.sum(axis=1)
    CH = CH[detected > 0]
    # Actual population size
    Nt = CH_sur.sum(axis=0)    

    out_dict = {'CH':CH, 'B':B, 'N':Nt}
    
    return out_dict