import pandas as pd
import numpy as np
import yaml

def scale_p(mean, p_max, p_min):

    scaled = ((p_max - p_min) * (mean - min(mean)) / 
            (max(mean) - min(mean))) + p_min
    
    return scaled

def main():

    SCENARIO = 'fully'

    # hyperparameters for configs 
    MARK_CHANGE_RATE = 0.2

    P_MAX = 0.66
    P_MIN = 0.33
    B0 = 0.35
    T = 10
    PHI = 0.9

    INPUT_DIR = 'input'

    # directories
    if SCENARIO == 'fully':
        path = f'{INPUT_DIR}/fully-automated-rates.csv'
    elif SCENARIO == 'semi5':
        path = f'{INPUT_DIR}/semi5-rates.csv'
    elif SCENARIO == 'semi10':
        path = f'{INPUT_DIR}/semi10-rates.csv'

    error_rates = pd.read_csv(path)

    # identifier for each catalog
    catalog_id = error_rates['catalog_id'].to_list()

    # error rates from each catalog
    false_reject_rate = error_rates['FN'].to_numpy() 
    alpha = (false_reject_rate * (1 - MARK_CHANGE_RATE)).tolist()
    beta = (false_reject_rate * MARK_CHANGE_RATE).tolist()
    gamma = error_rates['FP'].to_list()

    # superpopulation size 
    N = error_rates['id_count'].to_list()

    # translate number of training images per id to scale (P_MIN, P_MAX)
    train_img_per_id = error_rates['train_img_per_id'].to_numpy()
    p = scale_p(train_img_per_id, P_MAX, P_MIN).tolist()

    # entrance probabilities 
    b = np.zeros(T)
    b[0] = B0
    b[1:] = (1 - b[0]) / (len(b) - 1) # ensure sums to one 
    b = b.tolist()

    catalog_count = len(catalog_id)
    for i in range(catalog_count):
        
        yaml_path = f'config/{SCENARIO}-{catalog_id[i]}.yaml'

        scenario_dict = {
            'N': N[i],
            'T': T,
            'phi': PHI,
            'b': b,
            'p': p[i],
            'alpha': alpha[i],
            'beta': beta[i],
            'gamma': gamma[i]
        }

        with open(yaml_path, 'w') as outfile:
            yaml.dump(scenario_dict, outfile, default_flow_style=False)

    return None    

if __name__ == '__main__':
    main()