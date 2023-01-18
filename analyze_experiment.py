import numpy as np 
import json 

experiment_name = 'tmp'
experiment_dir = f'results/{experiment_name}'

trial_path = f'{experiment_dir}/trial_2.json'

with open(trial_path, 'r') as f:
  trial_results = json.loads(json.load(f))

capture_history = np.asarray(trial_results["N"])