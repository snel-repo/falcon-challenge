#%% 

import pickle
import numpy as np

#%% 

path = '/snel/share/runs/falcon/linear_baseline_scores/held_out_calib_only_scores.pkl'

with open(path, 'rb') as f:
    scores = pickle.load(f)
# %%

for key in scores: 
    print(key)
    values = np.array([float(x) for x in scores[key].values()])
    print(f'Mean: {values.mean()}')
    print(f'Std: {values.std()}')
# %%
