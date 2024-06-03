#%% 
# 

import yaml 
import numpy as np 


paths = '/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submissions.yaml'

# %%

path_dict = yaml.safe_load(open(paths, 'r'))
# %%

paths_to_copy = []
for track in path_dict['submissions']:
    for key in track.keys():
        if key != 'track': 
            paths_to_copy.append(track[key]['model'])
            paths_to_copy.append(track[key]['decoder'])

#%% 
unique_paths_to_copy = np.unique(paths_to_copy)

# %%
with open('paths.txt', 'w') as f:
    for path in unique_paths_to_copy:
        f.write(path + '\n')
# %%
