#%%
import os, json, glob, re
import numpy as np
import matplotlib.pyplot as plt 

#%% 

RUN_PATH = '/snel/share/runs/falcon/240516_M2_singlefile_NoMAD_coinspkrem'
IS_MUTLIK = False
if IS_MUTLIK: 
    dayk_key = 'DAYK_SESS'
else:
    dayk_key = 'DAYK_PATH'

workers = glob.glob(os.path.join(RUN_PATH, 'tuneAlign*'))

#%% 

hp_vals = []
day0s = []
dayks = []
perf_vals = []
perf_keys = ['day0_r2', 'unalign_dayk_r2', 'align_dayk_r2', 'val_nll', 'val_kl']

for wkr in workers: 
    param_path = os.path.join(wkr, 'params.json')
    align_path = os.path.join(wkr, 'align_out.json')
    
    try: 
        with open(param_path, 'r') as f:
            params = json.load(f)

        with open(align_path, 'r') as f:
            align = json.load(f)
    except:
        continue

    n_hps = len(params['CFG_UPDATES'])
    hps = list(params['CFG_UPDATES'].keys())
    hp_vals.append(list(params['CFG_UPDATES'].values()))
    dayks.append(params[dayk_key].split('/')[-1])
    day0s.append(params['DAY0_PATH'].split('/')[-1])
    perf_vals.append([
        np.mean(align[key]) for key in perf_keys
    ])

# %%

cm = plt.cm.RdYlBu

'''make a plot of the hyperparameters and performance values'''
fig, axs = plt.subplots(len(perf_vals[0]), len(hp_vals[0]), figsize=(20, 20), facecolor='w', sharey='row', sharex='col')

for i, _ in enumerate(hp_vals[0]):
    for j, _ in enumerate(perf_vals[0]):
        for k in range(len(hp_vals)):
            color = cm(perf_vals[k][2])
            if 'LR' in hps[i] or 'NLL.WEIGHT' in hps[i] or 'KL' in hps[i]:
                axs[j, i].semilogx(hp_vals[k][i], perf_vals[k][j], '.', markersize=8, color=color)
            else:
                axs[j, i].scatter(hp_vals[k][i], perf_vals[k][j], s=8, color=color)
            axs[j, i].set_ylabel(perf_keys[j])
            axs[j, i].set_xlabel(hps[i])

for i in range(len(hp_vals[0])):
    axs[2, i].set_ylim([0, 1])

plt.suptitle(RUN_PATH)
plt.savefig(os.path.join(RUN_PATH, 'analysis.png'))

# %%
plt.figure(facecolor='w')
for i in range(len(perf_vals)): 
    # get align_r2 and dayk id
    align_r2 = perf_vals[i][2]
    dayk = dayks[i].split('-')[-1].split('_')[0]
    plt.plot(dayk, align_r2, '.', markersize=8, color='b', alpha=0.7)
plt.xlabel('Day K')
plt.ylabel('Align R2')
plt.hlines(perf_vals[0][0], 0, 3, linestyle='--', color='k')
plt.ylim([0, 1])
# square axes 
plt.gca().set_aspect('equal', adjustable='box')

# %%
# get max performance for each pair of days 
import numpy as np 
max_perfs = np.full((len(set(day0s)),len(set(dayks))), np.nan)
d0_ids = np.full((len(set(day0s)),len(set(dayks))), np.nan)
dk_ids = np.full((len(set(day0s)),len(set(dayks))), np.nan)

for i, d0 in enumerate(set(day0s)): 
    for j, dk in enumerate(set(dayks)): 
        perf = [perf_vals[i][2] for i in range(len(perf_vals)) if day0s[i] == d0 and dayks[i] == dk]
        inds = [i for i in range(len(perf_vals)) if day0s[i] == d0 and dayks[i] == dk]
        max_perf = max(perf)
        max_ind = inds[np.argmax(perf)]
        # d0_id = d0.split('-')[-1].split('_')[0]
        # dk_id = dk.split('-')[-1].split('_')[0]

        d0_id = re.search(r'\d{4}-\d{2}-\d{2}', d0).group().replace('-', '')
        dk_id = re.search(r'\d{4}-\d{2}-\d{2}', dk).group().replace('-', '')

        print(f'{d0_id} - {dk_id}: {max_perf}')
        # print(hp_vals[max_ind])
        max_perfs[i][j] = max_perf
        if IS_MUTLIK:
            d0_ids[i][j] = int(d0_id[1:])
            dk_ids[i][j] = int(dk_id[1:])
        else:
            d0_ids[i][j] = int(d0_id)
            dk_ids[i][j] = int(dk_id)
       

# %%

y_sort_inds = np.argsort(d0_ids[:,0].astype('int'))
x_sort_inds = np.argsort(dk_ids[0,:].astype('int'))

import matplotlib.pyplot as plt 
plt.figure(facecolor='w')
plt.imshow(max_perfs[y_sort_inds, :][:, x_sort_inds], cmap='plasma', vmin=0, vmax=1)
plt.gca().set_xticks(range(0, len(set(dayks))))
plt.gca().set_xticklabels(dk_ids[0,:][x_sort_inds])
plt.gca().set_yticks(range(0, len(set(day0s))))
plt.gca().set_yticklabels(d0_ids[:,0][y_sort_inds])
plt.title('M1')
# add value in each square 
for i in range(len(set(day0s))):
    for j in range(len(set(dayks))):
        plt.text(j, i, round(max_perfs[y_sort_inds, :][:, x_sort_inds][i, j], 2), ha='center', va='center', color='w')

# %%
