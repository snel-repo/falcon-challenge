#%% 
import numpy as np
import matplotlib.pyplot as plt 
import pickle 

from styleguide import set_style
set_style()

#%% 
path = '/snel/share/runs/falcon/linear_baseline_scores/sweep_results_h1_m1_m2.pkl'
# %%
with open(path, 'rb') as f:
    results = pickle.load(f)
# %%

import colorcet as cc
import matplotlib.cm as cmaps
cm = cmaps.get_cmap('cet_kbc')

fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.5))

for ii, track in enumerate(['m1', 'm2', 'h1']): 
    ax = axs[ii]
    r2 = np.full((31, len(results[track][0])), np.nan)
    for hist in results[track].keys():
        hist_res = results[track][hist]
        values = []
        if track == 'h1': 
            ds = sorted([int(x.strip('S')) for x in list(hist_res.keys())])
            for d in ds: 
                values.append(float(hist_res[f'S{d}']))
        else: 
            ds = sorted([int(x) for x in list(hist_res.keys())])
            for d in ds: 
                values.append(float(hist_res[f'{d}']))
        r2[hist, :]= values
    
    for jj in range(r2.shape[1]): 
        ax.plot(np.arange(0, 31) * 20, r2[:, jj], '-', label=ds[jj], color=cm(jj/float(r2.shape[1])), linewidth=0.75, alpha=0.8)
    ax.set_title(track.upper())
    ax.grid(alpha=0.3)
    # remove tick marks 
    ax.tick_params(axis='both', which='both', length=0)
    # ax.set_ylim([0, 0.8])
    ax.set_xlabel('History (ms)')
    if ii==0:
        ax.set_ylabel('R2')
    if track == 'm1': 
        ax.set_ylim([0.35, 0.6])
        ax.vlines(30*20, 0.35, 0.6, color='r', linestyle='--')
    elif track == 'm2':
        ax.set_ylim([0.15, 0.35])
        ax.set_yticks([0.15, 0.2, 0.25, 0.3, 0.35])
        ax.vlines(7*20, 0.15, 0.35, color='r', linestyle='--')
    elif track == 'h1':
        ax.set_ylim([0.05, 0.3])
        ax.vlines(30*20, 0.05, 0.3, color='r', linestyle='--')
    
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=np.min(ds), vmax=np.max(ds)))
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.25)
    cbar.set_ticks(ds)
    ticklabels = np.full(len(ds), None)
    ticklabels[0] = str(np.min(ds))
    ticklabels[-1] = str(np.max(ds))
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(length=0)

plt.savefig('/snel/home/bkarpo2/projects/falcon_figs/wf_history.pdf')

# %%
