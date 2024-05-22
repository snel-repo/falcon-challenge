#%%
import os, glob, re, pickle
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime
from styleguide import set_style
set_style()
from pynwb import NWBHDF5IO
from falcon_challenge.evaluator import DATASET_HELDINOUT_MAP

#%% 

# oracle_results = '/snel/share/runs/falcon/linear_baseline_scores/sweep_results_h1_m1_m2.pkl'
results = '/snel/share/runs/falcon/linear_baseline_scores/baseline_scores_all.pkl'
base_path = '/snel/home/bkarpo2/bin/falcon-challenge/data'
tracks = ['m1', 'm2', 'h1'] #, 
# history = [7, 30, 30]
static_decoder = {
    'h1': 'S4',
    'm1': '20120927',
    'm2': '20201020'
}

#%% 

with open(results, 'rb') as f: 
    results = pickle.load(f)

#%% 
fig, ax = plt.subplots(1, 6, figsize=(5.5, 1.5), sharey=True)
axs = [[ax[0], ax[1]], [ax[2], ax[3]], [ax[4], ax[5]]]
for i, track in enumerate(tracks): 

    held_in_files = glob.glob(os.path.join(base_path, track, '*held-in-calib*', '*.nwb'))
    held_out_files = glob.glob(os.path.join(base_path, track, '*held-out-calib*', '*.nwb'))
    if track == 'm1':
        held_in_dates = [re.search(r"\d{8}", d).group() for d in held_in_files]
        held_out_dates = [re.search(r"\d{8}", d).group() for d in held_out_files]
        format_held_in_dates = sorted([datetime.strptime(date, '%Y%m%d') for date in held_in_dates])
        format_held_out_dates = sorted([datetime.strptime(date, '%Y%m%d') for date in held_out_dates])
    elif track == 'h2': 
        held_in_dates = [re.search(r"\d{4}.\d{2}.\d{2}", d).group() for d in held_in_files]
        held_out_dates = [re.search(r"\d{4}.\d{2}.\d{2}", d).group() for d in held_out_files]
        format_held_in_dates = sorted([datetime.strptime(date, '%Y.%m.%d') for date in held_in_dates])
        format_held_out_dates = sorted([datetime.strptime(date, '%Y.%m.%d') for date in held_out_dates])
    elif track == 'm2': 
        held_in_dates = [re.search(r"\d{4}-\d{2}-\d{2}", d).group() for d in held_in_files]
        held_out_dates = [re.search(r"\d{4}-\d{2}-\d{2}", d).group() for d in held_out_files]
        format_held_in_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in held_in_dates])
        format_held_out_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in held_out_dates])
    elif track == 'h1':
        format_held_in_dates = []
        format_held_out_dates = []
        for f in held_in_files: 
            with NWBHDF5IO(f, 'r') as io:
                nwbfile = io.read()
                format_held_in_dates.append(datetime.strptime(nwbfile.session_start_time.strftime('%Y-%m-%d'), '%Y-%m-%d'))
        for f in held_out_files:
            with NWBHDF5IO(f, 'r') as io:
                nwbfile = io.read()
                format_held_out_dates.append(datetime.strptime(nwbfile.session_start_time.strftime('%Y-%m-%d'), '%Y-%m-%d'))
        format_held_in_dates = sorted(format_held_in_dates)
        format_held_out_dates = sorted(format_held_out_dates)

    # all_dates = np.unique(format_held_in_dates + format_held_out_dates)
    held_in_dates = np.unique(format_held_in_dates)
    held_out_dates = np.unique(format_held_out_dates)
    held_in_days = np.array([x.days for x in held_in_dates - held_in_dates[0]])
    held_out_days = np.array([x.days for x in held_out_dates - held_in_dates[0]])

    held_in_keys = DATASET_HELDINOUT_MAP[track]['held_in']
    held_out_keys = DATASET_HELDINOUT_MAP[track]['held_out']
    held_in_sort_idx = np.argsort(held_in_keys)
    held_out_sort_idx = np.argsort(held_out_keys)
    if track == 'h1':
        held_in_keys = np.unique([k.split('_')[0] for k in held_in_keys])
        held_out_keys = np.unique([k.split('_')[0] for k in held_out_keys])
        held_in_sort_idx = np.argsort([int(k.strip('S')) for k in held_in_keys])
        held_out_sort_idx = np.argsort([int(k.strip('S')) for k in held_out_keys])
    elif track == 'm2': 
        held_in_keys = np.unique([k.split('_')[1] for k in held_in_keys])
        held_out_keys = np.unique([k.split('_')[1] for k in held_out_keys])
        held_in_sort_idx = np.argsort([int(k) for k in held_in_keys])
        held_out_sort_idx = np.argsort([int(k) for k in held_out_keys])

    static_vals = results[f'{track}_static']
    held_in_static = np.array([static_vals[k] for k in held_in_keys])
    held_out_static = np.array([static_vals[k] for k in held_out_keys])

    oracle_vals = results[f'{track}_oracle']
    held_in_oracle = np.array([oracle_vals[k] for k in held_in_keys])
    held_out_oracle = np.array([oracle_vals[k] for k in held_out_keys])

    seed_idx = np.where(np.array(held_in_keys) == static_decoder[track])[0][0]

    axs[i][0].plot(held_in_days, held_in_oracle[held_in_sort_idx], 'x', color='k', linewidth=0.5, markersize=5)
    axs[i][0].plot(held_in_days[seed_idx], held_in_static[seed_idx], 's', color='g', linewidth=0.5, markersize=5, zorder=0, alpha=0.5)
    axs[i][0].set_ylim(0, 1)
    axs[i][0].grid(alpha=0.3)
    axs[i][0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
    axs[i][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=i < 1, pad=-0.5)
    # set x-ticks to be held_in_days 
    axs[i][0].set_xticks(np.arange(0, max(held_in_days) + 1, max(held_in_days)//2))

    axs[i][1].plot(held_out_days, held_out_oracle[held_out_sort_idx], 'o-', color='k', linewidth=0.5, markersize=3)
    axs[i][1].plot(held_out_days, held_out_static[held_out_sort_idx], 'o-', color='g', linewidth=0.5, markersize=3)
    axs[i][1].set_ylim(0, 1)
    axs[i][1].grid(alpha=0.3)
    axs[i][1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
    axs[i][1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs[i][1].spines['left'].set_visible(False)
    axs[i][1].set_xticks(np.arange(min(held_out_days), max(held_out_days) + 1, (max(held_out_days) - min(held_out_days))//2))
    axs[i][0].set_title(f'{track.upper()} held-in')
    axs[i][1].set_title(f'{track.upper()} held-out')

    print(f'{track.upper()} static start: {held_in_static[seed_idx]}')
    print(f'{track.upper()} static max diff: {np.max(held_in_static[seed_idx] - held_out_static)}')
    print(f'{track.upper()} Held-In Oracle Summary: {np.mean(held_in_oracle):.2f} +/- {np.std(held_in_oracle):.2f}')
    print(f'{track.upper()} Held-Out Oracle Summary: {np.mean(held_out_oracle):.2f} +/- {np.std(held_out_oracle):.2f}')
    print(f'{track.upper()} Held-In Static Summary: {np.mean(held_in_static):.2f} +/- {np.std(held_in_static):.2f}')
    print(f'{track.upper()} Held-Out Static Summary: {np.mean(held_out_static):.2f} +/- {np.std(held_out_static):.2f}')

# plt.savefig('/snel/home/bkarpo2/projects/falcon_figs/oracle_static.pdf', bbox_inches='tight')
# %%
