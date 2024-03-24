#%%
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pynwb import NWBHDF5IO
from data_demos.styleguide import set_style
from falcon_challenge.dataloaders import bin_units
set_style()

#%% 
# Load the NWB file
test_path = Path('./data/h1/held_out/')
test_files = sorted(test_path.glob('*calib.nwb'))
print(test_path.exists())
print(f'Found {len(test_files)} test files')
#%% 

def load_nwb(fn: str):
    r"""
        Load NWB for H1.
    """
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematics'].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(kin.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
        eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)
        epochs = nwbfile.epochs.to_dataframe()
        trials = nwbfile.acquisition['TrialNum'].data[:]
        labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        return (
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            timestamps,
            labels,
            trials,
            eval_mask,
        )

from sklearn.metrics import r2_score
from decoder_demos.decoding_utils import (
    zscore_data,
    generate_lagged_matrix,
    apply_neural_behavioral_lag,
    fit_and_eval_decoder
)
from preproc.filtering import apply_exponential_filter

#%% 
n_hist = 7

n_trials = [1, 2, 3]

trials = np.full((len(test_files), len(n_trials)), np.nan)
perfs = np.full((len(test_files), len(n_trials)), np.nan)
print(trials)
for ii in range(len(test_files)):
    print(f'Processing {test_files[ii]}')
    spikes, emg, time, muscles, tinfo, eval_mask = load_nwb(test_files[ii])
    ds_n_trials = n_trials
    trials[ii, :] = ds_n_trials

    for jj in range(len(n_trials)): 
        print(f'Decoding {ds_n_trials[jj]} trials')
        ntr = ds_n_trials[jj]
        print(ntr)
        trial_mask = tinfo <= ntr
        print(trial_mask.shape, trial_mask.sum())
        # last_trial = tinfo.max()
        # end_time = last_trial['end_time']
        # end_ind = np.argmin(np.abs(time - end_time))
        rel_spikes = spikes[trial_mask]
        rel_emg = emg[trial_mask]
        rel_mask = eval_mask[trial_mask]
        print(rel_spikes.shape, spikes.shape)
        print(rel_emg.shape, rel_mask.shape)

        rel_ss = apply_exponential_filter(rel_spikes)
        emg_trials = rel_emg[rel_mask]
        ss_trials = rel_ss[rel_mask]

        split_ind = int(0.8 * len(emg_trials))
        smoothed_spikes_train, smoothed_spikes_valid = np.split(ss_trials, [split_ind])
        emg_train, emg_valid = np.split(emg_trials, [split_ind])

        train_neural = generate_lagged_matrix(zscore_data(smoothed_spikes_train), n_hist)
        valid_neural = generate_lagged_matrix(zscore_data(smoothed_spikes_valid), n_hist)
        train_behavioral = emg_train[n_hist:]
        valid_behavioral = emg_valid[n_hist:]

        valid_score, _ = fit_and_eval_decoder(
            train_neural,
            train_behavioral,
            valid_neural,
            valid_behavioral,
            return_preds=False
        )

        perfs[ii, jj] = valid_score


#%% 

plt.figure(facecolor='w')
print(trials)
for kk in range(len(test_files)): 
    plt.plot(trials[kk, :], perfs[kk, :], 'o-', label=f'Dataset {kk+1}')
# plt.hlines(0.5, 0, 150, color='k', linestyle='--')
# plt.ylim(-0.2, 1.2)
plt.xlabel('Number of trials provided')
plt.ylabel('R2 score')
plt.grid(alpha=0.4)

# %%
plt.figure(facecolor='w')
for kk in range(len(test_files)): 
    plt.plot(trials[kk, :] * 0.2, perfs[kk, :], 'o-', label=f'Dataset {kk+1}')
# plt.hlines(0.5, 0, 35, color='k', linestyle='--')
# plt.ylim(-0.2, 1.2)
plt.xlabel('Number of valid trials for scoring')
plt.ylabel('R2 score')
plt.grid(alpha=0.4)

# %%

# %%
