#%% 
import pickle, os, sys, json, h5py, glob
from yacs.config import CfgNode as CN
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
from snel_toolkit.interfaces import LFADSInterface
from snel_toolkit.analysis import PSTH
from decoder_demos.decoding_utils import fit_and_eval_decoder, generate_lagged_matrix, apply_neural_behavioral_lag

from sklearn.metrics import r2_score

#%% 

DAY0_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb'

DAYK_PATHS = sorted(
    glob.glob('/snel/share/share/derived/rouse/RTG/NWB_FALCON_v7_unsorted/held_out_oracle/*.nwb')
)

align_field = 'move_onset_time'
cond_sep_field = 'condition_id'
decoding_field = 'preprocessed_emg'

#%% 

ds_spk0 = NWBDataset(
    DAY0_PATH, 
    skip_fields=['preprocessed_emg', 'eval_mask'])
ds_spk0.resample(20)

ds0 = NWBDataset(DAY0_PATH)
ds0.data = ds0.data.dropna()
ds0.data.spikes = ds_spk0.data.spikes
ds0.bin_width = 20

# %% get EMG PSTHs 

# make trials 
ds0.trials = ds0.make_trial_data(
        align_field=align_field,
        align_range=(-250, 500),  # ms
        allow_overlap=False,
        ignored_trials=ds0.trial_info['result'] != 'R'
    )

# %% get condition separated data 

unique_conds = ds0.trial_info[cond_sep_field].unique()
unique_conds.sort()
d0_single_trial_emg = np.array(
    np.split(
        ds0.trials.pivot(index='align_time', columns='trial_id').preprocessed_emg.values, 
        ds0.data.preprocessed_emg.shape[-1], 
        axis=1)
    ) # ch x time x trials 
val_trials = ds0.trial_info.iloc[np.unique(ds0.trials.trial_id.values)]
d0_cond_sep_emg = [d0_single_trial_emg[:, :, val_trials[cond_sep_field] == k] for k in unique_conds]
d0_cond_avg_emg = np.dstack([np.mean(x, axis=-1) for x in d0_cond_sep_emg])

# %%

emg_psth_r2 = np.zeros((len(DAYK_PATHS), len(unique_conds), d0_cond_avg_emg.shape[0]))

# for each dayk dataset 
for DAYK_PATH in DAYK_PATHS: 
    # load the dataset 
    ds_spkK = NWBDataset(
        DAYK_PATH, 
        skip_fields=['preprocessed_emg', 'eval_mask'])
    ds_spkK.resample(20)

    dsK = NWBDataset(DAYK_PATH)
    dsK.data = dsK.data.dropna()
    dsK.data.spikes = ds_spkK.data.spikes
    dsK.bin_width = 20

    # compute the PSTHs 
    dsK.trials = dsK.make_trial_data(
        align_field=align_field,
        align_range=(-250, 500),  # ms
        allow_overlap=False,
        ignored_trials=dsK.trial_info['result'] != 'R'
    )
    dK_single_trial_emg = np.array(
        np.split(
            dsK.trials.pivot(index='align_time', columns='trial_id').preprocessed_emg.values, 
            dsK.data.preprocessed_emg.shape[-1], 
            axis=1)
        ) # ch x time x trials 
    val_trials = dsK.trial_info.iloc[np.unique(dsK.trials.trial_id.values)]
    dK_cond_sep_emg = [dK_single_trial_emg[:, :, val_trials[cond_sep_field] == k] for k in unique_conds]
    dK_cond_avg_emg = np.dstack([np.mean(x, axis=-1) for x in dK_cond_sep_emg])

    # compute the R2 values for each muscle and condition separately 
    for i, cond in enumerate(unique_conds): 
        for j in range(d0_cond_avg_emg.shape[0]): 
            emg_psth_r2[DAYK_PATHS.index(DAYK_PATH), i, j] = r2_score(d0_cond_avg_emg[j, :, i], dK_cond_avg_emg[j, :, i])

# %%

# plot the R2 values for each day pair separately 
for i, DAYK_PATH in enumerate(DAYK_PATHS): 
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w')
    p = ax.imshow(emg_psth_r2[i, :, :].T, vmin=0, vmax=1)
    ax.set_title(f'EMG PSTH R2 Between 0924 and {os.path.basename(DAYK_PATH)}')
    ax.set_xlabel('Condition ID')
    ax.set_ylabel('Muscle')
    #add a colorbar
    cbar = plt.colorbar(p)
    # set yticklabels to muscle names 
    ax.set_yticks(np.arange(ds0.data.preprocessed_emg.shape[1]))
    ax.set_yticklabels(ds0.data.preprocessed_emg.columns)
    plt.show()

# %%
