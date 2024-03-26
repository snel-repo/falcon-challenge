#%%
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
test_path = Path('/snel/share/share/derived/rouse/RTG/NWB_FALCON_v3/held_out_calib')
test_files = sorted(test_path.glob('*.nwb'))

#%% 
def load_nwb(fn: str):
    print(f'Loading {fn}')
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()

        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
        )

        raw_emg = nwbfile.acquisition['preprocessed_emg']
        muscles = [ts for ts in raw_emg.time_series]
        emg_data = []
        emg_timestamps = []
        for m in muscles: 
            mdata = raw_emg.get_timeseries(m)
            data = mdata.data[:]
            timestamps = mdata.timestamps[:]
            emg_data.append(data)
            emg_timestamps.append(timestamps)

        emg_data = np.vstack(emg_data).T 
        emg_timestamps = emg_timestamps[0]

        units = nwbfile.units.to_dataframe()
        binned_units = bin_units(units, bin_size_s=0.02, bin_end_timestamps=emg_timestamps)

        eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)

        return (
            binned_units,
            emg_data,
            emg_timestamps,
            muscles,
            trial_info,
            eval_mask
        )

#%% 
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

n_trials = [10, 20, 30, 50, 75, 100]

trials = np.full((len(test_files), len(n_trials)+1), np.nan)
perfs = np.full((len(test_files), len(n_trials)+1), np.nan)

for ii in range(len(test_files)): 
    print(f'Processing {test_files[ii]}')
    spikes, emg, time, muscles, tinfo, eval_mask = load_nwb(test_files[ii])

    ds_n_trials = n_trials + [len(tinfo)-1]
    trials[ii, :] = ds_n_trials

    for jj in range(len(n_trials)+ 1): 
        print(f'Decoding {ds_n_trials[jj]} trials')
        ntr = ds_n_trials[jj]
        last_trial = tinfo.iloc[ntr]
        end_time = last_trial['end_time']
        end_ind = np.argmin(np.abs(time - end_time))
        rel_spikes = spikes[:end_ind]
        rel_emg = emg[:end_ind]
        rel_mask = eval_mask[:end_ind]
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
for kk in range(len(test_files)): 
    plt.plot(trials[kk, :], perfs[kk, :], 'o-', label=f'Dataset {kk+1}')
plt.hlines(0.5, 0, 150, color='k', linestyle='--')
plt.ylim(-0.2, 1.2)
plt.xlabel('Number of trials provided')
plt.ylabel('R2 score')
plt.grid(alpha=0.4)

# %%
plt.figure(facecolor='w')
for kk in range(len(test_files)): 
    plt.plot(trials[kk, :] * 0.2, perfs[kk, :], 'o-', label=f'Dataset {kk+1}')
plt.hlines(0.5, 0, 35, color='k', linestyle='--')
plt.ylim(-0.2, 1.2)
plt.xlabel('Number of valid trials for scoring')
plt.ylabel('R2 score')
plt.grid(alpha=0.4)

# %%
