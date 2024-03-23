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
test_path = Path('../data/h1/test/')
test_files = sorted(test_path.glob('*calibration.nwb'))
print(test_path.exists())
print(f'Found {len(test_files)} test files')
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
    
def load_nwb(fn: str):
    r"""
        Load NWB for H1.
    """
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematics'].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(kin.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
        blacklist = nwbfile.acquisition['kin_blacklist'].data[:].astype(bool)
        epochs = nwbfile.epochs.to_dataframe()
        trials = nwbfile.acquisition['TrialNum'].data[:]
        labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        return (
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            timestamps,
            blacklist,
            epochs,
            trials,
            labels
        )

def load_files(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    r"""
        Load several, merge data by simple concat
    """
    binned, kin, timestamps, blacklist, epochs, trials, labels = zip(*[load_nwb(str(f)) for f in files])
    lengths = [binned.shape[0] for binned in binned]
    binned = np.concatenate(binned, axis=0)
    kin = np.concatenate(kin, axis=0)
    
    # Offset timestamps and epochs
    bin_size = timestamps[0][1] - timestamps[0][0]
    all_timestamps = [timestamps[0]]
    for current_epochs, current_times in zip(epochs[1:], timestamps[1:]):
        clock_offset = all_timestamps[-1][-1] + bin_size
        current_epochs['start_time'] += clock_offset
        current_epochs['stop_time'] += clock_offset
        all_timestamps.append(current_times + clock_offset)
    timestamps = np.concatenate(all_timestamps, axis=0)
    blacklist = np.concatenate(blacklist, axis=0)
    trials = np.concatenate(trials, axis=0)
    epochs = pd.concat(epochs, axis=0)
    for l in labels[1:]:
        assert l == labels[0]
    return binned, kin, timestamps, blacklist, epochs, trials, labels[0], lengths

binned_neural, all_kin, all_timestamps, all_blacklist, all_epochs, all_trials, all_labels, lengths = load_files(sample_files)
BIN_SIZE_S = all_timestamps[1] - all_timestamps[0]
BIN_SIZE_MS = BIN_SIZE_S * 1000
print(f"Bin size = {BIN_SIZE_S} s")
print(f"Neural data ({len(lengths)} days) of shape T={binned_neural.shape[0]}, N={binned_neural.shape[1]}")

train_bins, train_kin, train_timestamps, train_blacklist, train_epochs, train_trials, train_labels, _ = load_files(train_files)
test_bins, test_kin, test_timestamps, test_blacklist, test_epochs, test_trials, test_labels, _ = load_files(test_files)


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
