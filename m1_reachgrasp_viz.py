#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filtering import apply_butter_filt
from pynwb import NWBHDF5IO
import torch
import torch.nn.functional as F

root = Path('/snel/share/share/derived/rouse/RTG/NWB')
files = list(root.glob('*.nwb'))

# %%

def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.01,
        bin_end_timestamps: np.ndarray | None = None
    ) -> np.ndarray:
    r"""
        units: df with only index (spike index) and spike times (list of times in seconds). From nwb.units.
        bin_end_timestamps: array of timestamps indicating end of bin

        Returns:
        - array of spike counts per bin, per unit. Shape is (bins x units)
    """
    if bin_end_timestamps is None:
        end_time = units.spike_times.apply(lambda s: max(s) if len(s) else 0).max() + bin_size_s
        bin_end_timestamps = np.arange(0, end_time, bin_size_s)
    spike_arr = np.zeros((len(bin_end_timestamps), len(units)), dtype=np.uint8)
    bin_edges = np.concatenate([np.array([-np.inf]), bin_end_timestamps])
    for idx, (_, unit) in enumerate(units.iterrows()):
        spike_cnt, _ = np.histogram(unit.spike_times, bins=bin_edges)
        spike_arr[:, idx] = spike_cnt
    return spike_arr

def load_nwb(fn: str):
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()

        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
        )

        units = nwbfile.units.to_dataframe()
        # raw_emg = nwbfile.get_acquisition('emg')
        raw_emg = nwbfile.processing['emg_filtering_module'].get_data_interface('emg')
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

        fs = 1/(emg_timestamps[1] - emg_timestamps[0])
        # apply a lowpass filter to the emg_data 
        emg_data = apply_butter_filt(emg_data, fs, 'low', 10)

        return (
            bin_units(units),
            emg_data,
            emg_timestamps,
            muscles,
            trial_info,
        )

# %%
units, emg, time, muscle_names, trials = load_nwb(files[0])
# %% # plot EMG for some conditions 

for cond in [1, 2, 3]:

    condition_trials = trials.loc[trials['condition_id'] == cond]
    fig, ax = plt.subplots(len(muscle_names)//2, 2, figsize=(5, 5), sharex=True, sharey=True)
    axs = ax.flatten()

    for trial in range(condition_trials.shape[0]):
        tr = condition_trials.iloc[trial] 
        # get the timestamps between start and stop time 
        start = tr['start_time']
        stop = tr['end_time']
        start_idx = np.where(time == start)[0][0]
        stop_idx = np.where(time == stop)[0][0]
        for i, muscle in enumerate(muscle_names):
            signal = emg[start_idx:stop_idx, i]
            t = np.linspace(0, stop-start, len(signal))
            axs[i].plot(t, signal, label=muscle, alpha=0.7, linewidth=0.75, color='k')
            axs[i].set_ylabel(muscle)

    ax[-1, 0].set_xlabel('Time (s)')
    ax[-1, 1].set_xlabel('Time (s)')
    plt.suptitle(f'Single Trial EMG, Condition {cond}')
    plt.show()


# %% plot some continuous raster/EMG data 

fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True, sharey=False)
def rasterplot(spike_arr, bin_size_s=0.01, ax=None):
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(np.where(unit)[0] * bin_size_s, np.ones(np.sum(unit != 0)) * idx, s=1, c='k', marker='|')
    ax.set_xticks(np.arange(0, spike_arr.shape[0]*bin_size_s, 5))
    ax.set_xticklabels(np.arange(0, spike_arr.shape[0], 500) * bin_size_s)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 20))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
    # remove splines 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

rasterplot(units[:4001, :], ax=ax[0])

ax[1].plot(time[:40001], emg[:40001, :], linewidth=0.75, color='k', alpha=0.7)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('EMG')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
# %% histogram of mean firing rates 

mean_firing_rates = np.sum(units, axis=0) / (units.shape[0] * 0.01)
plt.figure()
plt.hist(mean_firing_rates, bins=20, alpha=0.7)
plt.xlabel('Mean Firing Rate (Hz)')
plt.ylabel('Channel Count')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# %% violin plot of ranges of EMG data 

plt.figure()
plt.violinplot(emg[:100000], showmeans=True, showextrema=False)
ax = plt.gca()
ax.set_ylabel('Recorded Units')
ax.set_xticks(np.arange(1, len(muscle_names) + 1))
ax.set_xticklabels(muscle_names, rotation=45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# %%
DEFAULT_TARGET_SMOOTH_MS = 60
BIN_SIZE_MS = 10

def gaussian_kernel(size, sigma):
    """
    Create a 1D Gaussian kernel.
    """
    size = int(size)
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def smooth(position, kernel_size=DEFAULT_TARGET_SMOOTH_MS / BIN_SIZE_MS, sigma=DEFAULT_TARGET_SMOOTH_MS / (3 * BIN_SIZE_MS)):
    """
    Apply Gaussian smoothing on the position data (dim 0)
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_left, pad_right = int(kernel_size // 2), int(kernel_size // 2)

    position = torch.as_tensor(position, dtype=torch.float32)
    position = F.pad(position.T, (pad_left, pad_right), 'replicate')
    smoothed = F.conv1d(position.unsqueeze(1), torch.tensor(kernel).float().unsqueeze(0).unsqueeze(0))
    return smoothed.squeeze().T.numpy()

#%% 
smoothed_spikes = smooth(units)

# %%
