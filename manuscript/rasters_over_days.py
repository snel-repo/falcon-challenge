#%%
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime
from styleguide import set_style
set_style()
from pynwb import NWBHDF5IO
from falcon_challenge.dataloaders import bin_units

#%% 
def rasterplot(spike_arr, bin_size_s=0.02, ax=None, spike_alpha=0.1, lw=0.1, s=1):
    """
    Plot a raster plot of the spike_arr

    Args:
    - spike_arr (np.ndarray): Array of shape (T, N) containing the spike times.
    - T expected in ms..?
    - bin_size_s (float): Size of the bin in seconds
    - ax (plt.Axes): Axes to plot on
    """
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(
            np.where(unit)[0], #* bin_size_s,
            np.ones(np.sum(unit != 0)) * idx,
            s=s,
            c='k',
            marker='|',
            linewidths=lw,
            alpha=spike_alpha
        )
    # plt.pcolor(spike_arr.T, cmap='Greys', vmin=0, vmax=2)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 20))
    ax.set_ylabel('Channel #')
    # ax.set_xlabel('Time (s)')
    # make ticks invisible 
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=True, pad=-10)
    # make splines invisible
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

#%% 

files_to_plot = [
    '/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb',
    '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb',
    '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys.nwb',
    '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120928_behavior+ecephys.nwb',
    '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-out-calib/sub-MonkeyL-held-out-calib_ses-20121004_behavior+ecephys.nwb',
    '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-out-calib/sub-MonkeyL-held-out-calib_ses-20121017_behavior+ecephys.nwb',
    '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-out-calib/sub-MonkeyL-held-out-calib_ses-20121024_behavior+ecephys.nwb'
]

binned_spikes = []

for f in files_to_plot:
    with NWBHDF5IO(f, 'r') as io:
        nwbfile = io.read()

        m1_trial_info = (
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

        m1_beh = np.vstack(emg_data).T 
        emg_timestamps = emg_timestamps[0]

        units = nwbfile.units.to_dataframe()
        m1_spikes = bin_units(units, bin_size_s=0.02, bin_timestamps=emg_timestamps)
        binned_spikes.append(m1_spikes)
# %%

inds = 100
trim_binned_neural = []
for bn in binned_spikes:
    trim_binned_neural.append(bn[:inds, :])
trim_binned_neural = np.vstack(trim_binned_neural)

f, axes = plt.subplots(1, 1, figsize=(2, 1))

rasterplot(trim_binned_neural, ax=axes)
trim_lengths = [x[:inds, :].shape[0] for x in binned_spikes]
day_intervals = np.cumsum(trim_lengths)

for i, length in enumerate(day_intervals[:-1]):
    axes.axvline(x = length, color='r', linewidth=1)
axes.set_xlabel('Time (s)')
axes.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-2)
axes.tick_params(axis='y', which='both', right=False, left=False, labelleft=True, pad=-2)
axes.set_xticks(np.concatenate([[0], day_intervals]))
# axes.set_xticklabels(np.array([0, 1500, 3000, 0, 1500, 3000]) * 0.02)

plt.savefig('/home/bkarpo2/projects/falcon_figs/raster_stability2.pdf', bbox_inches='tight')

# %%
