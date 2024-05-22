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
def rasterplot(spike_arr, bin_size_s=0.02, ax=None, spike_alpha=0.1, lw=0.2, s=1):
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
    # for idx, unit in enumerate(spike_arr.T):
    #     ax.scatter(
    #         np.where(unit)[0] * bin_size_s,
    #         np.ones(np.sum(unit != 0)) * idx,
    #         s=s,
    #         c='k',
    #         marker='|',
    #         linewidths=lw,
    #         alpha=spike_alpha
    #     )
    plt.pcolor(spike_arr.T, cmap='Greys', vmin=0, vmax=2)
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

t = int(10/0.02)
base_path = '/snel/home/bkarpo2/bin/falcon-challenge/data'

tracks = ['h1', 'h2', 'm1', 'm2']
for track in tracks: 
    held_in_files = glob.glob(os.path.join(base_path, track, '*held-in-calib*', '*.nwb'))

    with NWBHDF5IO(held_in_files[0], 'r') as io:
        nwbfile = io.read()
        if track == 'h1': 
            units = nwbfile.units.to_dataframe()
            h1_beh = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:]
            timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(h1_beh.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
            h1_spikes = bin_units(units, bin_size_s=0.02, bin_timestamps=timestamps)
            h1_trial_info = nwbfile.acquisition['TrialNum'].data[:]
        elif track == 'h2': 
            h2_spikes = nwbfile.acquisition['binned_spikes'].data[()]
            h2_trial_info = (
               nwbfile.trials.to_dataframe()
                .reset_index()
            )
            h2_beh = trial_info['cue'].values
        elif track == 'm1': 
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
        elif track == 'm2': 
            m2_trial_info = (
                nwbfile.trials.to_dataframe()
                .reset_index()
                .rename({"id": "trial_id", "tgt_pos": "target"}, axis=1)
            )
            vel_container = nwbfile.acquisition['finger_vel']
            labels = [ts for ts in vel_container.time_series]
            vel_data = []
            vel_timestamps = None
            for ts in labels: 
                ts_data = vel_container.get_timeseries(ts)
                vel_data.append(ts_data.data[:])
                vel_timestamps = ts_data.timestamps[:]
            m2_beh = np.vstack(vel_data).T
            units = nwbfile.units.to_dataframe()
            m2_spikes = bin_units(units, bin_size_s=0.02, bin_timestamps=vel_timestamps, is_timestamp_bin_start=True)


# #%% 
#     fig, ax = plt.subplots(figsize=(0.8, 0.7))
#     rasterplot(spikes[:t, :], bin_size_s=0.02, ax=ax, s=1)
#     fig.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', f'{track}_raster.pdf'), dpi=300)

# %% m2 behavior 
stop_time = m2_trial_info.stop_time[2]
t = int(stop_time/0.02)
trange = np.linspace(0, stop_time, t)
fig, ax = plt.subplots(m2_beh.shape[-1], 1, figsize=(0.8, 0.7), sharex=True, sharey=False)#(0.8, 0.7) (4,2)
ax[0].plot(trange, m2_beh[:t, 0], color='k', lw=0.5)
ax[1].plot(trange, m2_beh[:t, 1], color='k', lw=0.5)
# remove all components of ax[0]
for a in ax:
    a.set_yticks([])
    a.set_xticks([])
    a.spines['left'].set_visible(False)
    a.spines['bottom'].set_visible(False)
ax[0].set_ylabel('Index')
ax[1].set_ylabel('MRS')

ax[-1].hlines(-0.1, 0, 0.5, color='k', lw=0.5)
ax[-1].text(0.2, -0.12, '500 ms', ha='center', fontsize=7)
for a in ax:
    a.vlines(m2_trial_info.start_time[:3], -0.1, 0.1, lw=0.5, linestyle='--', color='teal')
plt.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/new_m2_beh.pdf'), dpi=300)

fig, ax = plt.subplots(figsize=(0.8, 0.7))
rasterplot(m2_spikes[:t, :], bin_size_s=0.02, ax=ax, s=1)
fig.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', f'new_m2_raster.pdf'), dpi=300)
# %% h1 behavior 
trial_changept = np.where(np.diff(h1_trial_info) != 0)[0]
t = trial_changept[0]
trange = np.arange(0, t) * 0.02

fig, ax = plt.subplots(h1_beh.shape[-1], 1, figsize=(0.8, 0.7), sharex=True, sharey=False)
for i in range(h1_beh.shape[-1]): 
    ax[i].plot(trange, h1_beh[:t, i], color='k', lw=0.5)
    ax[i].set_yticks([])
    ax[i].set_xticks([])
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
ax[-1].hlines(-0.1, 0, 5, color='k', lw=0.5)
ax[-1].text(1, -0.15, '5 s', ha='center', fontsize=5)

labels = ['Tx', 'Ty', 'Tz', 'Rx', 'G1', 'G2', 'G3']
for i, l in enumerate(labels): 
    ax[i].set_ylabel(l)
plt.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', 'new_h1_beh.pdf'), dpi=300)

fig, ax = plt.subplots(figsize=(0.8, 0.7))
rasterplot(h1_spikes[:t, :], bin_size_s=0.02, ax=ax, s=1)
fig.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', f'new_h1_raster.pdf'), dpi=300)

# %% m1 behavior 
stop_time = m1_trial_info.end_time[2]
end_t = int(stop_time/0.02)
start_time = m1_trial_info.start_time[0] - 0.5
start_t = int(start_time/0.02)
trange = np.linspace(start_time, stop_time, end_t-start_t)

# fig, ax = plt.subplots(m1_beh.shape[-1], 1, figsize=(6,3), sharex=True, sharey=True) #(0.87, 0.7)
# for i in range(m1_beh.shape[-1]): 
#     ax[i].plot(trange, m1_beh[start_t:end_t, i], color='k', lw=0.5)
#     ax[i].set_yticks([])
#     ax[i].set_xticks([])
#     ax[i].spines['left'].set_visible(False)
#     ax[i].spines['bottom'].set_visible(False)

from sklearn.preprocessing import MinMaxScaler

fig, ax = plt.subplots(figsize=(0.8, 1))
# Define the offset
offset = 1
# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
for i in range(m1_beh.shape[-1]): 
    # Scale the data to fall between 0 and 1
    scaled_data = scaler.fit_transform(m1_beh[start_t:end_t, i].reshape(-1, 1))
    # Flatten the data and add the offset
    scaled_data = np.squeeze(scaled_data) + i * offset
    ax.plot(trange, scaled_data, color='k', lw=0.5)

ax.set_yticks([])
ax.set_xticks([])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.hlines(-0.5, start_time, start_time+2, color='k', lw=0.5)
ax.text(start_time + 0.5, -1.2, '2 s', ha='center', fontsize=7)
ax.set_ylabel('Normalized EMG')
# for j, m in enumerate(muscles): 
#     ax.text(start_time - 0.5, j * offset, m, ha='right', va='center', fontsize=6)
    # ax.vlines(m1_trial_info.start_time[:3], j * offset, (j+1) * offset, lw=0.5, linestyle='--', color='teal')
#     ax[j].set_ylabel(m, rotation=0)
ax.vlines(m1_trial_info.start_time[:3], 0, 16, lw=0.5, linestyle='--', color='teal')
    # a.vlines(m1_trial_info.end_time[:3], 0, 1, lw=0.5, linestyle='-', color='teal')
plt.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', 'new_m1_beh.pdf'), dpi=300)

fig, ax = plt.subplots(figsize=(0.8, 0.7))
rasterplot(m1_spikes[start_t:end_t, :], bin_size_s=0.02, ax=ax, s=1)
fig.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', f'new_m1_raster.pdf'), dpi=300)
# %% h2 beh 
from collections import defaultdict

# Create a defaultdict of int
char_freq = defaultdict(int)

# Iterate over each string in the list
for string in h2_beh:
    # Iterate over each character in the string
    for char in string:
        # Increment the count of the character in the dictionary
        char_freq[char] += 1

fig, ax = plt.subplots(figsize=(4, 2))
# Create a bar plot of the character frequencies
sorted_inds = np.argsort(list(char_freq.keys()))
ax.bar(
    np.array(list(char_freq.keys()))[sorted_inds], 
    np.array(list(char_freq.values()))[sorted_inds],
    color='lightseagreen'
)

# Add labels and title
ax.set_xlabel('Characters')
ax.set_ylabel('Frequency')

#remove the tick markers 
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-3)
ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=True, pad=-3)
ax.grid(alpha=0.5, axis='y')
plt.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', 'h2_beh.pdf'), dpi=300)

# %%
stop_time = h2_trial_info.stop_time[1]
end_t = int(stop_time/0.02)
start_time = h2_trial_info.start_time[0]
start_t = int(start_time/0.02)
trange = np.linspace(start_time, stop_time, end_t-start_t)


fig, ax = plt.subplots(figsize=(0.8, 0.7))
rasterplot(h2_spikes[start_t:end_t, :], bin_size_s=0.02, ax=ax, s=1)
ax.vlines((h2_trial_info.start_time[:2]/0.02).astype('int'), 0, 192, lw=0.5, linestyle='--', color='teal')
ax.hlines(-0.5, 0, 50/0.02, color='k', lw=0.5)
ax.text(0.5, -35, '50 s', ha='left', fontsize=7)
ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=True, pad=-5)
ax.set_yticks(np.arange(0, 192, 40))
ax.set_ylabel('Channel #', labelpad=-2)
fig.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig/', f'new_h2_raster.pdf'))
# %%
