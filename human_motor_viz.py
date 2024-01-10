#%%

r"""
Demo notebook for human motor dataset.

    Data collected as open loop calibration for 5-7 DoF robot arm control.
    Different degrees of freedom are varied in phases for an arm in an virtual environment,
    and the subject is asked to move that degree in pace with the virtual arm.
    Multi-unit activity is provided for several calibration blocks.

- cf the aspirational NLB standard of https://github.com/neurallatents/neurallatents.github.io/blob/master/notebooks/mc_maze.ipynb
    - Overview of task, data collection/preprocessing.
    - Exploring each field in text, plotting reach conds, PSTH, hand decoding, and neural trajs.
- JY intends to show:
    - General viz
    - Decoding

- [ ] Apply base smoothing for labels
- [ ] Apply spike smoothing

spike smoothing
same day decoding
multiday aggregation decoding

test day decoding

TODO build out the utilities lib and upload lib we want to provide...

## Bug bash
- Not expecting any odd numbers in start/time times, but seeing odd numbers in epoch labels
- Expecting 5-7 DoF, but reliably receiving 7 active dimensions

"""

# 1. Extract raw datapoints
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBFile, NWBHDF5IO, TimeSeries

root = Path('./data/human_motor')

# List files
files = list(root.glob('*.nwb'))
sample = files[0]

#%%
# Load nwb file
def load_nwb(fn: str):
    r"""
        Load NWB for Human Motor ARAT dataset. Kinematic timestamps are provided at 100Hz.
    """
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        # print(nwbfile)
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematics'].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].timestamps[:]
        epochs = nwbfile.epochs.to_dataframe()
        return (
            units,
            kin,
            timestamps,
            epochs,
        )

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

units, kin, timestamps, epochs = load_nwb(sample)
binned = bin_units(units, bin_end_timestamps=timestamps) # T x C
print(timestamps)
print(epochs)
print(sample)
print(binned.shape)

#%%
# Basic qualitative

all_tags = [tag for sublist in epochs['tags'] for tag in sublist]
unique_tags = list(set(all_tags))
epoch_palette = sns.color_palette(n_colors=len(unique_tags))
# Mute colors of "Presentation" phases
for idx, tag in enumerate(unique_tags):
    if 'Presentation' in tag:
        epoch_palette[idx] = (0.5, 0.5, 0.5, 0.5)

def rasterplot(spike_arr, bin_size_s=0.01, ax=None):
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(np.where(unit)[0] * bin_size_s, np.ones(np.sum(unit != 0)) * idx, s=1, c='k', marker='|')
    # ax = sns.heatmap(spike_arr.T, cmap='gray_r', ax=ax) # Not sufficient - further autobinning occurs
    ax.set_xticks(np.arange(0, spike_arr.shape[0], 5000))
    ax.set_xticklabels(np.arange(0, spike_arr.shape[0], 5000) * bin_size_s)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
    # ax.set_title(sample.stem)

def kinplot(kin, timestamps, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(timestamps, kin)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kinematics')
    # ax.set_title(sample.stem)

def epoch_annote(epochs, ax=None):
    if ax is None:
        ax = plt.gca()
    for _, epoch in epochs.iterrows():
        # print(epoch.start_time, epoch.stop_time, epoch.tags)
        epoch_idx = unique_tags.index(epoch.tags[0])
        ax.axvspan(epoch['start_time'], epoch['stop_time'], color=epoch_palette[epoch_idx], alpha=0.5)

# Plot together
# Increase font sizes
plt.rcParams.update({'font.size': 16})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
rasterplot(binned, ax=ax1)
kinplot(kin, timestamps, ax=ax2)
epoch_annote(epochs, ax=ax1)
epoch_annote(epochs, ax=ax2)
plt.suptitle(sample.stem)
plt.tight_layout()
plt.xlim([0, 10])

#%%
# Basic quantitative
# Neural: check for dead channels, firing rate distribution
DEAD_THRESHOLD = 0.001 # Firing less than 0.1Hz is dead - which is 0.001 spikes per 10ms.
mean_firing_rates = binned.mean(axis=0)
dead_channels = np.where(mean_firing_rates < DEAD_THRESHOLD)[0]  # Define your threshold

# Plot firing rate distribution
ax = plt.gca()
ax.hist(mean_firing_rates, bins=30, alpha=0.75)
# Multiply xticks by 100
ax.set_xticklabels([f'{x*100:.0f}' for x in plt.xticks()[0]])
ax.set_xlabel('Mean Firing Rate (Hz)')
ax.set_ylabel('Channel Count')
ax.set_title(f'{sample.stem} Firing Rates')
ax.text(0.95, 0.95, f' {len(dead_channels)} Dead channel(s) < 0.1Hz:\n{dead_channels}', transform=ax.transAxes, ha='right', va='top')

#%%
# Kinematics: check for range on each dimension, overall time spent in active/nonactive phases

# Calculate range for each kinematic dimension (x,y,z,roll,pitch,yaw,grasp).
# Units inferred as virtual meters and radians. Grasp is arbitrary.
# TODO Ask: Grasp dimension isn't making a ton of sense to me. Why is there no weight at 1?
# TODO Feedback: Violinplot is not ideal
ax = plt.gca()
ax.scatter(np.arange(7), np.min(kin, axis=0), c='k', marker='_', s=100)
ax.scatter(np.arange(7), np.max(kin, axis=0), c='k', marker='_', s=100)
ax = sns.violinplot(kin, truncate=True, ax=ax)

ax.set_ylabel('Recorded Units')
ax.set_xticks(np.arange(7))
ax.set_xticklabels(['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp'])

# * Active times
print(f"Total time (s): {timestamps[-1]}")
# Label active if epoch does not contain "presentation"
active_phases = epochs.apply(lambda epoch: 'Presentation' not in epoch.tags[0], axis=1)
# Sum active phases
time_active = np.sum(epochs[active_phases].stop_time - epochs[active_phases].start_time)
time_inactive = np.sum(epochs[~active_phases].stop_time - epochs[~active_phases].start_time)
print(f"Seconds active (s) (Phase labeled): {time_active:.2f}")
print(f"Percent active (Phase labeled): {time_active / (time_active + time_inactive) * 100:.2f}%")

# Define active phase criteria and calculate time spent
velocity_threshold = 0.002 # ~ noise threshold
velocity = np.diff(kin, axis=0)  # Simple velocity estimate
active_phases = np.sum(np.abs(velocity) > velocity_threshold, axis=0)
time_active = np.sum(active_phases) * (timestamps[1] - timestamps[0])  # Total active time
print(f"Percent active (Variance inferred): {time_active / timestamps[-1] * 100:.2f}%")
#%%
# Smooth data, make base decoder



#%%
# Make multiday decoder