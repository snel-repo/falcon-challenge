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


    Need a normalization step for training…

# **Need to figure out why the cross val score is so low**

1. Test on new days
2. Merge multiday data
3. Update linear decoder until nontrivial

```python
## Bug bash
TODO build out the utilities lib and upload lib we want to provide...
- Not expecting any odd numbers in start/time times, but seeing odd numbers in epoch labels
- Expecting 5-7 DoF, but reliably receiving 7 active dimensions
- Data labels should come from NWB, not from custom hardcode
- [ ]  What is data discontinuity in S53?
```

- [ ]  Use Pitt OLE logic - see my OLE baselines in NDT2 codebase
    - [ ]  Then try to apply exponential smoothing to make it better etc

- Design choices to discuss
    - Temporally blocked splits
    - Causal smoothing in baseline
"""

# 1. Extract raw datapoints
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.signal import convolve

from pynwb import NWBFile, NWBHDF5IO, TimeSeries

root = Path('/ihome/rgaunt/joy47/share/stability/human_motor')
# root = Path('./data/human_motor')

# List files
files = list(root.glob('*.nwb'))
print(files)
sample = files[0]
print(sample)
session_query = 'S53_set_1'
# session_query = 'S77'
train_files = [f for f in files if session_query in str(f)]
print(train_files)

#%%
# Load nwb file
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
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            timestamps,
            epochs,
        )
all_binned = []
all_kin = []
all_timestamps = []
all_epochs = []
for f in train_files:
    binned, kin, timestamps, epochs = load_nwb(str(f))
    all_binned.append(binned)
    all_kin.append(kin)
    all_timestamps.append(timestamps)
    all_epochs.append(epochs)
binned, kin, timestamps, epochs = zip(*[load_nwb(str(f)) for f in train_files])
# Merge data by simple concat
binned = np.concatenate(binned, axis=0)
kin = np.concatenate(kin, axis=0)
# Offset timestamps and epochs
all_timestamps = [timestamps[0]]
for idx, current_times in enumerate(timestamps[1:]):
    epochs[idx]['start_time'] += all_timestamps[-1][-1] + 0.01 # 1 bin
    epochs[idx]['stop_time'] += all_timestamps[-1][-1] + 0.01 # 1 bin
    all_timestamps.append(current_times + all_timestamps[-1][-1] + 0.01)
timestamps = np.concatenate(all_timestamps, axis=0)
epochs = pd.concat(epochs, axis=0)
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

def kinplot(kin, timestamps, ax=None, palette=None):
    if ax is None:
        ax = plt.gca()
    num_dims = kin.shape[1]  # Assuming kin is a 2D array with shape (time, dimensions)

    if palette is None:
        palette = plt.cm.viridis(np.linspace(0, 1, num_dims))

    for i in range(num_dims):
        ax.plot(timestamps, kin[:, i], color=palette[i])
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
plt.suptitle(session_query)
plt.tight_layout()
# plt.xlim([0, 10])

print(binned.shape)
print(kin.shape)
print(timestamps.shape, timestamps.max())


#%%
# Basic quantitative
# Neural: check for dead channels, firing rate distribution
DEAD_THRESHOLD_HZ = 0.001 # Firing less than 0.1Hz is dead - which is 0.001 spikes per 10ms.
MUTE_THRESHOLD_HZ = 1 # Mute firing rates above 100Hz
mean_firing_rates = binned.mean(axis=0)
dead_channels = np.where(mean_firing_rates < DEAD_THRESHOLD_HZ)[0]
mute_channels = np.where(mean_firing_rates > MUTE_THRESHOLD_HZ)[0]
mean_firing_rates[dead_channels] = 0
mean_firing_rates[mute_channels] = 0
# Plot firing rate distribution
ax = plt.gca()
ax.hist(mean_firing_rates, bins=30, alpha=0.75)
# Multiply xticks by 100
ax.set_xticklabels([f'{x*100:.0f}' for x in plt.xticks()[0]])
ax.set_xlabel('Mean Firing Rate (Hz)')
ax.set_ylabel('Channel Count')
ax.set_title(f'{session_query} Firing Rates')
ax.text(0.95, 0.95, f' {len(dead_channels)} Dead channel(s) < 0.1Hz:\n{dead_channels}', transform=ax.transAxes, ha='right', va='top')
ax.text(0.95, 0.55, f' {len(mute_channels)} Mute channel(s) > 100Hz:\n{mute_channels}', transform=ax.transAxes, ha='right', va='top')

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
ax.set_title(f'{session_query} Kinematic Range')

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
# Smooth data for decoding make base linear decoder
BIN_SIZE_MS = 10
DEFAULT_TARGET_SMOOTH_MS = 490
palette = sns.color_palette(n_colors=kin.shape[1])

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

ax = plt.gca()
kin_targets = smooth(kin)
# Compute velocity
# Plot together to compare
kinplot(kin, timestamps, ax=ax, palette=palette)
kinplot(kin_targets - 0.05, timestamps, ax=ax, palette=palette) # Offset for visual clarity

velocity = np.gradient(kin_targets, axis=0)  # Simple velocity estimate
# kinplot(velocity, timestamps, ax=ax)
ax.set_xlim([5, 10])

#%%
NEURAL_TAU = 60. # exponential filter (per human motor practice), in units of ms

def apply_exponential_filter(signal, tau, bin_size=10):
    """
    Apply a **causal** exponential filter to the neural signal.

    :param signal: NumPy array of shape (time, channels)
    :param tau: Decay rate (time constant) of the exponential filter
    :param bin_size: Bin size in ms (default is 10ms)
    :return: Filtered signal
    """
    # Time array (considering the length to be sufficiently long for the decay)
    t = np.arange(0, 5 * tau, bin_size)  # 5*tau ensures capturing most of the decay
    # Exponential filter kernel
    kernel = np.exp(-t / tau)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    # Apply the filter
    filtered_signal = np.array([convolve(signal[:, ch], kernel, mode='full')[-len(signal):] for ch in range(signal.shape[1])]).T
    return filtered_signal

filtered_signal = apply_exponential_filter(binned, NEURAL_TAU)
f = plt.figure(figsize=(20, 10))
ax = f.gca()
kinplot(filtered_signal[:, 3:4] * 100, timestamps, ax=ax, palette=palette)
# kinplot(binned[:, 3:4] * 100, timestamps, ax=ax, palette=palette)
ax.set_xlim([0, 10])
#%%
# compute first pc of filtered signal
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(binned)
# pca.fit(filtered_signal)
print(pca.explained_variance_ratio_)
print(pca.components_)
# Plot timecourse of 1st PC
f = plt.figure(figsize=(20, 10))
ax = f.gca()
ax.plot(pca.transform(binned)[:100])

#%%
# Make and eval train test splits
# From NLB tools
# Make single day linear decoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def fit_and_eval_decoder(
    train_rates,
    train_behavior,
    eval_rates,
    eval_behavior,
    grid_search=True,
):
    """Fits ridge regression on train data passed
    in and evaluates on eval data

    Parameters
    ----------
    train_rates : np.ndarray
        2d array with 1st dimension being samples (time) and
        2nd dimension being input variables (units).
        Used to train regressor
    train_behavior : np.ndarray
        2d array with 1st dimension being samples (time) and
        2nd dimension being output variables (channels).
        Used to train regressor
    eval_rates : np.ndarray
        2d array with same dimension ordering as train_rates.
        Used to evaluate regressor
    eval_behavior : np.ndarray
        2d array with same dimension ordering as train_behavior.
        Used to evaluate regressor
    grid_search : bool
        Whether to perform a cross-validated grid search to find
        the best regularization hyperparameters.

    Returns
    -------
    float
        R2 score on eval data
    """
    if np.any(np.isnan(train_behavior)):
        train_rates = train_rates[~np.isnan(train_behavior)[:, 0]]
        train_behavior = train_behavior[~np.isnan(train_behavior)[:, 0]]
    if np.any(np.isnan(eval_behavior)):
        eval_rates = eval_rates[~np.isnan(eval_behavior)[:, 0]]
        eval_behavior = eval_behavior[~np.isnan(eval_behavior)[:, 0]]
    assert not np.any(np.isnan(train_rates)) and not np.any(np.isnan(eval_rates)), \
        "fit_and_eval_decoder: NaNs found in rate predictions within required trial times"

    if grid_search:
        decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 1, 9)})
    else:
        decoder = Ridge(alpha=1e-2)
    decoder.fit(train_rates, train_behavior)
    return decoder.score(eval_rates, eval_behavior), decoder

print(filtered_signal.shape)
print(kin_targets.shape)
TRAIN_TEST = (0.8, 0.2)
train_x, test_x = np.split(filtered_signal, [int(TRAIN_TEST[0] * filtered_signal.shape[0])])
train_y, test_y = np.split(velocity, [int(TRAIN_TEST[0] * kin_targets.shape[0])])
score, decoder = fit_and_eval_decoder(train_x, train_y, test_x, test_y)
pred_y = decoder.predict(test_x)
print(score)


#%%
print(pred_y.shape)
print(test_y.shape)
from sklearn.metrics import r2_score

decoder = Ridge(alpha=0.0)
is_nan_y = np.isnan(train_y).any(axis=1)
train_x = train_x[~is_nan_y]
train_y = train_y[~is_nan_y]
decoder.fit(train_x, train_y)

# compute r2
print(test_x.shape, test_y.shape)
is_nan_y = np.isnan(test_y).any(axis=1)
test_x = test_x[~is_nan_y]
test_y = test_y[~is_nan_y]


pred_y = decoder.predict(test_x)
train_pred_y = decoder.predict(train_x)
r2 = r2_score(test_y, pred_y, multioutput='raw_values')
train_r2 = r2_score(train_y, train_pred_y, multioutput='raw_values')
print(f"Test : {r2}")
print(f"Train: {train_r2}")

palette = sns.color_palette(n_colors=kin.shape[1])
f, axes = plt.subplots(kin.shape[1], figsize=(12, 10))
# Plot true vs predictions
for idx, (true, pred) in enumerate(zip(test_y.T, pred_y.T)):
    axes[idx].plot(true, label=f'{idx}', color=palette[idx])
    axes[idx].plot(pred, linestyle='--', color=palette[idx])


#%%
# Make multiday decoder