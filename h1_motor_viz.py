#%%

r"""
Demo notebook for human motor dataset.

    Data collected as open loop calibration for 5-7 DoF robot arm control.
    Different degrees of freedom are varied in phases for an arm in an virtual environment,
    and the subject is asked to move that degree in pace with the virtual arm.
    Multi-unit activity is provided for several calibration blocks.

1. Test on new days
2. Merge multiday data

```python
## Bug bash
TODO build out the utilities lib and upload lib we want to provide...
- Not expecting any odd numbers in start/time times, but seeing odd numbers in epoch labels
- Expecting 5-7 DoF, but reliably receiving 7 active dimensions
```
- Design choices to discuss
    - Temporally blocked splits
    - Causal smoothing in baseline
"""

# 1. Extract raw datapoints
from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.signal import convolve

from pynwb import NWBHDF5IO
from styleguide import set_style

root = Path('/ihome/rgaunt/joy47/share/stability/human_motor')
# root = Path('./data/human_motor')

# List files
files = list(root.glob('*.nwb'))
print(files)
sample = files[0]
print(sample)
# session_query = 'S53_set_1'
train_query = ['S53_set_1']
train_query = ['S53_set']
train_query = ['S53_set', 'S63_set']
test_query_short = ['S77']
test_query_long = ['S91', 'S95', 'S99']

def get_files(query):
    return [f for f in files if any(sq in str(f) for sq in query)]
train_files = get_files(train_query)
test_files_short = get_files(test_query_short)
test_files_long = get_files(test_query_long)
print(train_files)

is_5dof = lambda f: int(f.stem[len('pitt_thin_session_S'):].split('_')[0]) in range(364, 605)
is_5dof_files = np.array([is_5dof(f) for f in [*train_files, *test_files_short, *test_files_long]])

# check if not uniform and report
if not all(is_5dof_files) and not all(~is_5dof_files):
    print("Warning: Not all files are same DoF")
    uniform_dof = 5
else:
    uniform_dof = 5 if is_5dof_files[0] else 7
print(f"Uniform DoF: {uniform_dof}")

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
        labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        epochs = nwbfile.epochs.to_dataframe()
        return (
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            timestamps,
            epochs,
            labels
        )

def load_files(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    binned, kin, timestamps, epochs, labels = zip(*[load_nwb(str(f)) for f in train_files])
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
    for l in labels[1:]:
        assert l == labels[0]
    return binned, kin, timestamps, epochs, labels[0]

train_bins, train_kin, train_timestamps, train_epochs, train_labels = load_files(train_files)
test_bins_short, test_kin_short, test_timestamps_short, test_epochs_short, test_labels_short = load_files(test_files_short)
test_bins_long, test_kin_long, test_timestamps_long, test_epochs_long, test_labels_long = load_files(test_files_long)

#%%
# Basic qualitative
palette = [*sns.color_palette('rocket', n_colors=3), *sns.color_palette('viridis', n_colors=3), 'k']
to_plot = train_labels

all_tags = [tag for sublist in train_epochs['tags'] for tag in sublist]
unique_tags = list(set(all_tags))
epoch_palette = sns.color_palette(n_colors=len(unique_tags))
# Mute colors of "Presentation" phases
for idx, tag in enumerate(unique_tags):
    if 'Presentation' in tag:
        epoch_palette[idx] = (0.9, 0.9, 0.9, 0.1)
    else:
        # white
        epoch_palette[idx] = (1, 1, 1, 0.5)

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

def kinplot(kin, timestamps, ax=None, palette=None, reference_labels=[], to_plot=to_plot):
    if ax is None:
        ax = plt.gca()

    if palette is None:
        palette = plt.cm.viridis(np.linspace(0, 1, len(reference_labels)))
    for kin_label in to_plot:
        kin_idx = reference_labels.index(kin_label)
        ax.plot(timestamps, kin[:, kin_idx], color=palette[kin_idx])
    print(timestamps.min(), timestamps.max())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kinematics')

def epoch_annote(epochs, ax=None):
    if ax is None:
        ax = plt.gca()
    for _, epoch in epochs.iterrows():
        # print(epoch.start_time, epoch.stop_time, epoch.tags)
        epoch_idx = unique_tags.index(epoch.tags[0])
        # ax.axvspan(epoch['start_time'], epoch['stop_time'], color=epoch_palette[epoch_idx], alpha=0.5)

# Plot together
# Increase font sizes
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
})

def plot_qualitative(
    binned: np.ndarray,
    kin: np.ndarray,
    timestamps: np.ndarray,
    epochs: pd.DataFrame,
    labels: list,
    palette: list,
    to_plot: list,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    rasterplot(binned, ax=ax1)
    kinplot(kin, timestamps, palette=palette, ax=ax2, reference_labels=labels, to_plot=to_plot)
    ax2.set_ylabel('Position')
    epoch_annote(epochs, ax=ax1)
    epoch_annote(epochs, ax=ax2)
    fig.suptitle(f'(DoF: {labels})')
    fig.tight_layout()

    xticks = np.arange(0, 50, 10)
    plt.xlim(xticks[0], xticks[-1])
    plt.xticks(xticks, labels=xticks.round(2))

plot_qualitative(
    train_bins,
    train_kin,
    train_timestamps,
    train_epochs,
    train_labels,
    palette,
    to_plot
)
#%%
# Just show kinematics
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
kinplot(train_kin, train_timestamps, palette=palette, ax=ax, reference_labels=train_labels)
ax.set_ylabel('Raw Visual Position')
epoch_annote(train_epochs, ax=ax)
plt.suptitle(f'{train_query} (DoF: {uniform_dof})')
plt.tight_layout()

xticks = np.arange(0, 50, 10)
plt.xlim(xticks[0], xticks[-1])
plt.xticks(xticks, labels=xticks.round(2))


#%%
# Basic quantitative
# Neural: check for dead channels, firing rate distribution
DEAD_THRESHOLD_HZ = 0.001 # Firing less than 0.1Hz is dead - which is 0.001 spikes per 10ms.
MUTE_THRESHOLD_HZ = 1 # Mute firing rates above 100Hz
mean_firing_rates = train_bins.mean(axis=0)
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
ax.set_title(f'{train_query} Firing Rates')
ax.text(0.95, 0.95, f' {len(dead_channels)} Dead channel(s) < 0.1Hz:\n{dead_channels}', transform=ax.transAxes, ha='right', va='top')
ax.text(0.95, 0.55, f' {len(mute_channels)} Mute channel(s) > 100Hz:\n{mute_channels}', transform=ax.transAxes, ha='right', va='top')

#%%
# Kinematics: check for range on each dimension, overall time spent in active/nonactive phases

# Calculate range for each kinematic dimension (x,y,z,roll,pitch,yaw,grasp).
# Units inferred as virtual meters and radians. Grasp is arbitrary.
# TODO Feedback: Violinplot is not ideal
ax = plt.gca()
ax.scatter(np.arange(7), np.min(train_kin, axis=0), c='k', marker='_', s=100)
ax.scatter(np.arange(7), np.max(train_kin, axis=0), c='k', marker='_', s=100)
ax = sns.violinplot(train_kin, truncate=True, ax=ax)

ax.set_ylabel('Recorded Units')
ax.set_xticks(np.arange(7))
ax.set_xticklabels(['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp'])
ax.set_title(f'{train_query} Kinematic Range')

# * Active times
print(f"Total time (s): {train_timestamps[-1]}")
# Label active if epoch does not contain "presentation"
active_phases = train_epochs.apply(lambda epoch: 'Presentation' not in epoch.tags[0], axis=1)
# Sum active phases
time_active = np.sum(train_epochs[active_phases].stop_time - train_epochs[active_phases].start_time)
time_inactive = np.sum(train_epochs[~active_phases].stop_time - train_epochs[~active_phases].start_time)
print(f"Seconds active (s) (Phase labeled): {time_active:.2f}")
print(f"Percent active (Phase labeled): {time_active / (time_active + time_inactive) * 100:.2f}%")

# Define active phase criteria and calculate time spent
velocity_threshold = 0.002 # ~ noise threshold
velocity = np.diff(train_kin, axis=0)  # Simple velocity estimate
active_phases = np.sum(np.abs(velocity) > velocity_threshold, axis=0)
time_active = np.sum(active_phases) * (train_timestamps[1] - train_timestamps[0])  # Total active time
print(f"Percent active (Variance inferred): {time_active / train_timestamps[-1] * 100:.2f}%")
#%%
# Smooth data for decoding make base linear decoder
BIN_SIZE_MS = 10
DEFAULT_TARGET_SMOOTH_MS = 490
palette = [*sns.color_palette('rocket', n_colors=3), *sns.color_palette('viridis', n_colors=3), 'k']

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
kin_targets = smooth(train_kin)
# Compute velocity
# Plot together to compare
# kinplot(kin, timestamps, ax=ax, palette=palette)
# kinplot(kin_targets, train_timestamps, ax=ax, palette=palette, reference_labels=train_labels) # Offset for visual clarity
velocity = np.gradient(kin_targets, axis=0)  # Simple velocity estimate
kinplot(velocity, train_timestamps, ax=ax, palette=palette, reference_labels=train_labels)

xticks = np.arange(0, 50, 10)
plt.xlim(xticks[0], xticks[-1])
plt.xticks(xticks, labels=xticks.round(2))

#%%
NEURAL_TAU_MS = 240. # exponential filter from H1 Lab

def apply_exponential_filter(
        signal, tau, bin_size=10, extent: int=1
    ):
    """
    Apply a **causal** exponential filter to the neural signal.

    :param signal: NumPy array of shape (time, channels)
    :param tau: Decay rate (time constant) of the exponential filter
    :param bin_size: Bin size in ms (default is 10ms)
    :return: Filtered signal
    :param extent: Number of time constants to extend the filter kernel

    Implementation notes:
    # extent should be 3 for reporting parity, but reference hardcodes a kernel that's equivalent to extent=1
    """
    t = np.arange(0, extent * tau, bin_size)
    # Exponential filter kernel
    kernel = np.exp(-t / tau)
    kernel /= np.sum(kernel)
    # Apply the filter
    filtered_signal = np.array([convolve(signal[:, ch], kernel, mode='full')[-len(signal):] for ch in range(signal.shape[1])]).T
    return filtered_signal

filtered_signal = apply_exponential_filter(train_bins, NEURAL_TAU_MS)
f = plt.figure(figsize=(20, 10))
ax = f.gca()

sample = 1
palette = sns.color_palette(n_colors=filtered_signal.shape[1])
ax.plot(train_timestamps, filtered_signal[:, sample], color=palette[sample])
ax.set_xlabel('Time (s)')
ax.set_yticklabels([f'{x*(1000 / BIN_SIZE_MS):.0f}' for x in plt.yticks()[0]])
ax.set_ylabel('Firing Rate (Hz)')
ax.set_xlim([0, 10])

#%%
# Make single day linear decoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def fit_and_eval_decoder(
    train_rates: np.ndarray,
    train_behavior: np.ndarray,
    eval_rates: np.ndarray,
    eval_behavior: np.ndarray,
    grid_search: bool=True,
):
    """Fits ridge regression on train data passed
    in and evaluates on eval data

    Parameters
    ----------
    train_rates :
        2d array time x units.
    train_behavior :
        2d array time x output dims.
    eval_rates :
        2d array time x units
    eval_behavior :
        2d array time x output dims
    grid_search :
        Whether to perform a cross-validated grid search to find
        the best regularization hyperparameters.

    Returns
    -------
    float
        Uniform average R2 score on eval data
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
        decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-5, 5, 20)})
    else:
        decoder = Ridge(alpha=1e-2)
    decoder.fit(train_rates, train_behavior)
    return decoder.score(eval_rates, eval_behavior), decoder

TRAIN_TEST = (0.8, 0.2)

# Remove timepoints where nothing is happening in the kinematics
still_times = np.all(np.abs(velocity) < 0.001, axis=1)
# final_signal = filtered_signal
# final_target = velocity
final_signal = filtered_signal[~still_times]
final_target = velocity[~still_times]

train_x, test_x = np.split(final_signal, [int(TRAIN_TEST[0] * final_signal.shape[0])])
train_y, test_y = np.split(final_target, [int(TRAIN_TEST[0] * final_target.shape[0])])
x_mean, x_std = np.nanmean(train_x, axis=0), np.nanstd(train_x, axis=0)
x_std[x_std == 0] = 1
y_mean, y_std = np.nanmean(train_y, axis=0), np.nanstd(train_y, axis=0)
y_std[y_std == 0] = 1
train_x = (train_x - x_mean) / x_std
test_x = (test_x - x_mean) / x_std
train_y = (train_y - y_mean) / y_std
test_y = (test_y - y_mean) / y_std

# print(np.isnan(train_x).any())
is_nan_y = np.isnan(train_y).any(axis=1)
train_x = train_x[~is_nan_y]
train_y = train_y[~is_nan_y]

is_nan_y = np.isnan(test_y).any(axis=1)
test_x = test_x[~is_nan_y]
test_y = test_y[~is_nan_y]

score, decoder = fit_and_eval_decoder(train_x, train_y, test_x, test_y)
pred_y = decoder.predict(test_x)
train_pred_y = decoder.predict(train_x)
print(f"Final R2: {score}")

r2 = r2_score(test_y, pred_y, multioutput='raw_values')
r2_uniform = r2_score(test_y, pred_y, multioutput='uniform_average')
train_r2 = r2_score(train_y, train_pred_y, multioutput='raw_values')
print(f"Test : {r2}")
print(f"Train: {train_r2}")

palette = sns.color_palette(n_colors=train_kin.shape[1])
f, axes = plt.subplots(train_kin.shape[1], figsize=(6, 12))
# Plot true vs predictions
for idx, (true, pred) in enumerate(zip(test_y.T, pred_y.T)):
    axes[idx].plot(true, label=f'{idx}', color=palette[idx])
    axes[idx].plot(pred, linestyle='--', color=palette[idx])
    axes[idx].set_title(f"{train_labels[idx]} $R^2$: {r2[idx]:.2f}")
# f.supxlabel('Time (10ms)')
f.suptitle(f'Val $R^2$: {score:.2f}')
f.tight_layout()

#%%
# Multi-day decoder
test