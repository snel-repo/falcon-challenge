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

# root = Path('/ihome/rgaunt/joy47/share/stability/human_motor')
# root = Path('./data/h1')
root = Path('/snel/home/bkarpo2/pitt_data')

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
    binned, kin, timestamps, epochs, labels = zip(*[load_nwb(str(f)) for f in files])
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
set_style()

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

def create_targets(kin: np.ndarray):
    kin = smooth(kin)
    return np.gradient(kin, axis=0)
ax = plt.gca()
# Compute velocity
# Plot together to compare
# kinplot(train_kin, timestamps, ax=ax, palette=palette)
# kinplot(smooth(train_kin), train_timestamps, ax=ax, palette=palette, reference_labels=train_labels) # Offset for visual clarity
velocity = create_targets(train_kin)  # Simple velocity estimate
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
import numpy as np

def generate_lagged_matrix(input_matrix: np.ndarray, lag: int):
    """
    Generate a lagged version of an input matrix.

    Parameters:
    input_matrix (np.ndarray): The input matrix.
    lag (int): The number of lags to consider.

    Returns:
    np.ndarray: The lagged matrix.
    """
    # Initialize the lagged matrix
    lagged_matrix = np.zeros((input_matrix.shape[0] - lag, input_matrix.shape[1] * (lag + 1)))

    # Fill the lagged matrix
    for i in range(lag + 1):
        lagged_matrix[:, i*input_matrix.shape[1]:(i+1)*input_matrix.shape[1]] = input_matrix[lag-i : (-i if i != 0 else None)]

    return lagged_matrix


def apply_neural_behavioral_lag(neural_matrix: np.ndarray, behavioral_matrix: np.ndarray, lag: int):
    """
    Apply a lag to the neural matrix and the behavioral matrix.

    Parameters:
    neural_matrix (np.ndarray): The neural matrix.
    behavioral_matrix (np.ndarray): The behavioral matrix.
    lag (int): The number of lags to consider.

    Returns:
    np.ndarray: The lagged neural matrix.
    np.ndarray: The lagged behavioral matrix.
    """
    # Apply the lag to the neural matrix
    neural_matrix = neural_matrix[:-lag]

    # Apply the lag to the behavioral matrix
    behavioral_matrix = behavioral_matrix[lag:]

    return neural_matrix, behavioral_matrix


def prepare_train_test(
        binned_spikes: np.ndarray,
        behavior: np.ndarray,
        history: int=0,
        lag: int=0,
        ):
    signal = apply_exponential_filter(binned_spikes, NEURAL_TAU_MS)
    targets = create_targets(behavior)

    # Remove timepoints where nothing is happening in the kinematics
    still_times = np.all(np.abs(targets) < 0.001, axis=1)
    # final_signal = filtered_signal
    # final_target = velocity
    signal = signal[~still_times]
    targets = targets[~still_times]

    train_x, test_x = np.split(signal, [int(TRAIN_TEST[0] * signal.shape[0])])
    train_y, test_y = np.split(targets, [int(TRAIN_TEST[0] * targets.shape[0])])
    x_mean, x_std = np.nanmean(train_x, axis=0), np.nanstd(train_x, axis=0)
    x_std[x_std == 0] = 1
    y_mean, y_std = np.nanmean(train_y, axis=0), np.nanstd(train_y, axis=0)
    y_std[y_std == 0] = 1
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    # train_y = (train_y - y_mean) / y_std # don't standardize y if using var weighted r2
    # test_y = (test_y - y_mean) / y_std

    is_nan_y = np.isnan(train_y).any(axis=1)
    if np.any(is_nan_y):
        print(f"NaNs found in train_y, removing {np.sum(is_nan_y)} timepoints")
    train_x = train_x[~is_nan_y]
    train_y = train_y[~is_nan_y]

    is_nan_y = np.isnan(test_y).any(axis=1)
    if np.any(is_nan_y):
        print(f"NaNs found in test_y, removing {np.sum(is_nan_y)} timepoints")
    test_x = test_x[~is_nan_y]
    test_y = test_y[~is_nan_y]

    if lag > 0:
        train_x, train_y = apply_neural_behavioral_lag(train_x, train_y, lag)
        test_x, test_y = apply_neural_behavioral_lag(test_x, test_y, lag)

    if history > 0: 
        train_x = generate_lagged_matrix(train_x, history)
        test_x = generate_lagged_matrix(test_x, history)
        train_y = train_y[history:]
        test_y = test_y[history:]

    return train_x, train_y, test_x, test_y, x_mean, x_std, y_mean, y_std


def prepare_test(
        binned_spikes: np.ndarray,
        behavior: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        use_local_x_stats: bool = True # Minimal adaptation is to zscore with local statistics
        ):
    signal = apply_exponential_filter(binned_spikes, NEURAL_TAU_MS)
    targets = create_targets(behavior)

    # Remove timepoints where nothing is happening in the kinematics
    still_times = np.all(np.abs(targets) < 0.001, axis=1)
    signal = signal[~still_times]
    targets = targets[~still_times]

    if use_local_x_stats:
        x_mean = np.nanmean(signal, axis=0)
        x_std = np.nanstd(signal, axis=0)
        x_std[x_std == 0] = 1
    signal = (signal - x_mean) / x_std
    targets = (targets - y_mean) / y_std

    is_nan_y = np.isnan(targets).any(axis=1)
    if np.any(is_nan_y):
        print(f"NaNs found in test_y, removing {np.sum(is_nan_y)} timepoints")
    signal = signal[~is_nan_y]
    targets = targets[~is_nan_y]

    return signal, targets


#%% 
(
    train_x,
    train_y,
    test_x,
    test_y,
    x_mean,
    x_std,
    y_mean,
    y_std
) = prepare_train_test(train_bins, train_kin, history=5)

score, decoder = fit_and_eval_decoder(train_x, train_y, test_x, test_y)
pred_y = decoder.predict(test_x)
train_pred_y = decoder.predict(train_x)
print(f"Final R2: {score:.2f}")

r2 = r2_score(test_y, pred_y, multioutput='variance_weighted') #multioutput='raw_values')
r2_uniform = r2_score(test_y, pred_y, multioutput='uniform_average')
train_r2 = r2_score(train_y, train_pred_y, multioutput='variance_weighted') #multioutput='raw_values')
print(f"Val : {r2}")
print(f"Train: {train_r2}")

#%%
palette = sns.color_palette(n_colors=train_kin.shape[1])
f, axes = plt.subplots(train_kin.shape[1], figsize=(6, 12), sharex=True)
# Plot true vs predictions
for idx, (true, pred) in enumerate(zip(test_y.T, pred_y.T)):
    axes[idx].plot(true, label=f'{idx}', color=palette[idx])
    axes[idx].plot(pred, linestyle='--', color=palette[idx])
    axes[idx].set_title(f"{train_labels[idx]} $R^2$: {r2[idx]:.2f}")
    axes[idx].set_xticklabels([f'{x/1000 * BIN_SIZE_MS:.0f}' for x in axes[idx].get_xticks()])
axes[-1].set_xlabel('Time (s)')
f.suptitle(f'Val $R^2$: {score:.2f}')
f.tight_layout()

#%%
# Multi-day decoder
print(train_kin.shape)
print(test_kin_short.shape)
print(test_kin_long.shape)

x_short, y_short = prepare_test(test_bins_short, test_kin_short, x_mean, x_std, y_mean, y_std)
x_long, y_long = prepare_test(test_bins_long, test_kin_long, x_mean, x_std, y_mean, y_std)

score_short = decoder.score(x_short, y_short)
score_long = decoder.score(x_long, y_long)
print(f"Short Zero-shot: {score_short:.2f}") # 0.06
print(f"Long Zero-shot: {score_long:.2f}") # 0.0
