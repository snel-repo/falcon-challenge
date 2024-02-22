#%%

r"""
Demo notebook for human motor dataset.

    Data collected as open loop calibration for 5-7 DoF robot arm control.
    Different degrees of freedom are varied in phases for an arm in an virtual environment,
    and the subject is asked to move that degree in pace with the virtual arm.
    Multi-unit activity is provided for several calibration blocks.

TODO build out the utilities lib and upload lib we want to provide...
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

from pynwb import NWBHDF5IO
from data_demos.styleguide import set_style
from falcon_challenge.dataloaders import bin_units
from falcon_challenge.config import FalconTask

# root = Path('/ihome/rgaunt/joy47/share/stability/human_motor')
root = Path('./data/h1')
# root = Path('/snel/home/bkarpo2/pitt_data')

# List files
train_query = 'train'
test_query_short = 'test_short'
test_query_long = 'test_long'

def get_files(query):
    return sorted(list((root / query).glob('*.nwb')))
train_files = get_files(train_query)
test_files_short = get_files(test_query_short)
test_files_long = get_files(test_query_long)
print(train_files)

uniform_dof = 7
print(f"Uniform DoF: {uniform_dof}")

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
        blacklist = nwbfile.acquisition['Blacklist'].data[:].astype(bool)
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

def load_files(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    binned, kin, timestamps, blacklist, epochs, trials, labels = zip(*[load_nwb(str(f)) for f in files])
    # Merge data by simple concat
    binned = np.concatenate(binned, axis=0)
    kin = np.concatenate(kin, axis=0)
    # Offset timestamps and epochs
    all_timestamps = [timestamps[0]]
    for current_epochs, current_times in zip(epochs[1:], timestamps[1:]):
        clock_offset = all_timestamps[-1][-1] + 0.01
        current_epochs['start_time'] += clock_offset
        current_epochs['stop_time'] += clock_offset
        all_timestamps.append(current_times + clock_offset)
    timestamps = np.concatenate(all_timestamps, axis=0)
    blacklist = np.concatenate(blacklist, axis=0)
    trials = np.concatenate(trials, axis=0)
    epochs = pd.concat(epochs, axis=0)
    for l in labels[1:]:
        assert l == labels[0]
    return binned, kin, timestamps, blacklist, epochs, trials, labels[0]

train_bins, train_kin, train_timestamps, train_blacklist, train_epochs, train_trials, train_labels = load_files(train_files)
test_bins_short, test_kin_short, test_timestamps_short, test_blacklist_short, test_epochs_short, test_trials_short, test_labels_short = load_files(test_files_short)
test_bins_long, test_kin_long, test_timestamps_long, test_blacklist_long, test_epochs_long, test_trials_long, test_labels_long = load_files(test_files_long)
#%%
BIN_SIZE_MS = 10 # TODO derive from nwb

# Basic qualitative
palette = [*sns.color_palette('rocket', n_colors=3), *sns.color_palette('viridis', n_colors=3), 'k']
to_plot = train_labels

all_tags = [tag for sublist in train_epochs['tags'] for tag in sublist]
all_tags.extend([tag for sublist in test_epochs_short['tags'] for tag in sublist])
all_tags.extend([tag for sublist in test_epochs_long['tags'] for tag in sublist])
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

def kinplot(kin, timestamps, ax=None, palette=None, reference_labels=[], to_plot=to_plot):
    if ax is None:
        ax = plt.gca()

    if palette is None:
        palette = plt.cm.viridis(np.linspace(0, 1, len(reference_labels)))
    for kin_label in to_plot:
        kin_idx = reference_labels.index(kin_label)
        ax.plot(timestamps, kin[:, kin_idx], color=palette[kin_idx])
    # print(timestamps.min(), timestamps.max())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kinematics')

def epoch_annote(epochs, ax=None):
    if ax is None:
        ax = plt.gca()
    for _, epoch in epochs.iterrows():
        # print(epoch.start_time, epoch.stop_time, epoch.tags)
        epoch_idx = unique_tags.index(epoch.tags[0])
        ax.axvspan(epoch['start_time'], epoch['stop_time'], color=epoch_palette[epoch_idx], alpha=0.5)

set_style()

def plot_qualitative(
    binned: np.ndarray,
    kin: np.ndarray,
    timestamps: np.ndarray,
    epochs: pd.DataFrame,
    trials: np.ndarray,
    labels: list,
    palette: list,
    to_plot: list,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    rasterplot(binned, ax=ax1)
    kinplot(kin, timestamps, palette=palette, ax=ax2, reference_labels=labels, to_plot=to_plot)
    ax2.set_ylabel('Position')
    # epoch_annote(epochs, ax=ax1)
    # epoch_annote(epochs, ax=ax2)
    trial_changept = np.where(np.diff(trials) != 0)[0]
    for changept in trial_changept:
        ax2.axvline(timestamps[changept], color='k', linestyle='-', alpha=0.1)

    fig.suptitle(f'(DoF: {labels})')
    fig.tight_layout()
    return fig, (ax1, ax2)
    # xticks = np.arange(0, 50, 10)
    # xticks = np.arange(0, 800, 10)
    # plt.xlim(xticks[0], xticks[-1])
    # plt.xticks(xticks, labels=xticks.round(2))

f, axes = plot_qualitative(
    train_bins,
    train_kin,
    train_timestamps,
    train_epochs,
    train_trials,
    train_labels,
    palette,
    to_plot
)

# f, axes = plot_qualitative(
#     test_bins_short,
#     test_kin_short,
#     test_timestamps_short,
#     test_epochs_short,
#     test_trials_short,
#     test_labels_short,
#     palette,
#     to_plot
# )

# f, axes = plot_qualitative(
#     test_bins_long,
#     test_kin_long,
#     test_timestamps_long,
#     test_epochs_long,
#     test_trials_long,
#     test_labels_long,
#     palette,
#     to_plot
# )

#%%
# Just show kinematics with phases
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
non_nan = ~np.isnan(train_kin)
ax = sns.violinplot(train_kin, ax=ax)

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
from data_demos.filtering import smooth

palette = [*sns.color_palette('rocket', n_colors=3), *sns.color_palette('viridis', n_colors=3), 'k']
ax = plt.gca()
DEFAULT_TARGET_SMOOTH_MS = 490
KERNEL_SIZE = int(DEFAULT_TARGET_SMOOTH_MS / BIN_SIZE_MS)
KERNEL_SIGMA = DEFAULT_TARGET_SMOOTH_MS / (3 * BIN_SIZE_MS)
def create_targets(kin: np.ndarray):
    kin = smooth(kin, KERNEL_SIZE, KERNEL_SIGMA)
    out = np.gradient(kin, axis=0)
    return out

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
from data_demos.filtering import apply_exponential_filter
filtered_signal = apply_exponential_filter(train_bins)
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
        blacklist: np.ndarray | None=None,
        history: int=0,
        lag: int=0,
        ):
    signal = apply_exponential_filter(binned_spikes)
    targets = create_targets(behavior)

    # Remove timepoints where nothing is happening in the kinematics
    still_times = np.all(np.abs(targets) < 0.001, axis=1)
    if blacklist is not None:
        blacklist = still_times | blacklist
    else:
        blacklist = still_times

    train_x, test_x = np.split(signal, [int(TRAIN_TEST[0] * signal.shape[0])])
    train_y, test_y = np.split(targets, [int(TRAIN_TEST[0] * targets.shape[0])])
    train_blacklist, test_blacklist = np.split(blacklist, [int(TRAIN_TEST[0] * blacklist.shape[0])])

    x_mean, x_std = np.nanmean(train_x, axis=0), np.nanstd(train_x, axis=0)
    x_std[x_std == 0] = 1
    y_mean, y_std = np.nanmean(train_y[~train_blacklist], axis=0), np.nanstd(train_y[~train_blacklist], axis=0)
    y_std[y_std == 0] = 1
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std # don't standardize y if using var weighted r2
    test_y = (test_y - y_mean) / y_std

    train_blacklist = train_blacklist | np.isnan(train_y).any(axis=1)
    test_blacklist = test_blacklist | np.isnan(test_y).any(axis=1)
    if np.any(train_blacklist):
        print(f"Invalidating {np.sum(train_blacklist)} timepoints in train")
    if np.any(test_blacklist):
        print(f"Invalidating {np.sum(test_blacklist)} timepoints in test")

    # ? Where are the NaNs? We can't trivially lag if there are NaNs across data...
    if lag > 0:
        assert False, "not implemented"
        train_x, train_y = apply_neural_behavioral_lag(train_x, train_y, lag)
        test_x, test_y = apply_neural_behavioral_lag(test_x, test_y, lag)

    if history > 0:
        train_x = generate_lagged_matrix(train_x, history)
        test_x = generate_lagged_matrix(test_x, history)
        train_y = train_y[history:]
        test_y = test_y[history:]
        if blacklist is not None:
            train_blacklist = train_blacklist[history:]
            test_blacklist = test_blacklist[history:]

    # Now, finally, remove by blacklist
    train_x = train_x[~train_blacklist]
    train_y = train_y[~train_blacklist]
    test_x = test_x[~test_blacklist]
    test_y = test_y[~test_blacklist]

    return train_x, train_y, test_x, test_y, x_mean, x_std, y_mean, y_std

# HISTORY = 0
HISTORY = 5

(
    train_x,
    train_y,
    test_x,
    test_y,
    x_mean,
    x_std,
    y_mean,
    y_std
) = prepare_train_test(train_bins, train_kin, train_blacklist, history=HISTORY)

score, decoder = fit_and_eval_decoder(train_x, train_y, test_x, test_y)
print(f"CV Score: {score:.2f}")

# Same-day eval
pred_y = decoder.predict(test_x)
train_pred_y = decoder.predict(train_x)

r2 = r2_score(test_y, pred_y, multioutput='raw_values')
r2_weighted = r2_score(test_y, pred_y, multioutput='variance_weighted')
r2_uniform = r2_score(test_y, pred_y, multioutput='uniform_average')
train_r2 = r2_score(train_y, train_pred_y, multioutput='variance_weighted') #multioutput='raw_values')
print(f"Val R2 Weighted: {r2_weighted:.3f}")
print(f"Val R2 Uniform: {r2_uniform:.3f}")
print(f"Train: {train_r2:.3f}")

palette = sns.color_palette(n_colors=train_kin.shape[1])
f, axes = plt.subplots(train_kin.shape[1], figsize=(6, 12), sharex=True)
# Plot true vs predictions
for idx, (true, pred) in enumerate(zip(test_y.T, pred_y.T)):
    axes[idx].plot(true, label=f'{idx}', color=palette[idx])
    axes[idx].plot(pred, linestyle='--', color=palette[idx])
    axes[idx].set_title(f"{train_labels[idx]} $R^2$: {r2[idx]:.2f}")
    xticks = axes[idx].get_xticks()
    axes[idx].set_xticks(xticks)
    axes[idx].set_xticklabels([f'{x/1000 * BIN_SIZE_MS:.0f}' for x in xticks])
axes[-1].set_xlabel('Time (s)')
f.suptitle(f'Val $R^2$: {score:.2f}')
f.tight_layout()

#%%
def prepare_test(
        binned_spikes: np.ndarray,
        behavior: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        use_local_x_stats: bool = True, # Minimal adaptation is to zscore with local statistics
        history: int=0,
        blacklist: np.ndarray | None=None,
        ):
    signal = apply_exponential_filter(binned_spikes)
    targets = create_targets(behavior)

    # Remove timepoints where nothing is happening in the kinematics
    still_times = np.all(np.abs(targets) < 0.001, axis=1)
    if blacklist is not None:
        blacklist = still_times | blacklist
    else:
        blacklist = still_times

    if use_local_x_stats:
        x_mean = np.nanmean(signal[~blacklist], axis=0)
        x_std = np.nanstd(signal[~blacklist], axis=0)
        x_std[x_std == 0] = 1
    signal = (signal - x_mean) / x_std
    targets = (targets - y_mean) / y_std

    is_nan_y = np.isnan(targets).any(axis=1)
    if np.any(is_nan_y):
        print(f"NaNs found in test_y, removing {np.sum(is_nan_y)} timepoints")
        blacklist = blacklist | is_nan_y

    if history > 0:
        signal = generate_lagged_matrix(signal, history)
        targets = targets[history:]
        blacklist = blacklist[history:]

    signal = signal[~blacklist]
    targets = targets[~blacklist]

    return signal, targets

# Multi-day eval
x_short, y_short = prepare_test(
    test_bins_short,
    test_kin_short,
    x_mean,
    x_std,
    y_mean,
    y_std,
    history=HISTORY,
    blacklist=test_blacklist_short
    )
x_long, y_long = prepare_test(
    test_bins_long,
    test_kin_long,
    x_mean,
    x_std,
    y_mean,
    y_std,
    history=HISTORY,
    blacklist=test_blacklist_long
    )

#%%
r2_uniform_short = r2_score(y_short, decoder.predict(x_short), multioutput='uniform_average')
score_short = decoder.score(x_short, y_short)
score_long = decoder.score(x_long, y_long)
print(f"Short Zero-shot: {score_short:.2f}")
print(f"Short Zero-shot: {r2_uniform_short:.2f}")
print(f"Long Zero-shot: {score_long:.2f}")
#%%
# Save decoder for use in sklearn_decoder example agent
import pickle
with open(f'data/sklearn_h1.pkl', 'wb') as f:
    pickle.dump({
        'decoder': decoder,
        'task': FalconTask.h1,
        'history': HISTORY,
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }, f)