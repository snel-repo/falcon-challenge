#%%
# Self-contained sanity + decoding quality check in a given dataset
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
from pynwb import NWBHDF5IO
from data_demos.styleguide import set_style
set_style()

data_dir = Path("data/h1/")

train_query = 'train'
test_query_short = 'test_short'
test_query_long = 'test_long'

def get_files(query):
    if 'test' in query:
        return sorted(list((data_dir / query).glob('*full.nwb')))
    return sorted(list((data_dir / query).glob('*.nwb')))

train_files = get_files(train_query)
train_files.extend(get_files(test_query_short))
train_files.extend(get_files(test_query_long))
sample_files = train_files
def get_start_date_and_volume(fn: Path):
    with NWBHDF5IO(fn, 'r') as io:
        print(fn)
        nwbfile = io.read()
        start_date = nwbfile.session_start_time.strftime('%Y-%m-%d') # full datetime to just date
        return pd.to_datetime(start_date), nwbfile.acquisition['OpenLoopKinematics'].timestamps.shape[0]
start_dates, volume = zip(*[get_start_date_and_volume(fn) for fn in sample_files])

# Convert to pandas dataframe for easier manipulation
df = pd.DataFrame({'Date': start_dates, 'Dataset Size': volume})

fig, ax = plt.subplots()

# Scatter plot for visualizing each start date on the same y-value
sns.barplot(x='Date', y='Dataset Size', data=df, ax=ax, order=df['Date'].sort_values(), estimator=np.sum, errorbar=None)

fig.autofmt_xdate()  # Rotate dates for readability
day_unique = set([f.name.split('_')[0] for f in sample_files])

#%%
from typing import List, Tuple
from falcon_challenge.dataloaders import bin_units
from decoder_demos.filtering import smooth
from decoder_demos.filtering import apply_exponential_filter

from sklearn.metrics import r2_score
from decoder_demos.decoding_utils import (
    TRAIN_TEST,
    generate_lagged_matrix,
    fit_and_eval_decoder,
)

# Batch load all data for subsequent cells
def load_nwb(fn: str):
    r"""
        Load NWB for H1.
    """
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematics'].data[:]
        velocity = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].timestamps[:]
        blacklist = nwbfile.acquisition['Blacklist'].data[:].astype(bool)
        epochs = nwbfile.epochs.to_dataframe()
        trials = nwbfile.acquisition['TrialNum'].data[:]
        labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        return (
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            velocity,
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

    binned, kin, velocity, timestamps, blacklist, epochs, trials, labels = zip(*[load_nwb(str(f)) for f in files])
    lengths = [binned.shape[0] for binned in binned]
    binned = np.concatenate(binned, axis=0)
    kin = np.concatenate(kin, axis=0)
    velocity = np.concatenate(velocity, axis=0)

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
    return binned, kin, velocity, timestamps, blacklist, epochs, trials, labels[0], lengths

for subset_query in day_unique:
    train_files = [t for t in sample_files if subset_query in str(t)]

    binned_neural, all_kin, all_velocity, all_timestamps, all_blacklist, all_epochs, all_trials, all_labels, lengths = load_files(sample_files)
    BIN_SIZE_S = all_timestamps[1] - all_timestamps[0]
    BIN_SIZE_MS = BIN_SIZE_S * 1000
    print(f"Bin size = {BIN_SIZE_S} s")
    print(f"Neural data ({len(lengths)} days) of shape T={binned_neural.shape[0]}, N={binned_neural.shape[1]}")

    train_bins, train_kin, train_velocity, train_timestamps, train_blacklist, train_epochs, train_trials, train_labels, _ = load_files(train_files)

    all_tags = [tag for sublist in train_epochs['tags'] for tag in sublist]
    unique_tags = list(set(all_tags))

    def rasterplot(spike_arr: np.ndarray, bin_size_s=BIN_SIZE_S, ax=None):
        if ax is None:
            ax = plt.gca()
        for idx, unit in enumerate(spike_arr.T):
            ax.scatter(np.where(unit)[0] * bin_size_s, np.ones(np.sum(unit != 0)) * idx, s=1, c='k', marker='|', linewidths=0.2, alpha=0.3)
        ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron #')

    def kinplot(kin, timestamps, ax=None, palette=None, reference_labels=[], to_plot=to_plot, **kwargs):
        if ax is None:
            ax = plt.gca()

        if palette is None:
            palette = plt.cm.viridis(np.linspace(0, 1, len(reference_labels)))
        for kin_label in to_plot:
            kin_idx = reference_labels.index(kin_label)
            ax.plot(timestamps, kin[:, kin_idx], color=palette[kin_idx], **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Kinematics')

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
        trial_changept = np.where(np.diff(trials) != 0)[0]
        for changept in trial_changept:
            ax2.axvline(timestamps[changept], color='k', linestyle='-', alpha=0.1)
        fig.suptitle(f'DoF: {labels}')
        fig.tight_layout()
        return fig, (ax1, ax2)

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
    plt.savefig(f'data_demos/outputs/h1_{subset_query}_qual.png')
    plt.clf()


    DEFAULT_TARGET_SMOOTH_MS = 490
    KERNEL_SIZE = int(DEFAULT_TARGET_SMOOTH_MS / BIN_SIZE_MS)
    KERNEL_SIGMA = DEFAULT_TARGET_SMOOTH_MS / (3 * BIN_SIZE_MS)
    palette = [*sns.color_palette('rocket', n_colors=3), *sns.color_palette('viridis', n_colors=3), 'k']

    epoch_palette = sns.color_palette(n_colors=len(unique_tags))

    # Mute colors of "Presentation" phases
    for idx, tag in enumerate(unique_tags):
        if 'Presentation' in tag:
            epoch_palette[idx] = (0.9, 0.9, 0.9, 0.1)
        else:
            # white
            epoch_palette[idx] = (1, 1, 1, 0.5)

    def epoch_annote(epochs, ax=None):
        if ax is None:
            ax = plt.gca()
        for _, epoch in epochs.iterrows():
            epoch_idx = unique_tags.index(epoch.tags[0])
            ax.axvspan(epoch['start_time'], epoch['stop_time'], color=epoch_palette[epoch_idx], alpha=0.5)

    def rasterplot(spike_arr: np.ndarray, bin_size_s=BIN_SIZE_S, ax=None):
        if ax is None:
            ax = plt.gca()
        for idx, unit in enumerate(spike_arr.T):
            ax.scatter(np.where(unit)[0] * bin_size_s, np.ones(np.sum(unit != 0)) * idx, s=1, c='k', marker='|', linewidths=0.2, alpha=0.3)
        ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron #')

    to_plot = train_labels

    def kinplot(kin, timestamps, ax=None, palette=None, reference_labels=[], to_plot=to_plot, **kwargs):
        if ax is None:
            ax = plt.gca()

        if palette is None:
            palette = plt.cm.viridis(np.linspace(0, 1, len(reference_labels)))
        for kin_label in to_plot:
            kin_idx = reference_labels.index(kin_label)
            ax.plot(timestamps, kin[:, kin_idx], color=palette[kin_idx], **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Kinematics')

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    def plot_kin_pos(kin, timestamps, labels, epochs, ax, offset_y=0.1, offset_a=0.1):
        kinplot(kin + offset_y, timestamps, palette=palette, ax=ax, reference_labels=labels, alpha=offset_a)
        kinplot(smooth(kin, KERNEL_SIZE, KERNEL_SIGMA), timestamps, ax=ax, palette=palette, reference_labels=labels) # Offset for visual clarity
        ax.set_ylabel('Position')
        epoch_annote(epochs, ax=ax)
    # plot_kin_pos(sample_kin, sample_timestamps, sample_labels, sample_epochs, ax=axes[0])
    plot_kin_pos(train_kin, train_timestamps, train_labels, train_epochs, ax=axes[0])
    plt.suptitle(f'{train_query} (DoF: {len(train_labels)})')
    plt.tight_layout()

    def create_targets(kin: np.ndarray, target_smooth_ms=DEFAULT_TARGET_SMOOTH_MS, bin_size_ms=BIN_SIZE_MS, sigma=3):
        kernel_size = int(target_smooth_ms / bin_size_ms)
        kernel_sigma = target_smooth_ms / (sigma * bin_size_ms)
        kin = smooth(kin, kernel_size, kernel_sigma)
        out = np.gradient(kin, axis=0)
        return out

    # velocity = create_targets(sample_kin, bin_size_ms=BIN_SIZE_MS)  # Simple velocity estimate
    # kinplot(velocity, sample_timestamps, ax=axes[1], palette=palette, reference_labels=sample_labels)
    # epoch_annote(sample_epochs, ax=axes[1])
    velocity = create_targets(train_kin, bin_size_ms=BIN_SIZE_MS)  # Simple velocity estimate
    kinplot(velocity, train_timestamps, ax=axes[1], palette=palette, reference_labels=train_labels)
    epoch_annote(train_epochs, ax=axes[1])


    kinplot(train_velocity, train_timestamps, ax=axes[1], palette=palette, reference_labels=train_labels, linestyle='-', alpha=0.5)

    # trial_changept = np.where(np.diff(sample_trials) != 0)[0]
    # for changept in trial_changept:
    #     axes[1].axvline(sample_timestamps[changept], color='k', linestyle='-', alpha=1.0)
    trial_changept = np.where(np.diff(train_trials) != 0)[0]
    for changept in trial_changept:
        axes[1].axvline(train_timestamps[changept], color='k', linestyle='-', alpha=1.0)

    # xticks = np.arange(120, 200, 10)
    xticks = np.arange(0, 60, 10)
    plt.xlim(xticks[0], xticks[-1])
    plt.xticks(xticks, labels=xticks.round(2))
    plt.savefig(f'data_demos/outputs/h1_{subset_query}_kin_qual.png')
    plt.clf()
    filtered_signal = apply_exponential_filter(train_bins)

    def prepare_train_test(
            binned_spikes: np.ndarray,
            behavior: np.ndarray,
            blacklist: np.ndarray | None=None,
            history: int=0,
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
        # Note closed form ridge regression doesn't benefit from target rescaling, and we want to evaluate in native scale, variance weighted
        # train_y = (train_y - y_mean) / y_std
        # test_y = (test_y - y_mean) / y_std

        train_blacklist = train_blacklist | np.isnan(train_y).any(axis=1)
        test_blacklist = test_blacklist | np.isnan(test_y).any(axis=1)
        if np.any(train_blacklist):
            print(f"Invalidating {np.sum(train_blacklist)} timepoints in train")
        if np.any(test_blacklist):
            print(f"Invalidating {np.sum(test_blacklist)} timepoints in test")

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

    HISTORY = 0
    # HISTORY = 5

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
    train_r2 = r2_score(train_y, train_pred_y, multioutput='variance_weighted')
    print(f"Val R2 Variance Weighted: {r2_weighted:.3f}")
    print(f"Train R2 Variance Weighted: {train_r2:.3f}")

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
    axes[-1].set_xlim(0, test_y.shape[0])
    axes[-1].set_xlabel('Time (s)')
    f.suptitle(f'Val $R^2$: {r2_weighted:.2f}')
    f.tight_layout()
    f.savefig(f'data_demos/outputs/h1_{subset_query}_decoding.png', bbox_inches='tight')