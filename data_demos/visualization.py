import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plot_split_bars(df, fig, ax):
    """
    Plot the dataset sizes for each split type (train, test) for each start date

    Args:
    - df (pd.DataFrame): DataFrame containing the dataset sizes for each split type
    - fig (plt.Figure): Figure to plot on
    - ax (plt.Axes): Axes to plot on
    """
    # Scatter plot for visualizing each start date on the same y-value
    palette = sns.color_palette(n_colors=2)
    sns.barplot(
        x='Date',
        y='Dataset Size (s)',
        data=df,
        ax=ax,
        order=df['Date'].sort_values(),
        estimator=np.sum,
        errorbar=None,
        hue='Split Type',
        palette=palette
    )
    split_label = ('Train', 'Test') if 'Train' in df['Split Type'].unique() else ('Held In', 'Held Out')
    train_ratio = len(df[df['Split Type'] == split_label[0]]) / len(df)
    train_center_of_mass = train_ratio / 2
    test_center_of_mass = (1 - train_ratio) / 2 + train_ratio
    ax.get_legend().remove()
    ax.text(train_center_of_mass, 0.3, split_label[0], color=palette[0], fontsize=36, transform=ax.transAxes, backgroundcolor='white', ha='center')
    ax.text(test_center_of_mass, 0.3, split_label[1], color=palette[1], fontsize=36, transform=ax.transAxes, backgroundcolor='white', ha='center')
    fig.autofmt_xdate()  # Rotate dates for readability

def find_time_ratio(datestr_start, datestr_mid, datestr_end, year='2010'):
    # Assume strs are %m-%d, year doesn't matter
    if year != None:
        datetime_start = pd.to_datetime(f'{year}-{datestr_start}')
        datetime_mid = pd.to_datetime(f'{year}-{datestr_mid}')
        datetime_end = pd.to_datetime(f'{year}-{datestr_end}')
    else: 
        datetime_start = pd.to_datetime(datestr_start)
        datetime_mid = pd.to_datetime(datestr_mid)
        datetime_end = pd.to_datetime(datestr_end)
    total_time = datetime_end - datetime_start
    return (datetime_mid - datetime_start) / total_time

def plot_timeline(ax, sections, year='2010'):
    """
    Plot a colorized timeline of the dataset splits

    Args:
    - ax (plt.Axes): Axes to plot on
    - sections (dict): Dictionary containing the start dates for each dataset split
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    palette = sns.color_palette(n_colors=len(sections))
    # Creating an inset for the timeline
    ax_inset = inset_axes(ax, width="50%", height="20%", loc='upper right', borderpad=1)

    # Plotting the colorized timeline in the inset
    markers = ['o', 's']
    for i, (section, dates) in enumerate(sections.items()):
        if year != None:
            dates = list(map(lambda md: pd.to_datetime(f'{year}-{md}'), dates))
        else:
            dates = list(map(lambda md: pd.to_datetime(md), dates))
        ax_inset.scatter(dates, [0] * len(dates), marker=markers[i], color=palette[i], s=40)

    # Configuring the inset
    ax_inset.set_yticks([])
    ax_inset.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax_inset.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_inset.xaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(ax=ax_inset, left=True, bottom=True, right=True, top=True)
    ax_inset.axis('off')
    ax_inset.annotate('', xy=(1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='black'),
                xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'))

    if 'Train' in sections:
        split_label = ('Train', 'Test')
    else:
        split_label = ('Held In', 'Held Out')

    train_ratio = find_time_ratio(sections[split_label[0]][0], sections[split_label[1]][0], sections[split_label[1]][-1], year=year)
    train_center_of_mass = train_ratio / 2
    test_center_of_mass = (1 - train_ratio) / 2 + train_ratio
    ax_inset.text(
        train_center_of_mass, 0.7, split_label[0],
        color=palette[0],
        fontsize=20,
        transform=ax_inset.transAxes,
        ha='center'
    )
    ax_inset.text(
        test_center_of_mass, 0.7, split_label[1],
        color=palette[1],
        fontsize=20,
        transform=ax_inset.transAxes,
        ha='center'
    )

def rasterplot(spike_arr, bin_size_s=0.02, ax=None, spike_alpha=0.3, lw=0.2, s=1):
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
            np.where(unit)[0] * bin_size_s,
            np.ones(np.sum(unit != 0)) * idx,
            s=s,
            c='k',
            marker='|',
            linewidths=lw,
            alpha=spike_alpha,
        )
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 20))
    ax.set_ylabel('Channel #')

def plot_firing_rate_distributions(lengths, binned_neural, start_dates, axes, bin_size_s=0.02):
    """
    Plot the firing rate distributions for each day

    Args:
    - lengths (list): Dictionary with values of the lengths of each day
    - binned_neural (np.ndarray): Array of shape (T, N) containing the binned neural data
    - start_dates (list): List of the start dates
    - axes (plt.Axes): Axes to plot on
    """
    # Plot mean firing rates in different days
    daily_means = []
    day_intervals = [0, *np.cumsum([lengths[x] for x in lengths])]
    for start, end in zip(day_intervals[:-1], day_intervals[1:]):
        daily_means.append(np.mean(binned_neural[start:end], axis=0) / bin_size_s)
    # daily_means is a list of 8 arrays, each of shape (N,) - histplot them separately using sns and pd
    flattened_data = {'Day': [], 'Mean Firing Rate (Hz)': []}
    for day, means in zip(start_dates, daily_means):
        flattened_data['Day'].extend([day.strftime('%m-%d')] * len(means))  # Extend day labels - reformat for simplicity
        flattened_data['Mean Firing Rate (Hz)'].extend(means)  # Extend mean firing rates

    axes = sns.histplot(data=flattened_data, x='Mean Firing Rate (Hz)', hue='Day', kde=True, ax=axes, palette='viridis', bins=20, stat='percent', multiple='dodge')
    axes.set_title('Firing Rate Distribution per day', ha='center', va='center', transform=axes.transAxes, fontsize=18)
