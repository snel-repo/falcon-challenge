#%%
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime
from styleguide import set_style
set_style()
from pynwb import NWBHDF5IO

#%% 

base_path = '/snel/home/bkarpo2/bin/falcon-challenge/data'
track = 'b1'
if track != 'b1': 
    held_in_files = glob.glob(os.path.join(base_path, track, '*held-in-calib*', '*.nwb'))
    held_out_files = glob.glob(os.path.join(base_path, track, '*held-out-calib*', '*.nwb'))

#%% 
if track == 'm1':
    held_in_dates = [re.search(r"\d{8}", d).group() for d in held_in_files]
    held_out_dates = [re.search(r"\d{8}", d).group() for d in held_out_files]
    format_held_in_dates = sorted([datetime.strptime(date, '%Y%m%d') for date in held_in_dates])
    format_held_out_dates = sorted([datetime.strptime(date, '%Y%m%d') for date in held_out_dates])
elif track == 'h2': 
    held_in_dates = [re.search(r"\d{4}.\d{2}.\d{2}", d).group() for d in held_in_files]
    held_out_dates = [re.search(r"\d{4}.\d{2}.\d{2}", d).group() for d in held_out_files]
    format_held_in_dates = sorted([datetime.strptime(date, '%Y.%m.%d') for date in held_in_dates])
    format_held_out_dates = sorted([datetime.strptime(date, '%Y.%m.%d') for date in held_out_dates])
elif track == 'm2': 
    held_in_dates = [re.search(r"\d{4}-\d{2}-\d{2}", d).group() for d in held_in_files]
    held_out_dates = [re.search(r"\d{4}-\d{2}-\d{2}", d).group() for d in held_out_files]
    format_held_in_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in held_in_dates])
    format_held_out_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in held_out_dates])
elif track == 'h1':
    format_held_in_dates = []
    format_held_out_dates = []
    for f in held_in_files: 
        with NWBHDF5IO(f, 'r') as io:
            nwbfile = io.read()
            format_held_in_dates.append(datetime.strptime(nwbfile.session_start_time.strftime('%Y-%m-%d'), '%Y-%m-%d'))
    for f in held_out_files:
        with NWBHDF5IO(f, 'r') as io:
            nwbfile = io.read()
            format_held_out_dates.append(datetime.strptime(nwbfile.session_start_time.strftime('%Y-%m-%d'), '%Y-%m-%d'))
    format_held_in_dates = sorted(format_held_in_dates)
    format_held_out_dates = sorted(format_held_out_dates)
elif track == 'b1': 
    format_held_in_dates = [
        datetime(2021,6,26),
        datetime(2021,6,27),
        datetime(2021,6,28),
        datetime(2021,6,29),
        datetime(2021,6,30),
    ]
    format_held_out_dates = [
        datetime(2021,7,1),
        datetime(2021,7,4),
        datetime(2021,7,5),
    ]
    

#%% 
# Some corresponding y values
held_in_y = np.zeros(len(format_held_in_dates))
held_out_y = np.zeros(len(format_held_out_dates))

plt.figure(figsize=(1.25, 0.5))

# Plot the dates on the x-axis and y values on the y-axis
plt.plot_date(format_held_in_dates, held_in_y, 'o', color='k', markersize=4, zorder=1000)
plt.plot_date(format_held_out_dates, held_out_y, 's', color='lightseagreen', markersize=4, zorder=1000)

if track != 'h2': 
    for d in format_held_in_dates + format_held_out_dates: 
        plt.text(d, -0.025, d.strftime('%m/%d'), ha='center', fontsize=5, rotation=90)
else: 
    for ii, d in enumerate(format_held_in_dates):
        if ii % 2 == 0: 
            plt.text(d, -0.025, d.strftime('%m/%d'), ha='center', fontsize=5, rotation=90)
        else: 
            plt.text(d, 0.01, d.strftime('%m/%d'), ha='center', fontsize=5, rotation=90)
    for d in format_held_out_dates:
        plt.text(d, -0.025, d.strftime('%m/%d'), ha='center', fontsize=5, rotation=90)

# Improve the layout
plt.gcf().autofmt_xdate()

# Set the date format
date_format = mdates.DateFormatter('%m-%d')
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(len(y)))

plt.gca().axis('off')
# plt.gca().spines['left'].set_visible(False)
# plt.gca().set_yticks([])
plt.gca().annotate('', xy=(1, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color='black'),
            xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'))

plt.tight_layout()
plt.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig', f'{track}_timeline.pdf'), dpi=300)
# plt.show()

# %%
# plt.figure(figsize=(0.5, 1.5))
plt.figure(figsize=(1.5, 4.5))
# plt.gca().annotate('', xy=(0, 1), xytext=(0, 0), 
#                    xycoords='axes fraction', textcoords='offset points',
#                    arrowprops=dict(arrowstyle="->", color='k'))

held_in_dates_num = mdates.date2num(format_held_in_dates)
held_out_dates_num = mdates.date2num(format_held_out_dates)

plt.plot(held_in_y, held_in_dates_num, '_', color='k', markersize=10, zorder=1000)
plt.plot(held_out_y, held_out_dates_num, '_', color='teal', markersize=10, zorder=1000)
plt.arrow(
    0, np.min(held_in_dates_num) - 5, 
    0, np.max(held_out_dates_num) - np.min(held_in_dates_num) + 10, 
    color='gray',
    width=0.00001,
    head_width=0.01,
    head_length=1,
    overhang=0)
plt.gca().axis('off')
plt.gca().invert_yaxis()
plt.ylim(np.max(held_out_dates_num) + 6, np.min(held_in_dates_num) - 6)
plt.xlim(-0.05,0.05)

held_in_span = (format_held_in_dates[-1] - format_held_in_dates[0]).days
between_span = (format_held_out_dates[0] - format_held_in_dates[-1]).days
held_out_span = (format_held_out_dates[-1] - format_held_out_dates[0]).days

plt.vlines(0.01, held_in_dates_num[0], held_in_dates_num[-1], color='k', linestyle='-', linewidth=0.75)
plt.vlines(0.01, held_out_dates_num[0], held_out_dates_num[-1], color='teal', linestyle='-', linewidth=0.75)
plt.vlines(-0.01, held_in_dates_num[-1], held_out_dates_num[0], color='gray', linestyle='-', linewidth=0.75)

plt.text(0.015, (held_in_dates_num[0] + held_in_dates_num[-1])/2, f'{held_in_span} days', ha='left', va='center', fontsize=10, rotation=270)
plt.text(0.015, (held_out_dates_num[0] + held_out_dates_num[-1])/2, f'{held_out_span} days', ha='left', fontsize=10, rotation=270, va='center')
plt.text(-0.015, (held_out_dates_num[0] + held_in_dates_num[-1])/2, f'{between_span} days', ha='right', fontsize=10, rotation=90, va='center')
plt.text(0.015, np.max(held_out_dates_num) + 4, 't', fontsize=10, fontstyle='italic')
plt.savefig(os.path.join('/snel/home/bkarpo2/projects/falcon_figs/datasets_fig', f'{track}_timeline2.pdf'), dpi=300)

# %%
