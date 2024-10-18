#%%
import os, glob, re, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime
from styleguide import set_style
set_style()
from pynwb import NWBHDF5IO
from falcon_challenge.evaluator import DATASET_HELDINOUT_MAP

#%% 

# oracle_results = '/snel/share/runs/falcon/linear_baseline_scores/sweep_results_h1_m1_m2.pkl'
results = '/snel/share/runs/falcon/linear_baseline_scores/baseline_scores_all.pkl'
base_path = '/snel/home/bkarpo2/bin/falcon-challenge/data'
tracks = ['b1'] #, 
# tracks = ['h2']
# history = [7, 30, 30]
static_decoder = {
    'h1': 'S4',
    'm1': '20120927',
    'm2': '20201020',
    'h2': 'all',
    'h1-rnn': 'S5',
    'm1-rnn': '20120926',
    'm2-rnn': '20201028',
    'b1': '2021.06.27'
}

b1_dates = [
    '2021.06.26',
    '2021.06.27',
    '2021.06.28',
    '2021.06.30',
    '2021.07.01',
    '2021.07.05',
]

b1_results = {
    'oracle': [
               0.0006172111129,
               0.0005180985214,
               0.000546235208,
               0.0008874613933,
               0.0006895604189,
               0.0006643953386
            ],
    'static': [
        0.001652644805,
        0.002318245041,
        0.002582313362
    ]
}

#%% 

with open(results, 'rb') as f: 
    results = pickle.load(f)

# read the h2 csv using pandas 
h2_csv = '/home/bkarpo2/bin/falcon-challenge/manuscript/results/h2_model_eval.csv'
sess_cers_df = pd.read_csv(h2_csv)

#%% 
fig, ax = plt.subplots(1, 10, figsize=(6.25, 1), sharey=False)
axs = [[ax[0], ax[1]], [ax[2], ax[3]], [ax[4], ax[5]], [ax[6], ax[7]], [ax[8], ax[9]]]
for i, track in enumerate(tracks): 

    if track != 'h2' and track != 'b1':
        rnn_file = f'/snel/home/bkarpo2/bin/falcon-challenge/manuscript/results/{track}_rnn_results2.pkl'
        with open(rnn_file, 'rb') as f: 
            rnn_results = pickle.load(f)
    if track != 'b1':
        held_in_files = glob.glob(os.path.join(base_path, track, '*held-in-calib*', '*.nwb'))
        held_out_files = glob.glob(os.path.join(base_path, track, '*held-out-calib*', '*.nwb'))

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
    elif track == 'h2': 
        held_in_dates = [re.search(r"\d{4}.\d{2}.\d{2}", d).group() for d in held_in_files]
        held_out_dates = [re.search(r"\d{4}.\d{2}.\d{2}", d).group() for d in held_out_files]
        format_held_in_dates = sorted([datetime.strptime(date, '%Y.%m.%d') for date in held_in_dates])
        format_held_out_dates = sorted([datetime.strptime(date, '%Y.%m.%d') for date in held_out_dates])
    elif track == 'b1': 
        held_in_dates = b1_dates[:3]
        held_out_dates = b1_dates[3:]
        format_held_in_dates = np.array([datetime.strptime(date, '%Y.%m.%d') for date in held_in_dates])
        format_held_out_dates = np.array([datetime.strptime(date, '%Y.%m.%d') for date in held_out_dates])
        held_in_days = np.array([x.days for x in format_held_in_dates - format_held_in_dates[0]])
        held_out_days = np.array([x.days for x in format_held_out_dates - format_held_in_dates[0]])

    if track != 'b1': 
        # all_dates = np.unique(format_held_in_dates + format_held_out_dates)
        held_in_dates = np.unique(format_held_in_dates)
        held_out_dates = np.unique(format_held_out_dates)
        held_in_days = np.array([x.days for x in held_in_dates - held_in_dates[0]])
        held_out_days = np.array([x.days for x in held_out_dates - held_in_dates[0]])

        held_in_keys = DATASET_HELDINOUT_MAP[track]['held_in']
        held_out_keys = DATASET_HELDINOUT_MAP[track]['held_out']
        held_in_sort_idx = np.argsort(held_in_keys)
        held_out_sort_idx = np.argsort(held_out_keys)
        if track == 'h1':
            held_in_keys = np.unique([k.split('_')[0] for k in held_in_keys])
            held_out_keys = np.unique([k.split('_')[0] for k in held_out_keys])
            held_in_sort_idx = np.argsort([int(k.strip('S')) for k in held_in_keys])
            held_out_sort_idx = np.argsort([int(k.strip('S')) for k in held_out_keys])
        elif track == 'm2': 
            held_in_keys = np.unique([k.split('_')[1] for k in held_in_keys])
            held_out_keys = np.unique([k.split('_')[1] for k in held_out_keys])
            held_in_sort_idx = np.argsort([int(k) for k in held_in_keys])
            held_out_sort_idx = np.argsort([int(k) for k in held_out_keys])

    if track == 'h2': 
        held_in_static =  sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib') & (sess_cers_df.test_data == 'held_in_eval')]['rnn_wer_mean']
        held_out_static = sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib') & (sess_cers_df.test_data == 'held_out_eval')].groupby('seed')['rnn_wer_mean']
        held_in_oracle = held_in_static
        held_out_oracle = sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib+held_out_oracle') & (sess_cers_df.test_data == 'held_out_eval')].groupby('seed')['rnn_wer_mean']

        axs[i][0].hlines(held_in_days[0], held_in_days[-1], held_in_oracle.mean(), color='b', linestyle='--', linewidth=0.5)
        axs[i][0].fill_between(held_in_days, held_in_oracle.mean() - held_in_oracle.std(), held_in_oracle.mean() + held_in_oracle.std(), color='k', alpha=0.3)
        axs[i][0].hlines(held_in_days[0], held_in_days[-1], held_in_static.mean(), color='k', linestyle='--', linewidth=0.5)
        axs[i][0].fill_between(held_in_days, held_in_static.mean() - held_in_static.std(), held_in_static.mean() + held_in_static.std(), color='b', alpha=0.3)
        axs[i][0].grid(alpha=0.3)
        axs[i][0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
        axs[i][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, pad=-0.5)
        # set x-ticks to be held_in_days 
        axs[i][0].set_xticks(np.arange(0, max(held_in_days) + 1, max(held_in_days)//2))
        axs[i][0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        axs[i][0].set_ylim([0., 0.8])

        axs[i][1].errorbar(held_out_days, held_out_oracle.mean(), yerr=held_out_oracle.std(), fmt='o-', color='b', linewidth=0.5, markersize=3)
        axs[i][1].errorbar(held_out_days, held_out_static.mean(), yerr=held_out_static.std(), fmt='o-', color='k', linewidth=0.5, markersize=3)
        axs[i][1].grid(alpha=0.3)
        axs[i][1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
        axs[i][1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs[i][1].spines['left'].set_visible(False)
        axs[i][1].set_xticks(np.arange(min(held_out_days), max(held_out_days) + 1, (max(held_out_days) - min(held_out_days))//2))
        axs[i][0].set_title(f'{track.upper()} held-in')
        axs[i][1].set_title(f'{track.upper()} held-out')
        axs[i][1].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        axs[i][1].set_ylim([0., 0.8])

        print(f'{track.upper()} static start: {held_in_static.mean()}')
        print(f'{track.upper()} static max diff: {np.max(held_in_static.mean() - held_out_static.mean())}')
        print(f'{track.upper()} Held-In Oracle Summary: {held_in_oracle.mean():.2f}')
        print(f'{track.upper()} Held-Out Oracle Summary: {held_out_oracle.mean().mean():.2f} +/- {held_out_oracle.mean().std():.2f}')
        print(f'{track.upper()} Held-In Static Summary: {held_in_static.mean():.2f}')
        print(f'{track.upper()} Held-Out Static Summary: {held_out_static.mean().mean():.2f} +/- {held_out_static.mean().std():.2f}')
    elif track == 'b1': 
        held_in_oracle = b1_results['oracle'][:3]
        held_out_oracle = b1_results['oracle'][3:]
        held_in_static_idx = np.where(np.array(b1_dates) == static_decoder['b1'])[0][0]
        held_in_static = b1_results['oracle'][held_in_static_idx]
        held_out_static = b1_results['static']

        axs[i][0].plot(held_in_days, held_in_oracle, 'x', color='b', linewidth=0.5, markersize=5)
        axs[i][0].plot(held_in_days[held_in_static_idx], held_in_static, 'o', color='k', linewidth=0.5, markersize=3, alpha=0.8)
        axs[i][0].set_ylim(0, 0.003)
        axs[i][0].grid(alpha=0.3)
        axs[i][0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
        axs[i][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True, pad=-0.5)
        # set x-ticks to be held_in_days 
        axs[i][0].set_xticks(np.arange(0, max(held_in_days) + 1, max(held_in_days)//2))
        axs[i][0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        axs[i][1].plot(held_out_days, held_out_oracle, 'o-', color='b', linewidth=0.5, markersize=3, alpha=0.75)
        axs[i][1].plot(held_out_days, held_out_static, 'o-', color='k', linewidth=0.5, markersize=3, alpha=0.75)
        axs[i][1].grid(alpha=0.3)
        axs[i][1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
        axs[i][1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs[i][1].spines['left'].set_visible(False)
        axs[i][1].set_xticks(np.arange(min(held_out_days), max(held_out_days) + 1, (max(held_out_days) - min(held_out_days))//2))
        axs[i][0].set_title(f'{track.upper()} held-in')
        axs[i][1].set_title(f'{track.upper()} held-out')
        axs[i][1].set_ylim(0, 0.003)

        print(f'{track.upper()} static start: {held_in_static}')
        print(f'{track.upper()} static max diff: {np.max(held_in_static - np.array(held_out_static))}')
        print(f'{track.upper()} Held-In Oracle Summary: {np.mean(held_in_oracle):.2e} +/- {np.std(held_in_oracle):.2e}')
        print(f'{track.upper()} Held-Out Oracle Summary: {np.mean(held_out_oracle):.2e} +/- {np.std(held_out_oracle):.2e}')
        print(f'{track.upper()} Held-In Static Summary: {held_in_static:.2e}')
        print(f'{track.upper()} Held-Out Static Summary: {np.mean(held_out_static):.2e} +/- {np.std(held_out_static):.2e}')

    else: 
        static_vals = results[f'{track}_static']
        held_in_static = np.array([static_vals[k] for k in held_in_keys])
        held_out_static = np.array([static_vals[k] for k in held_out_keys])

        oracle_vals = results[f'{track}_oracle']
        held_in_oracle = np.array([oracle_vals[k] for k in held_in_keys])
        held_out_oracle = np.array([oracle_vals[k] for k in held_out_keys])

        seed_idx = np.where(np.array(held_in_keys) == static_decoder[track])[0][0]
        rnn_seed_ds = static_decoder[f'{track}-rnn']
        seed_idx_rnn = np.where(np.array(held_in_keys) == rnn_seed_ds)[0][0]

        rnn_static = np.array(rnn_results['static_performance']).squeeze()
        print(rnn_static)
        rnn_oracle = np.array(rnn_results['oracle_performance']).squeeze()

        axs[i][0].plot(held_in_days, held_in_oracle[held_in_sort_idx], 'x', color='b', linewidth=0.5, markersize=5)
        axs[i][0].plot(held_in_days[seed_idx], held_in_static[seed_idx], 'o', color='k', linewidth=0.5, markersize=3, alpha=0.8)
        axs[i][0].plot(held_in_days, rnn_oracle[:len(held_in_days)], 'x', color='r', linewidth=0.5, markersize=5)
        axs[i][0].plot(held_in_days[seed_idx_rnn], rnn_static[seed_idx_rnn], 'o', color='gray', linewidth=0.5, markersize=3, alpha=0.8)
        axs[i][0].grid(alpha=0.3)
        axs[i][0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
        axs[i][0].tick_params(axis='y', which='both', left=False, right=False, labelleft=i < 1, pad=-0.5)
        # set x-ticks to be held_in_days 
        axs[i][0].set_xticks(np.arange(0, max(held_in_days) + 1, max(held_in_days)//2))
        axs[i][0].set_ylim([0., 0.85])
        axs[i][0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        

        axs[i][1].plot(held_out_days, held_out_oracle[held_out_sort_idx], 'o-', color='b', linewidth=0.5, markersize=3, alpha=0.75)
        axs[i][1].plot(held_out_days, held_out_static[held_out_sort_idx], 'o-', color='k', linewidth=0.5, markersize=3, alpha=0.75)
        where_static_neg = np.where(held_out_static[held_out_sort_idx] < 0)[0]
        axs[i][1].plot(held_out_days[where_static_neg],np.ones(len(where_static_neg)) * 0.05, 'v', color='k', markersize=3, alpha=0.75)
        axs[i][1].plot(held_out_days, rnn_oracle[len(held_in_days):], 'o-', color='r', linewidth=0.5, markersize=3, alpha=0.75)
        axs[i][1].plot(held_out_days, rnn_static[len(held_in_days):], 'o-', color='gray', linewidth=0.5, markersize=3, alpha=0.75)
        where_rnn_neg = np.where(rnn_static[len(held_in_days):] < 0)[0]
        axs[i][1].plot(held_out_days[where_rnn_neg],np.ones(len(where_rnn_neg)) * 0.05, 'v', color='gray', markersize=3, alpha=0.75)
        # axs[i][1].set_ylim(0, 1)
        axs[i][1].grid(alpha=0.3)
        axs[i][1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
        axs[i][1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs[i][1].spines['left'].set_visible(False)
        axs[i][1].set_xticks(np.arange(min(held_out_days), max(held_out_days) + 1, (max(held_out_days) - min(held_out_days))//2))
        axs[i][0].set_title(f'{track.upper()} held-in')
        axs[i][1].set_title(f'{track.upper()} held-out')
        axs[i][1].set_ylim([0., 0.85])
        axs[i][1].set_yticks([0, 0.2, 0.4, 0.6, 0.8])

        print(f'{track.upper()} static start: {held_in_static[seed_idx]}')
        print(f'{track.upper()} static max diff: {np.max(held_in_static[seed_idx] - held_out_static)}')
        print(f'{track.upper()} static rnn max diff: {np.max(rnn_static[seed_idx_rnn] - rnn_static)}')
        print(f'{track.upper()} Held-In Oracle Summary: {np.mean(held_in_oracle):.2f} +/- {np.std(held_in_oracle):.2f}')
        print(f'{track.upper()} Held-Out Oracle Summary: {np.mean(held_out_oracle):.2f} +/- {np.std(held_out_oracle):.2f}')
        print(f'{track.upper()} Held-In Static Summary: {np.mean(held_in_static):.2f} +/- {np.std(held_in_static):.2f}')
        print(f'{track.upper()} Held-Out Static Summary: {np.mean(held_out_static):.2f} +/- {np.std(held_out_static):.2f}')

plt.savefig('/snel/home/bkarpo2/projects/falcon_figs/oracle_static_updb1.pdf', bbox_inches='tight')
# %%
# getting the numbers for oracle and static with the language model 
held_in_static =  sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib') & (sess_cers_df.test_data == 'held_in_eval')]['lm_wer_mean']
held_out_static = sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib') & (sess_cers_df.test_data == 'held_out_eval')].groupby('seed')['lm_wer_mean']
held_in_oracle = held_in_static
held_out_oracle = sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib+held_out_oracle') & (sess_cers_df.test_data == 'held_out_eval')].groupby('seed')['lm_wer_mean']

print(f'{track.upper()} static start + LM: {held_in_static.mean()}')
print(f'{track.upper()} static max diff + LM: {np.max(held_in_static.mean() - held_out_static.mean())}')
print(f'{track.upper()} Held-In Oracle Summary + LM: {held_in_oracle.mean():.2f}')
print(f'{track.upper()} Held-Out Oracle Summary + LM: {held_out_oracle.mean().mean():.2f} +/- {held_out_oracle.mean().std():.2f}')
print(f'{track.upper()} Held-In Static Summary + LM: {held_in_static.mean():.2f}')
print(f'{track.upper()} Held-Out Static Summary + LM: {held_out_static.mean().mean():.2f} +/- {held_out_static.mean().std():.2f}')

# %%
held_in_corp = sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib') & (sess_cers_df.test_data == 'held_in_eval')]['lm_wer_mean']
held_out_corp = sess_cers_df.loc[(sess_cers_df.train_data == 'held_in_calib+held_out_eval(CORP)') & (sess_cers_df.test_data == 'held_out_eval')].groupby('seed')['lm_wer_mean']

print(f'{track.upper()} Held-In CORP: {held_in_corp.mean():.2f}')
print(f'{track.upper()} Held-Out CORP: {held_out_corp.mean().mean():.2f} +/- {held_out_corp.mean().std():.2f}')

# %%
