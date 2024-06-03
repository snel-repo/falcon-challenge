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

held_in_results = {
    'ndt2': [0.744922239130974, 0.761236731408813, 0.7881243671158603, 0.8072260619126261],
    'nomad': [0.6205157241114031, 0.6484411273242108, 0.6348675663252332, 0.6589362043429672],
    'cyclegan': [0.58704366546381, 0.6063171981294609, 0.605259984986606, 0.6355977543036897],
    'static_wf': [0.37100878131669796, 0.4686046508083223, 0.539264382674297, 0.46664828970361366]
}
held_out_results = {
    'ndt2': [0.6355582331426868, 0.6578559607337571, 0.49440192306033404],
    'nomad': [0.5094512937850105, 0.5234957917131661, 0.44585368613994564],
    'cyclegan': [0.4795853623709667, 0.4033784511259925, 0.39898314642588134],
    'static_wf': [0.39259487067679766, 0.3762496700649608, 0.2612390566332373]
}

#%% 
base_path = '/snel/home/bkarpo2/bin/falcon-challenge/data/m1'

held_in_files = glob.glob(os.path.join(base_path, '*held-in-calib*', '*.nwb'))
held_out_files = glob.glob(os.path.join(base_path, '*held-out-calib*', '*.nwb'))
held_in_dates = [re.search(r"\d{8}", d).group() for d in held_in_files]
held_out_dates = [re.search(r"\d{8}", d).group() for d in held_out_files]
format_held_in_dates = sorted([datetime.strptime(date, '%Y%m%d') for date in held_in_dates])
format_held_out_dates = sorted([datetime.strptime(date, '%Y%m%d') for date in held_out_dates])

held_in_dates = np.unique(format_held_in_dates)
held_out_dates = np.unique(format_held_out_dates)
held_in_days = np.array([x.days for x in held_in_dates - held_in_dates[0]])
held_out_days = np.array([x.days for x in held_out_dates - held_in_dates[0]])

held_in_keys = DATASET_HELDINOUT_MAP[track]['held_in']
held_out_keys = DATASET_HELDINOUT_MAP[track]['held_out']
held_in_sort_idx = np.argsort(held_in_keys)
held_out_sort_idx = np.argsort(held_out_keys)

# %%
fig, axs = plt.subplots(1, 2, figsize=(1.5, 1.5), sharey=True)
colors = {
    'ndt2': '#648FFF',
    'nomad': '#DC267F',
    'cyclegan': '#FE6100',
    'static_wf': 'k'
}

for method in held_in_results.keys():
    held_in_r2 = np.array(held_in_results[method])
    held_out_r2 = np.array(held_out_results[method])

    axs[0].plot(held_in_days, held_in_r2, 'x', color=colors[method], linewidth=0.5, markersize=5, label=method)
    axs[0].set_ylim(0, 1)
    axs[0].grid(alpha=0.3)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True, pad=-0.5)
    axs[0].set_xticks(np.arange(0, max(held_in_days) + 1, max(held_in_days)//2))
    axs[0].set_title(f'held-in')
    axs[0].legend()

    axs[1].plot(held_out_days, held_out_r2, 'o-', color=colors[method], linewidth=0.5, markersize=3)
    axs[1].grid(alpha=0.3)
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
    axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axs[1].spines['left'].set_visible(False)
    axs[1].set_xticks(np.arange(min(held_out_days), max(held_out_days) + 1, (max(held_out_days) - min(held_out_days))//2))
    axs[1].set_title(f'held-out')

plt.savefig('/home/bkarpo2/projects/falcon_figs/fig4a.pdf', bbox_inches='tight')

#%% load a raw NWB file to get true EMG out 

f = '/home/bkarpo2/bin/falcon-challenge/data/m1/eval/L_20121024_held_out_eval.nwb'
from snel_toolkit.datasets.nwb import NWBDataset
# load and rebin the spikes 
ds_spk = NWBDataset(
    f, 
    skip_fields=['preprocessed_emg', 'eval_mask'])
ds_spk.resample(20)

ds = NWBDataset(f)
ds.data = ds.data.dropna()
ds.data.index = ds.data.index.round('20ms')
ds.data.spikes = ds_spk.data.loc[ds_spk.data.index >= ds.data.index[0]].spikes
ds.bin_width = 20

#%% 

muscles_to_plot = ['BCPs', 'FCR', 'EDC']
muscles = ds.data.preprocessed_emg.columns.values.tolist()
muscle_inds = [muscles.index(m) for m in muscles_to_plot]

# %% todo: load all the preds, merge back to the df 

preds = {}

preds_ndt2 = '/home/bkarpo2/bin/falcon-challenge/manuscript/results/preds_m1_ndt2.pkl'
with open(preds_ndt2, 'rb') as f: 
    preds['ndt2'] = pickle.load(f)

preds_nomad = '/home/bkarpo2/bin/falcon-challenge/manuscript/results/preds_m1_nomad.pkl'
with open(preds_nomad, 'rb') as f: 
    preds['nomad'] = pickle.load(f)

preds_cyclegan = '/home/bkarpo2/bin/falcon-challenge/manuscript/results/preds_m1_cgan.pkl'
with open(preds_cyclegan, 'rb') as f: 
    preds['cyclegan'] = pickle.load(f)

#%% 
eval_mask = ds.data.eval_mask.values.squeeze().astype('bool')

for k in preds: 
    full_res = np.full((len(ds.data), len(muscles)), np.nan)
    full_res[eval_mask, :] = preds[k]['20121024']

    # assign the array to the dataframe with multindex
    ds.data[[(f'{k}_pred', m) for m in muscles]] = full_res

#%% # make trialized data 
ds.trials = ds.make_trial_data(
    align_field='gocue_time',
    align_range=(0, 2000),
    ignored_trials=ds.trial_info['result'] != 'R' 
)

#%% 

pivot_trials = ds.trials.pivot(index='align_time', columns='trial_id')
trial_loc = ds.trial_info.tgt_loc.values
trial_obj = ds.trial_info.tgt_obj.values

#%% 

st_true_emg = np.array(np.split(pivot_trials.preprocessed_emg.values, len(muscles), axis=1))
st_ndt_emg = np.array(np.split(pivot_trials.ndt2_pred.values, len(muscles), axis=1))
st_nomad_emg = np.array(np.split(pivot_trials.nomad_pred.values, len(muscles), axis=1))
st_cyclegan_emg = np.array(np.split(pivot_trials.cyclegan_pred.values, len(muscles), axis=1))

#%% plot true muscle activity 
colors = {
    'ndt2': '#648FFF',
    'nomad': '#DC267F',
    'cyclegan': '#FE6100',
    'static_wf': 'k'
}

loc_to_plot = 22.5
obj_to_plot = 'Button'

to_plot_true_emg = [st_true_emg[i, :, (trial_loc == loc_to_plot) & (trial_obj == obj_to_plot)] for i in muscle_inds]
to_plot_ndt_emg = [st_ndt_emg[i, :, (trial_loc == loc_to_plot) & (trial_obj == obj_to_plot)] for i in muscle_inds]
to_plot_nomad_emg = [st_nomad_emg[i, :, (trial_loc == loc_to_plot) & (trial_obj == obj_to_plot)] for i in muscle_inds]
to_plot_cg_emg = [st_cyclegan_emg[i, :, (trial_loc == loc_to_plot) & (trial_obj == obj_to_plot)] for i in muscle_inds]

f, axes = plt.subplots(3, 3, figsize=(6,6), sharex=True, sharey=True) #todo: change to 2x2

for i in range(3):
    for j in range(3):
        axes[i, j].plot(np.array(to_plot_true_emg)[j, :, :].T, color='gray', alpha=0.5, linewidth=0.5)
        if i == 0: 
            axes[i, j].plot(np.array(to_plot_ndt_emg)[j, :, :].T, color=colors['ndt2'], alpha=0.5, linewidth=0.5)
        elif i == 1: 
            axes[i, j].plot(np.array(to_plot_nomad_emg)[j, :, :].T, color=colors['nomad'], alpha=0.5, linewidth=0.5)
        elif i == 2:
            axes[i, j].plot(np.array(to_plot_cg_emg)[j, :, :].T, color=colors['cyclegan'], alpha=0.5, linewidth=0.5)


# %%
f, axes = plt.subplots(9, 4, figsize=(6,6), sharex=True, sharey='row') #todo: change to 2x2
plt.subplots_adjust(wspace=0.05)
# plt.tight_layout()

trials_to_plot = [
    0, # sphere, 90
    100, # sphere, 67.5
    200, # button, 45
    300, # button, 22.5 
]

for i in range(3): # muscle
    for j in range(4): # trial 
        axes[i, j].plot(np.array(st_true_emg)[muscle_inds[i], :, trials_to_plot[j]].T, color='black', alpha=0.5, linewidth=1.5)
        axes[i + 3, j].plot(np.array(st_true_emg)[muscle_inds[i], :, trials_to_plot[j]].T, color='black', alpha=0.5, linewidth=1.5)
        axes[i + 6, j].plot(np.array(st_true_emg)[muscle_inds[i], :, trials_to_plot[j]].T, color='black', alpha=0.5, linewidth=1.5)

        axes[i, j].plot(np.array(st_ndt_emg)[muscle_inds[i], :, trials_to_plot[j]].T, color=colors['ndt2'], alpha=0.8, linewidth=1.5)
        axes[i + 3, j].plot(np.array(st_nomad_emg)[muscle_inds[i], :, trials_to_plot[j]].T, color=colors['nomad'], alpha=0.8, linewidth=1.5)
        axes[i + 6, j].plot(np.array(st_cyclegan_emg)[muscle_inds[i], :, trials_to_plot[j]].T, color=colors['cyclegan'], alpha=0.8, linewidth=1.5)
        # axes[i*3 + 1, j].plot(np.array(to_plot_ndt_emg)[j, :, :].T, color=colors['ndt2'], alpha=0.5, linewidth=0.5)
        # axes[i*3 + 2, j].plot(np.array(to_plot_nomad_emg)[j, :, :].T, color=colors['nomad'], alpha=0.5, linewidth=0.5)
        # axes[i*3 + 3, j].plot(np.array(to_plot_cg_emg)[j, :, :].T, color=colors['cyclegan'], alpha=0.5, linewidth=0.5)

# turn the splines all off 
for ax in axes.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # get rid of left and bottom ticks and tick labels 
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

for i in [2,5,8]:
    axes[i, 0].hlines(-0.5, 0, 25, color='k')
    axes[i, 0].text(0, -1.2, f'{25 * 20} ms')

plt.savefig('/home/bkarpo2/projects/falcon_figs/fig4b.pdf', bbox_inches='tight')

# %%

''' make a polar plot that has colored points at each angle in trial_loc'''
# import colormaps 
import matplotlib.cm as cm
cmap = cm.get_cmap('Purples')#, len(np.unique(trial_loc)))
plt.figure()
markers = ['o', 'x', 's', 'd', '^', '*', '2', 'X']
ax = plt.subplot(111, polar=True)
for ii,angle in enumerate(np.unique(trial_loc)):
    point = ax.plot(angle * np.pi / 180, 1, 'o', color=cmap((ii+1)/8.), markersize=8)
    # point = ax.plot(angle * np.pi / 180, 1, markers[ii], color='black', markersize=8)
    [p.set_clip_on(False) for p in point]
# turn off the grid lines for y only
ax.grid(axis='y')
# turn off the radius labels 
ax.set_yticklabels([])
ax.set_xticks(np.unique(trial_loc) * np.pi / 180)
ax.set_xticklabels(np.unique(trial_loc))
ax.set_xlim(0,np.max(trial_loc)/180*np.pi)
# change the font size of the axis labels 
for label in ax.get_xticklabels():
    label.set_fontsize(8)

plt.savefig('/home/bkarpo2/projects/falcon_figs/fig4b_key.pdf', bbox_inches='tight')

# %%
from sklearn.metrics import r2_score

# for each of the muscles in muscles_to_plot, get the individual muscle R2 for each method 
muscle_r2_ndt = []
muscle_r2_nomad = []
muscle_r2_cyclegan = []

for m in muscle_inds: 
    true_emg = ds.data.preprocessed_emg.values[:, m]
    ndt_emg = ds.data.ndt2_pred.values[:, m]
    nomad_emg = ds.data.nomad_pred.values[:, m]
    cyclegan_emg = ds.data.cyclegan_pred.values[:, m]
    nan_mask = ~np.isnan(ndt_emg)
    muscle_r2_ndt.append(r2_score(true_emg[nan_mask], ndt_emg[nan_mask]))
    muscle_r2_nomad.append(r2_score(true_emg[nan_mask], nomad_emg[nan_mask]))
    muscle_r2_cyclegan.append(r2_score(true_emg[nan_mask], cyclegan_emg[nan_mask]))

# %%
# make a triple bar plot where each bar is a method, and each group is a muscle 

fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
barWidth = 0.25
r1 = np.arange(len(muscle_inds))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

ax.bar(r1, muscle_r2_ndt, color=colors['ndt2'], width=barWidth, edgecolor='grey', label='ndt2')
ax.bar(r2, muscle_r2_nomad, color=colors['nomad'], width=barWidth, edgecolor='grey', label='nomad')
ax.bar(r3, muscle_r2_cyclegan, color=colors['cyclegan'], width=barWidth, edgecolor='grey', label='cyclegan')
ax.set_ylim([0, 1])
ax.set_xticks(np.array([barWidth, 5*barWidth, 9*barWidth]))
ax.set_xticklabels(muscles_to_plot)

# remove the actual ticks from both axes 
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=-0.5)
ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True, pad=-0.5)

ax.grid(axis='y', alpha=0.3)

# add text labels above each bar 
for i, r in enumerate(r1): 
    ax.text(r, muscle_r2_ndt[i] + 0.07, f'{muscle_r2_ndt[i]:.2f}', color='black', ha='center', fontsize=8, rotation=90)
    ax.text(r2[i], muscle_r2_nomad[i] + 0.07, f'{muscle_r2_nomad[i]:.2f}', color='black', ha='center', fontsize=8, rotation=90)
    ax.text(r3[i], muscle_r2_cyclegan[i] + 0.07, f'{muscle_r2_cyclegan[i]:.2f}', color='black', ha='center', fontsize=8, rotation=90)
plt.savefig('/home/bkarpo2/projects/falcon_figs/fig4c.pdf', bbox_inches='tight')

# %%
from sklearn.metrics import r2_score

# for each of the muscles in muscles_to_plot, get the individual muscle R2 for each method 
muscle_r2_ndt = []
muscle_r2_nomad = []
muscle_r2_cyclegan = []

for m in range(16): 
    true_emg = ds.data.preprocessed_emg.values[:, m]
    ndt_emg = ds.data.ndt2_pred.values[:, m]
    nomad_emg = ds.data.nomad_pred.values[:, m]
    cyclegan_emg = ds.data.cyclegan_pred.values[:, m]
    nan_mask = ~np.isnan(ndt_emg)
    muscle_r2_ndt.append(r2_score(true_emg[nan_mask], ndt_emg[nan_mask]))
    muscle_r2_nomad.append(r2_score(true_emg[nan_mask], nomad_emg[nan_mask]))
    muscle_r2_cyclegan.append(r2_score(true_emg[nan_mask], cyclegan_emg[nan_mask]))

# %%
plt.figure(facecolor='w')
plt.plot(muscle_r2_ndt, 'o-', label='NDT2')
plt.plot(muscle_r2_nomad, 'o-', label='NoMAD')
plt.plot(muscle_r2_cyclegan, 'o-', label='CycleGAN')
plt.ylim([-0.1, 1.0])
plt.grid()
plt.legend()
plt.ylabel('R2')
plt.xlabel('Muscle Index')