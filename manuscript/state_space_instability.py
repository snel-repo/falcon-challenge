#%% 
import pickle, os, sys, json, h5py
from yacs.config import CfgNode as CN
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from snel_toolkit.datasets.nwb import NWBDataset

#%%

DAY0_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb'
DAYK_PATH = '/snel/share/share/derived/rouse/RTG/NWB_FALCON_v7_unsorted/held_out_oracle/L_20121017_held_out_oracle.nwb' 
TRACK = 'M1'

align_field = 'move_onset_time'
cond_sep_field = 'tgt_loc'
decoding_field = 'preprocessed_emg'

#%% 

ds_spk0 = NWBDataset(
    DAY0_PATH, 
    skip_fields=['preprocessed_emg', 'eval_mask'])
ds_spk0.resample(20)

ds0 = NWBDataset(DAY0_PATH)
ds0.data = ds0.data.dropna()
ds0.data.spikes = ds_spk0.data.spikes
ds0.bin_width = 20

ds_spkK = NWBDataset(
    DAYK_PATH, 
    skip_fields=['preprocessed_emg', 'eval_mask'])
ds_spkK.resample(20)

dsK = NWBDataset(DAYK_PATH)
dsK.data = dsK.data.dropna()
dsK.data.spikes = ds_spkK.data.spikes
dsK.bin_width = 20
# %% smooth spikes 
ds0.smooth_spk(60, name='smooth')
dsK.smooth_spk(60, name='smooth')

#%% make trials 
ds0.trials = ds0.make_trial_data(
    align_field=align_field, 
    align_range=(-250, 500),
    ignored_trials=ds0.trial_info['result'] != 'R' 
)

dsK.trials = dsK.make_trial_data(
    align_field=align_field, 
    align_range=(-250, 500),
    ignored_trials=dsK.trial_info['result'] != 'R' 
)

#%% run PCA on day 0 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=3)
ss = StandardScaler()

pca.fit(ss.fit_transform(ds0.trials.spikes_smooth.values))

d0_pcs = pca.transform(ss.fit_transform(ds0.trials.spikes_smooth.values))
dk_pcs = pca.transform(ss.fit_transform(dsK.trials.spikes_smooth.values))

ds0.trials[[('pcs', f'PC{i}') for i in range(1, 4)]] = d0_pcs
dsK.trials[[('pcs', f'PC{i}') for i in range(1, 4)]] = dk_pcs

#%%
# also get oracle Day K PCs 
dk_or_pcs = pca.fit_transform(ss.fit_transform(dsK.trials.spikes_smooth.values))

dsK.trials[[('pcs_oracle', f'PC{i}') for i in range(1, 4)]] = dk_or_pcs

#%% 

d0_pivot_trials = ds0.trials.pivot(
    index='align_time',
    columns='trial_id',
)
st_d0_pcs = np.array(
    np.split(
        d0_pivot_trials.pcs.values,
        3,
        axis=1
    )
)

#%% 

dK_pivot_trials = dsK.trials.pivot(
    index='align_time',
    columns='trial_id',
)
st_dK_pcs = np.array(
    np.split(
        dK_pivot_trials.pcs.values,
        3,
        axis=1
    )
)

st_dK_o_pcs = np.array(
    np.split(
        dK_pivot_trials.pcs_oracle.values,
        3,
        axis=1
    )
)

#%% 
d0_conds = sorted(ds0.trial_info.tgt_loc.unique())
dK_conds = sorted(dsK.trial_info.tgt_loc.unique())

# separate the trials by condition
trial_sep_d0 = [
    st_d0_pcs[:, :, ds0.trial_info.tgt_loc.values == i] for i in d0_conds]

trial_sep_dK = [
    st_dK_pcs[:, :, dsK.trial_info.tgt_loc.values == i] for i in dK_conds]

trial_sep_dK_or = [
    st_dK_o_pcs[:, :, dsK.trial_info.tgt_loc.values == i] for i in dK_conds]

#%% 
# from sklearn.linear_model import Ridge
# r = Ridge(alpha=0.001)
# stack_or = np.hstack([np.mean(trial_sep_d0[c], axis=2) for c in range(8)]).T
# r.fit(
#     stack_or,
#     np.hstack([np.mean(trial_sep_d0[c], axis=2) for c in range(8)]).T)
# dk_or_pcs = r.predict(stack_or)
# or_avg = np.split(dk_or_pcs, 8, axis=0)
# or_single = []
# for c in range(8):
#     st = trial_sep_dK_or[c]
#     st_form = np.hstack(np.swapaxes(trial_sep_dK_or[c],0,1)).T
#     st_pred = r.predict(st_form)
#     or_single.append(
#         np.array(
#             np.split(st_form, st.shape[1])
#         )
#     )

#%% do some plotting - one trajectory per condition 
# make 3d axis 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(facecolor='white', figsize=(12, 4))
ax = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

import matplotlib.cm as cm
cmap = cm.get_cmap('hsv')

n_trials = 10

for c in range(8):
    avg = np.mean(trial_sep_d0[c], axis=2)
    trials = trial_sep_d0[c]
    ax.plot(
        avg[0, :],
        avg[1, :],
        avg[2, :],
        color=cmap(c/8.),
        linewidth=1.5,
        zorder=10
    )

    for i in range(n_trials):
        ax.plot(
            trials[0, :, i],
            trials[1, :, i],
            trials[2, :, i],
            color=cmap(c/8.),
            alpha=0.3,
            linewidth=0.4
        )

    avg = np.mean(trial_sep_dK[c], axis=2)
    trials = trial_sep_dK[c]
    ax2.plot(
        avg[0, :],
        avg[1, :],
        avg[2, :],
        color=cmap(c/8.),
        linewidth=1.5,
        zorder=10
    )

    for i in range(n_trials):
        ax2.plot(
            trials[0, :, i],
            trials[1, :, i],
            trials[2, :, i],
            color=cmap(c/8.),
            alpha=0.3,
            linewidth=0.4
        )

    avg = np.mean(trial_sep_dK_or[c], axis=2)
    trials = trial_sep_dK_or[c]
    ax3.plot(
        avg[0, :],
        avg[1, :],
        avg[2, :],
        color=cmap(c/8.),
        linewidth=1.5,
        zorder=10
    )
    for i in range(n_trials):
        ax3.plot(
            trials[0, :, i],
            trials[1, :, i],
            trials[2, :, i],
            color=cmap(c/8.),
            alpha=0.3,
            linewidth=0.4
        )

for a in [ax, ax2, ax3]:
    a.set_xlabel('PC1')
    a.set_ylabel('PC2')
    a.set_zlabel('PC3')
    a.set_xlim(-10, 10)
    a.set_ylim(-5, 5)
    a.set_zlim(-2, 2)
    elev = 30
    azim = 65
    print(f'Elev: {elev}, Azim: {azim}')
    a.view_init(elev=elev, azim=azim)
    a.grid(False)
    a.set_xticks([])
    a.set_yticks([])
    a.set_zticks([])

ax.set_title('Day 0')
ax2.set_title('Day K Static')
ax3.set_title('Day K Oracle')

plt.savefig('/snel/home/bkarpo2/projects/falcon_figs/pca_stability.pdf')
# %%

plt.figure()
ax = plt.subplot(111, polar=True)
for ii,angle in enumerate(dK_conds):
    point = ax.plot(angle * np.pi / 180, 1, 'o', color=cmap((ii+1)/8.), markersize=8)
    # point = ax.plot(angle * np.pi / 180, 1, markers[ii], color='black', markersize=8)
    [p.set_clip_on(False) for p in point]
# turn off the grid lines for y only
ax.grid(axis='y')
# turn off the radius labels 
ax.set_yticklabels([])
ax.set_xticks(np.array(dK_conds) * np.pi / 180)
ax.set_xticklabels(dK_conds)
ax.set_xlim(0,np.max(dK_conds)/180*np.pi)
# change the font size of the axis labels 
for label in ax.get_xticklabels():
    label.set_fontsize(8)
plt.savefig('/snel/home/bkarpo2/projects/falcon_figs/pca_stability_key.pdf')


# %%
