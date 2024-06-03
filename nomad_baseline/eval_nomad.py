#%% 
import pickle, os, sys, json, h5py
from yacs.config import CfgNode as CN
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
from snel_toolkit.interfaces import LFADSInterface
from snel_toolkit.analysis import PSTH
from decoder_demos.decoding_utils import fit_and_eval_decoder, generate_lagged_matrix, apply_neural_behavioral_lag

from sklearn.metrics import r2_score

import tensorflow as tf
tf.config.run_functions_eagerly(
    True
)

from lfads_tf2.utils import restrict_gpu_usage
IPYTHON = True if 'ipykernel' in sys.modules else False 
if len(sys.argv) <= 1 or IPYTHON:
    restrict_gpu_usage(gpu_ix=9)

from lfads_tf2.utils import load_posterior_averages, unflatten 
sys.path.insert(0, '/home/bkarpo2/bin/stability-benchmark/align_tf2')
for p in sys.path: 
    if 'nomad_dev' in p: 
        sys.path.remove(p)
from align_tf2.models import AlignLFADS
from align_tf2.tuples import AlignInput, SingleModelOutput, AlignmentOutput
from align_tf2.defaults import DEFAULT_CONFIG_DIR, get_cfg_defaults

#%%

DAY0_PATH = '/home/bkarpo2/bin/stability-benchmark/data/h1/held_in_calib/S2_set_1_calib.nwb'
DAY0_LFADS = '/snel/share/runs/falcon/H1_S2_set1/'

DAYK_PATH = '/home/bkarpo2/bin/stability-benchmark/data/h1/held_out_calib/S9_set_2_calib.nwb'
# ALIGN_PATH = '/snel/share/runs/falcon/240531_nomad_h1_multi0_multik/tuneAlign_38_TRAIN.BATCH_SIZE=914,TRAIN.KL.CO_WEIGHT=0.00027197,TRAIN.KL.IC_WEIGHT=0.0009324,TRAIN.KL.INCREASE_EPOCH=81,TRAIN.LR.I_2024-05-31_21-05-32qjjo1lk7'
ALIGN_PATH = '/snel/share/runs/falcon/240510_H1_NoMAD_multidayK/tuneAlign_17_TRAIN.BATCH_SIZE=973,TRAIN.KL.CO_WEIGHT=0.0002188,TRAIN.KL.IC_WEIGHT=1.022e-05,TRAIN.KL.INCREASE_EPOCH=100,TRAIN.LR.I_2024-05-10_14-05-02yc5r87l3'

TRACK = 'H1'

CHOP_LEN = 1000
OLAP_LEN = 200
VALID_RATIO = 0.2
NORM_SMOOTH_MS = 20

#%%

if TRACK == 'M1': 
    align_field = 'move_onset_time'
    cond_sep_field = 'tgt_loc'
    decoding_field = 'preprocessed_emg'
    trialized = True
elif TRACK == 'H1': 
    decoding_field = 'OpenLoopKinematicsVelocity'
    trialized = False

#%% 
if TRACK == 'M1':
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
else: 
    ds0 = H1Dataset(DAY0_PATH)
    dsK = H1Dataset(DAYK_PATH)

#%%  otherwise just get the causally sampled outputs of the model 

from causal_samp import get_causal_model_output, merge_data

align_model = AlignLFADS(align_dir=ALIGN_PATH)

#%% day0 output - data is way too long so just get the acausally sampled outputs 

print('Merging Day 0 LFADS Output...') 
# load lfi object 
with open(os.path.join(DAY0_LFADS, 'input_data', 'interface.pkl'), 'rb') as f:
    lfi = pickle.load(f)

# update merging map 
lfi.merge_fields_map = {
    'rates': 'lfads_rates',
    'factors': 'lfads_factors',
    'gen_states': 'lfads_gen_states'
}

# merge back to dataframe 
ds0.data = lfi.load_and_merge(
    os.path.join(DAY0_LFADS, 'pbt_run', 'model_output', 'posterior_samples.h5'),
    ds0.data,
    smooth_pwr=2
)

#%% dayk unaligned output 

unalign_out = get_causal_model_output(
    model = align_model.lfads_day0,
    binsize = dsK.bin_size,
    input_data = dsK.data.spikes.values,
    out_fields = ['rates', 'factors', 'gen_states'],
    output_dim = {'rates': align_model.lfads_dayk.cfg.MODEL.DATA_DIM, 
                'factors': align_model.lfads_dayk.cfg.MODEL.FAC_DIM, 
                'gen_states': align_model.lfads_dayk.cfg.MODEL.GEN_DIM}
)

for k in unalign_out:
    dsK = merge_data(dsK, unalign_out[k], 'unaligned_lfads_{}'.format(k))

#%% 
align_out = get_causal_model_output(
    model = align_model.lfads_dayk,
    binsize = dsK.bin_size,
    input_data = dsK.data.spikes.values,
    out_fields = ['rates', 'factors', 'gen_states'],
    output_dim = {'rates': align_model.lfads_dayk.cfg.MODEL.DATA_DIM, 
                'factors': align_model.lfads_dayk.cfg.MODEL.FAC_DIM, 
                'gen_states': align_model.lfads_dayk.cfg.MODEL.GEN_DIM}
    )

for k in align_out:
    dsK = merge_data(dsK, align_out[k], 'aligned_lfads_{}'.format(k))


dsK.data = dsK.data.dropna()
ds0.data = ds0.data.dropna()

#%% history sweep 

history = np.arange(0, 15)
day0_r2 = []
dayk_align_r2 = []
dayk_unalign_r2 = []

for N_HIST in history:
    # train decoder on Day 0 LFADS gen states  
    print(N_HIST)
    # N_HIST = 4
    VAL_RATIO = 0.2
    eval_mask = ds0.data.eval_mask.values.squeeze().astype('bool')
    X = generate_lagged_matrix(ds0.data.lfads_gen_states.values[eval_mask, :], N_HIST)
    y = ds0.data[decoding_field].values[eval_mask, :][N_HIST:, :]
    # y = (y - np.nanmean(y, axis=0)) / np.nanstd(y, axis=0)

    n_train = int(X.shape[0] * (1 - VAL_RATIO))

    r2, decoder, y_pred = fit_and_eval_decoder(
        X[:n_train, :], 
        y[:n_train, :], 
        X[n_train:, :], 
        y[n_train:, :], 
        grid_search=True, 
        param_grid=np.logspace(2, 3, 20),
        # cv=10,
        return_preds=True)

    print(f'Day 0 R2: {r2}')
    day0_r2.append(r2)

    # evaluate on unaligned gen states and aligned gen states 

    eval_mask_K = dsK.data.eval_mask.values.squeeze().astype('bool')
    unalignX = generate_lagged_matrix(dsK.data.unaligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
    alignX = generate_lagged_matrix(dsK.data.aligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
    y_K = dsK.data[decoding_field].values[eval_mask_K, :][N_HIST:, :]
    # y_K = (y_K - np.nanmean(y_K, axis=0)) / np.nanstd(y_K, axis=0)

    unalign_preds = decoder.predict(unalignX)
    align_preds = decoder.predict(alignX)

    unalign_r2 = r2_score(y_K, unalign_preds, multioutput='variance_weighted')
    align_r2 = r2_score(y_K, align_preds, multioutput='variance_weighted')
    dayk_align_r2.append(align_r2)
    dayk_unalign_r2.append(unalign_r2)

    print(f'Unaligned R2: {unalign_r2}')
    print(f'Aligned R2: {align_r2}')

plt.figure(facecolor='w')
plt.plot(history, day0_r2, 'o-', label='Day 0', color='k')
plt.plot(history, dayk_unalign_r2, 'o-', label='Unaligned', color='b')
plt.plot(history, dayk_align_r2, 'o-', label='Aligned', color='g')
plt.grid(alpha=0.3)
plt.legend()
plt.ylabel('R2')
plt.xlabel('# Bins WF History')
plt.savefig('h1_history_sweep.pdf')

# %%

n_outputs = y.shape[1]
fig, ax = plt.subplots(n_outputs, 3, figsize=(15, 15), sharex=True, sharey='row', facecolor='w')

for i in range(n_outputs):
    ax[i][0].plot(y[n_train:, i], label='True', color='k')
    ax[i][0].plot(y_pred[:, i], label='Pred', color='r')
ax[-1][0].legend()
ax[0][0].set_title(f'Day 0 Decoding Valid R2: {np.around(r2, 3)}')

for i in range(n_outputs):
    ax[i][1].plot(y_K[:, i], label='True', color='k')
    ax[i][1].plot(unalign_preds[:, i], label='Pred', color='b')
ax[-1][1].legend()
ax[0][1].set_title(f'Unalign Decoding Valid R2: {np.around(unalign_r2, 3)}')

for i in range(n_outputs):
    ax[i][2].plot(y_K[:, i], label='True', color='k')
    ax[i][2].plot(align_preds[:, i], label='Pred', color='g')
ax[-1][2].legend()
ax[0][2].set_title(f'Align Decoding Valid R2: {np.around(align_r2, 3)}')

ax[0][0].set_ylabel('Tx Vel')
ax[1][0].set_ylabel('Ty Vel')
ax[2][0].set_ylabel('Tz Vel')
ax[3][0].set_ylabel('Rx Vel')
ax[4][0].set_ylabel('G1 Vel')
ax[5][0].set_ylabel('G2 Vel')
ax[6][0].set_ylabel('G3 Vel')
# %%

from mpl_toolkits.mplot3d import Axes3D  # This import is necessary
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


day0_gen_states = ds0.data.lfads_gen_states.values[eval_mask, :]
day0_kin = ds0.data[decoding_field].values[eval_mask, :]

nonzero_pts = ~(np.abs(day0_kin) < 0.0001).all(axis=1)
day0_gen_states = day0_gen_states[nonzero_pts, :]
day0_kin = day0_kin[nonzero_pts, :]

pca = PCA(n_components=3)
ss = StandardScaler()
pca.fit(ss.fit_transform(day0_gen_states))

day0_pcs = pca.transform(ss.fit_transform(day0_gen_states))

# make a set of 3d axes 
fig = plt.figure(figsize=(20, 5), facecolor='w')
ax = fig.add_subplot(131, projection='3d')
ax.scatter(
    day0_pcs[:, 0], 
    day0_pcs[:, 1], 
    day0_pcs[:, 2], 
    c=np.sqrt(np.sum([day0_kin[:, 0]**2, day0_kin[:, 1]**2, day0_kin[:, 2]**2], axis=0)), 
    cmap='GnBu', 
    alpha=0.4,
    vmin=0,
) 
ax.set_title('Day 0')
# change view angle 
ax.view_init(elev=45, azim=30)

unalign_gen_states = dsK.data.unaligned_lfads_gen_states.values[eval_mask_K, :]
align_gen_states = dsK.data.aligned_lfads_gen_states.values[eval_mask_K, :]
dayk_kin = dsK.data[decoding_field].values[eval_mask_K, :]

nonzero_pts = ~(np.abs(dayk_kin) < 0.0001).all(axis=1)
unalign_gen_states = unalign_gen_states[nonzero_pts, :]
align_gen_states = align_gen_states[nonzero_pts, :]
dayk_kin = dayk_kin[nonzero_pts, :]

unalign_pcs = pca.transform(ss.fit_transform(unalign_gen_states))
align_pcs = pca.transform(ss.fit_transform(align_gen_states))

ax=fig.add_subplot(132, projection='3d')
ax.scatter(
    unalign_pcs[:, 0], 
    unalign_pcs[:, 1], 
    unalign_pcs[:, 2], 
    c=np.sqrt(np.sum([dayk_kin[:, 0]**2, dayk_kin[:, 1]**2, dayk_kin[:, 2]**2], axis=0)), 
    cmap='GnBu', 
    alpha=0.4,
    vmin=0,
)
ax.set_title('Unaligned')
ax.view_init(elev=45, azim=30)

ax = fig.add_subplot(133, projection='3d')
ax.scatter(
    align_pcs[:, 0], 
    align_pcs[:, 1], 
    align_pcs[:, 2], 
    c=np.sqrt(np.sum([dayk_kin[:, 0]**2, dayk_kin[:, 1]**2, dayk_kin[:, 2]**2], axis=0)), 
    cmap='GnBu', 
    alpha=0.4,
    vmin=0
)
ax.set_title('Aligned')
ax.view_init(elev=45, azim=30)

# %%
