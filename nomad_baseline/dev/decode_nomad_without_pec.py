#%% 
import pickle, os, sys, json, h5py, glob
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

DAY0_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb'
DAY0_LFADS = '/snel/share/runs/falcon/M1_sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys'

DAYK_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-out-calib/sub-MonkeyL-held-out-calib_ses-20121022_behavior+ecephys.nwb'
ALIGN_PATH = '/snel/share/runs/falcon/240509_M1_day0gridsearch_new0924/tuneAlign_9_DAY0_LFADS=_snel_share_runs_falcon_M1_sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys,DAY0_PATH=_home_bkarpo2__2024-05-09_17-08-303xq39bd9'

TRACK = 'M1'

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

sys.path.append('/snel/home/bkarpo2/bin/falcon-challenge/nomad_baseline/')
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

# %% train decoder on Day 0 LFADS gen states  

N_HIST = 4
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
    return_preds=True)

print(f'Day 0 R2: {r2}')

# %% evaluate on unaligned gen states and aligned gen states 

eval_mask_K = dsK.data.eval_mask.values.squeeze().astype('bool')
unalignX = generate_lagged_matrix(dsK.data.unaligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
alignX = generate_lagged_matrix(dsK.data.aligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
y_K = dsK.data[decoding_field].values[eval_mask_K, :][N_HIST:, :]
# y_K = (y_K - np.nanmean(y_K, axis=0)) / np.nanstd(y_K, axis=0)

unalign_preds = decoder.predict(unalignX)
align_preds = decoder.predict(alignX)

unalign_r2 = r2_score(y_K, unalign_preds, multioutput='variance_weighted')
align_r2 = r2_score(y_K, align_preds, multioutput='variance_weighted')

print(f'Unaligned R2: {unalign_r2}')
print(f'Aligned R2: {align_r2}')

# %%

with_pec = [r2, unalign_r2, align_r2]

#%% 

N_HIST = 4
VAL_RATIO = 0.2
eval_mask = ds0.data.eval_mask.values.squeeze().astype('bool')
X = generate_lagged_matrix(ds0.data.lfads_gen_states.values[eval_mask, :], N_HIST)
y = ds0.data[decoding_field].values[eval_mask, :][N_HIST:, :][:, [0,1,2,4,5,6,7,8,9,10,11,12,14,15]]
# y = (y - np.nanmean(y, axis=0)) / np.nanstd(y, axis=0)

n_train = int(X.shape[0] * (1 - VAL_RATIO))

r2, decoder, y_pred = fit_and_eval_decoder(
    X[:n_train, :], 
    y[:n_train, :], 
    X[n_train:, :], 
    y[n_train:, :], 
    grid_search=True, 
    param_grid=np.logspace(2, 3, 20),
    return_preds=True)

print(f'Day 0 R2: {r2}')

#  evaluate on unaligned gen states and aligned gen states 

eval_mask_K = dsK.data.eval_mask.values.squeeze().astype('bool')
unalignX = generate_lagged_matrix(dsK.data.unaligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
alignX = generate_lagged_matrix(dsK.data.aligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
y_K = dsK.data[decoding_field].values[eval_mask_K, :][N_HIST:, :][:, [0,1,2,4,5,6,7,8,9,10,11,12,14,15]]
# y_K = (y_K - np.nanmean(y_K, axis=0)) / np.nanstd(y_K, axis=0)

unalign_preds = decoder.predict(unalignX)
align_preds = decoder.predict(alignX)

unalign_r2 = r2_score(y_K, unalign_preds, multioutput='variance_weighted')
align_r2 = r2_score(y_K, align_preds, multioutput='variance_weighted')

print(f'Unaligned R2: {unalign_r2}')
print(f'Aligned R2: {align_r2}')


# %%
without_pec = [r2, unalign_r2, align_r2]
# %%
# make a double bar plot of with_pec and without_pec 
fig, ax = plt.subplots(facecolor='w')
barWidth = 0.25
r1 = np.arange(len(with_pec))
r2 = [x + barWidth for x in r1]
plt.bar(r1, with_pec, color='lightblue', width=barWidth, edgecolor='grey', label='All Muscles')
plt.bar(r2, without_pec, color='lightgreen', width=barWidth, edgecolor='grey', label='Without Pec, DLTa')
plt.ylabel('R2 Score')
plt.xticks([r + barWidth/2 for r in range(len(with_pec))], ['Day 0', 'Unaligned', 'Aligned'])
plt.legend(loc='lower right')
# add values above bars 
for i in range(len(with_pec)):
    plt.text(i, with_pec[i] + 0.03, round(with_pec[i], 2), ha='center')
    plt.text(i + barWidth, without_pec[i] + 0.03, round(without_pec[i], 2), ha='center')
# %%
