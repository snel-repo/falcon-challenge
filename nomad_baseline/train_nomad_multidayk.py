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
sys.path.insert(0, '/home/bkarpo2/bin/falcon-challenge/align_tf2')
for p in sys.path: 
    if 'nomad_dev' in p: 
        sys.path.remove(p)
from align_tf2.models import AlignLFADS
from align_tf2.tuples import AlignInput, SingleModelOutput, AlignmentOutput
from align_tf2.defaults import DEFAULT_CONFIG_DIR, get_cfg_defaults

#%%

DAY0_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/000954/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250113T120811.nwb'
DAY0_LFADS = '/snel/share/runs/falcon/H1_S2_set1/'

DAYK_SESS = '19250126'
DAYK_BASE_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/h1/held-out-calib'
# DAYK_PATH = '/home/bkarpo2/bin/stability-benchmark/data/h1/held_out_calib/S9_set_1_calib.nwb'
CONFIG_PATH = '/home/bkarpo2/bin/stability-benchmark/nomad_baseline/config/nomad_config.yaml'

TRACK = 'H1'
RUN_FLAG = 'test_new_data'

CHOP_LEN = 1000
OLAP_LEN = 200
VALID_RATIO = 0.2
NORM_SM_MS = 20

if len(sys.argv) > 1 and not IPYTHON:
    # load the tune configuration passed as an argument
    with open(sys.argv[1], 'r') as f:
        tune_config = json.load(f)
    # Overwrite the above local variables (e.g. VALID_RATIO)
    locals().update(tune_config)

DAYK_PATHS = glob.glob(os.path.join(DAYK_BASE_PATH, f'*{DAYK_SESS}*.nwb'))

if len(sys.argv) > 1 and not IPYTHON:
    # Store the model at the tune logdir
    run_path = tune_config['logdir']
else: 
    run_path = os.path.join('/snel/share/runs/falcon', f'NoMAD_{TRACK}_{RUN_FLAG}')

if TRACK == 'M1': 
    align_field = 'move_onset_time'
    cond_sep_field = 'tgt_loc'
    decoding_field = 'preprocessed_emg'
    trialized = True
elif TRACK == 'H1': 
    decoding_field = 'OpenLoopKinematicsVelocity'
    trialized = False
elif TRACK == 'M2': 
    decoding_field = 'finger_vel'
    trialized = True
    cond_sep_field='index_pos'
    align_field='start_time'

if not os.path.exists(run_path):
    os.makedirs(run_path, mode=0o755)

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
elif TRACK == 'M2': 
    skip_fields = ['eval_mask', 'finger_pos', 'finger_vel']
    # load and rebin the spikes 
    ds_spk = NWBDataset(
        DAY0_PATH, 
        skip_fields=skip_fields)
    ds_spk.resample(20)

    ds0 = NWBDataset(DAY0_PATH)
    ds0.data = ds0.data.dropna()
    ds0.data.index = ds0.data.index.round('20ms')
    ds0.data.spikes = ds_spk.data.spikes
    ds0.bin_width = 20
    ds0.trial_info['index_pos'] = [tgt[0] for tgt in ds0.trial_info.tgt_loc.values]

    dsKs = []
    for DAYK_PATH in DAYK_PATHS:
        ds_spk = NWBDataset(
            DAYK_PATH, 
            skip_fields=skip_fields)
        ds_spk.resample(20)

        dsK = NWBDataset(DAYK_PATH)
        dsK.data = dsK.data.dropna()
        dsK.data.index = dsK.data.index.round('20ms')
        dsK.data.spikes = ds_spk.data.spikes
        dsK.bin_width = 20
        dsK.trial_info['index_pos'] = [tgt[0] for tgt in dsK.trial_info.tgt_loc.values]
        dsKs.append(dsK)
else: 
    ds0 = H1Dataset(DAY0_PATH)
    dsKs = []
    for DAYK_PATH in DAYK_PATHS:
        dsK = H1Dataset(DAYK_PATH)
        dsKs.append(dsK)

# %%

NEEDS_TO_TRAIN = not os.path.exists(os.path.join(run_path, 'align_ckpts'))

#%% 
if NEEDS_TO_TRAIN:
    if NORM_SM_MS > 0:
        all_ss = []
        for ds in dsKs:
            ds.smooth_spk(NORM_SM_MS, name='smooth')
            ss = ds.data.spikes_smooth.values
            all_ss.append(ss)
        ss = np.concatenate(all_ss, axis=0)
        # dsK.smooth_spk(NORM_SMOOTH_MS, name='smooth')
        # ss = dsK.data.spikes_smooth.values
        # ss = model_ni.dataset.data['spikes'].values
        ch_mean = np.nanmean(ss, axis=0)
        ch_std = np.nanstd(ss, axis=0)
        ch_std[ch_std == 0] = 1 #avoid NaNs by dealing with columns of 0's 
        mat = np.diag(1./ch_std)

        norm_ss = np.dot(ss, mat) - np.dot(ch_mean, mat)
        assert (all(np.nanmean(norm_ss, axis=0) < 1e-12))
        check_std = np.nanstd(norm_ss, axis=0)
        assert (all(check_std[check_std != 0] >= 0.99999) and all(check_std[check_std != 0] <= 1.0001)) # controlling for zero-filled channels
        # save the weights and biases 
        print('Saving normalization weights and biases...')
        hf = h5py.File(
            os.path.join(run_path, 'normalization_dayk.h5'), 'w')
        hf.create_dataset('matrix', data=mat)
        hf.create_dataset('bias', data=np.dot(ch_mean, mat))
        hf.close()


    # NoMAD configs 
    #Load the desired alignment parameters
    align_cfg = get_cfg_defaults()
    align_cfg.merge_from_file(CONFIG_PATH)
    if len(sys.argv) > 1 and not IPYTHON:
        # Add the sampled HP's from the random search
        cfg_update = CN(unflatten(CFG_UPDATES))
        align_cfg.merge_from_other_cfg(cfg_update)
    align_cfg.MODEL.DATA_DIM = dsK.data.spikes.values.shape[1]
    align_cfg.MODEL.DAY0_MODEL_TYPE = 'dimreduced'
    align_cfg.MODEL.SEQ_LEN =  int(CHOP_LEN /dsK.bin_size/1000)
    align_cfg.MODEL.DATA_TYPE = 'T5'
    align_cfg.TRAIN.MODEL_DIR = os.path.join(DAY0_LFADS, 'pbt_run', 'model_output')
    align_cfg.TRAIN.ALIGN_DIR = run_path
    align_cfg.TRAIN.NI_MODE = True
    align_cfg.TRAIN.USE_TB = False
    align_cfg.freeze()

    # prep input data 

    day0_lfi = LFADSInterface(
        window=CHOP_LEN, 
        overlap=OLAP_LEN,
        chop_fields_map={'spikes': 'data'})
    dayk_lfi = LFADSInterface(
        window=CHOP_LEN, 
        overlap=OLAP_LEN,
        chop_fields_map={'spikes': 'data'})
    day0_datadict = day0_lfi.chop(ds0.data)

    dayk_data = []
    for dsK in dsKs:
        dayk_datadict = dayk_lfi.chop(dsK.data)
        dayk_data.append(dayk_datadict['data'])
    
    # create a concatenated datadict 
    dayk_datadict = {}
    dayk_datadict['data'] = np.concatenate(dayk_data, axis=0)

    for datadict in [day0_datadict, dayk_datadict]:
        # drop any chops with NaNs in them 
        where_nan = np.isnan(datadict['data']).sum(axis=(1,2)) > 0
        datadict['data'] = datadict['data'][~where_nan, :, :]

        train_input, valid_input = \
            np.split(datadict['data'], [np.ceil(datadict['data'].shape[0]*0.8).astype(int)])
        train_inds, valid_inds = \
            np.split(np.arange(0, datadict['data'].shape[0]), [np.ceil(datadict['data'].shape[0]*0.8).astype(int)])
        datadict['train_data'] = train_input
        datadict['valid_data'] = valid_input
        datadict['train_inds'] = train_inds
        datadict['valid_inds'] = valid_inds
    # Alignment model input tuple, model initialization, loading norm weights 
    align_input = AlignInput(
        day0_train_data=day0_datadict['train_data'],
        day0_valid_data=day0_datadict['valid_data'],
        dayk_train_data=dayk_datadict['train_data'],
        dayk_valid_data=dayk_datadict['valid_data'],
        day0_train_inds=day0_datadict['train_inds'],
        day0_valid_inds=day0_datadict['valid_inds'],
        dayk_train_inds=dayk_datadict['train_inds'],
        dayk_valid_inds=dayk_datadict['valid_inds'])
    model = AlignLFADS(cfg_node=align_cfg)
    model.load_datasets(align_input)
    model.smart_init()

    # Train alignment and get the aligned rates and factors
    done = False
    i = 0
    while not done and i < 1000:
        results = model.train_epoch()
        i += 1
        done = results.get('done', False)

#%%  otherwise just get the causally sampled outputs of the model 

from causal_samp import get_causal_model_output, merge_data

align_model = AlignLFADS(align_dir=run_path)

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
for dsK in dsKs:
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

for dsK in dsKs:
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

N_HIST = 3
VAL_RATIO = 0.2
eval_mask = ds0.data.eval_mask.values.squeeze().astype('bool')
X = generate_lagged_matrix(ds0.data.lfads_gen_states.values[eval_mask, :], N_HIST)
y = ds0.data[decoding_field].values[eval_mask, :][N_HIST:, :]

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

all_unalign = []
all_align = []
for dsK in dsKs:
    eval_mask_K = dsK.data.eval_mask.values.squeeze().astype('bool')
    unalignX = generate_lagged_matrix(dsK.data.unaligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
    alignX = generate_lagged_matrix(dsK.data.aligned_lfads_gen_states.values[eval_mask_K, :], N_HIST)
    y_K = dsK.data[decoding_field].values[eval_mask_K, :][N_HIST:, :]

    unalign_preds = decoder.predict(unalignX)
    align_preds = decoder.predict(alignX)

    unalign_r2 = r2_score(y_K, unalign_preds, multioutput='variance_weighted')
    align_r2 = r2_score(y_K, align_preds, multioutput='variance_weighted')

    print(f'Unaligned R2: {unalign_r2}')
    print(f'Aligned R2: {align_r2}')

    all_unalign.append(unalign_r2)
    all_align.append(align_r2)


# %%
# Save to align_out.json
train_out = pd.read_csv(os.path.join(run_path, 'train_data.csv'))

results_out = {
    'loss': train_out['loss'].values[-1],
    'kl': train_out['kl'].values[-1],
    'val_loss': train_out['val_loss'].values[-1],
    'val_kl': train_out['val_kl'].values[-1],
    'nll': train_out['nll'].values[-1],
    'val_nll': train_out['val_nll'].values[-1],
    'day0_r2': r2,
    'unalign_dayk_r2': all_unalign,
    'align_dayk_r2': all_align
}
results_path = os.path.join(run_path, 'align_out.json')
with open(results_path, 'w') as f:
    json.dump(results_out, f, sort_keys=True, indent=4)
print('Saved results to ' + results_path)
# %%
