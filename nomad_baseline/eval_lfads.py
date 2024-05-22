#%% 
import pickle, os, sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
from snel_toolkit.interfaces import LFADSInterface
from snel_toolkit.analysis import PSTH
from decoder_demos.decoding_utils import fit_and_eval_decoder, generate_lagged_matrix, apply_neural_behavioral_lag
from tune_tf2.pbt.utils import plot_pbt_hps, plot_pbt_log

np.random.seed(731)
#%%
ds_str = '2020-10-19-Run2'

DATA_PATH = f'/snel/home/bkarpo2/bin/falcon-challenge/data/m2/sub-MonkeyN-held-in-calib/sub-MonkeyN-held-in-calib_ses-{ds_str}_behavior+ecephys.nwb'
BASE_PATH = '/snel/share/runs/falcon'
TRACK = 'M2'
RUN_FLAG = f'{ds_str}_coinspkrem'
INTERFACE_KEY = ''
IS_COMBO_MODEL = False

# SESSION_NUMBER = 1
# SET_NUMBER = 1

# DATA_PATH = f'/snel/home/bkarpo2/bin/falcon-challenge/data/h1/held_in_calib/S{SESSION_NUMBER}_set_{SET_NUMBER}_calib.nwb'
# BASE_PATH = '/snel/share/runs/falcon'
# TRACK = 'H1'
# RUN_FLAG = f'S{SESSION_NUMBER}_combined_day0_larger_capacity'
# INTERFACE_KEY = f'S{SESSION_NUMBER}_set_{SET_NUMBER}_calib_'
# IS_COMBO_MODEL = True

if len(sys.argv) > 2:
    DATA_PATH = sys.argv[1]
    TRACK = sys.argv[2]
    RUN_FLAG = sys.argv[3]

CHOP_LEN = 1000 #ms
OLAP_LEN = 200 #ms 
VALID_RATIO = 0.2 
REMOVE_COIN_SPK = True

run_path = os.path.join(BASE_PATH, f'{TRACK}_{RUN_FLAG}')
# analysis_path = os.path.join(run_path, f'analysis_set{SET_NUMBER}')
analysis_path = os.path.join(run_path, 'analysis_path_removal')

if not os.path.exists(analysis_path):
    os.makedirs(analysis_path, mode=0o775)

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

#%% 
if TRACK == 'M1': 
    # load and rebin the spikes 
    ds_spk = NWBDataset(
        DATA_PATH, 
        skip_fields=['preprocessed_emg', 'eval_mask'])
    ds_spk.resample(20)

    ds = NWBDataset(DATA_PATH)
    ds.data = ds.data.dropna()
    ds.data.spikes = ds_spk.data.spikes
    ds.bin_width = 20
elif TRACK == 'M2': 
    skip_fields = ['eval_mask', 'finger_pos', 'finger_vel']
    # load and rebin the spikes 
    ds_spk = NWBDataset(
        DATA_PATH, 
        skip_fields=skip_fields)
    ds_spk.resample(20)

    ds = NWBDataset(DATA_PATH)
    ds.data = ds.data.dropna()
    ds.data.index = ds.data.index.round('20ms')
    ds.data.spikes = ds_spk.data.spikes
    ds.bin_width = 20
    ds.trial_info['index_pos'] = [tgt[0] for tgt in ds.trial_info.tgt_loc.values]
elif TRACK == 'H1': 
    ds = H1Dataset(DATA_PATH)

#%% 
if REMOVE_COIN_SPK:
    threshold = 0.5

    max_val = int(np.max(ds.data.spikes.values))
    rem_spikes = 0
    for i in range(max_val):
        spikes = ds.data.spikes
        # Calculate channel threshold
        frac_channels = int(threshold * spikes.shape[1])
        # Find bins of spike coincidences
        time_idx = []
        for i, spk_row in spikes.iterrows():
            # Get indices of channels that spiked
            spk_chans = np.where(spk_row > 0)[0]
            # Check if number of channels that spiked is greater than threshold
            if len(spk_chans) > frac_channels:
                time_idx.append(i)

        if len(time_idx) > 0: 
            spikes.loc[time_idx, :] = spikes.loc[time_idx, :].subtract(1).clip(lower=0)
            ds.data['spikes'] = spikes

        rem_spikes += len(time_idx)

    print(f'Removed {rem_spikes} coincident spikes')

#%% 

print('Merging LFADS Output...') 
# load lfi object 
with open(os.path.join(run_path, 'input_data', INTERFACE_KEY + 'interface.pkl'), 'rb') as f:
    lfi = pickle.load(f)

# update merging map 
lfi.merge_fields_map = {
    'rates': 'lfads_rates',
    'factors': 'lfads_factors',
    'gen_states': 'lfads_gen_states'
}

if IS_COMBO_MODEL:
    from lfads_tf2.subclasses.dimreduced.models import DimReducedLFADS
    from lfads_tf2.tuples import LoadableData
    chopped_data = lfi.chop(ds.data)['data']
    dataset_name = 'lfads_' + TRACK + RUN_FLAG[2:] + '.h5'
    input_tuple = LoadableData(
        train_data={dataset_name: chopped_data[:150, :, :]},
        valid_data={dataset_name: chopped_data[150:, :, :]},
        train_ext_input=None,
        valid_ext_input=None,
        train_behavior=None,
        valid_behavior=None,
        train_inds={dataset_name: np.arange(150)},
        valid_inds={dataset_name: np.arange(150, chopped_data.shape[0])},
    )
    model = DimReducedLFADS(model_dir=os.path.join(run_path, 'pbt_run', 'model_output'))
    model.sample_and_average(
        loadable_data=input_tuple, 
        ps_filename=INTERFACE_KEY+'posterior_samples.h5'
    )

# merge back to dataframe 
ds.data = lfi.load_and_merge(
    os.path.join(run_path, 'pbt_run', 'model_output', INTERFACE_KEY+'posterior_samples.h5'),
    ds.data,
    smooth_pwr=2
)

#%% plot loss curves 
pbt_exp_dir = os.path.join(run_path, 'pbt_run')
fields = ['MODEL.DROPOUT_RATE', 'TRAIN.KL.CO_WEIGHT', 
        'TRAIN.KL.IC_WEIGHT', 'TRAIN.L2.CON_SCALE', 'TRAIN.L2.GEN_SCALE',
        'TRAIN.LR.INIT']
_ = [plot_pbt_hps(pbt_exp_dir, f, save_dir=analysis_path) for f in fields]

fields = ['val_loss', 'val_nll_heldin', 'nll_heldin', 'loss']
_ = [plot_pbt_log(pbt_exp_dir, f, save_dir=analysis_path) for f in fields]

#%% 

if trialized:
    ds.trials = ds.make_trial_data(
        align_field=align_field,
        align_range=(-250, 500),  # ms
        allow_overlap=False,
        ignored_trials=ds.trial_info['result'] != 'R' if TRACK=='M1' else None
    )

    # PSTHs

    #Find channels to plot
    where_zero = np.where(ds.trials.spikes.sum(axis=0) == 0)
    zeroed_lfads_chans = ds.trials.lfads_rates.columns[where_zero]
    lfads_neurons = np.setdiff1d(ds.data.lfads_rates.columns, zeroed_lfads_chans)

    print('Plotting Cond Avg LFADS rates PSTHs...') 
    psth = PSTH(ds.trial_info[cond_sep_field])

    psth_means, psth_sems = psth.compute_trial_average(ds.trials, 'lfads_rates')
    psth.plot(
        psth_means, 
        psth_sems,
        neurons=lfads_neurons,
        max_neurons=ds.data.spikes.shape[1],
        ncols=12,
        save_path=os.path.join(analysis_path, 'psths.pdf')
    )
    # single trial rates 

    trial_labels = ds.trial_info[cond_sep_field].unique()
    single_trial_rates = np.array(
        np.split(
            ds.trials.pivot(index='align_time', columns='trial_id').lfads_rates.values, 
            ds.data.lfads_rates.shape[-1], 
            axis=1)
        ) # ch x time x trials 
    single_trial_rates = single_trial_rates[~(ds.trials.spikes.sum(axis=0) == 0), :, :]

    val_trials = ds.trial_info.iloc[np.unique(ds.trials.trial_id.values)]
    cond_sep_rates = [single_trial_rates[:, :, val_trials[cond_sep_field] == k] for k in trial_labels]

    
    cmap = plt.cm.rainbow
    n_conds = len(cond_sep_rates)
    n_chans = len(lfads_neurons)

    ncols = 8
    fig, axs = plt.subplots(int(np.ceil(n_chans/float(ncols))), ncols, figsize=(20, 12), sharex=True)
    ax = axs.flatten()

    t = ds.trials.pivot(index='align_time', columns='trial_id').index

    for i, cond in enumerate(cond_sep_rates):
        for j, chan in enumerate(lfads_neurons):
            ax[j].plot(t, cond[j, :, :], linewidth=0.5, alpha=0.7, color=cmap(i/float(n_conds)))
            ax[j].set_title(f'Chan {chan}')

    plt.savefig(os.path.join(analysis_path, 'single_trial_rates.pdf'))
else: 
    eval_mask = ds.data.eval_mask.values.squeeze().astype('bool')
    rates = ds.data.lfads_rates.values
    ds.smooth_spk(50, name='smooth')
    true_rates = ds.data.spikes_smooth.values 
    rates[~eval_mask, :] = np.nan
    n_neurons = rates.shape[1]
    neurons_to_plot = np.random.choice(n_neurons, 10, replace=False)

    fig, axs = plt.subplots(5, 2, figsize=(15, 15), sharex=True)
    for i, n in enumerate(neurons_to_plot):
        axs[i//2, i%2].plot(rates[:1000, n], label=f'Neuron {n}', color='b')
        axs[i//2, i%2].plot(true_rates[:1000, n], label='True', color='k', alpha=0.5)
        axs[i//2, i%2].set_title(f'Neuron {n}')
    plt.savefig(os.path.join(analysis_path, 'single_trial_rates.pdf'))


# %% decoding 

N_HIST = 3
VAL_RATIO = 0.2
ds.data = ds.data.dropna()
eval_mask = ds.data.eval_mask.values.squeeze().astype('bool')
X = generate_lagged_matrix(ds.data.lfads_gen_states.values[eval_mask, :], N_HIST)
y = ds.data[decoding_field].values[eval_mask, :][N_HIST:, :]

# from sklearn.preprocessing import StandardScaler
# y = StandardScaler().fit_transform(y)

n_train = int(X.shape[0] * (1 - VAL_RATIO))

r2, decoder, y_pred = fit_and_eval_decoder(
    X[:n_train, :], 
    y[:n_train, :], 
    X[n_train:, :], 
    y[n_train:, :], 
    grid_search=True, 
    return_preds=True)

# %% decoding_plot 

if trialized: 
    y_preds_full = np.full((y.shape[0] + N_HIST, y.shape[1]), np.nan)
    y_preds_full[(n_train+N_HIST):, :] = y_pred 
    y_preds_df = np.full((len(ds.data), y.shape[1]), np.nan)
    y_preds_df[eval_mask, :] = y_preds_full
    ds.data[[('beh_pred', x) for x in ds.data[decoding_field].columns]] = y_preds_df

    ds.trials = ds.make_trial_data(
        align_field=align_field,
        align_range=(-250, 500),  # ms
        allow_overlap=False,
        ignored_trials=ds.trial_info['result'] != 'R' if TRACK=='M1' else None
    )

    n_dims = y.shape[1]

    true_emg = np.array(
        np.split(
            ds.trials.pivot(index='align_time', columns='trial_id')[decoding_field].values, 
            ds.data[decoding_field].shape[-1], 
            # len(ds.trials.trial_id.unique()),
            axis=1)
        ) # ch x time x trials 
    pred_emg = np.array(
        np.split(
            ds.trials.pivot(index='align_time', columns='trial_id').beh_pred.values, 
            ds.data[decoding_field].shape[-1], 
            # len(ds.trials.trial_id.unique()),
            axis=1)
        ) # ch x time x trials

    fig, axs = plt.subplots(n_dims, n_conds, figsize=(15, 15), sharex=True, sharey=True)
    cond_ids = sorted(ds.trial_info[cond_sep_field].unique())
    val_trials = ds.trial_info.iloc[np.unique(ds.trials.trial_id.values)]

    for i in range(n_conds):
        for j in range(n_dims):
            valid_trials = np.where(val_trials[cond_sep_field] == cond_ids[i])[0]
            for tr in valid_trials: 
                axs[j, i].plot(t, true_emg[j, :, tr], label='True', alpha=0.5, linewidth=0.5, color='k')
                axs[j, i].plot(t, pred_emg[j, :, tr], label='Pred', alpha=0.5, linewidth=0.5, color='r')

    for i, c in enumerate(cond_ids): 
        axs[0, i].set_title(f'Cond {c}')

    for i, d in enumerate(ds.data[decoding_field].columns): 
        axs[i, 0].set_ylabel(d)

    plt.suptitle(f'Decoding Valid R2: {r2}')
    plt.savefig(os.path.join(analysis_path, 'decoding.pdf'))
else: 
    n_outputs = y.shape[1]
    fig, ax = plt.subplots(n_outputs, 1, figsize=(15, 15), sharex=True)

    for i in range(n_outputs):
        ax[i].plot(y[n_train:, i], label='True')
        ax[i].plot(y_pred[:, i], label='Pred')
        ax[i].set_title(f'Output {i}')
    ax[-1].legend()
    plt.suptitle(f'Decoding Valid R2: {r2}')
    plt.savefig(os.path.join(analysis_path, 'decoding.pdf'))
# %%
