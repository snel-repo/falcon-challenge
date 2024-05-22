#%% 
import pickle, os, h5py, sys
import numpy as np
from pynwb import NWBHDF5IO
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
from snel_toolkit.interfaces import LFADSInterface

#%% 

DATA_PATHS = [
    '/snel/share/share/derived/rouse/RTG/NWB_FALCON_v7_unsorted/held_out_oracle/L_20121022_held_out_oracle.nwb',
    '/snel/share/share/derived/rouse/RTG/NWB_FALCON_v7_unsorted/held_out_eval/L_20121022_held_out_eval.nwb'
]
BASE_PATH = '/snel/share/runs/falcon'
TRACK = 'M1'

CHOP_LEN = 1000 #ms
OLAP_LEN = 200 #ms 
VALID_RATIO = 0.2
NORM_SM_MS = 20
CORR_CH_THRESH = 0.6

#%%
datasets =[] 

for DATA_PATH in DATA_PATHS:

    if TRACK == 'M1': 
        skip_fields = ['preprocessed_emg', 'eval_mask']
        # load and rebin the spikes 
        ds_spk = NWBDataset(
            DATA_PATH, 
            skip_fields=skip_fields)
        ds_spk.data = ds_spk.data.dropna()
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
    elif TRACK == 'H1': 
        ds = H1Dataset(DATA_PATH)
    
    datasets.append(ds)

# %%

import matplotlib.pyplot as plt
fig, ax = plt.subplots(16, 2, figsize=(10, 10), facecolor='w', sharex=True, sharey='row')
for col, ds in enumerate(datasets): 
    for m in range(16): 
        ax[m, col].plot(ds.data.preprocessed_emg.values[:2500, m], 'k', alpha=0.5)

muscle_names = ds.data.preprocessed_emg.columns.values
for i in range(16): 
    ax[i, 0].set_ylabel(muscle_names[i], rotation=0, labelpad=20)

#%% 

for ds in datasets:
    ds.smooth_spk(50, name='smooth')

#%% 

def fit_and_eval_decoder(
    train_rates: np.ndarray,
    train_behavior: np.ndarray,
    eval_rates: np.ndarray,
    eval_behavior: np.ndarray,
    grid_search: bool=True,
    param_grid: np.ndarray=np.logspace(-5, 5, 20),
    return_preds: bool=False
):
    """Fits ridge regression on train data passed
    in and evaluates on eval data

    Parameters
    ----------
    train_rates :
        2d array time x units.
    train_behavior :
        2d array time x output dims.
    eval_rates :
        2d array time x units
    eval_behavior :
        2d array time x output dims
    grid_search :
        Whether to perform a cross-validated grid search to find
        the best regularization hyperparameters.

    Returns
    -------
    float
        Uniform average R2 score on eval data
    """
    if np.any(np.isnan(train_behavior)):
        train_rates = train_rates[~np.isnan(train_behavior)[:, 0]]
        train_behavior = train_behavior[~np.isnan(train_behavior)[:, 0]]
    if np.any(np.isnan(eval_behavior)):
        eval_rates = eval_rates[~np.isnan(eval_behavior)[:, 0]]
        eval_behavior = eval_behavior[~np.isnan(eval_behavior)[:, 0]]
    assert not np.any(np.isnan(train_rates)) and not np.any(np.isnan(eval_rates)), \
        "fit_and_eval_decoder: NaNs found in rate predictions within required trial times"

    if grid_search:
        decoder = GridSearchCV(Ridge(), {"alpha": param_grid})
    else:
        decoder = Ridge(alpha=1e-2)
    decoder.fit(train_rates, train_behavior)
    if return_preds:
        return decoder.score(eval_rates, eval_behavior), decoder, decoder.predict(eval_rates)
    else:
        return decoder.score(eval_rates, eval_behavior), decoder

# %%
from decoder_demos.decoding_utils import fit_and_eval_decoder, generate_lagged_matrix, apply_neural_behavioral_lag

perfs = []
for N_HIST in [0]:
    # VAL_RATIO = 0.2
    for ds in datasets:
        ds.data = ds.data.dropna()
    eval_mask = datasets[0].data.eval_mask.values.squeeze().astype('bool')
    X_train = generate_lagged_matrix(datasets[0].data.spikes.values[eval_mask, :], N_HIST)
    y_train = datasets[0].data.preprocessed_emg.values[eval_mask, :][N_HIST:, :]

    eval_mask_test = datasets[1].data.eval_mask.values.squeeze().astype('bool')
    X_test = generate_lagged_matrix(datasets[1].data.spikes.values[eval_mask_test, :], N_HIST)
    y_test = datasets[1].data.preprocessed_emg.values[eval_mask_test, :][N_HIST:, :]

    # from sklearn.preprocessing import StandardScaler
    # y = StandardScaler().fit_transform(y)

    # n_train = int(X.shape[0] * (1 - VAL_RATIO))

    r2, decoder, y_pred = fit_and_eval_decoder(
        X_train, 
        y_train, 
        X_test, 
        y_test,
        grid_search=True, 
        return_preds=True)
    perfs.append(r2)

# %%

plt.plot([0, 1, 2, 3, 4, 5, 6], perfs, '.-')
plt.title('with Pec_maj')

# %%

perfs = []
for N_HIST in [0]:
    # VAL_RATIO = 0.2
    for ds in datasets:
        ds.data = ds.data.dropna()
    eval_mask = datasets[0].data.eval_mask.values.squeeze().astype('bool')
    X_train = generate_lagged_matrix(datasets[0].data.spikes.values[eval_mask, :], N_HIST)
    y_train = datasets[0].data.preprocessed_emg.values[eval_mask, :][N_HIST:, :][:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15]]

    eval_mask_test = datasets[1].data.eval_mask.values.squeeze().astype('bool')
    X_test = generate_lagged_matrix(datasets[1].data.spikes.values[eval_mask_test, :], N_HIST)
    y_test = datasets[1].data.preprocessed_emg.values[eval_mask_test, :][N_HIST:, :][:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15]]

    # from sklearn.preprocessing import StandardScaler
    # y = StandardScaler().fit_transform(y)

    # n_train = int(X.shape[0] * (1 - VAL_RATIO))

    r2, decoder, y_pred = fit_and_eval_decoder(
        X_train, 
        y_train, 
        X_test, 
        y_test,
        grid_search=True, 
        return_preds=True)
    perfs.append(r2)

# %%
