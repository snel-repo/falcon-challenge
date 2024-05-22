#%% 
import pickle, os, h5py, sys
import numpy as np
from pynwb import NWBHDF5IO
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
from snel_toolkit.interfaces import LFADSInterface

#%% 
ds_str = '2020-10-28-Run1'

DATA_PATH = f'/snel/home/bkarpo2/bin/falcon-challenge/data/m2/sub-MonkeyN-held-in-calib/sub-MonkeyN-held-in-calib_ses-{ds_str}_behavior+ecephys.nwb'
BASE_PATH = '/snel/share/runs/falcon'
TRACK = 'M2'
RUN_FLAG = f'{ds_str}_coinspkrem'

if len(sys.argv) > 2:
    DATA_PATH = sys.argv[1]
    TRACK = sys.argv[2]
    RUN_FLAG = sys.argv[3]

CHOP_LEN = 1000 #ms
OLAP_LEN = 200 #ms 
VALID_RATIO = 0.2
NORM_SM_MS = 20
CORR_CH_THRESH = 0.6
REMOVE_COIN_SPK = True

#%% 

run_path = os.path.join(BASE_PATH, f'{TRACK}_{RUN_FLAG}')
input_data_path = os.path.join(run_path, 'input_data')
input_data_file = os.path.join(input_data_path, f'lfads_{TRACK}_{RUN_FLAG}.h5')

for p in [run_path, input_data_path]:
    if not os.path.exists(p):
        os.makedirs(p)

#%%
if TRACK == 'M1': 
    skip_fields = ['preprocessed_emg', 'eval_mask']
    # load and rebin the spikes 
    ds_spk = NWBDataset(
        DATA_PATH, 
        skip_fields=skip_fields)
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
corrs, dropped_chans = ds.get_pair_xcorr(
        'spikes',
        threshold=CORR_CH_THRESH,
        zero_chans=True
    )

#%% 
if NORM_SM_MS > 0:
    ds.smooth_spk(NORM_SM_MS, name='smooth')
    ss = ds.data.spikes_smooth.values
    ch_mean = np.nanmean(ss, axis=0)
    ch_std = np.nanstd(ss, axis=0)
    ch_std[ch_std == 0] = 1 #avoid NaNs by dealing with channels of 0's 
    mat = np.diag(1./ch_std)
    norm_ss = np.dot(ss, mat) - np.dot(ch_mean, mat)
    assert (all(np.nanmean(norm_ss, axis=0) < 1e-12))
    check_std = np.nanstd(norm_ss, axis=0)
    assert (all(check_std[check_std != 0] >= 0.99999) and all(check_std[check_std != 0] <= 1.0001)) # controlling for zero-filled channels
    # save the weights and biases 
    print('Saving normalization weights and biases...')
    hf = h5py.File(
        os.path.join(input_data_path, 'normalization.h5'), 'w')
    hf.create_dataset('matrix', data=mat)
    hf.create_dataset('bias', data=np.dot(ch_mean, mat))
    hf.close()

# %%
print(f'Saving input data...')
lfi = LFADSInterface(
    window=CHOP_LEN, 
    overlap=OLAP_LEN,
    chop_fields_map={
                    'spikes': 'data', 
                    }
)

lfi.chop_and_save(ds.data, input_data_file, valid_ratio=VALID_RATIO, overwrite=True)
print('Saved input data to ' + input_data_file)

with open(os.path.join(input_data_path, 'interface.pkl'), 'wb') as f: 
    pickle.dump(lfi, f)

# %%
