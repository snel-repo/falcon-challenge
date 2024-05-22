#%% 
import pickle, os, h5py, sys, copy
import numpy as np
from pynwb import NWBHDF5IO
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
import matplotlib.pyplot as plt

#%% 

DATA_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/m2/sub-MonkeyN-held-in-calib/sub-MonkeyN-held-in-calib_ses-2020-10-28-Run1_behavior+ecephys.nwb'
BASE_PATH = '/snel/share/runs/falcon'
TRACK = 'M2'

CORR_CH_THRESH = 0.6

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
orig_spikes = copy.deepcopy(ds.data.spikes.values)

#%% 

threshold = 0.5
zero_bins = False

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

# print(time_idx)

coin_rem_spikes = copy.deepcopy(ds.data.spikes.values)

#%% 
corrs, dropped_chans = ds.get_pair_xcorr(
        'spikes',
        threshold=CORR_CH_THRESH,
        zero_chans=True
    )

print(dropped_chans)

#%% 

fig, ax = plt.subplots(3,1,facecolor='w', figsize=(10,10), sharex=True)
ax[0].imshow(orig_spikes.T, aspect='auto', cmap='gray_r', origin='lower', vmin=0, vmax=1)
ax[0].set_title('Original')
ax[1].imshow(coin_rem_spikes.T, aspect='auto', cmap='gray_r', origin='lower', vmin=0, vmax=1)
ax[1].set_title(f'Coincident Spike Removal, Threshold={threshold}, Removed {rem_spikes} bins')
ax[2].imshow(ds.data.spikes.values.T, aspect='auto', cmap='gray_r', origin='lower', vmin=0, vmax=1)
ax[2].set_title(f'Cross Correlation Removal, Threshold={CORR_CH_THRESH}, Removed {len(dropped_chans)} channels')
plt.show()

# %%
