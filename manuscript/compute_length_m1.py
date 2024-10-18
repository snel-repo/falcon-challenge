#%% 

import numpy as np 
from pynwb import NWBHDF5IO
import glob
from snel_toolkit.datasets.nwb import NWBDataset

#%% 

files = glob.glob('/snel/home/bkarpo2/bin/falcon-challenge/data/m1/eval/*.nwb')


# %%

lengths = []
for f in files: 
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
    lengths.append(ds.data.shape[0] * 0.02) # seconds 
    
# %%
# lengths are in seconds

total_len = np.sum(lengths)
len_min = total_len/60
len_hours = len_min/60
# %%
