#%% 
import pickle, os, h5py, sys, glob
import numpy as np 
from pynwb import NWBHDF5IO
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
from snel_toolkit.interfaces import LFADSInterface

#%% 

DATA_PATH = '/snel/home/bkarpo2/bin/falcon-challenge/data/h1/held_in_calib'
SESS_ID = 'S1'
BASE_PATH = '/snel/share/runs/falcon'
TRACK = 'H1'
RUN_FLAG = 'combined_day0_larger_capacity'

if len(sys.argv) > 2:
    DATA_PATH = sys.argv[1]
    TRACK = sys.argv[2]
    RUN_FLAG = sys.argv[3]

CHOP_LEN = 1000 #ms
OLAP_LEN = 200 #ms 
VALID_RATIO = 0.2 
NORM_SM_MS = 20
CORR_CH_THRESH = 0.6

#%% 

run_path = os.path.join(BASE_PATH, f'{TRACK}_{SESS_ID}_{RUN_FLAG}')
input_data_path = os.path.join(run_path, 'input_data')
input_data_file = os.path.join(input_data_path, f'lfads_{TRACK}_{RUN_FLAG}.h5')

for p in [run_path, input_data_path]:
    if not os.path.exists(p):
        os.makedirs(p)

#%% 

dataset_files = glob.glob(
    os.path.join(DATA_PATH, f'{SESS_ID}*.nwb')
)

#%%

datasets = []
if TRACK == 'H1': 
    for f in dataset_files:
        ds = H1Dataset(f)
        datasets.append(ds)
#%% 

for ds in datasets: 
    corrs, dropped_chans = ds.get_pair_xcorr(
            'spikes',
            threshold=CORR_CH_THRESH,
            zero_chans=True
        )
    print(f'Dropping {len(dropped_chans)} channels due to high correlations.')

#%% 
if NORM_SM_MS > 0:
    all_ss = []
    for ds in datasets:
        ds.smooth_spk(NORM_SM_MS, name='smooth')
        ss = ds.data.spikes_smooth.values
        all_ss.append(ss)
    ss = np.concatenate(all_ss, axis=0)
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

chops = []

for ds in datasets:
    lfi = LFADSInterface(
        window=CHOP_LEN, 
        overlap=OLAP_LEN,
        chop_fields_map={
                        'spikes': 'data', 
                        }
    )

    datadict = lfi.chop(ds.data)
    chops.append(datadict['data'])

    with open(os.path.join(input_data_path, ds.fpath.split('/')[-1].split('.')[0]+'_interface.pkl'), 'wb') as f: 
        pickle.dump(lfi, f)


# %%

all_chops = np.concatenate(chops, axis=0)
# split into train and valid 
n_chops = all_chops.shape[0]
n_valid = int(n_chops * VALID_RATIO)
n_train = n_chops - n_valid
inds = np.arange(n_chops)
np.random.shuffle(inds)
train_inds = inds[:n_train]
valid_inds = inds[n_train:]

train_data = all_chops[train_inds, :, :]
valid_data = all_chops[valid_inds, :, :]

hf = h5py.File(input_data_file, 'w')
hf.create_dataset('train_data', data=train_data)
hf.create_dataset('valid_data', data=valid_data)
hf.create_dataset('train_inds', data=train_inds)
hf.create_dataset('valid_inds', data=valid_inds)
hf.close()

# %%
