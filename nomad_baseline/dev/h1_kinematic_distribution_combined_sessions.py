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

#%% 

DAY0_PATH = '/home/bkarpo2/bin/stability-benchmark/data/h1/held_in_calib/S2_set_1_calib.nwb'
eval_sessions = sorted(glob.glob(os.path.join('/home/bkarpo2/bin/stability-benchmark/data/h1/held_out_calib', '*.nwb')))

ds0 = H1Dataset(DAY0_PATH)

kin0 = ds0.data.OpenLoopKinematicsVelocity.values 
n_dims = kin0.shape[-1]

# test keys are S6-S12

#%% 

dim_labels = ['Tx', 'Ty', 'Tz', 'R', 'G1', 'G2', 'G3']

for sess in range(6, 13):
    rel_sets = [f for f in eval_sessions if f.split('/')[-1].split('_')[0] == f'S{sess}']
    all_kin_k = []
    for es in rel_sets:
        ds = H1Dataset(es)
        kin_k = ds.data.OpenLoopKinematicsVelocity.values
        all_kin_k.append(kin_k)

    kin_k = np.concatenate(all_kin_k, axis=0)
    fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharey=True)
    plt.subplots_adjust(hspace=0.35)
    ax = axs.flatten()
    for i in range(n_dims):
        ax[i].hist(kin0[:,i], bins=50, color='b', alpha=0.5, label='S2_set_1')
        ax[i].hist(kin_k[:,i], bins=50, color='r', alpha=0.5, label=f'S{sess}')
        ax[i].set_ylim([0, 750])
        ax[i].set_title(dim_labels[i])
        ax[i].set_xlabel('Velocity')
        ax[i].set_ylabel('Count')
    ax[6].legend()
    plt.savefig(f'kinematic_distribution_S{sess}.png')

# %%
