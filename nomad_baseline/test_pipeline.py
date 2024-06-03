#%% 
import pickle, os, sys, glob, yaml
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.falcon_h1 import H1Dataset
from snel_toolkit.interfaces import LFADSInterface
from snel_toolkit.analysis import PSTH
from decoder_demos.decoding_utils import fit_and_eval_decoder, generate_lagged_matrix, apply_neural_behavioral_lag
from tune_tf2.pbt.utils import plot_pbt_hps, plot_pbt_log
from causal_samp import get_causal_model_output, merge_data
from lfads_tf2.subclasses.dimreduced.models import DimReducedLFADS
from decoder_demos.filtering import apply_exponential_filter

sys.path.insert(0, '/home/bkarpo2/bin/stability-benchmark/align_tf2')
for p in sys.path: 
    if 'nomad_dev' in p: 
        sys.path.remove(p)
from align_tf2.models import AlignLFADS

np.random.seed(731)

#%% 

TRACK = 'h1'
# KEY = 'Run1_20201019'
# KEY = 'Run1_20201118'
KEY = 'S12_set_2'

datapath = f'/home/bkarpo2/bin/falcon-challenge/data/{TRACK}'
datafile = glob.glob(os.path.join(datapath, 'eval', f'*{KEY}*.nwb'))[0]

# KEY = 'Run1_20201030'

# %%
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.config import FalconTask

neural, kin, trial_change, eval_mask = load_nwb(datafile, dataset=FalconTask.h1)
# %%

with open('/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submissions.yaml', 'rb') as f: 
    submissions = yaml.safe_load(f)
track_subs = [x for x in submissions['submissions'] if x['track'] == TRACK][0]
model_path = track_subs[KEY]['model']
decoder_path = track_subs[KEY]['decoder']

# %%

# model = DimReducedLFADS(model_dir=model_path)
model = AlignLFADS(align_dir=model_path)
model = model.lfads_dayk
lfads_out = get_causal_model_output(
    model = model,
    binsize = 20,
    input_data = neural.astype('float64'),
    out_fields = ['rates', 'factors', 'gen_states'],
    output_dim = {'rates': model.cfg.MODEL.DATA_DIM, 
                'factors': model.cfg.MODEL.FAC_DIM, 
                'gen_states': model.cfg.MODEL.GEN_DIM}
)

#%% 
gen_states = lfads_out['gen_states']

# %%
from decoder_demos.decoding_utils import generate_lagged_matrix

with open(decoder_path, 'rb') as f:
    decoder_info = pickle.load(f)

history = decoder_info['history']
decoder = decoder_info['decoder']

decoder_input = generate_lagged_matrix(gen_states, history)
# %%
preds = decoder.predict(decoder_input)
# %%
from sklearn.metrics import r2_score

r2_score(kin[eval_mask,:][history:], preds[eval_mask[history:],:], multioutput='variance_weighted')
# %%
# from decoder_demos.decoding_utils import fit_and_eval_decoder
# N_HIST = 12
# VAL_RATIO = 0.2
# eval_mask = eval_mask.squeeze().astype('bool')
# X = generate_lagged_matrix(gen_states[eval_mask, :], N_HIST)
# y = kin[eval_mask,:][N_HIST:, :]

# n_train = int(X.shape[0] * (1 - VAL_RATIO))

# r2, decoder, y_pred = fit_and_eval_decoder(
#     X[:n_train, :], 
#     y[:n_train, :], 
#     X[n_train:, :], 
#     y[n_train:, :], 
#     grid_search=True, 
#     param_grid=np.logspace(2, 3, 20),
#     return_preds=True)

# print(f'Day 0 R2: {r2}')
# %%
