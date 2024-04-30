#%%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pynwb import NWBHDF5IO
from data_demos.styleguide import set_style
set_style()

data_dir = Path('data/')
import pickle

with open('local_gt.pkl', 'rb') as f:
    gt = pickle.load(f)
with open('local_prediction.pkl', 'rb') as f:
    pred = pickle.load(f)
# MonkeyNRun1_20201019
session = '2020-10-19-Run1'
session = '2020-10-20-Run1'
labels = gt['m2'][session]['data']
mask = gt['m2'][session]['mask']
preds = pred['m2'][session]

plt.plot(labels[:,0])
plt.plot(preds[:, 0])
# train_files = sorted((data_dir / 'held_in_calib').glob('*calib.nwb'))
# test_files = sorted((data_dir / 'held_out_calib').glob('*calib.nwb'))
# from falcon_challenge.dataloaders import bin_units, load_nwb
# from falcon_challenge.config import FalconTask
# BIN_SIZE_S = 0.02

# spikes, vel, time, eval_mask = load_nwb(train_files[0], dataset=FalconTask.h1)
# breakpoint()