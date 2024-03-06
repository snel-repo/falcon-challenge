#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from filtering import apply_butter_filt
from pynwb import NWBHDF5IO
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

#%% 
def load_nwb(fn: str):
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()

        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
        )

        units = nwbfile.units.to_dataframe()
        # raw_emg = nwbfile.get_acquisition('emg')
        raw_emg = nwbfile.processing['emg_filtering_module'].get_data_interface('preprocessed_emg')
        muscles = [ts for ts in raw_emg.time_series]
        emg_data = []
        emg_timestamps = []
        for m in muscles: 
            mdata = raw_emg.get_timeseries(m)
            data = mdata.data[:]
            timestamps = mdata.timestamps[:]
            emg_data.append(data)
            emg_timestamps.append(timestamps)

        emg_data = np.vstack(emg_data).T 
        emg_timestamps = emg_timestamps[0]

        trial_info.start_time = trial_info.start_time.round(2)
        trial_info.end_time = trial_info.end_time.round(2)

        return (
            bin_units(units, bin_size_s=0.02),
            emg_timestamps[::20],
            muscles,
            trial_info,
        )
# %%

root = Path('/snel/share/share/derived/rouse/RTG/NWB_FALCON/full')
files = list(root.glob('*.nwb'))

neur_t = []
neur_n = []
emg_t = []
trials = []

for f in files: 
    out = load_nwb(f)
    neur_t.append(out[0].shape[0])
    neur_n.append(out[0].shape[1])
    emg_t.append(out[1].shape[0])
    trials.append(out[3].shape[0])

print('full')
print(neur_t)
print(neur_n)
print(emg_t)
print(trials)

# %%

root = Path('/snel/share/share/derived/rouse/RTG/NWB_FALCON/minival')
files = list(root.glob('*.nwb'))

neur_t = []
neur_n = []
emg_t = []
trials = []

for f in files: 
    out = load_nwb(f)
    neur_t.append(out[0].shape[0])
    neur_n.append(out[0].shape[1])
    emg_t.append(out[1].shape[0])
    trials.append(out[3].shape[0])

print('minival')
print(neur_t)
print(neur_n)
print(emg_t)
print(trials)
# %%
root = Path('/snel/share/share/derived/rouse/RTG/NWB_FALCON/in_day_oracle')
files = list(root.glob('*.nwb'))

neur_t = []
neur_n = []
emg_t = []
trials = []

for f in files: 
    out = load_nwb(f)
    neur_t.append(out[0].shape[0])
    neur_n.append(out[0].shape[1])
    emg_t.append(out[1].shape[0])
    trials.append(out[3].shape[0])

print('oracle')
print(neur_t)
print(neur_n)
print(emg_t)
print(trials)
# %%
root = Path('/snel/share/share/derived/rouse/RTG/NWB_FALCON/eval')
files = list(root.glob('*.nwb'))

neur_t = []
neur_n = []
emg_t = []
trials = []

for f in files: 
    out = load_nwb(f)
    neur_t.append(out[0].shape[0])
    neur_n.append(out[0].shape[1])
    emg_t.append(out[1].shape[0])
    trials.append(out[3].shape[0])

print('eval')
print(neur_t)
print(neur_n)
print(emg_t)
print(trials)

# %%
root = Path('/snel/share/share/derived/rouse/RTG/NWB_FALCON/calibration')
files = list(root.glob('*.nwb'))

neur_t = []
neur_n = []
emg_t = []
trials = []

for f in files: 
    out = load_nwb(f)
    neur_t.append(out[0].shape[0])
    neur_n.append(out[0].shape[1])
    emg_t.append(out[1].shape[0])
    trials.append(out[3].shape[0])

print('calib')
print(neur_t)
print(neur_n)
print(emg_t)
print(trials)
# %%
