#%%
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime
from styleguide import set_style
set_style()
from pynwb import NWBHDF5IO
from falcon_challenge.dataloaders import bin_units

#%%
base_path = '/snel/home/bkarpo2/bin/falcon-challenge/data'
track = 'm2'
held_in_files = glob.glob(os.path.join(base_path, track, '*held-in-calib*', '*.nwb'))
held_out_files = glob.glob(os.path.join(base_path, track, '*held-out-calib*', '*.nwb'))

#%%

if track == 'm1' or track == 'h2':
    heldin_time_s = []
    heldout_time_s = []
elif track == 'h1' or track == 'm2':
    heldin_time_s = {}
    heldout_time_s = {}

for files, time_s in zip([held_in_files, held_out_files], [heldin_time_s, heldout_time_s]):
    for f in files:
        with NWBHDF5IO(f, 'r') as io:
            nwbfile = io.read()
            if track == 'h1': 
                code = f.split('/')[-1].split('_')[0]
                units = nwbfile.units.to_dataframe()
                h1_beh = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:]
                timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(h1_beh.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
                h1_spikes = bin_units(units, bin_size_s=0.02, bin_timestamps=timestamps)
                if code in time_s: 
                    time_s[code] += h1_spikes.shape[0] * 0.02
                else:
                    time_s[code] = h1_spikes.shape[0] * 0.02
            elif track == 'h2': 
                h2_spikes = nwbfile.acquisition['binned_spikes'].data[()]
                time_s.append(h2_spikes.shape[0] * 0.02)
            elif track == 'm1': 
                m1_trial_info = (
                    nwbfile.trials.to_dataframe()
                    .reset_index()
                    .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
                )
                raw_emg = nwbfile.acquisition['preprocessed_emg']
                muscles = [ts for ts in raw_emg.time_series]
                emg_data = []
                emg_timestamps = []
                for m in muscles: 
                    mdata = raw_emg.get_timeseries(m)
                    data = mdata.data[:]
                    timestamps = mdata.timestamps[:]
                    emg_data.append(data)
                    emg_timestamps.append(timestamps)

                m1_beh = np.vstack(emg_data).T 
                emg_timestamps = emg_timestamps[0]
                units = nwbfile.units.to_dataframe()
                m1_spikes = bin_units(units, bin_size_s=0.02, bin_timestamps=emg_timestamps)
                time_s.append(m1_spikes.shape[0] * 0.02)
            elif track == 'm2': 
                code = ''.join(f.split('/')[-1].split('_')[1].split('-')[1:4])
                vel_container = nwbfile.acquisition['finger_vel']
                labels = [ts for ts in vel_container.time_series]
                vel_data = []
                vel_timestamps = None
                for ts in labels: 
                    ts_data = vel_container.get_timeseries(ts)
                    vel_data.append(ts_data.data[:])
                    vel_timestamps = ts_data.timestamps[:]
                    units = nwbfile.units.to_dataframe()
                m2_spikes = bin_units(units, bin_size_s=0.02, bin_timestamps=vel_timestamps, is_timestamp_bin_start=True)
                if code in time_s: 
                    time_s[code] += m2_spikes.shape[0] * 0.02
                else:
                    time_s[code] = m2_spikes.shape[0] * 0.02
# %%
if track == 'h1' or track == 'm2':
    heldin_time_s = np.array(list(heldin_time_s.values()))
    heldout_time_s = np.array(list(heldout_time_s.values()))

heldin_time_s = np.array(heldin_time_s)
heldout_time_s = np.array(heldout_time_s)

print(np.min(heldin_time_s/60), np.max(heldin_time_s/60))
print(np.min(heldout_time_s/60), np.max(heldout_time_s/60))
# %%
