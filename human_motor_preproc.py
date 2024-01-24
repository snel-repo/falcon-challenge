#%%

# Preprocessing to go from relatively raw server transfer to NWB format.

# 1. Extract raw datapoints
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d # Upsample by simple interp

from datetime import datetime
from uuid import uuid4

from dateutil.tz import tzlocal

from pynwb import NWBFile, NWBHDF5IO, TimeSeries

root = Path('./data/human_motor')

# List files
files = list(root.glob('*.mat'))

sample = files[0]
# Files are of the form

#%%
CHANNELS_PER_SOURCE = 128
BIN_SIZE_MS = 20
def to_nwb(fn):
    # if fn.with_suffix('.nwb').exists():
        # return
    payload = loadmat(str(fn), simplify_cells=True, variable_names=['thin_data'])['thin_data']
    raw_spike_channel = payload['source_index'] * CHANNELS_PER_SOURCE + payload['channel']
    raw_spike_time = payload['source_timestamp']
    bin_time = payload['spm_source_timestamp'] # Indicates end of 20ms bin, in seconds. Shape is (recording devices x Timebin), each has own clock
    bin_kin = payload['open_loop_kin'] # Shape is (K x Timebin)
    bin_state = payload['state_num'] # Shape is (1 x Timebin)

    if 'S608_set_1' in str(fn): # Fail
        return
    if 'S615_set_1' in str(fn): # Fail
        return
    if 'S594_set_1' in str(fn): # Fail
        return

    state_names = payload['state_names'] # Shape is ~15 (arbitrary, exp dependent)

    # Align all times to start of 1st bin of data.
    for recording_box in range(bin_time.shape[0]):
        raw_spike_time[payload['source_index'] == recording_box] -= bin_time[recording_box][0] - BIN_SIZE_MS/1000
        bin_time[recording_box] -= bin_time[recording_box][0] - BIN_SIZE_MS/1000
        # Clip other recorded spikes to timestep 0, assuming not too negative
        # assert np.all(raw_spike_time[payload['source_index'] == recording_box] >= -(BIN_SIZE_MS/1000)/4)
        # raw_spike_time[payload['source_index'] == recording_box] = np.clip(raw_spike_time[payload['source_index'] == recording_box], 0, None)
    bin_time_native = bin_time[0] # Aligned, there's only need for one clock
    r"""
        Create NWB metadata
    """
    nwbfile = NWBFile(
        session_description="Open loop 5-7DoF calibration.",
        identifier=str(uuid4()),
        # This experiment occurred on 5/2/17
        session_start_time=datetime(2017, 5, 2, 12, tzinfo=tzlocal()),
        experimenter=[
            "Sharlene Flesher",
        ],
        lab="Rehab Neural Engineering Labs",
        institution="University of Pittsburgh",
        experiment_description="Open loop calibration for Action Research Arm Test (ARAT) for human motor BCI",
        session_id=f'P2_{sample.stem}',
    )

    motor_units = np.concatenate([
        np.arange(64) + 1,
        np.arange(32) + 96 + 1,
        np.arange(64) + 128 + 1,
        np.arange(32) + 128 + 64 + 1,
    ])
    # Subtract 114, 116, 118, 120, 122, 124, 126, 128, which are not wired for P2
    # Do this for both pedestals
    motor_units = np.setdiff1d(motor_units, np.arange(114, 128+1, 2))
    motor_units = np.setdiff1d(motor_units, np.arange(114+64, 128+64+1, 2))
    for unit_id in motor_units:
        spike_times = raw_spike_time[raw_spike_channel == unit_id]
        nwbfile.add_unit(spike_times=spike_times)

    # Add position information - upsample from 50 to 100Hz.
    # print(bin_time)
    assert(bin_kin.shape[0] == bin_time_native.shape[0])
    bin_kin = interp1d(bin_time_native, bin_kin, axis=0)(np.arange(0, bin_time_native[-1], 0.01))
    bin_time = np.arange(0, bin_time_native[-1], 0.01)

    position_spatial_series = TimeSeries(
        name="OpenLoopKinematics",
        description="Open Loop Kinematic Labels (x,y,z, roll, pitch, yaw, grasp)",
        timestamps=bin_time,
        data=bin_kin,
        unit="arbitrary",
    )
    nwbfile.add_acquisition(position_spatial_series)

    # Create a TimeSeries for task states, semi-hack - states should likely not be its own TimeSeries
    # Convert states to epoch data
    bin_state = bin_state - 1 # 1-indexed to 0-indexed
    state_names[0] = 'Pretrial'
    epoch_start = bin_time_native[0]
    cur_state = bin_state[0]
    for i in range(1, len(bin_state)):
        if bin_state[i] != cur_state:
            # End of state, add interval
            nwbfile.add_epoch(start_time=epoch_start, stop_time=bin_time_native[i], tags=[state_names[cur_state]], timeseries=[position_spatial_series])
            cur_state = bin_state[i]
            epoch_start = bin_time_native[i]

    # Save the NWB file
    with NWBHDF5IO(str(
            sample.with_suffix('.nwb')
    ), 'w') as io:
        io.write(nwbfile)

#%%
for sample in files:
    print(sample)
    to_nwb(sample)