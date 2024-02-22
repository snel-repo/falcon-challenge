#%%

# Preprocessing to go from relatively raw server transfer to NWB format.

# 1. Extract raw datapoints
import math
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d # Upsample by simple interp

from datetime import datetime
from uuid import uuid4

from dateutil.tz import tzlocal

from pynwb import NWBFile, NWBHDF5IO, TimeSeries

root = Path('./data/h1')
# List files
files = list(root.glob('*.mat'))

sample = files[0]
# Files are of the form

#%%
CHANNELS_PER_SOURCE = 128
BIN_SIZE_MS = 20
TARGET_BIN_SIZE_MS = 10
TARGET_BIN_SIZE_S = TARGET_BIN_SIZE_MS / 1000
FEW_SHOT_CALIBRATION_RATIO = 0.2
EVAL_RATIO = 0.2

CURATED_SETS = {
    'S53_set_1': 'train',
    'S53_set_2': 'train',
    'S63_set_1': 'train',
    'S63_set_2': 'train',
    'S77_set_1': 'test_short',
    'S77_set_2': 'test_short',
    'S91_set_1': 'test_long',
    'S91_set_2': 'test_long',
    'S95_set_1': 'test_long',
    'S95_set_2': 'test_long',
    'S99_set_1': 'test_long',
    'S99_set_2': 'test_long',
}

def create_nwb_name(mat_name: Path) -> Path:
    root = mat_name.parent
    name = mat_name.stem
    name = name.split('session_')[-1]
    return (root / name).with_suffix('.nwb')

def create_nwb_shell():
    # Note identifying info is redacted
    return NWBFile(
        session_description="Open loop 7DoF calibration.",
        identifier=str(uuid4()),
        session_start_time=datetime(2017, 1, 1, 1, tzinfo=tzlocal()),
        lab="Rehab Neural Engineering Labs",
        institution="University of Pittsburgh",
        experiment_description="Open loop calibration for Action Research Arm Test (ARAT) for human motor BCI",
    )

def write_to_nwb(nwbfile: NWBFile, fn: Path):
    fn.parent.mkdir(exist_ok=True, parents=True)
    with NWBHDF5IO(str(fn), 'w') as io:
        io.write(nwbfile)

def crop_spikes(spike_times, spike_channels, bin_times):
    r"""
        spike_times: 1D array of spike time events
        spike_channels: 1D array of spike channels (associated with spike times)
        bin_times: Separate dense clock of target bin times. Assumed continuous.
    """
    return (
        spike_times[(spike_times >= bin_times[0]) & (spike_times < bin_times[-1] + TARGET_BIN_SIZE_S)] - bin_times[0],
        spike_channels[(spike_times >= bin_times[0]) & (spike_times < bin_times[-1] + TARGET_BIN_SIZE_S)]
    )


def to_nwb(fn):
    tag = create_nwb_name(fn).stem
    if tag not in CURATED_SETS:
        return
    payload = loadmat(str(fn), simplify_cells=True, variable_names=['thin_data'])['thin_data']
    raw_spike_channel = payload['source_index'] * CHANNELS_PER_SOURCE + payload['channel']
    raw_spike_time = payload['source_timestamp']
    bin_time = payload['spm_source_timestamp'] # Indicates end of 20ms bin, in seconds. Shape is (recording devices x Timebin), each has own clock
    bin_kin = payload['open_loop_kin'] # Shape is (K x Timebin)
    bin_state = payload['state_num'] # Shape is (1 x Timebin)
    bin_trial = payload['trial_num'] # Shape is (1 x Timebin)

    # Mark pretrial period as first trial - it's a small buffer
    # Assert nans only occur at start of array, and are only a few
    nan_mask = np.isnan(bin_trial)
    assert np.all(nan_mask[:np.argmax(~nan_mask)])
    assert np.sum(nan_mask) < 10
    bin_trial[nan_mask] = 1
    bin_trial = bin_trial.astype(int)

    if 'S608_set_1' in str(fn): # Fail
        return
    if 'S615_set_1' in str(fn): # Fail
        return
    if 'S594_set_1' in str(fn): # Fail
        return

    state_names = payload['state_names'] # Shape is ~15 (arbitrary, exp dependent)
    state_names[0] = 'Pretrial'

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

    # Resample position information - upsample from 50 to 100Hz.
    assert(bin_kin.shape[0] == bin_time_native.shape[0])
    bin_time = np.arange(0, bin_time_native[-1], 0.01)
    bin_kin = interp1d(bin_time_native, bin_kin, axis=0, bounds_error=False)(bin_time)
    bin_trial_native = bin_trial
    bin_trial = interp1d(bin_time_native, bin_trial, bounds_error=False, kind='nearest', fill_value='extrapolate')(bin_time)
    # bin_state kept at native time
    bin_state = bin_state - 1 # 1-indexed to 0-indexed
    def create_nwb_container(
        spike_time: np.ndarray,
        spike_channel: np.ndarray,
        bin_time: np.ndarray, # Resampled
        bin_time_native: np.ndarray, # For state
        bin_kin: np.ndarray, # Resampled
        bin_trial: np.ndarray, # Resampled
        bin_state: np.ndarray, # Native resolution
    ):
        nwbfile = create_nwb_shell()
        for unit_id in motor_units:
            spike_times = spike_time[spike_channel == unit_id]
            nwbfile.add_unit(spike_times=spike_times)

        position_spatial_series = TimeSeries(
            name="OpenLoopKinematics",
            description="tx,ty,tz,rx,ry,rz,grasp",
            timestamps=bin_time,
            data=bin_kin,
            unit="arbitrary",
        )
        nwbfile.add_acquisition(position_spatial_series)

        # Create a TimeSeries for trial num, semi-hack - states should likely not be its own TimeSeries
        trial_series = TimeSeries(
            name="TrialNum",
            description="Trial number",
            timestamps=bin_time,
            data=bin_trial,
            unit="arbitrary",
        )
        nwbfile.add_acquisition(trial_series)

        # Convert states to epoch data
        epoch_start = bin_time_native[0]
        cur_state = bin_state[0]
        for i in range(1, len(bin_state)):
            if bin_state[i] != cur_state:
                # End of state, add interval
                # ! bin_state is not resampled to bin_time_native
                nwbfile.add_epoch(start_time=epoch_start, stop_time=bin_time_native[i], tags=[state_names[cur_state]], timeseries=[position_spatial_series])
                cur_state = bin_state[i]
                epoch_start = bin_time_native[i]
        return nwbfile

    if CURATED_SETS[tag] != "train":
        out_fn = create_nwb_name(fn)

        def create_cropped_container(trial_mask, trial_mask_native):
            print(trial_mask)
            sub_bin_time = bin_time[trial_mask]
            sub_spike_times, sub_channels = crop_spikes(raw_spike_time, raw_spike_channel, sub_bin_time)
            return create_nwb_container(
                sub_spike_times,
                sub_channels,
                sub_bin_time,
                bin_time_native[trial_mask_native],
                bin_kin[trial_mask],
                bin_trial[trial_mask],
                bin_state[trial_mask_native],
            )
        def create_and_write(trial_mask, trial_mask_native, suffix):
            nwbfile = create_cropped_container(trial_mask, trial_mask_native)
            out = fn.parent / CURATED_SETS[tag] / f"{out_fn.stem}_{suffix}.nwb"
            write_to_nwb(nwbfile, out)

        trials = sorted(np.unique(bin_trial))
        calibration_num = math.ceil(len(trials) * FEW_SHOT_CALIBRATION_RATIO)
        calibration_trials = trials[:calibration_num]
        create_and_write(bin_trial < calibration_trials[-1],
                         bin_trial_native < calibration_trials[-1],
                         'calibration')
        eval_num = int(len(trials) * EVAL_RATIO)
        eval_trials = trials[-eval_num:]
        create_and_write(bin_trial >= eval_trials[0],
                         bin_trial_native >= eval_trials[0],
                         'eval')

        in_day_oracle_num = int(len(trials) * EVAL_RATIO)
        in_day_oracle = trials[:-in_day_oracle_num]
        create_and_write(bin_trial < in_day_oracle[-1],
                         bin_trial_native < in_day_oracle[-1],
                         'in_day_oracle')
        print(f"[0-{calibration_num}) calibration | [-{eval_num}:] eval | [:-{in_day_oracle_num}) oracle")
    else:
        nwbfile = create_nwb_container(
            raw_spike_time,
            raw_spike_channel,
            bin_time,
            bin_time_native,
            bin_kin,
            bin_trial,
            bin_state,
        )

        out_fn = create_nwb_name(fn)
        out = fn.parent / CURATED_SETS[tag] / out_fn.name
        write_to_nwb(nwbfile, out)
for sample in files:
    to_nwb(sample)