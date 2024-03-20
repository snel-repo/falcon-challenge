#%%
r"""
    SN:
    FingerAnglesTIMRL: Contains the normalized positions for all five digits in the range [0, 1] (0 is full extension, one is full flexion). There are five columns (i.e. for thumb, index, middle, ring, and small fingers), though only two are relevant here. The position of the index finger is in index 1 (in Python indices), and the positions of the middle, ring, and small fingers (which were always grouped together) are in index 3. These samples are collected at 1kHz resolution
    Channels: Contains the millisecond times at which a spike was detected (times relative to block beginning, see ExperimentTime below) for each channel. Spikes were detected by the Cerebus at a -4.5RMS threshold after a 250Hz high-pass filter on raw 30kHz data. No re-referencing schemes were applied
    NeuralFeature: Contains the 2kHz samples collected in each 1ms cycle absolute-valued and summed. The number of samples within each 1ms is in the SampleWidth field. One could calculate a spike-band power measure by summing across the time dimension in NeuralFeature and dividing by the sum of the same time window in SampleWidth (i.e. to get a mean). 2kHz samples were filtered to the 300-1,000Hz band by the Cerebus before transmission to our task hardware
    ExperimentTime: Contains the values of a counter that increments every 1ms since the beginning of the block.
    TrialSuccess: Indicates whether the trial's targets were acquired successfully (likely to be all 1 since this was physically controlled by the monkey's fingers)
    BlankTrial: Indicates whether the display was provided to the monkey. Any trials where this is set to 1 should be ignored (i.e. the monkey's screen was off)
    TargetPos: The center positions of each 1D target (5D vector per trial). As with FingerAnglesTIMRL, only Python indices 1 and 3 matter here
    TargetScaling: A pseudo-arbitrary number indicating the size of targets. One can compute the width of targets in the same. units as FingerAnglesTIMRL with width = 0.0375 * (1 + TargetScaling / 100)
    Each element of mat['z'][0] corresponds to one trial
    Note that, due to how the system was developed, there are 2ms missing between each trial. The spike data for these was collected if absolutely needed, but would be a pain to fill it in. These absent 2ms are reflected in ExperimentTimes
"""
# 1. Extract raw datapoints
import math
from pathlib import Path
import numpy as np
from datetime import datetime
from pprint import pprint

from scipy.io import loadmat
from scipy.signal import resample_poly


from dateutil.tz import tzlocal

import pynwb
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralTimeSeries

from decoder_demos.filtering import smooth
from preproc.nwb_create_utils import (
    FEW_SHOT_CALIBRATION_RATIO, EVAL_RATIO, SMOKETEST_NUM, BIN_SIZE_MS,
    create_multichannel_timeseries,
    write_to_nwb
)

root = Path('./data/m2/raw')
files = list(root.glob('*.mat'))
KIN_LABELS = ['index', 'mrs']
CHANNEL_EXPECTATION = 96
TARGET_FINGERS = [1, 3] # Index finger and middle finger group
FS = 1000
pprint(files)

DATA_SPLITS = {
    'Z_Joker_2020-10-19_Run-002': 'train',
    'Z_Joker_2020-10-19_Run-003': 'train',
    'Z_Joker_2020-10-20_Run-002': 'train',
    'Z_Joker_2020-10-20_Run-003': 'train',
    'Z_Joker_2020-10-27_Run-002': 'train',
    'Z_Joker_2020-10-27_Run-003': 'train',
    'Z_Joker_2020-10-28_Run-001': 'train',
    'Z_Joker_2020-10-30_Run-001': 'test',
    'Z_Joker_2020-10-30_Run-002': 'test',
    'Z_Joker_2020-11-18_Run-001': 'test',
    'Z_Joker_2020-11-19_Run-001': 'test',
    'Z_Joker_2020-11-24_Run-001': 'test',
    'Z_Joker_2020-11-24_Run-002': 'test'
}

DATA_RUN_SET = {
    'Z_Joker_2020-10-19_Run-002': 1,
    'Z_Joker_2020-10-19_Run-003': 2,
    'Z_Joker_2020-10-20_Run-002': 1,
    'Z_Joker_2020-10-20_Run-003': 2,
    'Z_Joker_2020-10-27_Run-002': 1,
    'Z_Joker_2020-10-27_Run-003': 2,
    'Z_Joker_2020-10-28_Run-001': 1,
    'Z_Joker_2020-10-30_Run-001': 1,
    'Z_Joker_2020-10-30_Run-002': 2,
    'Z_Joker_2020-11-18_Run-001': 1,
    'Z_Joker_2020-11-19_Run-001': 1,
    'Z_Joker_2020-11-24_Run-001': 1,
    'Z_Joker_2020-11-24_Run-002': 2
}
# Note that intertrial spikes are not available
#%%
def create_nwb_shell(path: Path, split, suffix='full'):
    start_date = path.stem.split('_')[2] # YYYY-MM-DD
    hash = '_'.join(path.stem.split('_')[2:])
    subject = pynwb.file.Subject(
        subject_id=f'MonkeyN-set{DATA_RUN_SET[path.stem]}-{split}-{suffix}',
        description='MonkeyN, Chestek Lab, number indicates experimental set from day',
        species='Rhesus macaque',
        sex='M',
        age='P8Y',
    )
    f = NWBFile(
        session_description='M2 data',
        identifier=f'MonkeyN_{start_date}_{split}',
        subject=subject,
        session_start_time=datetime.strptime(start_date, '%Y-%m-%d'),
        experimenter='Samuel R. Nason and Matthew J. Mender',
        lab='Chestek Lab',
        institution='University of Michicgan',
        experiment_description='Two finger group movement in NHP',
        session_id=hash
    )
    device = f.create_device(name='Blackrock Utah Array', description='96-channel array')
    main_group = f.create_electrode_group(
        name='M1_array',
        description='Hand area 96-channel array',
        location='M1',
        device=device,
    )
    f.add_trial_column(name="tgt_loc", description="location of target (0-1 AU)")
    for i in range(CHANNEL_EXPECTATION):
        f.add_electrode(
            id=i,
            x=np.nan, y=np.nan, z=np.nan,
            imp=np.nan,
            location='M1',
            group=main_group,
            filtering='250Hz HPF, -4.5RMS threshold'
        )
    return f

def filt_single_trial(trial):
    # reduce raw
    assert len(trial['Channel']) == CHANNEL_EXPECTATION
    return {
        'fingers': trial['FingerAnglesTIMRL'][:, TARGET_FINGERS],
        'spikes': trial['Channel'], # spike time to nearest ms
        'time': trial['ExperimentTime'], # Clock time since start of block
        'target': trial['TargetPos'][TARGET_FINGERS]
    }

def to_nwb(path: Path, ):
    full_payload = loadmat(str(path), simplify_cells=True)['z']
    # Skip trial 0 - no data, used for block setup
    full_payload = full_payload[1:]
    full_payload = [i for i in full_payload if not i['BlankTrial']] # 1 if screen was off, no trial run
    full_payload = list(map(filt_single_trial, full_payload))

    def create_and_write(
        payload, suffix
    ):
        nwbfile = create_nwb_shell(path, DATA_SPLITS[path.stem], suffix=suffix)
        all_bhvr = []
        all_spikes = [[] for _ in range(CHANNEL_EXPECTATION)]
        all_time = []
        start_time = 0
        for i, trial_data in enumerate(payload):
            time = trial_data['time'].astype(float)
            if not start_time:
                start_time = time[0]
            time -= start_time
            nwbfile.add_trial(
                start_time=time[0],
                stop_time=time[-1],
                tgt_loc=trial_data['target'],
            )

            # Downsample to 20ms
            time = time[::math.ceil(FS * BIN_SIZE_MS / 1000)] # This effectively rounds up
            bhvr = trial_data['fingers']
            EDGE_PAD = 160 # reduce edge ringing, in ms
            y_padded = np.pad(bhvr, ((EDGE_PAD, EDGE_PAD), (0, 0)), mode='edge',)
            y_resampled_padded = resample_poly(y_padded, math.ceil(FS / BIN_SIZE_MS), 1000)
            bhvr = y_resampled_padded[int(EDGE_PAD / BIN_SIZE_MS):int(-EDGE_PAD / BIN_SIZE_MS)]
            all_bhvr.append(bhvr)
            all_time.append(time)

            for j, spike in enumerate(trial_data['spikes']):
                spike_data = spike['SpikeTimes']
                if isinstance(spike_data, int):
                    spike_data = [spike_data]
                all_spikes[j].extend(spike_data)

        for i, spikes in enumerate(all_spikes):
            nwbfile.add_unit(
                id=i,
                spike_times=spikes - start_time,
                electrodes=[i]
            )

        ts = create_multichannel_timeseries(
            data_name='finger_pos',
            chan_names=KIN_LABELS,
            data=np.concatenate(all_bhvr, axis=0),
            timestamps=np.concatenate(all_time, axis=0),
        )
        nwbfile.add_acquisition(ts)
        write_to_nwb(nwbfile, path.parent.parent / DATA_SPLITS[path.stem] / f"{path.stem}_{suffix}.nwb")
        print(f"Written {path.stem}_{suffix}.nwb")

    eval_num = int(len(full_payload) * EVAL_RATIO)
    eval_trials = full_payload[-eval_num:]
    create_and_write(eval_trials, 'eval')
    create_and_write(full_payload, 'full')
    in_day_full_trials = full_payload[:-eval_num]
    if DATA_SPLITS[path.stem] == 'train':
        create_and_write(in_day_full_trials, 'calibration')

        minival_trials = full_payload[:SMOKETEST_NUM]
        create_and_write(minival_trials, 'minival')
    else:
        calibration_num = math.ceil(len(full_payload) * FEW_SHOT_CALIBRATION_RATIO)
        calibration_trials = full_payload[:calibration_num]
        create_and_write(calibration_trials, 'calibration')

        create_and_write(in_day_full_trials, 'in_day_oracle')


for f in files:
    to_nwb(f)
# ! Ok...
# 1. create splits
# 2. create acq downsample
#%%
# Check!
payload = loadmat(str(files[0]), simplify_cells=True)['z']
# Skip trial 0 - no data, used for block setup
payload = payload[1:]
payload = [i for i in payload if not i['BlankTrial']] # 1 if screen was off, no trial run
payload = list(map(filt_single_trial, payload))
from matplotlib import pyplot as plt
time = payload[0]['time'].astype(float)
fingers = payload[0]['fingers']
# from scipy.interpolate import CubicSpline

downsample_time = time[::math.ceil(FS * BIN_SIZE_MS / 1000)] # This effectively rounds up
print(fingers.shape)
# print(downsample_fingers.shape)
EDGE_PAD = 160 # reduce edge ringing, in ms
y_padded = np.pad(fingers, ((EDGE_PAD, EDGE_PAD), (0, 0)), mode='edge',)

from scipy.signal import resample_poly
# Resample the padded signal
y_resampled_padded = resample_poly(y_padded, math.ceil(FS / BIN_SIZE_MS), 1000)
y_resampled = y_resampled_padded[int(EDGE_PAD / BIN_SIZE_MS):int(-EDGE_PAD / BIN_SIZE_MS)]
downsample_fingers = y_resampled

# Resampling methods produce edge artifacts
# downsample_fingers = resample_poly(fingers, math.ceil(FS / BIN_SIZE_MS), 1000) # This should round up
# downsample_fingers = resample(fingers, math.ceil(len(fingers) / (FS * BIN_SIZE_MS / 1000)), axis=0) # This should round up
# plt.plot(time, fingers)
plt.plot(downsample_time, downsample_fingers)