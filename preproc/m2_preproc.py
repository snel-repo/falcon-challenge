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
import matplotlib.pyplot as plt

from dateutil.tz import tzlocal

import pynwb
from pynwb import NWBFile, TimeSeries

from decoder_demos.filtering import smooth
from preproc.nwb_create_utils import (
    FEW_SHOT_CALIBRATION_RATIO_M2 as FEW_SHOT_CALIBRATION_RATIO, EVAL_RATIO, SMOKETEST_NUM, BIN_SIZE_MS, BIN_SIZE_S,
    create_multichannel_timeseries,
    write_to_nwb
)

root = Path('./data/m2/raw')
out_root = Path('./data/m2/preproc_src')
files = list(root.glob('*.mat'))
KIN_LABELS = ['index', 'mrs']
CHANNEL_EXPECTATION = 96
TARGET_FINGERS = [1, 3] # Index finger and middle finger group
FS = 1000
pprint(files)

DATA_SPLITS = {
    'Z_Joker_2020-10-19_Run-002': 'held_in',
    'Z_Joker_2020-10-19_Run-003': 'held_in',
    'Z_Joker_2020-10-20_Run-002': 'held_in',
    'Z_Joker_2020-10-20_Run-003': 'held_in',
    'Z_Joker_2020-10-27_Run-002': 'held_in',
    'Z_Joker_2020-10-27_Run-003': 'held_in',
    'Z_Joker_2020-10-28_Run-001': 'held_in',
    'Z_Joker_2020-10-30_Run-001': 'held_out',
    'Z_Joker_2020-10-30_Run-002': 'held_out',
    'Z_Joker_2020-11-18_Run-001': 'held_out',
    'Z_Joker_2020-11-19_Run-001': 'held_out',
    'Z_Joker_2020-11-24_Run-001': 'held_out',
    'Z_Joker_2020-11-24_Run-002': 'held_out'
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
    session_start_time = datetime.strptime(start_date, '%Y-%m-%d')
    # inject arbitrary hour to distinguish sets
    session_start_time = session_start_time.replace(hour=12 + DATA_RUN_SET[path.stem])
    # hash = '_'.join(path.stem.split('_')[2:]) # includes date and run, remap run
    session_hash = start_date + '_' + f'Run{DATA_RUN_SET[path.stem]}'
    # breakpoint()
    subject = pynwb.file.Subject(
        subject_id=f'MonkeyN-{split}-{suffix}',
        description='MonkeyN, Chestek Lab, number indicates experimental set from day',
        species='Rhesus macaque',
        sex='M',
        age='P8Y',
    )
    file_id = f'MonkeyN_{session_start_time.strftime("%m-%d-%H:%M")}_{split}'
    # file_id = f'MonkeyN_{session_start_time.strftime("%m-%d-%H:%M")}_run{DATA_RUN_SET[path.stem]}_{split}'
    f = NWBFile(
        session_description='M2 data',
        identifier=file_id,
        subject=subject,
        session_start_time=session_start_time,
        experimenter='Samuel R. Nason-Tomaszewski and Matthew J. Mender',
        lab='Chestek Lab',
        institution='University of Michigan',
        experiment_description='Two finger group movement in NHP. Behavior provided in 20ms bins, observation interval at trial ends may include a bit of neural data from partial bin.',
        session_id=session_hash # determines the fn
    )
    device = f.create_device(name='Blackrock Utah Array', description='2x64-channel array, 96 active')
    main_group = f.create_electrode_group(
        name='M1_array',
        description='Hand area 2x64-channel array (96 active channels in recording system)',
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
        'time': trial['ExperimentTime'], # Clock time since start of block. 0 on first step i.e. start of bin.
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
        all_vel = []
        all_spikes = [[] for _ in range(CHANNEL_EXPECTATION)]
        all_time = []
        start_time = 0
        # breakpoint()
        for i, trial_data in enumerate(payload):
            time = trial_data['time'].astype(float) / 1000 # To ms
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
            y_padded = smooth(y_padded, 120, 40) # Sam says Gaussian is fine

            y_resampled_padded = resample_poly(y_padded, math.ceil(FS / BIN_SIZE_MS), 1000)
            bhvr = y_resampled_padded[int(EDGE_PAD / BIN_SIZE_MS):int(-EDGE_PAD / BIN_SIZE_MS)]
            all_bhvr.append(bhvr)
            all_vel.append(np.gradient(bhvr, axis=0))
            assert np.isclose(np.diff(time), BIN_SIZE_S).all(), "Expecting timestamps to be about 20ms apart"
            all_time.append(time)

            for j, spike in enumerate(trial_data['spikes']):
                spike_data = spike['SpikeTimes']
                if isinstance(spike_data, int):
                    spike_data = [spike_data]
                all_spikes[j].extend(spike_data)

        for i, spikes in enumerate(all_spikes):
            nwbfile.add_unit(
                id=i,
                spike_times=np.array(spikes) / 1000 - start_time,
                electrodes=[i],
                obs_intervals=[[i[0], i[-1] + BIN_SIZE_S] for i in all_time]
            )
            # OK...
        trial_diffs = [all_time[i+1][0] - all_time[i][-1] for i in range(len(all_time) - 1)]
        trial_diff_raw = [payload[i+1]['time'][0] - payload[i]['time'][-1] for i in range(len(payload) - 1)]
        print(f"Max diff b/n consecutive trials: {max(trial_diffs):.4f}")
        print(f"Max diff b/n consecutive trials (raw): {max(trial_diff_raw):.4f}")
        # Note there's one long drop in `20201019`
        # if max(trial_diff_raw) > 3:
            # breakpoint()
        all_time = np.concatenate(all_time, axis=0)
        diff_check = np.diff(all_time).max()
        assert (diff_check > 0), "Expecting time to be monotonically increasing across trials."
        # print(f"Max diff b/n consecutive timebins: {diff_check}")

        
        ts = create_multichannel_timeseries(
            data_name='finger_pos',
            chan_names=KIN_LABELS,
            data=np.concatenate(all_bhvr, axis=0),
            timestamps=all_time, # timestamps denote wall-clock sample is drawn - not evenly spaced
            unit='AU'
        )
        nwbfile.add_acquisition(ts)

        ts = create_multichannel_timeseries(
            data_name='finger_vel',
            chan_names=KIN_LABELS,
            data=np.concatenate(all_vel, axis=0),
            timestamps=all_time,
            unit='AU'
        )
        nwbfile.add_acquisition(ts)

        nwbfile.add_acquisition(
            TimeSeries(
                name='eval_mask',
                description='Timesteps to keep covariates (for training, eval).',
                timestamps=all_time,
                data=np.ones_like(all_time, dtype=bool),
                unit='bool'
            )
        )
        date_str = path.stem.split('_')[2].replace('-', '')
        run = DATA_RUN_SET[path.stem]
        # new_prefix = f'sub-MonkeyN_{date_str}_{DATA_SPLITS[path.stem]}'
        new_prefix = f'sub-MonkeyNRun{run}_{date_str}_{DATA_SPLITS[path.stem]}'
        write_to_nwb(nwbfile, out_root / DATA_SPLITS[path.stem] / f"{new_prefix}_{suffix}.nwb")
        # breakpoint()
        print(f"Written ", out_root / DATA_SPLITS[path.stem] / f"{new_prefix}_{suffix}.nwb")
    # breakpoint()
    eval_num = int(len(full_payload) * EVAL_RATIO)
    eval_trials = full_payload[-eval_num:]
    create_and_write(eval_trials, 'eval')
    create_and_write(full_payload, 'full')
    in_day_full_trials = full_payload[:-eval_num]
    if DATA_SPLITS[path.stem] == 'held_in':
        create_and_write(in_day_full_trials, 'calib')

        minival_trials = full_payload[:SMOKETEST_NUM]
        create_and_write(minival_trials, 'minival')
    else:
        calibration_num = math.ceil(len(full_payload) * FEW_SHOT_CALIBRATION_RATIO)
        calibration_trials = full_payload[:calibration_num]
        create_and_write(calibration_trials, 'calib')

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
trial = 0

time = payload[trial]['time'].astype(float)
fingers = payload[trial]['fingers']

SRC_BINS_PER_TGT = math.ceil(FS * BIN_SIZE_MS / 1000)
downsample_time = time[::SRC_BINS_PER_TGT] # This effectively rounds up
# crop, then bin
if len(fingers) % SRC_BINS_PER_TGT:
    bin_fingers = fingers[:-(len(fingers) % SRC_BINS_PER_TGT)]
    bin_time = time[:-(len(time) % SRC_BINS_PER_TGT)]
bin_time = bin_time.reshape(-1, SRC_BINS_PER_TGT).mean(axis=1)
bin_fingers = bin_fingers.reshape(-1, SRC_BINS_PER_TGT, bin_fingers.shape[-1]).mean(axis=1)
# plt.plot(bin_time, np.gradient(bin_fingers, axis=0))

print(fingers.shape)
# print(downsample_fingers.shape)
EDGE_PAD = 160 # reduce edge ringing, in ms
y_padded = np.pad(fingers, ((EDGE_PAD, EDGE_PAD), (0, 0)), mode='edge',)

# Low pass
print(y_padded.shape)
y_padded = smooth(y_padded, 100, 25)

from scipy.signal import resample_poly
# Resample the padded signal
y_resampled_padded = resample_poly(y_padded, math.ceil(FS / BIN_SIZE_MS), 1000)
y_resampled = y_resampled_padded[int(EDGE_PAD / BIN_SIZE_MS):int(-EDGE_PAD / BIN_SIZE_MS)]
downsample_fingers = y_resampled

# Resampling methods produce edge artifacts
# downsample_fingers = resample_poly(fingers, math.ceil(FS / BIN_SIZE_MS), 1000) # This should round up
# downsample_fingers = resample(fingers, math.ceil(len(fingers) / (FS * BIN_SIZE_MS / 1000)), axis=0) # This should round up
# plt.plot(time, fingers)
# plt.plot(downsample_time, downsample_fingers)
plt.plot(downsample_time, np.gradient(downsample_fingers, axis=0), label='grad')