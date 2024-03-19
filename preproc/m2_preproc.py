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
from scipy.io import loadmat
from scipy.interpolate import interp1d # Upsample by simple interp

from datetime import datetime
from uuid import uuid4

from dateutil.tz import tzlocal

from pynwb import NWBFile, TimeSeries

from decoder_demos.filtering import smooth
from preproc.nwb_create_utils import (
    FEW_SHOT_CALIBRATION_RATIO, EVAL_RATIO, SMOKETEST_NUM,
    write_to_nwb
)

root = Path('./data/m2/raw')
files = list(root.glob('*.mat'))
KIN_LABELS = ['index', 'mrs']
CHANNEL_EXPECTATION = 96
TARGET_FINGERS = [1, 3] # Index finger and middle finger group
print(files)

#%%
def create_nwb_shell(path: Path):
    start_date = path.stem.split('_')[2] # YYYY-MM-DD
    hash = '_'.join(path.stem.split('_')[2:])
    return NWBFile(
        session_description='M2 data',
        identifier=str(uuid4()),
        session_start_time=datetime.strptime(start_date, '%Y-%m-%d'),
        experimenter='Samuel R. Nason and Matthew J. Mender',
        lab='Chestek Lab',
        institution='University of Michicgan',
        experiment_description='Two finger group movement in NHP',
        session_id=hash
    )

def filt_single_trial(trial):
    # reduce raw
    assert len(trial['Channel']) == CHANNEL_EXPECTATION
    return {
        'fingers': trial['FingerAnglesTIMRL'][:, TARGET_FINGERS],
        'spikes': trial['Channel'], # spike time to nearest ms
        'time': trial['ExperimentTime'], # Clock time since start of block
        'target': trial['TargetPos'][TARGET_FINGERS]
    }

def to_nwb(path: Path):
    payload = loadmat(str(path), simplify_cells=True)['z']
    # Skip trial 0 - no data, used for block setup
    payload = payload[1:]
    payload = [i for i in payload if not i['BlankTrial']] # 1 if screen was off, no trial run
    payload = list(map(filt_single_trial, payload))


print(payload[0]['time'])
print(payload[1]['time'])
print(payload[1]['spikes'][2])
#%%
from pprint import pprint
list_1 = list(payload[0].keys())
list_2 = list(payload[1].keys())
for i in range(len(list_1)):
    print(f'{list_1[i]}: {list_2[i]}')
print(len(payload[1]['Channel'])) # Spike times relative to block beginning - 96 channels
print(payload[1]['BlankTrial'])
print(payload[1]['FingerAnglesTIMRL'].shape) # T x 5 - 1 kHz finger position, normalized to 0-1. e.g T=1258
print(payload[1]['ExperimentTime'].shape) # T
print(payload[1]['TargetPos'].shape) # 5 - target positions for the trial
# print(len(payload['z']))