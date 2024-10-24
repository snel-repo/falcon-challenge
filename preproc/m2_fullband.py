#%%
r"""
    SN:
    In the .mat files, the ExperimentTime field is a 1ms uint32 counter since the block began, and is split into chunks representing which 1ms timestamps belong to each trial. Each 1ms, our behavioral system transmitted the 1ms counter value via serial connection directly to the NSP, which logged it as 4 uint8s in the NEV files along with the NSP's internal uint32 counter. That NSP's internal uint32 counter should be used to index into the main NSx data array. Now our NEV and NSx files recorded across the whole session, so you'll need to look for resets in the NEV's ExperimentTimes to determine which block based on the .mat's block number (final 3 digits). If I recall correctly, block numbers were 1 indexed. So to summarize:
    1. convert vector of NEV file's serial uint8s to ExperimentTime (uint32) and obtain corresponding NSP uint32 timestamps
    2. slice ExperimentTime and NSP uint32 timestamp vectors to the current block based on ExperimentTime restarts
    - optionally verify ExperimentTime vector approximately matches duration of the block according to .mat file
    3. for each trial in .mat file:
    1. get first ExperimentTime value in trial
    2. find first ExperimentTime value in NEV ExperimentTime vector
    3. startNspTs = corresponding NSP uint32 timestamp
    4. get last ExperimentTime value in trial
    5. find last ExperimentTime value in NEV ExperimentTime vector
    6. endNspTs = corresponding NSP uint32 timestamp
    7. broadband data for trial is NSx.data[:, startNspTs:endNspTs]
"""
# 0. Load mat and corresponding NEV and NSX
import math
from pathlib import Path
import numpy as np
from datetime import datetime
from pprint import pprint

from scipy.io import loadmat

#%%
mat_date = '2020-10-19'
nev_date = '20201019'
run = '002'
# run = '003'
mat_dir = Path("./data/m2/chestek/")
fullband_dir = Path("./data/m2/fullband/")

mat_path = list(fullband_dir.glob(f"*{mat_date}*{run}*.mat"))[0]
# mat_path = list(mat_dir.glob(f"*{mat_date}*{run}*.mat"))[0]
# mat = loadmat(mat_path)
nev_path = list(fullband_dir.glob(f"*{nev_date}*{run}.nev"))[0]
payload = loadmat(mat_path, simplify_cells=True)['z']
CHANNEL_EXPECTATION = 96
TARGET_FINGERS = [1, 3]
def filt_single_trial(trial):
    # ? What are the new keys
    # print(trial['Channel'])
    # print(len(trial['CerebusTimes']))
    # print(trial['NEVFile']) # ? What does Sam mean by files split? Are there multiple NSX
    # print(trial['GoodTrial'])
    # reduce raw
    assert len(trial['Channel']) == CHANNEL_EXPECTATION, f"Expected {CHANNEL_EXPECTATION} channels, got {len(trial['Channel'])}"
    print(trial['CerebusTimes'])
    return {
        'fingers': trial['FingerAnglesTIMRL'][:, TARGET_FINGERS],
        'spikes': trial['Channel'], # spike time to nearest ms
        'time': trial['ExperimentTime'], # Clock time since start of block. 0 on first step i.e. start of bin.
        'target': trial['TargetPos'][TARGET_FINGERS],
        'trial_num': trial['TrialNumber'],
        'good_trial': trial['GoodTrial'],
        'nev_file': str(trial['NEVFile']),
        'cerebus_sample_start': trial['CerebusTimes'][0] if trial['GoodTrial'] else None,
        'cerebus_sample_end': trial['CerebusTimes'][-1] if trial['GoodTrial'] else None,
    }
payload = [i for i in payload if not i['BlankTrial']] # 1 if screen was off, no trial run
payload = list(map(filt_single_trial, payload))
#%%
# Print unique nev files
print(mat_path)
print(payload[0]['nev_file'])
print(set([i['nev_file'] for i in payload]))
# Print number of bad trials
# print(len([i for i in payload if not i['good_trial']]))

# Load up an nsx
from brpylib.brpylib import NsxFile
sample_nsx = f"data/m2/fullband/{payload[0]['nev_file'] + '.ns6'}"
nsx = NsxFile(datafile=sample_nsx)
print(nsx.basic_header.keys())
print(nsx.basic_header['SampleResolution'])
nsx_data = nsx.getdata()
assert len(nsx_data['data']) == 1, "Expecting one contiguous chunk of NS6 recordings."
print(len(nsx_data['data']))
print(nsx_data['data'][0].shape) # 96 x 18052340
nsx.close()
print(payload[0]['cerebus_sample_start'])
print(payload[0]['cerebus_sample_end'])
print(payload[0]['nev_file'])
print(payload[80]['cerebus_sample_start'])
print(payload[80]['cerebus_sample_end'])
print(payload[80]['nev_file'])
print(nsx_data['data'][0].max())
print(nsx_data['data'][0].min())

#%%
# Load NEV
from brpylib.brpylib import NevFile
nev = NevFile(str(nev_path))
ev_header = nev.basic_header
ev_data = nev.getdata()
nev.close()
print(ev_data.keys())
#%%
print(ev_header.keys())
#%%
print(ev_data['digital_events'].keys())
print('Timestamps: ', ev_data['digital_events']['TimeStamps'][0:3])
print('Insertion Reason: ', (ev_data['digital_events']['InsertionReason'][0:3]))
print('Unparsed Data: ' , (ev_data['digital_events']['UnparsedData'][0:3]))
# print(ev_data['spike_events'].keys())
# print(np.unique(np.array(ev_data['spike_events']['Channel'])))
# print(ev_data['spike_events'].keys())
#%%
