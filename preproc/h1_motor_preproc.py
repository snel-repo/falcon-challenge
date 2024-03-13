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

from decoder_demos.filtering import smooth

root = Path('./data/h1')
files = list(root.glob('*.mat'))

KIN_SUBSET = {
    0: 'tx',
    1: 'ty',
    2: 'tz',
    3: 'rx',
    6: 'g1',
    7: 'g2',
    9: 'g3',
}

#%%
CHANNELS_PER_SOURCE = 128
BIN_SIZE_MS = 20
TARGET_BIN_SIZE_MS = 20
TARGET_BIN_SIZE_S = TARGET_BIN_SIZE_MS / 1000
FEW_SHOT_CALIBRATION_RATIO = 0.2
EVAL_RATIO = 0.4
INTERTRIAL_NAME = 'Intertrial'

DEFAULT_TARGET_SMOOTH_MS = 490
KERNEL_SIZE = int(DEFAULT_TARGET_SMOOTH_MS / BIN_SIZE_MS)
KERNEL_SIGMA = DEFAULT_TARGET_SMOOTH_MS / (3 * TARGET_BIN_SIZE_MS)

def create_targets(kin: np.ndarray, target_smooth_ms=DEFAULT_TARGET_SMOOTH_MS, bin_size_ms=TARGET_BIN_SIZE_MS, sigma=3):
    kernel_size = int(target_smooth_ms / bin_size_ms)
    kernel_sigma = target_smooth_ms / (sigma * bin_size_ms)
    kin = smooth(kin, kernel_size, kernel_sigma)
    out = np.gradient(kin, axis=0)
    return out

USE_SHORT_LONG_DISTINCTION = True
USE_SHORT_LONG_DISTINCTION = False
CURATED_SETS = {
    # 'S53_set_1': 'train',
    # 'S53_set_2': 'train',
    # 'S63_set_1': 'train',
    # 'S63_set_2': 'train',
    # 'S77_set_1': 'test_short',
    # 'S77_set_2': 'test_short',
    # 'S91_set_1': 'test_long',
    # 'S91_set_2': 'test_long',
    # 'S95_set_1': 'test_long',
    # 'S95_set_2': 'test_long',
    # 'S99_set_1': 'test_long',
    # 'S99_set_2': 'test_long',

    # 'S591_set_1': 'train',
    # 'S591_set_2': 'train',
    # 'S594_set_1': 'train',
    # 'S594_set_2': 'train',
    # 'S600_set_1': 'test_short',
    # 'S600_set_2': 'test_short',
    # 'S602_set_1': 'test_long',
    # 'S602_set_2': 'test_long',

    'S608_set_1': 'train',
    'S608_set_2': 'train',
    'S610_set_1': 'train',
    'S610_set_2': 'train',
    'S610_set_3': 'train',
    'S613_set_1': 'train',
    'S613_set_2': 'train',
    'S615_set_1': 'train',
    'S615_set_2': 'train',
    'S619_set_1': 'train',
    'S619_set_2': 'train',
    'S625_set_1': 'train',
    'S625_set_2': 'train',
    'S627_set_1': 'test_short',
    'S627_set_2': 'test_short',
    'S631_set_1': 'test_short',
    'S631_set_2': 'test_short',
    'S633_set_1': 'test_short',
    'S633_set_2': 'test_short',
    'S636_set_1': 'test_long',
    'S636_set_2': 'test_long',
    'S639_set_1': 'test_long',
    'S639_set_2': 'test_long',
    'S641_set_1': 'test_long',
    'S641_set_2': 'test_long',
    'S644_set_1': 'test_long',
    'S644_set_2': 'test_long',
}

# Misfires according to test log. Trials are one-indexed.
DROP_SET_TRIALS = {
    'S594_set_1': [1, 2],
    'S608_set_1': [1, 2],
    'S615_set_1': [1, 2], # Severely corrupted
}

def create_nwb_name(mat_name: Path) -> Path:
    root = mat_name.parent
    name = mat_name.stem
    name = name.split('session_')[-1]
    return (root / name).with_suffix('.nwb')

def create_nwb_shell(start_date: datetime = datetime(2017, 1, 1, 1, tzinfo=tzlocal())):
    # Note identifying info is redacted
    return NWBFile(
        session_description="Open loop 7DoF calibration.",
        identifier=str(uuid4()),
        session_start_time=start_date,
        lab="Rehab Neural Engineering Labs",
        institution="University of Pittsburgh",
        experiment_description="Open loop calibration for Action Research Arm Test (ARAT) for human motor BCI",
    )

def write_to_nwb(nwbfile: NWBFile, fn: Path):
    fn.parent.mkdir(exist_ok=True, parents=True)
    with NWBHDF5IO(str(fn), 'w') as io:
        io.write(nwbfile)

def crop_spikes(spike_times, spike_channels, bin_times, bin_size_s=TARGET_BIN_SIZE_S):
    r"""
        spike_times: 1D array of spike time events
        spike_channels: 1D array of spike channels (associated with spike times)
        bin_times: Separate dense clock of target bin times. Assumed continuous.
    """
    return (
        spike_times[(spike_times >= bin_times[0] - bin_size_s) & (spike_times < bin_times[-1])] - bin_times[0],
        spike_channels[(spike_times >= bin_times[0] - bin_size_s) & (spike_times < bin_times[-1])]
    )

# H1 consts
motor_units = np.concatenate([
    np.arange(64) + 1,
    np.arange(32) + 96 + 1,
    np.arange(64) + 128 + 1,
    np.arange(32) + 128 + 96 + 1,
])
# Subtract 114, 116, 118, 120, 122, 124, 126, 128, which are not wired for P2
# Do this for both pedestals
motor_units = np.setdiff1d(motor_units, np.arange(114, 128+1, 2))
motor_units = np.setdiff1d(motor_units, np.arange(114+128, 128+128+1, 2))
# NaN filtering for kinematics.
LEFT_CROP_BEGIN_MAX_NAN = 60
RIGHT_CROP_BEGIN_MAX_NAN = 20
MAX_CONSEC_NAN = 3
# Assert nans only occur at start of array, and are only a few.
# If at edge of trial, then crop trial
# If in middle of trial, and not too many consecutive, then interpolate
# Else flag for inspection

def to_nwb(fn):
    tag = create_nwb_name(fn).stem
    if tag not in CURATED_SETS:
        # Move file to fn.parent / 'archive'
        # os.makedirs(fn.parent / 'archive', exist_ok=True)
        # os.rename(fn, fn.parent / 'archive' / fn.name)
        return

    r"""
        Load raws and align clocks
    """
    print(tag)

    payload = loadmat(str(fn), simplify_cells=True, variable_names=['thin_data'])['thin_data']
    state_names = payload['state_names'] # Shape is ~15 (arbitrary, exp dependent)
    state_names[0] = INTERTRIAL_NAME
    blacklist_states = [INTERTRIAL_NAME, 'FailSafe', 'FailSafe1', 'SnapTo']
    blacklist_index = []
    for name in blacklist_states:
        if name in state_names:
            blacklist_index.append(list(state_names).index(name))

    # Align raw spike times - each on their own clocks - to the 20ms bin clock, where all covariates are recorded
    # The unifying field is `spm_source_timestamp`, which is the end of the 20ms bin.
    # However, we essentially assume SPM_SPIKECOUNT module was always online + functional.
    # This is not always true, even in our curated data.
    # When it's not, we assume that the last timestamp is consistent.
    raw_spike_channel = payload['source_index'] * CHANNELS_PER_SOURCE + payload['channel']
    raw_spike_time = payload['source_timestamp']
    bin_time = payload['spm_source_timestamp'] # Indicates end of 20ms bin, in seconds. Shape is (recording devices x Timebin), each has own clock
    # This code breaks when we have gaps in source timestamp - fill it in
    bin_kin = payload['open_loop_kin'] # Shape is (Timebin x K)
    if bin_time.shape[-1] != bin_kin.shape[0]:
        print(f"Warning: {tag} has mismatch in kinematics and timestamps. Filling in missing timestamps by assuming SPM timestamp is reliable at endpoint.")
        padded_bin_time = []
        for nsp_bin_time in bin_time:
            assert abs((nsp_bin_time[-1] - nsp_bin_time[0]) * 1000 / BIN_SIZE_MS - bin_kin.shape[0]) < 4, 'Substantial mismatch in start/end of NSP clocks and binned kinematics'
            end_time = nsp_bin_time[-1]
            padded_nsp_bin_time = np.arange(end_time - ((bin_kin.shape[0] - 1) * BIN_SIZE_MS / 1000), end_time + BIN_SIZE_MS / 1000, BIN_SIZE_MS/1000)
            padded_bin_time.append(padded_nsp_bin_time)
        bin_time = np.stack(padded_bin_time)
    assert bin_time.shape[-1] == bin_kin.shape[0], 'Mismatch in SPM timestamps and SPM kinematics - make sure to extract SPM timestamps from data, not iData.'
    bin_kin = bin_kin[:, np.array(list(KIN_SUBSET.keys()))] # Subset kinematics
    bin_state = payload['state_num'] -1 # Shape is (Timebin), 1-index -> 0-index
    blacklist_timesteps = np.isin(bin_state, blacklist_index)
    print(f'% blacklist phases: {np.sum(blacklist_timesteps) / len(blacklist_timesteps) * 100:.2f}')

    bin_trial = payload['trial_num'] # Shape is (Timebin)

    # Align all times to start of 1st bin of data.
    for recording_box in range(bin_time.shape[0]):
        raw_spike_time[payload['source_index'] == recording_box] -= bin_time[recording_box][0] - BIN_SIZE_MS/1000
        bin_time[recording_box] -= bin_time[recording_box][0] - BIN_SIZE_MS/1000
        # Clip other recorded spikes to timestep 0, assuming not too negative
        # assert np.all(raw_spike_time[payload['source_index'] == recording_box] >= -(BIN_SIZE_MS/1000)/4)
        # raw_spike_time[payload['source_index'] == recording_box] = np.clip(raw_spike_time[payload['source_index'] == recording_box], 0, None)
    bin_time_native = bin_time[0] # Aligned, there's only need for one clock
    bin_time = None # Null out until it's redefined later

    # Time axis is showing some repeats in some data. Remove repeats...
    rep_times, rep_counts = np.unique(bin_time_native, return_counts=True)
    rep_times = rep_times[rep_counts > 1]
    keep_indices = np.arange(len(bin_time_native))
    for time in rep_times:
        # Find all indices of the current duplicate time
        indices = np.where(bin_time_native == time)[0]
        # Keep the first index, mark the rest for removal
        keep_indices = np.setdiff1d(keep_indices, indices[1:])  # Exclude the first occurrence from removal
    bin_time_native = bin_time_native[keep_indices]
    bin_kin = bin_kin[keep_indices]
    bin_state = bin_state[keep_indices]
    bin_trial = bin_trial[keep_indices]
    blacklist_timesteps = blacklist_timesteps[keep_indices]

    # Mark pretrial period as first trial - it's a small buffer
    nan_mask = np.isnan(bin_trial)
    assert np.all(nan_mask[:np.argmax(~nan_mask)])
    assert np.sum(nan_mask) < 10
    bin_trial[nan_mask] = 1
    bin_trial = bin_trial.astype(int)

    # count number of dead channels with no spikes
    # channel_cts = np.unique(raw_spike_channel, return_counts=True)
    # dead_channels = np.setdiff1d(motor_units, channel_cts[0])
    # print(dead_channels)

    if tag in DROP_SET_TRIALS:
        # These are serious (neural) corruptions. Drop from data entirely.
        print(np.isnan(bin_kin).any(-1).nonzero())
        print(np.unique(bin_trial[np.isnan(bin_kin).any(-1)]))
        bad_trial_bins = np.isin(bin_trial, np.array(DROP_SET_TRIALS[tag]))
        blacklist_timesteps[bad_trial_bins] = True
        bin_kin[bad_trial_bins] = np.nan
        print(np.nonzero(bad_trial_bins)[0])
        cut_trial_left = np.nonzero(bad_trial_bins)[0][-1] + 1
        print(f"Cutting {cut_trial_left} bins from {bin_trial.shape[0]} total.")
        bin_kin = bin_kin[cut_trial_left:]
        bin_trial = bin_trial[cut_trial_left:]
        bin_state = bin_state[cut_trial_left:]
        bin_time_native = bin_time_native[cut_trial_left:]
        blacklist_timesteps = blacklist_timesteps[cut_trial_left:]

        raw_spike_time, raw_spike_channel = crop_spikes(raw_spike_time, raw_spike_channel, bin_time_native, bin_size_s=BIN_SIZE_MS)
        print(raw_spike_time)
        bin_time_native = bin_time_native - bin_time_native[0]


    r"""
        Clean kin NaNs from data
    """
    nan_kin_mask = np.isnan(bin_kin).any(-1)
    # Remove left edge
    if nan_kin_mask[:LEFT_CROP_BEGIN_MAX_NAN].any():
        crop_left = np.nonzero(nan_kin_mask[:LEFT_CROP_BEGIN_MAX_NAN])[0][-1]
        # Scan right
        for i in range(crop_left, len(nan_kin_mask)):
            if not nan_kin_mask[i]:
                crop_left = i
                break
    else:
        crop_left = 0
    # Remove right edge
    if nan_kin_mask[-RIGHT_CROP_BEGIN_MAX_NAN:].any():
        # Find where NaN starts
        crop_right = - np.nonzero(nan_kin_mask[-RIGHT_CROP_BEGIN_MAX_NAN:])[0][0]
        for i in range(crop_right, len(nan_kin_mask), -1):
            if not nan_kin_mask[i]:
                crop_right = i
                break
    else:
        crop_right = 0

    # Instead of true cropping, we merely notify that these kin time labels are not reliable by adding to blacklist.
    # Models are responsible for acknowledging this.

    if crop_left:
        blacklist_timesteps[:crop_left] = True
        bin_kin[:crop_left] = bin_kin[crop_left]
        nan_kin_mask[:crop_left] = False
    if crop_right:
        blacklist_timesteps[crop_right:] = True
        bin_kin[crop_right:] = bin_kin[crop_right - 1]
        nan_kin_mask[crop_right:] = False

    # if crop_right:
    #     bin_kin = bin_kin[crop_left:crop_right]
    #     bin_trial = bin_trial[crop_left:crop_right]
    #     bin_state = bin_state[crop_left:crop_right]
    #     bin_time_native = bin_time_native[crop_left:crop_right]
    #     nan_kin_mask = nan_kin_mask[crop_left:crop_right]
    #     blacklist_timesteps = blacklist_timesteps[crop_left:crop_right]
    # else:
    #     bin_kin = bin_kin[crop_left:]
    #     bin_trial = bin_trial[crop_left:]
    #     bin_state = bin_state[crop_left:]
    #     bin_time_native = bin_time_native[crop_left:]
    #     nan_kin_mask = nan_kin_mask[crop_left:]
    #     blacklist_timesteps = blacklist_timesteps[crop_left:]

    # raw_spike_time, raw_spike_channel = crop_spikes(raw_spike_time, raw_spike_channel, bin_time_native, bin_size_s=BIN_SIZE_MS)

    bin_time_native = bin_time_native - bin_time_native[0]

    if crop_left or crop_right:
        print(f"Edge kin Nans: {crop_left} bins on left, {crop_right} on right, masking")
    if nan_kin_mask.any():
        print(nan_kin_mask.nonzero())
        print(f"Remaining kin NaNs: {nan_kin_mask.sum()} (% {np.mean(nan_kin_mask)*100:.2f})")
    cum_nan_kin_mask = np.zeros_like(nan_kin_mask)
    count = 0
    # Iterate through the NaN mask and calculate cumulative NaNs
    for i, is_nan in enumerate(nan_mask):
        if is_nan:
            count += 1  # Increment count if current is NaN
            cum_nan_kin_mask[i] = count  # Assign count to the cumulative array
        else:
            count = 0  # Reset count if current is not NaN
    if np.any(cum_nan_kin_mask > MAX_CONSEC_NAN):
        raise ValueError(f"Warning: {tag} has NaNs in the middle of the trial. Inspect.")

    r"""
        Resample position information - upsample from 50 to 100Hz.
        Wrap in kin NaN interp
    """
    print(f'Pre NaNs: {np.isnan(bin_kin).sum()}')
    assert(bin_kin.shape[0] == bin_time_native.shape[0])
    bin_time = np.arange(0, bin_time_native[-1], TARGET_BIN_SIZE_S)
    if nan_kin_mask.any():
        print(f"interpolating out {nan_kin_mask.sum()} NaNs in kinematics")
        bin_kin = interp1d(bin_time_native[~nan_kin_mask], bin_kin[~nan_kin_mask], axis=0, bounds_error=False)(bin_time)
    else:
        bin_kin = interp1d(bin_time_native, bin_kin, axis=0, bounds_error=False)(bin_time)
    bin_vel = create_targets(bin_kin)
    # print(bin_kin.shape, bin_vel.shape)
    bin_trial_native = bin_trial
    bin_trial = interp1d(bin_time_native, bin_trial, bounds_error=False, kind='nearest', fill_value='extrapolate')(bin_time)
    bin_blacklist = interp1d(bin_time_native, blacklist_timesteps, bounds_error=False, kind='nearest', fill_value='extrapolate')(bin_time)

    # bin_state kept at native time
    def create_nwb_container(
        spike_time: np.ndarray,
        spike_channel: np.ndarray,
        bin_time: np.ndarray, # Resampled
        bin_time_native: np.ndarray, # For state
        bin_kin: np.ndarray, # Resampled
        bin_vel: np.ndarray,
        bin_trial: np.ndarray, # Resampled
        bin_state: np.ndarray, # Native resolution
        bin_blacklist: np.ndarray,
    ):
        # Recenter timestamps in case data is subsetted
        date_str = payload['start_date']
        date_obj = datetime.strptime(date_str, "%d-%b-%Y_%H_%M_%S")
        nwbfile = create_nwb_shell(date_obj)
        for unit_id in motor_units:
            spike_times = spike_time[spike_channel == unit_id]
            nwbfile.add_unit(spike_times=spike_times)
        position_spatial_series = TimeSeries(
            name="OpenLoopKinematics",
            description=','.join(KIN_SUBSET.values()),
            timestamps=bin_time,
            data=bin_kin,
            unit="arbitrary",
        )
        nwbfile.add_acquisition(position_spatial_series)

        # Add filtered kinematics
        # print(bin_vel.shape, bin_time.shape)
        velocity_spatial_series = TimeSeries(
            name="OpenLoopKinematicsVelocity",
            description=','.join(KIN_SUBSET.values()),
            timestamps=bin_time,
            data=bin_vel,
            unit="arbitrary",
        )

        nwbfile.add_acquisition(velocity_spatial_series)
        # Note we are abusing the acquisition field to store derivative data, which is not ideal

        # Create a TimeSeries for trial num, semi-hack - states should likely not be its own TimeSeries
        trial_series = TimeSeries(
            name="TrialNum",
            description="Trial number",
            timestamps=bin_time,
            data=bin_trial,
            unit="arbitrary",
        )
        nwbfile.add_acquisition(trial_series)
        blacklist_series = TimeSeries(
            name="Blacklist",
            description="Timesteps to ignore covariates (for training, eval). Neural data is not affected.",
            timestamps=bin_time,
            data=bin_blacklist,
            unit="bool",
        )
        nwbfile.add_acquisition(blacklist_series)

        # Convert states to epoch data
        epoch_start = bin_time_native[0]
        cur_state = bin_state[0]
        for i in range(1, len(bin_state)):
            if bin_state[i] != cur_state:
                # End of state, add interval
                nwbfile.add_epoch(
                    start_time=epoch_start,
                    stop_time=bin_time_native[i], # ! bin_state is not resampled
                    tags=[state_names[cur_state]],
                    timeseries=[position_spatial_series])
                cur_state = bin_state[i]
                epoch_start = bin_time_native[i]
        # Append a final posttrial state
        if epoch_start < bin_time_native[-1]:
            nwbfile.add_epoch(
                start_time=epoch_start,
                stop_time=bin_time_native[-1],
                tags=[INTERTRIAL_NAME],
                timeseries=[position_spatial_series])
        return nwbfile

    def create_cropped_container(trial_mask, trial_mask_native):
        # print(trial_mask)
        sub_bin_time = bin_time[trial_mask]
        sub_spike_times, sub_channels = crop_spikes(raw_spike_time, raw_spike_channel, sub_bin_time)
        start_time = sub_bin_time[0]
        return create_nwb_container(
            sub_spike_times,
            sub_channels,
            sub_bin_time - start_time,
            bin_time_native[trial_mask_native] - start_time,
            bin_kin[trial_mask],
            bin_vel[trial_mask],
            bin_trial[trial_mask],
            bin_state[trial_mask_native],
            bin_blacklist[trial_mask],
        )
    def create_and_write(trial_mask, trial_mask_native, suffix, folder=CURATED_SETS[tag]):
        nwbfile = create_cropped_container(trial_mask, trial_mask_native)
        if not USE_SHORT_LONG_DISTINCTION and "test" in folder:
            folder = "test"
        out = fn.parent / folder / f"{out_fn.stem}_{suffix}.nwb"
        write_to_nwb(nwbfile, out)

    trials = sorted(np.unique(bin_trial))
    out_fn = create_nwb_name(fn)
    eval_num = int(len(trials) * EVAL_RATIO)
    eval_trials = trials[-eval_num:]
    create_and_write(bin_trial >= eval_trials[0],
                     bin_trial_native >= eval_trials[0],
                     'eval')
    if CURATED_SETS[tag] != "train":
        calibration_num = math.ceil(len(trials) * FEW_SHOT_CALIBRATION_RATIO)
        calibration_trials = trials[:calibration_num]
        create_and_write(bin_trial < calibration_trials[-1],
                         bin_trial_native < calibration_trials[-1],
                         'calibration')

        in_day_oracle = trials[:-eval_num]
        create_and_write(bin_trial <= in_day_oracle[-1], # Note <= because array end already exclusive
                         bin_trial_native <= in_day_oracle[-1],
                         'in_day_oracle')

        create_and_write(np.ones_like(bin_trial, dtype=bool),
                         np.ones_like(bin_trial_native, dtype=bool),
                         'full')
    else:
        calibration_trials = trials[:-eval_num]
        create_and_write(bin_trial <= calibration_trials[-1],
                         bin_trial_native <= calibration_trials[-1],
                         'calibration')

        # Use first two trials for minival
        minival_num = 2
        minival_trials = trials[:minival_num]
        create_and_write(bin_trial <= minival_trials[-1],
                         bin_trial_native <= minival_trials[-1],
                         'minival')
        # nwbfile = create_nwb_container(
        #     raw_spike_time,
        #     raw_spike_channel,
        #     bin_time,
        #     bin_time_native,
        #     bin_kin,
        #     bin_vel,
        #     bin_trial,
        #     bin_state,
        #     bin_blacklist,
        # )

        # out = fn.parent / CURATED_SETS[tag] / out_fn.name
        # write_to_nwb(nwbfile, out)

print(f"Processing {len(files)} files")
for sample in files:
    to_nwb(sample)
print("Done")