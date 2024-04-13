r"""
    Dataloading utilities for evaluator
"""
from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from falcon_challenge.config import FalconTask

# Load nwb file
def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.02,
        bin_timestamps: Optional[np.ndarray] = None,
        is_timestamp_bin_start: bool = False,
    ) -> np.ndarray:
    r"""
        Bin spikes given by an nwb units dataframe.
        There is one bin per input timestamp. If timestamps are not provided, they are inferred from the spike times.
        Timestamps are ideally provided spaced bin_size_s apart.
        If not:
        - if consecutive interval is greater than bin_size_s, then only the proximal bin_size_s interval is used.
        - if consecutive interval is less than bin_size_s, spikes will be binned in the provided interval (i.e. those bins will be smaller than bin_size_s apart).
            - Not outputting repeated spikes mainly because the implementation would be more complex (no single call to np.histogram)
        Args:
            units: df with only index (spike index) and spike times (list of times in seconds). From nwb.units.
            bin_size_s: size of each bin to output in seconds.
            bin_timestamps: array of timestamps indicating bin time in seconds.
            is_timestamp_bin_start: if True, the bin is considered to start at the timestamp, otherwise it ends at the timestamp.
        Returns:
            array of spike counts per bin, per unit. Shape is (bins x units).
    """
    # Make even bins
    if bin_timestamps is None:
        end_time = units.spike_times.apply(lambda s: max(s) if len(s) else 0).max() + bin_size_s
        bin_end_timestamps = np.arange(0, end_time, bin_size_s)
        bin_mask = np.ones(len(bin_end_timestamps), dtype=bool)
    else:
        if is_timestamp_bin_start:
            bin_end_timestamps = bin_timestamps + bin_size_s
        else:
            bin_end_timestamps = bin_timestamps
        # Check contiguous else force cropping for even bins
        gaps = np.diff(bin_end_timestamps)
        if (gaps <= 0).any():
            raise ValueError("bin_end_timestamps must be monotonically increasing.")
        if not np.allclose(gaps, bin_size_s):
            print(f"Warning: Input timestamps not spaced like requested {bin_size_s}. Outputting proximal bin spikes.")
            # Adjust bin_end_timestamps to include bins at the end of discontinuities
            new_bin_ends = [bin_end_timestamps[0]]
            bin_mask = [True] # bool, True if bin ending at this timepoint should be included post mask (not padding)
            for i, gap in enumerate(gaps):
                if not np.isclose(gap, bin_size_s) and gap > bin_size_s:
                    cur_bin_end = bin_end_timestamps[i+1]
                    new_bin_ends.extend([cur_bin_end - bin_size_s, cur_bin_end])
                    bin_mask.extend([False, True])
                else:                        
                    new_bin_ends.append(bin_end_timestamps[i+1])
                    bin_mask.append(True)
            bin_end_timestamps = np.array(new_bin_ends)
            bin_mask = np.array(bin_mask)
        else:
            bin_mask = np.ones(len(bin_end_timestamps), dtype=bool)

    # Make spikes
    spike_arr = np.zeros((bin_mask.sum(), len(units)), dtype=np.uint8)
    bin_edges = np.concatenate([np.array([bin_end_timestamps[0] - bin_size_s]), bin_end_timestamps])
    for idx, (_, unit) in enumerate(units.iterrows()):
        spike_cnt, _ = np.histogram(unit.spike_times, bins=bin_edges)
        if bin_mask is not None:
            spike_cnt = spike_cnt[bin_mask]
        spike_arr[:, idx] = spike_cnt
    return spike_arr


def load_nwb(fn: Union[str, Path], dataset: FalconTask = FalconTask.h1, bin_old=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
        Load data for evaluation.

        Returns:
        - neural_data: binned spike counts. Shape is (time x units)
        - covariates (e.g. kinematics). Shape is (time x n_kinematic_dims)
        - trial_change: boolean of shape (time,) true if the trial has changed
        - eval_mask: boolean array indicating whether to evaluate each time step, True if should
    """
    if not dataset in [FalconTask.h1, FalconTask.h2, FalconTask.m1, FalconTask.m2]:
        raise ValueError(f"Unknown dataset {dataset}")
    with NWBHDF5IO(str(fn), 'r') as io:
        nwbfile = io.read()
        if dataset == FalconTask.h1:
            units = nwbfile.units.to_dataframe()
            kin = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:]
            timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(kin.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
            eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)
            binned_units = bin_units(units, bin_size_s=0.02, bin_timestamps=timestamps)
            return binned_units, kin, np.zeros(kin.shape[0]), eval_mask
        elif dataset == FalconTask.h2: 
            binned_spikes = nwbfile.acquisition['binned_spikes'].data[()]
            time = nwbfile.acquisition['binned_spikes'].timestamps[()]
            eval_mask = nwbfile.acquisition['eval_mask'].data[()].astype(bool)
            trial_info = (
                nwbfile.trials.to_dataframe()
                .reset_index()
            )
            return binned_spikes, trial_info.cue.values, np.concatenate([[False], np.diff(time) > 0.021]), eval_mask
        elif dataset == FalconTask.m1:
            units = nwbfile.units.to_dataframe()
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
            emg_data = np.vstack(emg_data).T
            emg_timestamps = emg_timestamps[0]
            binned_units = bin_units(units, bin_size_s=0.02, bin_timestamps=emg_timestamps)

            eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)

            trial_info = (
                nwbfile.trials.to_dataframe()
                .reset_index()
                .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
            )
            switch_inds = np.searchsorted(emg_timestamps, trial_info.start_time)
            trial_change = np.zeros(emg_timestamps.shape[0], dtype=bool)
            trial_change[switch_inds] = True

            return binned_units, emg_data, trial_change, eval_mask
        elif dataset == FalconTask.m2:
            units = nwbfile.units.to_dataframe()
            vel_container = nwbfile.acquisition['finger_vel']
            labels = [ts for ts in vel_container.time_series]
            vel_data = []
            vel_timestamps = None
            for ts in labels:
                ts_data = vel_container.get_timeseries(ts)
                vel_data.append(ts_data.data[:])
                vel_timestamps = ts_data.timestamps[:]
            vel_data = np.vstack(vel_data).T
            binned_units = bin_units(units, bin_size_s=0.02, bin_timestamps=vel_timestamps, is_timestamp_bin_start=True)

            eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)

            trial_change = np.zeros(vel_timestamps.shape[0], dtype=bool)
            trial_info = nwbfile.trials.to_dataframe().reset_index()
            switch_inds = np.searchsorted(vel_timestamps, trial_info.start_time)
            trial_change[switch_inds] = True
            return binned_units, vel_data, vel_timestamps, eval_mask
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
            breakpoint()
