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
        bin_size_s: float = 0.01,
        bin_end_timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
    r"""
        units: df with only index (spike index) and spike times (list of times in seconds). From nwb.units.
        bin_end_timestamps: array of timestamps indicating end of bin

        Returns:
        - array of spike counts per bin, per unit. Shape is (bins x units)
    """
    if bin_end_timestamps is None:
        end_time = units.spike_times.apply(lambda s: max(s) if len(s) else 0).max() + bin_size_s
        bin_end_timestamps = np.arange(0, end_time, bin_size_s)
    spike_arr = np.zeros((len(bin_end_timestamps), len(units)), dtype=np.uint8)
    bin_edges = np.concatenate([np.array([bin_end_timestamps[0] - bin_size_s]), bin_end_timestamps])
    for idx, (_, unit) in enumerate(units.iterrows()):
        spike_cnt, _ = np.histogram(unit.spike_times, bins=bin_edges)
        spike_arr[:, idx] = spike_cnt
    return spike_arr


def load_nwb(fn: Union[str, Path], dataset: FalconTask = FalconTask.h1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
        Load data for evaluation.

        Returns:
        - neural_data: binned spike counts. Shape is (time x units)
        - covariates (e.g. kinematics). Shape is (time x n_kinematic_dims)
        - trial_change: boolean of shape (time,) true if the trial has changed
        - eval_mask: boolean array indicating whether to evaluate each time step
    """
    if not dataset in [FalconTask.h1, FalconTask.h2, FalconTask.m1, FalconTask.m2]:
        raise ValueError(f"Unknown dataset {dataset}")
    with NWBHDF5IO(str(fn), 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        if dataset == FalconTask.h1:
            kin = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:]
            timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(kin.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
            blacklist = nwbfile.acquisition['kin_blacklist'].data[:].astype(bool)
            binned_units = bin_units(units, bin_end_timestamps=timestamps)
            return binned_units, kin, np.zeros(kin.shape[0]), ~blacklist
        elif dataset == FalconTask.m1:
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

            binned_units = bin_units(units, bin_size_s=0.02, bin_end_timestamps=emg_timestamps)

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
            vel_container = nwbfile.acquisition['finger_vel']
            labels = [ts for ts in vel_container.time_series]
            vel_data = []
            vel_timestamps = None
            for ts in labels:
                ts_data = vel_container.get_timeseries(ts)
                vel_data.append(ts_data.data[:])
                vel_timestamps = ts_data.timestamps[:]
            vel_data = np.vstack(vel_data).T
            binned_units = bin_units(units, bin_size_s=0.02, bin_end_timestamps=vel_timestamps)

            eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)

            trial_change = np.zeros(vel_timestamps.shape[0], dtype=bool)
            # TODO trial change
            return binned_units, vel_data, vel_timestamps, eval_mask
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
            breakpoint()
