r"""
    Dataloading utilities for evaluator
"""
from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

# Load nwb file
def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.01,
        bin_end_timestamps: np.ndarray | None = None
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


def load_nwb(fn: str | Path, dataset: str = 'h1') -> Tuple[np.ndarray, np.ndarray]:
    r"""
        Load data for evaluation.

        Returns:
        - neural_data: binned spike counts. Shape is (time x units)
        - covariates (e.g. kinematics). Shape is (time x n_kinematic_dims)
        - eval_mask: boolean array indicating whether to evaluate each time step
    """
    if dataset == 'h1':
        with NWBHDF5IO(str(fn), 'r') as io:
            nwbfile = io.read()
            # print(nwbfile)
            units = nwbfile.units.to_dataframe()
            kin = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:]
            timestamps = nwbfile.acquisition['OpenLoopKinematics'].timestamps[:]
            blacklist = nwbfile.acquisition['Blacklist'].data[:].astype(bool)
            return bin_units(units, bin_end_timestamps=timestamps), kin, ~blacklist
    else:
        raise ValueError(f"Unknown dataset {dataset}")
