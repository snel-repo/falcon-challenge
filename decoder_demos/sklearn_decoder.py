r"""
    Load an sklearn decoder.
    To train, for example:
    `python decoder_demos/sklearn_decoder.py --training_dir data/h1/train --calibration_dir data/h1/test --mode all`
    To evaluate, see `sklearn_sample.py`
"""
from typing import List, Union, Optional
import argparse
import pickle
import numpy as np
from pathlib import Path

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.interface import BCIDecoder

from filtering import (
    apply_exponential_filter,
    NEURAL_TAU_MS,
)
from decoding_utils import (
    TRAIN_TEST,
    generate_lagged_matrix,
    fit_and_eval_decoder,
)

HISTORY = 0

def prepare_train_test(
        binned_spikes: np.ndarray,
        targets: np.ndarray,
        blacklist: Optional[np.ndarray]=None,
        history: int=0,
        ):
    signal = apply_exponential_filter(binned_spikes)

    # Remove timepoints where nothing is happening in the kinematics
    still_times = np.all(np.abs(targets) < 0.001, axis=1)
    if blacklist is not None:
        blacklist = still_times | blacklist
    else:
        blacklist = still_times

    train_x, test_x = np.split(signal, [int(TRAIN_TEST[0] * signal.shape[0])])
    train_y, test_y = np.split(targets, [int(TRAIN_TEST[0] * targets.shape[0])])
    train_blacklist, test_blacklist = np.split(blacklist, [int(TRAIN_TEST[0] * blacklist.shape[0])])

    x_mean, x_std = np.nanmean(train_x, axis=0), np.nanstd(train_x, axis=0)
    x_std[x_std == 0] = 1
    y_mean, y_std = np.nanmean(train_y[~train_blacklist], axis=0), np.nanstd(train_y[~train_blacklist], axis=0)
    y_std[y_std == 0] = 1
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    # train_y = (train_y - y_mean) / y_std # don't standardize y if using var weighted r2
    # test_y = (test_y - y_mean) / y_std

    train_blacklist = train_blacklist | np.isnan(train_y).any(axis=1)
    test_blacklist = test_blacklist | np.isnan(test_y).any(axis=1)
    if np.any(train_blacklist):
        print(f"Invalidating {np.sum(train_blacklist)} timepoints in train")
    if np.any(test_blacklist):
        print(f"Invalidating {np.sum(test_blacklist)} timepoints in test")

    if history > 0:
        train_x = generate_lagged_matrix(train_x, history)
        test_x = generate_lagged_matrix(test_x, history)
        train_y = train_y[history:]
        test_y = test_y[history:]
        if blacklist is not None:
            train_blacklist = train_blacklist[history:]
            test_blacklist = test_blacklist[history:]

    # Now, finally, remove by blacklist
    train_x = train_x[~train_blacklist]
    train_y = train_y[~train_blacklist]
    test_x = test_x[~test_blacklist]
    test_y = test_y[~test_blacklist]

    return train_x, train_y, test_x, test_y, x_mean, x_std, y_mean, y_std

class SKLearnDecoder(BCIDecoder):
    r"""
        Load an sklearn decoder. Assumes the dimensionality is correct.
    """
    def __init__(self, task_config: FalconConfig, model_path: str):
        self._task_config = task_config
        with open(model_path, 'rb') as f:
            payload = pickle.load(f)
            assert payload['task'] == task_config.task
            self.clf = payload['decoder']
            self.history = payload['history'] + 1
            MAX_HISTORY = int(NEURAL_TAU_MS / task_config.bin_size_ms) * 5 # bin size ms
            self.x_mean = payload['x_mean']
            self.x_std = payload['x_std']
            self.raw_history_buffer = np.zeros((MAX_HISTORY, task_config.n_channels))
            self.observation_buffer = np.zeros((self.history, task_config.n_channels))

    def reset(self, dataset: Path = ""):
        if isinstance(self.x_mean, dict):
            dataset_tag =  self.get_file_tag(dataset)
            if dataset_tag not in self.x_mean:
                raise ValueError(f"Dataset tag {dataset_tag} not found in calibration set {self.x_mean.keys()} - did you calibrate on this dataset?")
            self.local_x_mean = self.x_mean[dataset_tag]
            self.local_x_std = self.x_std[dataset_tag]
        else:
            self.local_x_mean = self.x_mean
            self.local_x_std = self.x_std
        self.raw_history_buffer = np.zeros_like(self.raw_history_buffer)
        self.observation_buffer = np.zeros_like(self.observation_buffer)

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), binned spike counts
        """
        # breakpoint()
        self.raw_history_buffer = np.roll(self.raw_history_buffer, -1, axis=0)
        self.raw_history_buffer[-1] = neural_observations
        smth_history = apply_exponential_filter(self.raw_history_buffer, NEURAL_TAU_MS)
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = (smth_history[-1] - self.local_x_mean) / self.local_x_std
        decoder_in = self.observation_buffer[::-1].copy().flatten().reshape(1, -1) # Reverse since this happens to be how the lagged matrix is formatted
        out = self.clf.predict(decoder_in)[0]
        return out

def fit_sklearn_decoder(
    datafiles: List[Path],
    calibration_datafiles: List[Path],
    task_config: FalconConfig,
    save_path: Path,
    history = 0,
):
    r"""
        Fit an sklearn (ridge regresssion) decoder.
    """
    (
        all_neural_data,
        all_covariates,
        all_trial_change,
        all_eval_mask
    ) = zip(*[load_nwb(fn, task_config.task) for fn in datafiles])
    all_neural_data = np.concatenate(all_neural_data)
    all_covariates = np.concatenate(all_covariates)
    all_trial_change = np.concatenate(all_trial_change)
    all_eval_mask = np.concatenate(all_eval_mask)
    (
        train_x,
        train_y,
        test_x,
        test_y,
        x_mean,
        x_std,
        y_mean,
        y_std
    ) = prepare_train_test(all_neural_data, all_covariates, ~all_eval_mask, history=HISTORY)
    score, decoder = fit_and_eval_decoder(train_x, train_y, test_x, test_y)
    print(f"CV Fit score: {score:.2f}")
    (
        cal_neural_data,
        _,
        _,
        _,
    ) = zip(*[load_nwb(fn, task_config.task) for fn in calibration_datafiles])
    fns = [BCIDecoder.get_file_tag(fn) for fn in calibration_datafiles]
    x_means = {}
    x_stds = {}
    for fn, neural_data in zip(fns, cal_neural_data):
        x_means[fn], x_stds[fn] = np.mean(neural_data, axis=0), np.std(neural_data, axis=0)
        if np.any(x_stds[fn] == 0):
            x_stds[fn][x_stds[fn] == 0] = 1
    decoder_obj = {
        'decoder': decoder,
        'task': task_config.task,
        'history': history,
        'x_mean': x_means,
        'x_std': x_stds,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(decoder_obj, f)
    return save_path

def fit_calibration(
    datafiles: List[Path],
    calibration_datafiles: List[Path],
    task_config: FalconConfig,
    save_path: Path
):
    return fit_sklearn_decoder(
        calibration_datafiles,
        calibration_datafiles,
        task_config,
        save_path
    )

def fit_last_session(
    datafiles: List[Path],
    calibration_datafiles: List[Path],
    task_config: FalconConfig,
    save_path: Path
):
    day_unique = set([f.stem.split('_')[0] for f in datafiles])
    last_day = sorted(day_unique)[-1]
    fit_datafiles = [d for d in datafiles if last_day in d.stem]
    return fit_sklearn_decoder(
        fit_datafiles,
        calibration_datafiles,
        task_config,
        save_path
    )

def main(training_dir, calibration_dir, mode):
    # Your main function logic here
    if mode == 'all':
        fit_fn = fit_sklearn_decoder
    elif mode == 'last':
        fit_fn = fit_last_session
    else:
        raise ValueError(f"Unknown mode: {mode}")
    training_dir = Path(training_dir)
    calibration_dir = Path(calibration_dir)
    task_config = FalconConfig(
        task=FalconTask.h1,
    )
    save_path = Path(f'local_data/sklearn_{task_config.task}.pkl')
    datafiles = list(training_dir.glob('*.nwb'))
    calibration_datafiles = list(calibration_dir.glob('*calibration.nwb'))
    fit_fn(datafiles, calibration_datafiles, task_config, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sklearn decoder')

    parser.add_argument('--training_dir', '-t', type=str, help='Root directory for training files')
    parser.add_argument('--calibration_dir', '-c', type=str, help='Root directory for calibration files')
    parser.add_argument('--mode', '-m', type=str, choices=['all', 'last'], help='Mode for training')

    args = parser.parse_args()

    main(args.training_dir, args.calibration_dir, args.mode)