from typing import List
from functools import partial
import os
import pickle
import torch
import re
from collections import defaultdict
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from edit_distance import SequenceMatcher
from time import time, sleep

from falcon_challenge.config import FalconTask, FalconConfig
from falcon_challenge.interface import BCIDecoder
from falcon_challenge.dataloaders import load_nwb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data hash map, what participant submits, keys of comparison dict
DATASET_HELDINOUT_MAP = {
    'h1': {
        'held_in': [
            'S0_set_1', 
            'S0_set_2', 
            'S1_set_1', 
            'S1_set_2', 
            'S1_set_3',
            'S2_set_1',
            'S2_set_2',
            'S3_set_1',
            'S3_set_2',
            'S4_set_1',
            'S4_set_2',
            'S5_set_1',
            'S5_set_2',
        ],
        'held_out': [
            'S6_set_1', 
            'S6_set_2', 
            'S7_set_1', 
            'S7_set_2', 
            'S8_set_1', 
            'S8_set_2', 
            'S9_set_1', 
            'S9_set_2', 
            'S10_set_1', 
            'S10_set_2', 
            'S11_set_1', 
            'S11_set_2', 
            'S12_set_1',
            'S12_set_2',
        ],
    },
    'm1': {
        'held_in': ['20120924', '20120926', '20120927', '20120928'],
        'held_out': ['20121004', '20121017', '20121024'],
    },
    'h2': {
        'held_in': [
            '2022.05.18', 
            '2022.05.23', 
            '2022.05.25', 
            '2022.06.01', 
            '2022.06.03', 
            '2022.06.06', 
            '2022.06.08', 
            '2022.06.13', 
            '2022.06.15', 
            '2022.06.22', 
            '2022.09.01', 
            '2022.09.29', 
            '2022.10.06',
            '2022.10.18', 
            '2022.10.25', 
            '2022.10.27', 
            '2022.11.01', 
            '2022.11.03', 
            '2022.12.08',
            '2022.12.15', 
            '2023.02.28',
        ],
        'held_out': [
            '2023.04.17', 
            '2023.05.31', 
            '2023.06.28', 
            '2023.08.16', 
            '2023.10.09'
        ]
    },
    'm2': {
        'held_in': ['Run1_20201019', 'Run2_20201019', 'Run1_20201020', 'Run2_20201020', 'Run1_20201027', 'Run2_20201027', 'Run1_20201028'],
        'held_out': ['Run1_20201030', 'Run2_20201030', 'Run1_20201118', 'Run1_20201119', 'Run1_20201124', 'Run2_20201124'],
    },
    'b1': {
        'held_in': ['20210626', '20210627', '20210628'],
        'held_out': ['20210630', '20210701', '20210705']
    },
}

# Used to label test server data file names to look for
# These act as a _reduction_ set of the full list of datafiles considered, to specific sessions. Relevant for H1/M2.
def reduce_key(key):
    if key.startswith('Run'):
        return key.split('_')[1]
    if key.startswith('L_'):
        return key
    if key.startswith('S'):
        return key.split('_')[0]    
    return key

HELD_IN_KEYS = {
    FalconTask.h1: ['S0_', 'S1_', 'S2_', 'S3_', 'S4_', 'S5_'],
    FalconTask.m1: ['L_20120924', 'L_20120926', 'L_20120927', 'L_20120928'],
    FalconTask.m2: ['20201019', '20201020', '20201027', '20201028'],
    FalconTask.h2: DATASET_HELDINOUT_MAP['h2']['held_in'],
    FalconTask.b1: DATASET_HELDINOUT_MAP['b1']['held_in'],
}

HELD_OUT_KEYS = {
    FalconTask.h1: ['S6_', 'S7_', 'S8_', 'S9_', 'S10_', 'S11_', 'S12_'],
    FalconTask.m1: ['L_20121004', 'L_20121017', 'L_20121024'],
    FalconTask.m2: ['20201030', '20201118', '20201119', '20201124'],
    FalconTask.h2: DATASET_HELDINOUT_MAP['h2']['held_out'],
    FalconTask.b1: DATASET_HELDINOUT_MAP['b1']['held_out'],
}

RECOMMENDED_BATCH_SIZES = {
    FalconTask.h1: 8,
    FalconTask.m1: 4,
    FalconTask.h2: 1,
    FalconTask.m2: 7, # max
    FalconTask.b1: 1, 
}

# Development time flag. False allows direct evaluation without payload writing, only usable for local minival.
# Should be set to true for remote evaluation.
# USE_PKLS = False
USE_PKLS = True

BIN_SIZE = 0.02

HELDIN_OR_OUT_MAP = {
    'held_in': "Held In",
    'held_out': "Held Out",
}

def evaluate(
    test_annotation_file: str, # The annotation file for the phase
    user_submission_file: str, # * JY: This appears to always be /submission/submission.csv on EvalAI. No matter - load it as a pickle.
    phase_codename: str, # e.g. minival or test
    verbose: bool = False,
    **kwargs
):
    r"""
        Test struct:
        {
            'h1': {
                'hash': {
                    'data': tgt,
                    'mask': mask,
                }
            }
        }
        User submission struct:
        {
            'h1': {
                'hash': pred,

                'normalized_latency': 1,
            }
        }
        Evaluate payloads with potentially multiple splits worth of data
        - Low pri: can I provide all results or just one split's worth entry? Currently providing 1, examples just provide 1, but in general would be nice to provide all. User shouldn't be able to submit more than 1, though.
    """
    # ! Want: Locally, test_annotation should be somewhere safe (tmp)
    # ! Remotely, it shoudl be /submission/submission.csv exactly.
    # Ignore explicit annotations provided and directly search for concatenated answers
    logger.info(f"Evaluation: Docker side")
    # test_annotation_file = '/dataset/evaluation_data/answer_key/minival.pkl'
    logger.info(f"Loading GT from {test_annotation_file}")
    logger.info(f"Loading submission from {user_submission_file}")
    logger.info(f"Phase: {phase_codename}")

    result = []
    # Load pickles
    try:
        with open(test_annotation_file, 'rb') as test_annotation_file, open(user_submission_file, 'rb') as user_submission_file:
            test_annotations = pickle.load(test_annotation_file)
            user_submission = pickle.load(user_submission_file)
    except Exception as e:
        logger.error(f"Checking root: {os.listdir('/')}")
        raise ValueError(f"Failed to load submission pickles: {e}. dir is {os.getcwd()}; contents {os.listdir()}.")
    for datasplit in user_submission: # datasplit e.g. h1, m1
        if datasplit not in test_annotations:
            raise ValueError(f"Missing {datasplit} in GT labels.")
        split_annotations = test_annotations[datasplit]
        split_result = {}
        split_result["Normalized Latency"] = user_submission[datasplit]["normalized_latency"]
        del user_submission[datasplit]["normalized_latency"]
        pred_dict = defaultdict(list)
        tgt_dict = defaultdict(list)
        mask_dict = defaultdict(list)
        if 'h2' not in datasplit:
            dset_len_dict = defaultdict(lambda: defaultdict(list))
        if 'm2' in datasplit:
            grouped = {}
            for set_name in user_submission[datasplit]:
                run, date = set_name.split('_')
                if date not in grouped:
                    grouped[date] = []
                grouped[date].append(set_name)
            for date in grouped:
                grouped[date].sort()
            all_datasets = [run for runs in grouped.values() for run in runs]
        else:
            all_datasets = user_submission[datasplit]
        for dataset in all_datasets:
            dataset_pred = user_submission[datasplit][dataset]
            dataset_tgt = split_annotations[dataset]['data']
            dataset_mask = split_annotations[dataset]['mask']
            dataset_pred = dataset_pred[:dataset_mask.shape[0]] # In case excess timesteps are predicted due to batching, reduce
            
            if dataset in DATASET_HELDINOUT_MAP[datasplit]['held_in']:
                if 'h2' not in datasplit:
                    # For splits with multiple datasets per session (H1 and M2), we need to map predictions, targets, and masks for each dataset to the session ID 
                    session_id = reduce_key(dataset)
                    dset_len_dict['held_in'][session_id].append(dataset_mask.shape[0])
                pred_dict['held_in'].append(dataset_pred)
                tgt_dict['held_in'].append(dataset_tgt)
                mask_dict['held_in'].append(dataset_mask)
            elif dataset in DATASET_HELDINOUT_MAP[datasplit]['held_out']:
                if not 'h2' in datasplit:
                    # For splits with multiple datasets per session (H1 and M2), we need to map predictions, targets, and masks for each dataset to the session ID 
                    session_id = reduce_key(dataset)
                    dset_len_dict['held_out'][session_id].append(dataset_mask.shape[0])
                pred_dict['held_out'].append(dataset_pred)
                tgt_dict['held_out'].append(dataset_tgt)
                mask_dict['held_out'].append(dataset_mask)
            else:
                raise ValueError(f"Dataset {dataset} submitted but not found in held-in or held-out list of split {datasplit}.")
        
        for in_or_out in pred_dict:

            if len(pred_dict[in_or_out]) < len(DATASET_HELDINOUT_MAP[datasplit][in_or_out]):
                raise ValueError(f"Missing predictions for {datasplit} {in_or_out}. User submitted: {user_submission[datasplit].keys()}. Expecting more like: {HELDIN_OR_OUT_MAP[datasplit][in_or_out]}.")

            if 'b1' in datasplit:
                # B1 computes metrics across sessions independently. Don't concatenate.
                pred, tgt, mask = pred_dict[in_or_out], tgt_dict[in_or_out], mask_dict[in_or_out]            
            elif 'h2' in datasplit:
                pred = pred_dict[in_or_out]
                tgt = tgt_dict[in_or_out]
                mask = np.concatenate(mask_dict[in_or_out])
            else:
                pred = np.concatenate(pred_dict[in_or_out])
                tgt = np.concatenate(tgt_dict[in_or_out])
                dset_lens = dset_len_dict[in_or_out]
                mask = np.concatenate(mask_dict[in_or_out])
            
            try:
                if 'b1' in datasplit:
                    metrics = FalconEvaluator.compute_metrics_spectrogram_distance(pred, tgt, mask)
                elif 'h2' in datasplit:
                    metrics = FalconEvaluator.compute_metrics_edit_distance(pred, tgt, mask)
                else:
                    metrics = FalconEvaluator.compute_metrics_regression(pred, tgt, mask, dset_lens, verbose=verbose)
            except Exception as e:
                raise ValueError(f"Failed to compute metrics for {datasplit} {in_or_out}: {e}. Lengths submitted: {[len(piece) for piece in pred_dict[in_or_out]]}")
            for k in metrics:
                split_result[f'{HELDIN_OR_OUT_MAP[in_or_out]} {k}'] = metrics[k]
        result.append({f'{phase_codename}_split_{datasplit}': split_result})

    print(f"Returning result from phase: {phase_codename}: {result}")
    # Out struct according to https://evalai.readthedocs.io/en/latest/evaluation_scripts.html
    return {"result": result, 'submission_result': result[0]}

class EvalDataset(Dataset):
    r"""
        Simple class to help with batched evaluation.
        data: List of numpy arrays, one per datafile
        datafiles: str names, for cuing models which dataset is being evaluated
        trial_mode: serve trialized data (padded) or just parallelize among datafiles
    """
    def __init__(self, 
                 data: List[np.ndarray], 
                 trial_change: List[np.ndarray], 
                 targets: List[np.ndarray], 
                 eval_mask: List[np.ndarray], 
                 datafiles: List[str],
                 trial_mode=False):
        self.data = data
        self.trial_change = trial_change
        self.targets = targets
        self.eval_mask = eval_mask
        self.datafiles = datafiles
        assert not trial_mode, "Trial mode not implemented."

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            self.data[idx], 
            self.targets[idx], 
            self.trial_change[idx], 
            self.eval_mask[idx], 
            idx # needed to index for datafiles which are not natively loaded for pytorch dataloader
        )
    
    def get_datafile(self, idx):
        return self.datafiles[idx]

def simple_collater(batch, task):
    r"""
        Collates a batch of data, targets, trial_change, eval_mask, and datafile_idx.
        data: List of numpy arrays, one per datafile
        datafiles: str names, for cuing models which dataset is being evaluated
    """
    data, targets, trial_change, eval_mask, datafile_idx = zip(*batch)
    data = pad_sequence([torch.tensor(d) for d in data], batch_first=False).numpy()
    if task == FalconTask.h2:
        targets = pad_sequence([torch.tensor(t) for t in targets[0]], batch_first=True).numpy()
    else:
        targets = pad_sequence([torch.tensor(t) for t in targets], batch_first=False).numpy()
    trial_change = pad_sequence([torch.tensor(t) for t in trial_change], batch_first=False).numpy()
    eval_mask = pad_sequence([torch.tensor(t) for t in eval_mask], batch_first=False).numpy() # Serves as a mask for padding as well
    datafile_idx = np.array(datafile_idx)
    return data, targets, trial_change, eval_mask, datafile_idx

class FalconEvaluator:

    def __init__(self, eval_remote=False, split='h1', verbose=False, dataloader_workers=8):
        r"""
            verbose: Print out dataset specific metrics for movement tasks.
            dataloader_workers: Number of workers to use for dataloading, only meaningful up to # of datasets in a split. Set to 0 to run multiple Evaluators in multiprocessing
        """
        self.eval_remote = eval_remote
        assert split in ['h1', 'h2', 'm1', 'm2', 'b1'], "Split must be h1, h2, m1, m2 or b1."
        if split in ['h1', 'm1', 'm2']:
            self.continual = True
        else:
            self.continual = False
        self.verbose = verbose
        self.num_workers = dataloader_workers
        self.dataset: FalconTask = getattr(FalconTask, split)
        self.cfg = FalconConfig(self.dataset)

    @staticmethod
    def get_eval_handles(is_remote: bool, dataset: FalconTask, phase: str = 'minival'):
        if is_remote: # i.e. definitely docker
            data_dir = os.environ.get("EVAL_DATA_PATH")
        else: # possibly docker or local
            if os.path.exists(f"./data/{dataset.name}"):
                logger.info("Using local data directory.")
                data_dir = "data"
            else:
                data_dir = os.environ.get("EVAL_DATA_PATH") # a local docker eval
        data_dir = Path(data_dir) / dataset.name
        if not data_dir.exists():
            raise FileNotFoundError(f"Evaluation data directory {data_dir} not found.")
        if phase == 'test': # TODO wire wherever test is actually stored on remote
            eval_dir = data_dir / f"eval"
        else:
            eval_dir = data_dir / "minival"
        if not eval_dir.exists():
            raise FileNotFoundError(f"Evaluation directory {eval_dir} found but requested phase {phase} not found.")
        print(sorted(list(eval_dir.glob("*val*.nwb"))))
        return sorted(list(eval_dir.glob("*val*.nwb")))

    def get_eval_files(self, phase: str = 'minival'):
        logger.info("Searching for evaluation data.")
        handles = self.get_eval_handles(self.eval_remote, self.dataset, phase=phase)
        logger.info(f"Found {len(handles)} files.")
        if len(handles) == 0:
            raise FileNotFoundError(f"No files found in {self.dataset.name} for phase {phase}. Note test phase data is only available on EvalAI remote.")
        return handles
    
    def predict_files(self, decoder: BCIDecoder, eval_files: List):
        # returns triple dict, keyed by datafile hash and contains preds, targets, and eval_mask respective. also returns lists compute_time and neural_time
        # TODO this does not return uniquely identifiable data if eval_files is partial, e.g. if we only has set 2 of a day with 2 sets, we'll happily just provide partial predictions.
        # Pre-loop before starting batch loads
        
        file_neural_data = []
        file_trial_change = []
        file_targets = []
        file_eval_mask = []
        all_neural_times = []
        datafiles = list(sorted(eval_files))

        for datafile in datafiles:
            if not datafile.exists():
                raise FileNotFoundError(f"File {datafile} not found.")
        
            neural_data, decoding_targets, trial_change, eval_mask = load_nwb(datafile, dataset=self.dataset)
            
            file_neural_data.append(neural_data)
            file_trial_change.append(trial_change)
            file_targets.append(decoding_targets)
            file_eval_mask.append(eval_mask)

            if self.dataset == FalconTask.b1: BIN_SIZE = 1/30000
            all_neural_times.append(neural_data.shape[0] * BIN_SIZE)
    
        dataset = EvalDataset(
            data=file_neural_data,
            trial_change=file_trial_change,
            targets=file_targets,
            eval_mask=file_eval_mask,
            datafiles=datafiles,
        )
        
        all_preds = defaultdict(list)
        all_targets = defaultdict(list)
        all_eval_mask = defaultdict(list)
        all_compute_times = []

        print('Decoder batch size: ', decoder.batch_size)
        
        simple_collater_partial = partial(simple_collater, task=self.dataset)
        dataloader = DataLoader(
            dataset, shuffle=False,
            batch_size=decoder.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=simple_collater_partial,
        )
        # from time import time
        # for neural_data, decoding_targets, trial_change, eval_mask, datafile in tqdm(dataset):
        
        for neural_data, decoding_targets, trial_change, eval_mask, datafile_idx in tqdm(dataloader):
            neural_data: np.ndarray
            decoding_targets: np.ndarray
            trial_change: np.ndarray
            eval_mask: np.ndarray
            datafile_idx: np.ndarray
            
            decoder.reset(dataset_tags=[dataset.datafiles[idx] for idx in datafile_idx])
            
            trial_preds = []
            # loop_times = []
            # breakpoint()

            # B1 predicts using a whole-trial worth of neural data and neural_observations and eval_mask have different sampling rates.
            if self.dataset == FalconTask.b1:
                
                trial_neural_observations = []
                # neural_data and eval_mask have different sampling rates in B1
                for neural_observations, trial_delta_obs in zip(neural_data, trial_change):
                    
                    if not trial_delta_obs[0]:
                        trial_neural_observations.append(neural_observations) # Samples x 1 x channels
                    else: 
                        trial_neural_observations.append(neural_observations)
                        trial_neural_observations = np.array(trial_neural_observations).transpose(1, 2, 0)
                        
                        start_time = time()
                        # decoder.predict expects a whole-trial worth of neural data and returns a wholde-trial worth of reconstructed audio.
                        trial_preds.append(decoder.predict(np.array(trial_neural_observations)))
                        end_time = time()
                        all_compute_times.append(end_time - start_time)
                        
                        trial_neural_observations = []
            else:
                for neural_observations, trial_delta_obs, step_mask in zip(neural_data, trial_change, eval_mask):
                    
                    neural_observations: np.ndarray
                    trial_delta_obs: np.ndarray
                    step_mask: np.ndarray
                    if self.dataset == FalconTask.h2:
                        assert neural_data.shape[1] == 1, "H2 expects batch size 1."
                        if step_mask[0]:
                            start_time = time()
                            decoder.predict(neural_observations)
                            end_time = time()
                            all_compute_times.append(end_time - start_time)
                        if trial_delta_obs[0]:
                            trial_preds.append(decoder.on_done(trial_delta_obs))
                    else:
                        if not self.continual:
                            decoder.on_done(trial_delta_obs)
                        start_time = time()
                        step_prediction = decoder.predict(neural_observations)
                        end_time = time()
                        all_compute_times.append(end_time - start_time)
                        assert step_prediction.shape[1] == self.cfg.out_dim, f"Prediction shape mismatch: {step_prediction.shape[1]} vs {self.cfg.out_dim}."
                        trial_preds.append(step_prediction)
                    
                    # if step_mask.any():
                    #     trial_preds.append(decoder.predict(neural_observations))
                    # else:
                    #     decoder.observe(neural_observations)
                    #     trial_preds.append(np.full((decoder.batch_size, self.cfg.out_dim), np.nan))
                # loop_times.append(time() - loop_start)
            # loop_times = np.array(loop_times)
            # print(f"Loop {len(loop_times)}: {loop_times.mean()} +/- {loop_times.std()}")
            
            if self.dataset == FalconTask.b1:
                datafile_hash = self.cfg.hash_dataset(dataset.get_datafile(datafile_idx[0]))
                trial_preds = np.concatenate(trial_preds, axis=1).transpose()[:, np.newaxis, :]
                all_preds[datafile_hash].append(trial_preds)
                all_targets[datafile_hash].append(decoding_targets)
                all_eval_mask[datafile_hash].append(eval_mask)
            elif self.dataset == FalconTask.h2:
                datafile_hash = self.cfg.hash_dataset(dataset.get_datafile(datafile_idx[0]))
                all_preds[datafile_hash].append(trial_preds)
                all_targets[datafile_hash].append(decoding_targets)
                all_eval_mask[datafile_hash].append(eval_mask)
            else:
                trial_preds = np.stack(trial_preds) # -> T x B x H
                for idx in range(len(datafile_idx)):
                    datafile_hash = self.cfg.hash_dataset(dataset.get_datafile(datafile_idx[idx]))
                    all_preds[datafile_hash].append(trial_preds[:, idx])
                    all_targets[datafile_hash].append(decoding_targets[:, idx])
                    all_eval_mask[datafile_hash].append(eval_mask[:, idx])
        for k in all_preds:
            if self.dataset != FalconTask.h2:
                all_preds[k] = np.concatenate(all_preds[k])
                all_targets[k] = np.concatenate(all_targets[k])
            all_eval_mask[k] = np.concatenate(all_eval_mask[k])
        return all_preds, all_targets, all_eval_mask, all_compute_times, all_neural_times

    def evaluate_files(self, decoder: BCIDecoder, eval_files: List):
        all_preds, all_targets, all_eval_mask = self.predict_files(decoder, eval_files)
        metrics = self.compute_metrics(all_preds, all_targets, all_eval_mask)
        return metrics

    def evaluate(
            self, 
            decoder: BCIDecoder, 
            phase: str, 
            held_out_only: bool = False, 
            specific_keys: List = []
        ):
        r"""
            Note: Locally, this can produce metrics, but locally and remotely it should also write a submission file 
            that the actual evaluator on remote uses. The evaluation is done separately on remote.

            held_out_only: Only run predictions on held out
            specific_keys: Overrides held_out_only. Only run predictions on datafiles with specific keys.
        """
        assert phase in ['minival', 'test'], "Phase must be minival or test."
        if phase == 'minival' and (held_out_only or specific_keys):
            logger.warning("Ignoring held_out_only and specific_keys for minival phase.")
            held_out_only = False
        if decoder.batch_size > RECOMMENDED_BATCH_SIZES[self.dataset]:
            logger.warning(f"Decoder batch size {decoder.batch_size} is larger than limit {RECOMMENDED_BATCH_SIZES[self.dataset]} for {self.dataset}, clipping down.")
            decoder.set_batch_size(RECOMMENDED_BATCH_SIZES[self.dataset])
        
        np.random.seed(0)
        # ! TODO ideally seed other libraries as well...? Is that our responsibility?
        
        eval_files = self.get_eval_files(phase=phase)
        metrics = {}
        prediction_env_var = "PREDICTION_PATH" if self.eval_remote else "PREDICTION_PATH_LOCAL"
        prediction_path = os.environ.get(prediction_env_var, './local_prediction.pkl')
        if not prediction_path:
            raise ValueError("PREDICTION_PATH not set in remote env which expects it. Cannot forward to separate evaluate runscript.")
        gt_path = os.environ.get("GT_PATH", './local_gt.pkl')
        if not gt_path:
            raise ValueError("GT_PATH not set in remote env which expects it. Cannot forward to separate evaluate runscript.")
            
        if phase == 'test':
            eval_files_held_in = [f for f in eval_files if any(k in f.name for k in HELD_IN_KEYS[self.dataset])]
            eval_files_held_out = [f for f in eval_files if any(k in f.name for k in HELD_OUT_KEYS[self.dataset])]
            assert len(eval_files) == len(eval_files_held_in) + len(eval_files_held_out), f"Mismatch in extracted eval #: Eval file state is not consistent with benchmark creation settings. Found {len(eval_files)} files, {len(eval_files_held_in)} held in, {len(eval_files_held_out)} held out."

            if specific_keys:
                raise NotImplementedError("not sure what metrics to compute for specific keys yet.")
            elif held_out_only:
                eval_files_held_in = []
            all_preds, all_targets, all_eval_mask, all_compute_times, all_neural_times = self.predict_files(decoder, eval_files_held_out)
            if eval_files_held_in:
                all_preds_held_in, all_targets_held_in, all_eval_mask_held_in, _, _ = self.predict_files(decoder, eval_files_held_in)
                all_preds.update(all_preds_held_in)
                all_targets.update(all_targets_held_in)
                all_eval_mask.update(all_eval_mask_held_in)

        else:            
            all_preds, all_targets, all_eval_mask, all_compute_times, all_neural_times = self.predict_files(decoder, eval_files)
        # Indirect remote setup to satisfy EvalAI interface. Save metrics / comparison to file.
        if USE_PKLS:
            inner_pred = {**all_preds}
            inner_tgt_spoof = { # spoof for local mirror of eval ai path, in reality targets are already compiled on eval ai side.
                k: {
                    'data': all_targets[k] if (self.dataset == FalconTask.h2 or self.dataset == FalconTask.b1) else all_targets[k][all_eval_mask[k]],
                    'mask': all_eval_mask[k],
                } for k in all_targets
            }
            inner_pred['normalized_latency'] = np.sum(all_compute_times) / np.sum(all_neural_times)
            pred_payload = {self.dataset.name: inner_pred}
            truth_payload = {self.dataset.name: inner_tgt_spoof}
        else:
            pass
            
        if USE_PKLS:
            Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
            with open(prediction_path, 'wb') as f:
                pickle.dump(pred_payload, f)
            Path(gt_path).parent.mkdir(parents=True, exist_ok=True)
            with open(gt_path, 'wb') as f:
                pickle.dump(truth_payload, f)

            if self.eval_remote:
                print("Sleeping before exiting for remote eval - feel free to interrupt for local eval.", flush=True)
                # Gunjan, EvalAI contact says that current static code eval has an issue where the submission dump is only polled by the EvalAI worker comparison script every 5 minutes
                # Sleep so it's definitely available
                sleep(300) 
            else:
                return evaluate(
                    test_annotation_file=gt_path,
                    user_submission_file=prediction_path,
                    phase_codename=phase,
                    verbose=self.verbose
                )
        else:
            for k, v in metrics.items():
                logger.info("{}: {}".format(k, v))

    
    @staticmethod
    def compute_metrics_spectrogram_distance(preds, targets, eval_mask):
        '''
        preds, targets, eval_mask must have shapes [sessions x samples x 1 x frequencies]
        '''
        
        def normalize_signal(x):
            """"
            Normalizes signal x between 0 and 1.
            """
            return (x-np.min(x))/(np.max(x)-np.min(x))    
          
        error_per_session = []
        trial_len = 880
        
        for sess_idx in range(len(preds)):
            prd, tgt, msk = np.array(preds[sess_idx]), np.array(targets[sess_idx]), np.array(eval_mask[sess_idx])
            
            if prd.shape != tgt.shape or prd.shape != msk.shape:
                raise ValueError(f"Targets and predictions have different lengths: {len(tgt)} vs {len(prd)}.")

            # Reshape to normalize and compute error at the trial level:
            prd = prd.reshape(-1, trial_len, prd.shape[-2], prd.shape[-1])
            tgt = tgt.reshape(-1, trial_len, tgt.shape[-2], tgt.shape[-1])
            msk = msk.reshape(-1, trial_len, msk.shape[-2], msk.shape[-1])
            
            samples, frequencies = prd.shape[1], prd.shape[-1]

            error_per_trial = []
            for trial in range(len(prd)):
            
                sess_sxx_eval_mask = msk[trial].reshape(samples, frequencies)
                original_sxx_masked = tgt[trial].reshape(samples, frequencies) * sess_sxx_eval_mask
                reconstructed_sxx_masked = prd[trial].reshape(samples, frequencies) * sess_sxx_eval_mask
            
                # Calculate spectrogram reconstruction error
                error_per_trial.append(mean_squared_error(normalize_signal(original_sxx_masked), normalize_signal(reconstructed_sxx_masked)))

            error_per_session.append(np.mean(error_per_trial))
        
        base_metrics = {
            "MSE Mean": np.mean(error_per_session),
            "MSE Std.": np.std(error_per_session)
        }
        
        return base_metrics
    
    @staticmethod
    def compute_metrics_regression(preds, targets, eval_mask, dset_lens, verbose=False): # Verbose drop-in
        dsets = sorted(dset_lens.keys())
        dset_bounds = np.cumsum([sum(dset_lens[key]) for key in dsets])
        masked_points = np.cumsum(~eval_mask)
        dset_bounds = [0] + [dset_len - masked_points[dset_len - 1] for dset_len in dset_bounds]
        # assumes targets are already masked
        preds = preds[eval_mask]
        if not targets.shape[0] == preds.shape[0]:
            raise ValueError(f"Targets and predictions have different lengths: {targets.shape[0]} vs {preds.shape[0]}.")
        r2_scores = [r2_score(targets[dset_bounds[i]:dset_bounds[i+1]], preds[dset_bounds[i]:dset_bounds[i+1]], 
                              multioutput='variance_weighted') for i in range(len(dset_bounds) - 1)]
        base_metrics = {
            "R2 Mean": np.mean(r2_scores),
            "R2 Std.": np.std(r2_scores)
        }
        if verbose:
            for k, r2 in zip(dsets, r2_scores):
                print(f"{k}: {r2}")
                base_metrics[f"{k} R2"] = r2
            preds_dict = {k: preds[dset_bounds[i]:dset_bounds[i+1]] for i, k in enumerate(dsets)}
            with open(f'preds_{dsets}.pkl', 'wb') as f:
                pickle.dump(preds_dict, f)
        return base_metrics


    @staticmethod
    def compute_metrics_edit_distance(preds, targets, eval_mask):
        if len(preds) != len(targets):
            raise ValueError(f"Targets and predictions have different lengths: {len(targets)} vs {len(preds)}.")
        
        def _to_words(s):
            s = s.replace(">", " ")  # Remove space token when computing WER
            s = re.sub(r"([~,!?])", r" \1", s)
            s = s.split(" ")
            return s
        
        cers = []
        wers = []
        for sess_idx, (sess_preds, sess_tgts) in enumerate(zip(preds, targets)):
            if len(sess_preds) != len(sess_tgts):
                raise ValueError(f"Targets and predictions have different lengths: {len(sess_tgts)} vs {len(sess_preds)}.")
            char_counts = []
            word_counts = []
            char_distances = []
            word_distances = []

            # Batch size == 1
            sess_preds = sess_preds[0]
            sess_tgts = sess_tgts[0]

            for pred, target in zip(sess_preds, sess_tgts):
                target = "".join([chr(c) for c in target if c != 0])
                matcher = SequenceMatcher([c for c in pred], [c for c in target])
                char_distances.append(matcher.distance())
                char_counts.append(len(target))

                pred = _to_words(pred)
                target = _to_words(target)
                matcher = SequenceMatcher(pred, target)
                word_distances.append(matcher.distance())
                word_counts.append(len(target))

            cers.append(np.sum(char_distances) / np.sum(char_counts))
            wers.append(np.sum(word_distances) / np.sum(word_counts))

        return {
            "WER": np.mean(wers),
            "CER": np.mean(cers),
            "WER Std.": np.std(wers),
            "CER Std.": np.std(cers),
        }

    def compute_metrics(self, all_preds, all_targets, all_eval_mask=None):
        r"""
            all_preds: array of shape (n_timesteps, k_dim)
            all_targets: array of shape (n_timesteps, k_dim)
            all_eval_mask: array of shape (n_timesteps, k_dim). True if we should evaluate this timestep.
        """
        if self.dataset in [FalconTask.h1, FalconTask.m1, FalconTask.m2]:
            metrics = self.compute_metrics_regression(all_preds, all_targets, all_eval_mask, verbose=self.verbose)
        elif self.dataset in [FalconTask.h2]:
            metrics = self.compute_metrics_edit_distance(all_preds, all_targets, all_eval_mask)
        elif self.dataset in [FalconTask.b1]:
            metrics = self.compute_metrics_spectrogram_distance(all_preds, all_targets, all_eval_mask)
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")
        return metrics


r"""
    JY isn't too clear what happens eval-server side on submission but 1 of 2 patterns seem plausible:
    1. Server receives dockerfile and runs it blind.
    - Submitted docker code should call evaluation with config that points to GT data that only lives remotely.
    - This is the pattern that the Habitat challenge uses.
    In order for the image to access the test data, we can mount the data when we run on eval side.
    `docker run -v /path/on/host:/path/in/container your_image_name`
    2. Evaluator
    - Server receives dockerfile, and runs python evaluate.py --code Dockerfile
    And somehow the evaluator will dynamically import the agent code to run it.
"""