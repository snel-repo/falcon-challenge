from typing import List
import os
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

from falcon_challenge.config import FalconTask
from falcon_challenge.interface import BCIDecoder
from falcon_challenge.dataloaders import load_nwb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HELD_IN_KEYS = {
    FalconTask.h1: ['S608', 'S610', 'S613', 'S615', 'S619', 'S625'],
}

HELD_OUT_KEYS = {
    FalconTask.h1: ['S627', 'S631', 'S633', 'S636', 'S639', 'S641', 'S644']
}

# Development time flag. False allows direct evaluation without payload writing, only usable for local minival.
# Should be set to true for remote evaluation.
# USE_PKLS = False
USE_PKLS = True

HELDIN_OR_OUT_MAP = {
    'held_in': "Held In",
    'held_out': "Held Out",
}

def evaluate(
    test_annotation_file: str,
    user_submission_file: str,
    phase_codename: str,
    **kwargs
):
    r"""
        Evaluate payloads with potentially multiple splits worth of data
        - Low pri: can I provide all results or just one split's worth entry? Currently providing 1, examples just provide 1, but in general would be nice to provide all. User shouldn't be able to submit more than 1, though.
    """
    result = []
    # Load pickles
    with open(test_annotation_file, 'rb') as test_annotation_file, open(user_submission_file, 'rb') as user_submission_file:
        test_annotations = pickle.load(test_annotation_file)
        user_submission = pickle.load(user_submission_file)
    
    for datasplit in test_annotations:
        split_annotations = test_annotations[datasplit]
        if datasplit not in user_submission:
            raise ValueError(f"Missing {datasplit} in user submission.")
        split_result = {}
        split_result["Normalized Latency"] = user_submission[datasplit]["normalized_latency"]
        for in_or_out in split_annotations.keys():
            if f'{in_or_out}_pred' in user_submission[datasplit]:
                pred = user_submission[datasplit][f'{in_or_out}_pred']
                mask = user_submission[datasplit][f'{in_or_out}_eval_mask']
                # User submission should be in an expected format because we force predictions through our codepack interface... right? They could hypothetically spoof. But we see dockerfile.
                eval_fn = FalconEvaluator.compute_metrics_classification if 'h2' in datasplit else FalconEvaluator.compute_metrics_regression
                metrics_held_in = eval_fn(pred, split_annotations[in_or_out], mask)
                for k in metrics_held_in:
                    split_result[f'{HELDIN_OR_OUT_MAP[in_or_out]} {k}'] = metrics_held_in[k]
        result.append({datasplit: split_result})
            
    # Out struct according to https://evalai.readthedocs.io/en/latest/evaluation_scripts.html
    return {"result": result, 'submission_result': result[0]}


class FalconEvaluator:

    def __init__(self, eval_remote=False, split='h1'):
        self.eval_remote = eval_remote
        assert split in ['h1', 'h2', 'm1', 'm2'], "Split must be h1, h2, m1, or m2."
        self.dataset: FalconTask = getattr(FalconTask, split)

    @staticmethod
    def get_eval_handles(is_remote: bool, dataset: FalconTask, phase: str = 'minival'):
        if is_remote: # i.e. definitely docker
            data_dir = os.environ.get("EVAL_DATA_PATH")
        else: # possibly docker or local
            if os.path.exists("data"):
                logger.info("Using local data directory.")
                data_dir = "data"
            else:
                data_dir = os.environ.get("EVAL_DATA_PATH") # a local docker eval
        data_dir = Path(data_dir) / dataset.name
        if phase == 'test': # TODO wire wherever test is actually stored on remote
            eval_dir = data_dir / f"eval"
        else:
            eval_dir = data_dir / "minival"
        return sorted(list(eval_dir.glob("*val*.nwb")))

    def get_eval_files(self, phase: str = 'minival'):
        logger.info("Searching for evaluation data.")
        handles = self.get_eval_handles(self.eval_remote, self.dataset, phase=phase)
        logger.info(f"Found {len(handles)} files.")
        if len(handles) == 0:
            raise FileNotFoundError(f"No files found in {self.dataset.name} for phase {phase}. Note test phase data is only available on EvalAI remote.")
        return handles
    
    def predict_files(self, decoder: BCIDecoder, eval_files: List):
        all_preds = []
        all_targets = []
        all_eval_mask = []

        for datafile in eval_files:
            if not datafile.exists():
                raise FileNotFoundError(f"File {datafile} not found.")
            neural_data, decoding_targets, trial_change, eval_mask = load_nwb(datafile, dataset=self.dataset)
            decoder.reset(dataset=datafile)
            trial_preds = []
            for neural_observations, trial_delta_obs in zip(neural_data, trial_change):
                trial_preds.append(decoder.predict(neural_observations))
                if trial_delta_obs:
                    decoder.on_trial_end()
            all_preds.append(np.stack(trial_preds))
            all_targets.append(decoding_targets)
            all_eval_mask.append(eval_mask)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_eval_mask = np.concatenate(all_eval_mask)
        return all_preds, all_targets, all_eval_mask

    def evaluate_files(self, decoder: BCIDecoder, eval_files: List):
        all_preds, all_targets, all_eval_mask = self.predict_files(decoder, eval_files)
        metrics = self.compute_metrics(all_preds, all_targets, all_eval_mask)
        return metrics

    def evaluate(self, decoder: BCIDecoder, phase: str):
        r"""
            Note: Locally, this can produce metrics, but locally and remotely it should also write a submission file 
            that the actual evaluator on remote uses. The evaluation is done separately on remote.
        """
        assert phase in ['minival', 'test'], "Phase must be minival or test."
        
        np.random.seed(0)
        # ! TODO ideally seed other libraries as well...? Is that our responsibility?

        eval_files = self.get_eval_files(phase=phase)
        metrics = {}

        prediction_path = os.environ.get("PREDICTION_PATH", './local_prediction.pkl')
        if not prediction_path:
            raise ValueError("PREDICTION_PATH not set in remote env which expects it. Cannot forward to separate evaluate runscript.")
        gt_path = os.environ.get("GT_PATH", './local_gt.pkl')
        if not gt_path:
            raise ValueError("GT_PATH not set in remote env which expects it. Cannot forward to separate evaluate runscript.")
            
        if phase == 'test':
            eval_files_held_in = [f for f in eval_files if any(k in f.name for k in HELD_IN_KEYS[self.dataset])]
            eval_files_held_out = [f for f in eval_files if any(k in f.name for k in HELD_OUT_KEYS[self.dataset])]
            assert len(eval_files) == len(eval_files_held_in) + len(eval_files_held_out), "Mismatch in extracted eval files: Eval file state is not consistent with benchmark creation settings."
            all_preds_held_in, all_targets_held_in, all_eval_mask_held_in = self.predict_files(decoder, eval_files_held_in)
            all_preds_held_out, all_targets_held_out, all_eval_mask_held_out = self.predict_files(decoder, eval_files_held_out)

            # Indirect remote setup to satisfy EvalAI interface. Save metrics / comparison to file.
            if USE_PKLS:
                pred_payload = {self.dataset.name: {
                    'held_in_pred': all_preds_held_in,
                    'held_in_eval_mask': all_eval_mask_held_in,
                    'held_out_pred': all_preds_held_out,
                    'held_out_eval_mask': all_eval_mask_held_out,
                    'normalized_latency': 1, # TODO - CW insert timing code
                }}
                truth_payload = {self.dataset.name: {
                    'held_in': all_targets_held_in,
                    'held_out': all_targets_held_out,
                }}
            else:
                metrics_held_in = self.compute_metrics(all_preds_held_in, all_targets_held_in, all_eval_mask_held_in)
                metrics_held_out = self.compute_metrics(all_preds_held_out, all_targets_held_out, all_eval_mask_held_out)
                for k, v in metrics_held_in.items():
                    metrics[f'{HELDIN_OR_OUT_MAP["held_in"]} {k}'] = v
                for k, v in metrics_held_out.items():
                    metrics[f'{HELDIN_OR_OUT_MAP["held_out"]} {k}'] = v            
        else:
            all_preds, all_targets, all_eval_mask = self.predict_files(decoder, eval_files)
            if USE_PKLS:
                pred_payload = {self.dataset.name: {
                    'held_in_pred': all_preds,
                    'held_in_eval_mask': all_eval_mask,
                    'normalized_latency': 1, # TODO - CW insert timing code
                }}
                truth_payload = {self.dataset.name: {
                    'held_in': all_targets,
                }}
            else:
                metrics_minival = self.compute_metrics(all_preds, all_targets, all_eval_mask)
                for k, v in metrics_minival.items():
                    metrics[f'{HELDIN_OR_OUT_MAP["held_in"]} {k}'] = v
        
        if USE_PKLS:
            with open(prediction_path, 'wb') as f:
                pickle.dump(pred_payload, f)
            with open(gt_path, 'wb') as f:
                pickle.dump(truth_payload, f)

            # TODO - this subsequent line of logic needs to be owned by challenge worker - currently in here for Beta testing.
            print(evaluate(
                test_annotation_file=gt_path,
                user_submission_file=prediction_path,
                phase_codename=phase
            ))
        else:
            for k, v in metrics.items():
                logger.info("{}: {}".format(k, v))
        

    @staticmethod
    def compute_metrics_regression(preds, targets, eval_mask):
        targets = targets[eval_mask]
        preds = preds[eval_mask]
        return {
            "R2": r2_score(targets, preds, multioutput='variance_weighted'),
            "R2 Std.": 0, # TODO Clay
        }

    @staticmethod
    def compute_metrics_classification(preds, targets, eval_mask):
        return {
            "WER": 1-(preds == targets)[eval_mask].mean(),
            "WER Std.": 0, # TODO Clay
        }

    def compute_metrics(self, all_preds, all_targets, all_eval_mask=None):
        r"""
            all_preds: array of shape (n_timesteps, k_dim)
            all_targets: array of shape (n_timesteps, k_dim)
            all_eval_mask: array of shape (n_timesteps, k_dim). True if we should evaluate this timestep.
        """
        if self.dataset in [FalconTask.h1, FalconTask.m1, FalconTask.m2]:
            metrics = self.compute_metrics_regression(all_preds, all_targets, all_eval_mask)
        elif self.dataset in [FalconTask.h2]:
            metrics = self.compute_metrics_classification(all_preds, all_targets, all_eval_mask)
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