import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

from falcon_challenge.config import FalconTask
from falcon_challenge.interface import BCIDecoder
from falcon_challenge.dataloaders import load_nwb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FalconEvaluator:

    def __init__(self, eval_remote=False, phase='h1_short'):
        self.eval_remote = eval_remote
        self.phase = phase
        self.dataset: FalconTask = getattr(FalconTask, phase.split('_')[0])
        self.eval_term = phase.split('_')[1]

    def get_eval_files(self):
        if self.eval_remote:
            eval_dir = f"data/{self.dataset.name}/test_{self.eval_term}" # TODO is this secure? Not sure if this is the right pattern
            suffix = "*eval.nwb"
        else:
            logger.info(f"Local evaluation, running minival.")
            eval_dir = f"data/{self.dataset.name}/minival/"
            suffix = "*minival.nwb"
        return sorted(list(Path(eval_dir).glob(suffix)))

    def evaluate(self, decoder: BCIDecoder):
        r"""
            prints set of metrics, which we would look at to rank submissions
            # TODO how does eval-ai specifically get metrics beyond what we print?
            # * That should exist in the original NLB logic, or check Habitat further
        """
        np.random.seed(0)
        # ! TODO ideally seed other libraries as well...? Is that our responsibility?

        eval_files = self.get_eval_files()
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
        metrics = self.compute_metrics(all_preds, all_targets, all_eval_mask)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))
        return metrics

    @staticmethod
    def compute_metrics_regression(preds, targets, eval_mask):
        targets = targets[eval_mask]
        preds = preds[eval_mask]
        return {
            "r2": r2_score(targets, preds, multioutput='uniform_average')
        }

    @staticmethod
    def compute_metrics_classification(preds, targets, eval_mask):
        return {
            "accuracy": np.mean(preds == targets)
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