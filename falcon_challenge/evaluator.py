import logging
import pickle
import numpy as np

from sklearn.metrics import r2_score

from falcon_challenge.interface import BCIDecoder

logger = logging.getLogger(__name__)

class Evaluator:

    def __init__(self, eval_remote=False, phase='h1_short'):
        self.eval_remote = eval_remote
        self.phase = phase
        self.dataset = phase.split('_')[0]
        self.eval_short = phase.split('_')[1] == 'short'

    def retrieve_gt_for_eval(self):
        if self.eval_remote:
            decoding_gt = pickle.load(f"GT_{self.phase}.pkl") # TODO is this secure? Not sure if this is the right pattern
        else:
            decoding_gt = pickle.load(f"GT_{self.dataset}_minival.pkl")
        return decoding_gt

    def evaluate(self, decoder: BCIDecoder):
        r"""
            prints set of metrics, which we would look at to rank submissions
            # TODO how does eval-ai specifically get metrics beyond what we print?
            # * That should exist in the original NLB logic, or check Habitat further
        """
        decoding_gt = self.retrieve_gt_for_eval()
        all_preds = []
        for trial in decoding_gt:
            decoder.reset()
            trial_preds = []
            for neural_observations in trial:
                trial_preds.append(decoder.predict(neural_observations))
            all_preds.append(np.array(trial_preds))
        metrics = self.compute_metrics(all_preds, decoding_gt)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))
        return metrics

    @staticmethod
    def compute_metrics_regression(all_preds, decoding_gt):
        return {
            "r2": r2_score(decoding_gt, all_preds, multioutput='uniform_average')
        }

    @staticmethod
    def compute_metrics_classification(all_preds, decoding_gt):
        return {
            "accuracy": np.mean(all_preds == decoding_gt)
        }

    def compute_metrics(self, all_preds, decoding_gt):
        r"""
            all_preds: list of arrays of shape (n_timesteps, n_channels)
            decoding_gt: list of arrays of shape (n_timesteps, n_channels)
        """
        if self.dataset in ['h1', 'm1', 'm2']:
            return self.compute_metrics_regression(all_preds, decoding_gt)
        elif self.dataset in ['h2']:
            return self.compute_metrics_classification(all_preds, decoding_gt)
        raise ValueError(f"Unknown dataset {self.dataset}")

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