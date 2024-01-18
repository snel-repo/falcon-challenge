import logging
import pickle
import numpy as np

from bci_stability_challenge.interface import BCIDecoder


class Evaluator:

    def __init__(self, eval_remote=False):
        self.eval_remote = eval_remote

    def evaluate(self, decoder: BCIDecoder):
        r"""
            prints set of metrics, which we would look at to rank submissions
            # TODO how does eval-ai specifically get metrics beyond what we print?
            # * That should exist in the original NLB logic, or check Habitat further
        """
        logger = logging.getLogger(__name__)
        if self.eval_remote:
            decoding_gt = pickle.load("decoding_gt.pkl") # TODO is this secure? Not sure if this is the right pattern
        else:
            decoding_gt = pickle.load("decoding_gt_sanity.pkl")
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

    def compute_metrics(self, all_preds, decoding_gt):
        r"""
            all_preds: list of arrays of shape (n_timesteps, n_channels)
            decoding_gt: list of arrays of shape (n_timesteps, n_channels)
        """
        from sklearn.metrics import r2_score
        return {
            "mse": np.mean((all_preds - decoding_gt) ** 2),
            "corr": np.mean(np.corrcoef(all_preds, decoding_gt)),
            "r2": np.mean(r2_score(all_preds, decoding_gt)),
        }
