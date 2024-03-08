r"""
    Sample ridge regression decoder for the Falcon Challenge.
"""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from ndt2_decoder import NDT2Decoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, default='ndt2_sample.ckpt'
    )
    parser.add_argument(
        "--config-path", type=str, default='ndt2_sample.yaml'
    )
    parser.add_argument(
        '--phase', type=str, required=False, default='h1'
    )
    args = parser.parse_args()

    dataset = args.phase.split('_')[0]
    phase = f'{args.phase}_short' # TODO remove once terms are removed

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        phase=phase)

    task = getattr(FalconTask, dataset)
    config = FalconConfig(
        task=task,
        n_channels=176,
        dataset_handles=evaluator.get_eval_files()
    )

    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=args.model_path,
        model_cfg_path=args.config_path
    )


    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()