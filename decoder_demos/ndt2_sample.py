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
        "--model-path", type=str, default='./local_data/ndt2_h1_sample.pth'
    )
    parser.add_argument(
        "--config-stem", type=str, default='falcon/h1/h1_nopool_cross',
        help="Name in context-general-bci codebase for config. \
            Currently, directly referencing e.g. a local yaml is not implemented unclear how to get Hydra to find it in search path."
    )
    parser.add_argument(
        "--zscore-path", type=str, default='./local_data/ndt2_zscore_h1.pt'
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
        dataset_handles=[x.stem for x in evaluator.get_eval_files()]
    )

    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=args.model_path,
        model_cfg_stem=args.config_stem,
        zscore_path=args.zscore_path
    )


    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()