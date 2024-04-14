r"""
    Sample NDT2 decoder for the Falcon Challenge.

    H1: https://wandb.ai/joelye9/context_general_bci/runs/edf4h5ym
    M1: https://wandb.ai/joelye9/context_general_bci/runs/93snpffp
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
        "--config-stem", type=str, default='falcon/h1/h1',
        help="Name in context-general-bci codebase for config. \
            Currently, directly referencing e.g. a local yaml is not implemented unclear how to get Hydra to find it in search path."
    )
    parser.add_argument(
        "--zscore-path", type=str, default='./local_data/ndt2_zscore_h1.pt'
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1
    )
    args = parser.parse_args()

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split)

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)

    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=args.model_path,
        model_cfg_stem=args.config_stem,
        zscore_path=args.zscore_path,
        dataset_handles=[x.stem for x in evaluator.get_eval_files(phase=args.phase)],
        batch_size=args.batch_size
    )


    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()