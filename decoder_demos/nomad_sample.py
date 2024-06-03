import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from nomad_decoder import NoMAD_Decoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--docker', choices=['true', 'false'], default='false'
    )
    
    args = parser.parse_args()

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
    )

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)

    if args.docker == "false":
        submission_dict = '/snel/home/bkarpo2/bin/falcon-challenge/nomad_baseline/submissions.yaml'
    else: 
        submission_dict = 'submissions.yaml'

    decoder = NoMAD_Decoder(
        task_config=config,
        submission_dict=submission_dict
    )

    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()