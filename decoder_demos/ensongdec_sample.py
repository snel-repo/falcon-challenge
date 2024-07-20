r"""
    Sample Ensongdec decoder for the Falcon Challenge.
    
    Oracle commands:
    
    python ensongdec_sample.py --evaluation "local" --model-path '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.26_held_in_calib.nwb_FFNN_20240604_092642.pt' --model_cfg_path '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.26_held_in_calib.nwb_FFNN_20240604_092642_metadata.json' --split 'b1' --phase 'minival' --batch-size 1
    
    python ensongdec_sample.py --evaluation "local" --model-path '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705.pt' --model_cfg_path '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705_metadata.json' --split 'b1' --phase 'test' --batch-size 1
        
"""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from decoder_demos.ensongdec_decoder import EnSongdecDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False, default='./local_data/ndt2_h1_sample.pth'
    )
    parser.add_argument(
        "--model_cfg_path", type=str, required=False, default='falcon/h1/h1',
        help="Name in context-general-bci codebase for config. \
            Currently, directly referencing e.g. a local yaml is not implemented unclear how to get Hydra to find it in search path."
    )
    parser.add_argument(
        "--zscore-path", type=str, required=False, default='./local_data/ndt2_zscore_h1.pt'
    )
    parser.add_argument(
        '--split', type=str, required=False, choices=['h1', 'h2', 'm1', 'm2', 'b1'], default='h1',
    )
    parser.add_argument(
        '--phase', required=False, choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--batch-size', type=int, required=False, default=1 
    )
    parser.add_argument(
        '--force-static-key', required=False, type=str, default='', help="Specify session to enforce session parameters for, i.e. ignore session label from evaluator."
    )
    parser.add_argument(
        '--model-paths', required=False, type=str, nargs='+', default=[] # triggers unified path. WIP.
    )
    args = parser.parse_args()
    
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split
    )

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)
    # max_bins = 50 if task in [FalconTask.m1, FalconTask.m2] else 200 # These are recommended defaults

    print('PRINTING CONGIG: ', config, config.n_channels)
    
    if args.model_paths:
        model_paths = args.model_paths
        print(model_paths)
    else:
        model_paths = args.model_path
        
    decoder = EnSongdecDecoder(
        task_config = config,
        model_ckpt_path = model_paths,
        model_cfg_path = args.model_cfg_path,
        batch_size = 1, 
        force_static_key = args.force_static_key
    )

    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()