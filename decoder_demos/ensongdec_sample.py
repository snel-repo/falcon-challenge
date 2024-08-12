r"""
    Sample Ensongdec decoder for the Falcon Challenge.
    
    - Oracle commands -

    MULTI-MODEL:

        python ensongdec_sample.py --evaluation "local" --model-paths '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.26_held_in_calib.nwb_FFNN_20240604_092642.pt' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705.pt' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.28_held_in_calib.nwb_FFNN_20240604_093216.pt' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.30_held_out_oracle.nwb_FFNN_20240604_093453.pt' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.07.01_held_out_oracle.nwb_FFNN_20240604_095827.pt' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.07.05_held_out_oracle.nwb_FFNN_20240604_095944.pt' --model-cfg-paths '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.26_held_in_calib.nwb_FFNN_20240604_092642_metadata.json' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705_metadata.json' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.28_held_in_calib.nwb_FFNN_20240604_093216_metadata.json' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.30_held_out_oracle.nwb_FFNN_20240604_093453_metadata.json' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.07.01_held_out_oracle.nwb_FFNN_20240604_095827_metadata.json' '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.07.05_held_out_oracle.nwb_FFNN_20240604_095944_metadata.json' --split 'b1' --phase 'test' --batch-size 1
    

    SINGLE MODEL:
    
    python ensongdec_sample.py --evaluation "local" --model-paths '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705.pt' --model-cfg-paths '/home/jovyan/pablo_tostado/bird_song/enSongDec/ensongdec/models_checkpoints/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705_metadata.json' --split 'b1' --phase 'test' --batch-size 1
        
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
        "--model-paths", type=str, required=True, nargs='+', default='falcon/model.pt', help="One or more model paths."
    )
    parser.add_argument(
        "--model-cfg-paths", type=str, required=True, nargs='+', default='falcon/model_metadata.json',
        help="One or more model configuration paths."
    )
    parser.add_argument(
        '--split', type=str, required=False, choices=['h1', 'h2', 'm1', 'm2', 'b1'], default='b1',
    )
    parser.add_argument(
        '--phase', required=False, choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--batch-size', type=int, required=False, default=1 
    )
    args = parser.parse_args()
    
    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split
    )

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)
    
    model_paths = args.model_paths
    config_paths = args.model_cfg_paths

    dataset_handles=[x.stem for x in evaluator.get_eval_files(phase=args.phase)]
        
    decoder = EnSongdecDecoder(
        task_config = config,
        model_ckpt_paths = model_paths,
        model_cfg_paths = config_paths,
        dataset_handles = dataset_handles,
        batch_size = 1
    )

    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()