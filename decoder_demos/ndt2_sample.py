r"""
    Sample NDT2 decoder for the Falcon Challenge.
    
    Oracle commands:
    python decoder_demos/ndt2_sample.py --evaluation local --model-path '' --config-stem falcon/m1/m1_oracle_chop --zscore-path ./local_data/ndt2_zscore_m1.pt --split m1 --phase test --batch-size 1 --model-paths local_data/m1_single_oracle_ndt2_2mz1bysq.ckpt  local_data/m1_single_oracle_ndt2_awe4ln1c.ckpt  local_data/m1_single_oracle_ndt2_e980ervy.ckpt local_data/m1_single_oracle_ndt2_976acfc7.ckpt  local_data/m1_single_oracle_ndt2_dh2xwzi0.ckpt  local_data/m1_single_oracle_ndt2_hpuopdhc.ckpt  local_data/m1_single_oracle_ndt2_u8rt3ciq.ckpt
    
    python decoder_demos/ndt2_sample.py --evaluation local --model-path '' --config-stem falcon/m1/m1_rnn_oracle_chop --zscore-path ./local_data/ndt2_zscore_m1.pt --split m1 --phase test --batch-size 1 --model-paths local_data/m1_single_oracle_rnn_45neu57a.ckpt  local_data/m1_single_oracle_rnn_9ea31gav.ckpt  local_data/m1_single_oracle_rnn_gw1er3az.ckpt  local_data/m1_single_oracle_rnn_y0s9f8nh.ckpt local_data/m1_single_oracle_rnn_73gm6s7v.ckpt  local_data/m1_single_oracle_rnn_aj76s8lj.ckpt  local_data/m1_single_oracle_rnn_s3880opv.ckpt
    
    python decoder_demos/ndt2_sample.py --evaluation local --model-path '' --config-stem falcon/m2/m2_oracle_chop --zscore-path ./local_data/ndt2_zscore_m2.pt --split m2 --phase test --batch-size 1 --model-paths local_data/m2_single_oracle_ndt2_1yzaju1y.ckpt local_data/m2_single_oracle_ndt2_ngmpyjoa.ckpt  local_data/m2_single_oracle_ndt2_qd071akm.ckpt  local_data/m2_single_oracle_ndt2_w0xmp3zi.ckpt local_data/m2_single_oracle_ndt2_e90qnj0g.ckpt  local_data/m2_single_oracle_ndt2_lf5dop9h.ckpt  local_data/m2_single_oracle_ndt2_noc2gxfx.ckpt  local_data/m2_single_oracle_ndt2_rwr90vau.ckpt
    
    python decoder_demos/ndt2_sample.py --evaluation local --model-path '' --config-stem falcon/m2/m2_oracle_chop --zscore-path ./local_data/ndt2_zscore_m2.pt --split m2 --phase test --batch-size 1 --model-paths local_data/m2_single_oracle_rnn_04yxz3ls.ckpt  local_data/m2_single_oracle_rnn_8en5p25l.ckpt  local_data/m2_single_oracle_rnn_k0e7g14s.ckpt  local_data/m2_single_oracle_rnn_q3kii5uw.ckpt local_data/m2_single_oracle_rnn_0ojd346m.ckpt  local_data/m2_single_oracle_rnn_e1xp8g1i.ckpt  local_data/m2_single_oracle_rnn_m3ezdb44.ckpt  local_data/m2_single_oracle_rnn_rzlc1bj9.ckpt
    
    python decoder_demos/ndt2_sample.py --evaluation local --model-path '' --config-stem falcon/h1/h1_oracle_chop --zscore-path ./local_data/ndt2_zscore_h1.pt --split h1 --phase test --batch-size 1 --model-paths local_data/h1_single_oracle_ndt2_3jfsv0i8.ckpt  local_data/h1_single_oracle_ndt2_afztyy6r.ckpt  local_data/h1_single_oracle_ndt2_icdat9oj.ckpt  local_data/h1_single_oracle_ndt2_q8w8q5ml.ckpt  local_data/h1_single_oracle_ndt2_wo627blr.ckpt local_data/h1_single_oracle_ndt2_3k0hutli.ckpt  local_data/h1_single_oracle_ndt2_c6s9kim8.ckpt  local_data/h1_single_oracle_ndt2_nhkipouk.ckpt  local_data/h1_single_oracle_ndt2_sg4xrrrx.ckpt  local_data/h1_single_oracle_ndt2_wvdfscwx.ckpt local_data/h1_single_oracle_ndt2_672fjw7h.ckpt  local_data/h1_single_oracle_ndt2_hrggoznz.ckpt  local_data/h1_single_oracle_ndt2_p3k1ulxa.ckpt 
    
    python decoder_demos/ndt2_sample.py --evaluation local --model-path '' --config-stem falcon/h1/h1_oracle_chop --zscore-path ./local_data/ndt2_zscore_h1.pt --split h1 --phase test --batch-size 1 --model-paths local_data/h1_single_oracle_rnn_39csa51t.ckpt  local_data/h1_single_oracle_rnn_7r3tsg42.ckpt  local_data/h1_single_oracle_rnn_ikg7qjfw.ckpt  local_data/h1_single_oracle_rnn_undb9ibb.ckpt  local_data/h1_single_oracle_rnn_z09y8rwu.ckpt local_data/h1_single_oracle_rnn_4o0rwcdk.ckpt  local_data/h1_single_oracle_rnn_8zh5jglv.ckpt  local_data/h1_single_oracle_rnn_l4mh9x4l.ckpt  local_data/h1_single_oracle_rnn_v1qq1woe.ckpt local_data/h1_single_oracle_rnn_5rdrk3db.ckpt  local_data/h1_single_oracle_rnn_dk37awax.ckpt  local_data/h1_single_oracle_rnn_n44ylznx.ckpt  local_data/h1_single_oracle_rnn_y1n2qdsi.ckpt
    
    ZS commands:
    python decoder_demos/ndt2_sample.py --evaluation local --config-stem falcon/m1/m1_rnn_oracle_chop --zscore-path ./local_data/ndt2_zscore_m1.pt --split m1 --phase test --batch-size 1 --model-path local_data/m1_single_oracle_rnn_s3880opv.ckpt --force-static-key 20120926
    
    python decoder_demos/ndt2_sample.py --evaluation local --config-stem falcon/m2/m2_rnn_oracle_chop --zscore-path ./local_data/ndt2_zscore_m2.pt --split m2 --phase test --batch-size 1 --model-path local_data/m2_single_oracle_rnn_m3ezdb44.ckpt --force-static-key 2020-10-28
    
    python decoder_demos/ndt2_sample.py --evaluation local --config-stem falcon/h1/h1_rnn_oracle_chop --zscore-path ./local_data/ndt2_zscore_h1.pt --split h1 --phase test --batch-size 1 --model-path local_data/h1_single_oracle_rnn_y1n2qdsi.ckpt --force-static-key S5
    
    
"""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from context_general_bci.falcon_decoder import NDT2Decoder

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
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2', 'b1'], default='h1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1
    )
    parser.add_argument(
        '--force-static-key', type=str, default='', help="Specify session to enforce session parameters for, i.e. ignore session label from evaluator."
    )
    parser.add_argument(
        '--model-paths', type=str, nargs='+', default=[] # triggers unified path. WIP.
    )
    
    args = parser.parse_args()


    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
        # verbose=True
        # continual=args.continual
    )

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)
    max_bins = 50 if task in [FalconTask.m1, FalconTask.m2] else 200 # These are recommended defaults
    
    if args.model_paths:
        model_paths = args.model_paths
        print(model_paths)
    else:
        model_paths = args.model_path
    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=model_paths,
        model_cfg_stem=args.config_stem,
        zscore_path=args.zscore_path,
        max_bins=max_bins,
        dataset_handles=[x.stem for x in evaluator.get_eval_files(phase=args.phase)],
        batch_size=args.batch_size,
        force_static_key=args.force_static_key
    )


    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()