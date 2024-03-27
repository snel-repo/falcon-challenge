#%% 

from pynwb import NWBFile
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd 
import scipy.signal as signal

from scipy.io import loadmat
import h5py, glob, logging, sys, os
from datetime import datetime
from dateutil.tz import tzlocal, gettz
import matplotlib.pyplot as plt
from os import path

from pynwb import TimeSeries
from pynwb.file import Subject
from pynwb.misc import Units
from pynwb.behavior import Position
from pynwb.behavior import BehavioralTimeSeries
from pynwb import ProcessingModule

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

#%% 
base_dir = "/snel/share/share/data/bg2/t5_handwriting/CORP_data/stability_benchmark"
SAVE_PATH = os.path.join(base_dir, 'nwb')
EXT = 'online_recalibration_data'
EXP_DATE = '2023.04.17'

if len(sys.argv) > 2:
    EXP_DATE = sys.argv[1]
    EXT = sys.argv[2]
 
train_dates = [
            '2022.09.29', '2022.11.01', '2022.10.06', '2022.11.03', 
            '2022.10.18', '2022.12.08', '2022.10.25', '2022.12.15', 
            '2022.10.27', '2023.02.28', '2022.05.18', '2022.06.03', 
            '2022.06.15', '2022.05.23', '2022.06.06', '2022.06.22', 
            '2022.05.25', '2022.06.08', '2022.09.01', '2022.06.01', 
            '2022.06.13'
        ]
test_dates = ['2023.04.17', '2023.05.31', '2023.06.28', '2023.08.16', '2023.10.09']

if EXP_DATE in train_dates: 
    IS_TEST_DS = False
if EXP_DATE in test_dates: 
    IS_TEST_DS = True

mat_file = os.path.join(base_dir, EXT, f't5.{EXP_DATE}.mat')

def convert_datestr_to_datetime(collect_date):
    date_time = datetime.strptime(collect_date, "%Y.%m.%d").replace(
        tzinfo=gettz("America/Chicago")
    )

    return date_time

# %%

f = loadmat(mat_file)

# %%
spikes = f['tx_feats'][0] # list of trials, t x ch 
cues = f['sentences'][0] # list of trials
blocks = np.array([x[0][0] for x in f['blocks'][0]]) # n_trials

#%% 
times = []
start_time = 0
trial_offset = 10 / 0.02
for trial in spikes: 
    tr_time = np.arange(start_time, start_time+trial.shape[0]) * 0.02
    times.append(tr_time)
    start_time += trial.shape[0] + trial_offset

# %%

def convert_to_NWB(
    date,
    spikes,
    cues,
    blocks,
    trial_times,
    split_label=''
): 

    file_id = f'T5_{date}_{split_label}'
    logger.info("Creating new NWBFile")
    nwbfile = NWBFile(
        session_description="human BCI participant performing attempted handwriting task",
        identifier=file_id,
        session_start_time=convert_datestr_to_datetime(date),
        experiment_description="handwriting with prompted sentences",
        file_create_date=datetime.now(tzlocal()),
        lab="NPTL",
        institution="Stanford University",
        experimenter="Chaofei Fan",
    )

    subject = Subject(subject_id=f'T5_{split_label}', species='Homo sapiens', sex='M', age='P69Y')
    nwbfile.subject = subject

    # === NWBFile Step: add trial info
    logger.info("Adding trial info")
    # add trial information
    nwbfile.add_trial_column(name="cue", description="sentence participant was prompted to write")
    nwbfile.add_trial_column(name="block_num", description="experimental block trial belonged to")

    n_trials = len(blocks)
    
    for ix in range(n_trials): 
        t = trial_times[ix]
        nwbfile.add_trial(
            start_time=t[0],
            stop_time=t[-1],
            cue=cues[ix][0],
            block_num=blocks[ix]
        )

    t_cont = np.concatenate(trial_times)
    stacked_spikes = np.vstack(spikes)

    nwbfile.add_acquisition(
        TimeSeries(
            name="binned_spikes",
            description="Threshold crossings aggregated in 20ms bins",
            timestamps=t_cont,
            data=stacked_spikes,
            unit="int",
        )
    )

    eval_mask = np.full(t_cont.size, True)

    nwbfile.add_acquisition(
        TimeSeries(
            name="eval_mask",
            description="Timesteps to KEEP covariates (for training, eval).",
            timestamps=t_cont,
            data=eval_mask,
            unit="bool",
        )
    )

    nwb_path = path.join(SAVE_PATH, split_label)
    if not path.exists(nwb_path):
        os.makedirs(nwb_path, mode=0o755)
    save_fname = path.join(nwb_path, file_id + ".nwb")
    logger.info(f"Saving NWB file to {save_fname}")
    # write processed file
    with NWBHDF5IO(save_fname, "w") as io:
        io.write(nwbfile)


#%% 

n_trials = len(blocks)
EVAL_RATIO = 0.4
n_eval_trials = int(n_trials * EVAL_RATIO)
n_fewshot_trials = 3
n_smoketest_trials = 2

if IS_TEST_DS:
    logger.info("Creating few-shot calibration split")
    convert_to_NWB(
        EXP_DATE,
        spikes[:n_fewshot_trials],
        cues[:n_fewshot_trials],
        blocks[:n_fewshot_trials],
        times[:n_fewshot_trials],
        split_label='held_out_calib'
    )
    logger.info("Creating evaluation split")
    convert_to_NWB(
        EXP_DATE,
        spikes[-n_eval_trials:],
        cues[-n_eval_trials:],
        blocks[-n_eval_trials:],
        times[-n_eval_trials:],
        split_label='held_out_eval'
    )
    logger.info("Creating in-day oracle split")
    convert_to_NWB(
        EXP_DATE,
        spikes[:-n_eval_trials],
        cues[:-n_eval_trials],
        blocks[:-n_eval_trials],
        times[:-n_eval_trials],
        split_label='held_out_oracle'
    )
else: 
    logger.info("Creating held-in calibration dataset")
    convert_to_NWB(
        EXP_DATE,
        spikes[:-n_eval_trials],
        cues[:-n_eval_trials],
        blocks[:-n_eval_trials],
        times[:-n_eval_trials],
        split_label='held_in_calib'
    )
    logger.info("Creating eval split")
    convert_to_NWB(
        EXP_DATE,
        spikes[-n_eval_trials:],
        cues[-n_eval_trials:],
        blocks[-n_eval_trials:],
        times[-n_eval_trials:],
        split_label='held_in_eval'
    )
    logger.info("Creating smoketest data")
    convert_to_NWB(
        EXP_DATE,
        spikes[:n_smoketest_trials],
        cues[:n_smoketest_trials],
        blocks[:n_smoketest_trials],
        times[:n_smoketest_trials],
        split_label='held_in_minival'
    )