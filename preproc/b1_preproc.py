import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np
import pickle as pkl

# Spikeinterface
import spikeinterface.full as si

# Songbirdcore
from songbirdcore.utils import speech_bci_struct as fs
from songbirdcore.utils.params import BirdSpecificParams as BSP

# Ensongdec
import ensongdec.utils.train_utils as tu

from b1_falcon_utils import get_rasters_audio, extract_threshold_crossings
from b1_nwb_utils import convert_to_NWB_b1



def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def process_sessions(sessions_metadata, save_dir, logger):
    
    for day in sessions_metadata:
        logger.info(f"Creating new NWBFile for {day}")
        date = day['date']
        sess = day['sess']
        HELDOUT_DS = day['HELDOUT_DS']
    
        #------- LOCATE SESSION -------#
        sess_par = {'bird': 'z_r12r13_21',
                    'date': date,
                    'ephys_software': 'sglx',
                    'sess': sess,
                    'probe': 'probe_0',
                    'sort': 'ksort3_pt',
                    }
        
        exp_struct = fs.get_experiment_struct(bird=sess_par['bird'], 
                                    date=sess_par['date'], 
                                    ephys_software=sess_par['ephys_software'], 
                                    sess=sess_par['sess'], 
                                    sort=sess_par['sort'])
        
        base_folder = Path('/net2/expData/speech_bci/raw_data/z_r12r13_21') / date / 'sglx' / sess
        spikeglx_folder = base_folder / f'{sess}_imec0'
    
        #------- EXTRACT DATA -------#
        # Audio
        mic_file_name = os.path.join(exp_struct['folders']['derived'], 'wav_mic.npy')
        audio = np.load(mic_file_name)
        
        # Neural traces
        raw_rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
        traces = raw_rec.get_traces()
        
        # Audio dict
        audio_dict_path = os.path.join(exp_struct['folders']['derived'], 'mot_dict_curated.pkl')
        with open(audio_dict_path, 'rb') as f:
            audio_dict = pkl.load(f) 
        
        # Sampling frequencies
        fs_ap = round(audio_dict['s_f_ap_0'])
        fs_audio = round(audio_dict['s_f'])
    
        print('len motif: ', BSP.data['z_r12r13_21']['len_motif'])
        
        # Select time window around the bout
        t_pre = 0.1
        t_post = BSP.data['z_r12r13_21']['len_motif'] + 0.1
        
        # Retrieve start/end samples in the spiking/audio signals
        start_ap = audio_dict['start_sample_ap_0'] - round(fs_ap * t_pre)
        end_ap = audio_dict['start_sample_ap_0'] + round(fs_ap * t_post)
        start_audio = audio_dict['start_sample_nidq'] - round(fs_audio * t_pre)
        end_audio = audio_dict['start_sample_nidq'] + round(fs_audio * t_post)
        
        
        # --------- PROCESS NEURAL --------- #
        
        ra_channels = np.array(range(18, 103))
        lowcut = 300
        highcut = 3000
        th_cross = 3 # Number of standard deviations avobe average voltages
        refractory_th = 1
        
        # Build array of spiking events [trials x m_clusters x n_timestamps] for the clusters of interes, for each period specified by the start_samp_list
        spike_matrix = [extract_threshold_crossings(traces, ra_channels, fs_ap,
                                                       start_ap[i], end_ap[i],
                                                       th_cross, refractory_th=refractory_th,
                                                       filt=True, lowcut=lowcut, highcut=highcut) for i in range(len(start_ap))]
        spike_matrix = np.stack(spike_matrix, axis=-1).transpose(2,0,1)
        
        # --------- PROCESS AUDIO --------- #
        
        # Audio array
        audio_motifs = np.array(get_rasters_audio(audio, start_audio, end_audio))
        audio_motifs = np.array(tu.preprocess_audio(audio_motifs, fs_audio))
    
    
        # --------- TO NWB --------- #
        # Array of synthetic spike-times for each trial
        spike_matrix = spike_matrix.transpose(0,2,1) # Trials, T x Ch
        spike_times = np.array([np.arange(start_time, start_time + trial.shape[0]) * 1/fs_ap for start_time, trial in zip(np.cumsum([0] + [t.shape[0] for t in spike_matrix[:-1]]), spike_matrix)])
        # Array of synthetic audio samples for each trial
        audio_times = np.array([np.arange(start_time, start_time + trial.shape[0]) * 1/fs_audio for start_time, trial in zip(np.cumsum([0] + [t.shape[0] for t in audio_motifs]), audio_motifs)])
        
        print(spike_matrix.shape, spike_times.shape, audio_times.shape)
        
        EXP_DATE = sess_par['date'].replace('-', '.')
        bird = sess_par['bird']
        
        n_trials = len(audio_motifs)
        EVAL_RATIO = 0.4
        n_eval_trials = int(n_trials * EVAL_RATIO)
        n_fewshot_trials = 3 # Few-shot to recalibrate in held-out sessions
        n_smoketest_trials = 2 # Few trials in held-in sessions to test functionality
        
        if not HELDOUT_DS:
            logger.info("Creating smoketest data")
            convert_to_NWB_b1(
                bird,
                EXP_DATE,
                spike_matrix[:n_smoketest_trials],
                spike_times[:n_smoketest_trials],
                fs_ap,
                audio_motifs[:n_smoketest_trials],
                audio_times[:n_smoketest_trials],
                fs_audio,
                split_label='held_in_minival',
                save_dir = save_dir     
            )
            logger.info("Creating held-in calibration dataset")
            nwbfile = convert_to_NWB_b1(
                bird,
                EXP_DATE,
                spike_matrix[:-n_eval_trials],
                spike_times[:-n_eval_trials],
                fs_ap,
                audio_motifs[:-n_eval_trials],
                audio_times[:-n_eval_trials],
                fs_audio,
                split_label='held_in_calib',
                save_dir = save_dir        
            )
            logger.info("Creating eval split")
            convert_to_NWB_b1(
                bird,
                EXP_DATE,
                spike_matrix[-n_eval_trials:],
                spike_times[-n_eval_trials:],
                fs_ap,
                audio_motifs[-n_eval_trials:],
                audio_times[-n_eval_trials:],
                fs_audio,
                split_label='held_in_eval',
                save_dir = save_dir     
            )
        else: 
            logger.info("Creating few-shot calibration split")
            convert_to_NWB_b1(
                bird,
                EXP_DATE,
                spike_matrix[:n_fewshot_trials],
                spike_times[:n_fewshot_trials],
                fs_ap,
                audio_motifs[:n_fewshot_trials],
                audio_times[:n_fewshot_trials],
                fs_audio,
                split_label='held_out_calib',
                save_dir = save_dir     
            )
            logger.info("Creating in-day oracle split")
            convert_to_NWB_b1(
                bird,
                EXP_DATE,
                spike_matrix[:-n_eval_trials],
                spike_times[:-n_eval_trials],
                fs_ap,
                audio_motifs[:-n_eval_trials],
                audio_times[:-n_eval_trials],
                fs_audio,
                split_label='held_out_oracle',
                save_dir = save_dir     
            )
            logger.info("Creating evaluation split")
            convert_to_NWB_b1(
                bird,
                EXP_DATE,
                spike_matrix[-n_eval_trials:],
                spike_times[-n_eval_trials:],
                fs_ap,
                audio_motifs[-n_eval_trials:],
                audio_times[-n_eval_trials:],
                fs_audio,
                split_label='held_out_eval',
                save_dir = save_dir     
            )

def main():
    parser = argparse.ArgumentParser(description="Process data and save NWB files.")
    parser.add_argument("--save_dir", default = os.getcwd(), help="Directory where the NWB files will be saved. Default is the current working directory.")
    args = parser.parse_args()
    
    logger = setup_logger()
    sessions_metadata = [
        {'date': '2021-06-26', 'sess': '1056_g0', 'HELDOUT_DS': False},
        {'date': '2021-06-27', 'sess': '0727_g0', 'HELDOUT_DS': False},
        {'date': '2021-06-28', 'sess': '1006_g0', 'HELDOUT_DS': False},
        {'date': '2021-06-30', 'sess': '1842_g0', 'HELDOUT_DS': True},
        {'date': '2021-07-01', 'sess': '0724_g0', 'HELDOUT_DS': True},
        {'date': '2021-07-05', 'sess': '1209_g0', 'HELDOUT_DS': True}
    ]
    process_sessions(sessions_metadata, args.save_dir, logger)

if __name__ == '__main__':
    main()