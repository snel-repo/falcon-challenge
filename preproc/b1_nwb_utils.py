import logging

from pynwb import NWBFile
from pynwb.core import DynamicTable
from pynwb.file import Subject
from pynwb import TimeSeries
from pynwb.file import Subject
from pynwb.misc import Units
from pynwb.behavior import Position
from pynwb.behavior import BehavioralTimeSeries
from pynwb import ProcessingModule

from pynwb import NWBHDF5IO

import numpy as np
import os
from os import path
from datetime import datetime
from dateutil.tz import tzlocal, gettz

from scipy import signal

# Create a logger object
logger = logging.getLogger(__name__)


def falcon_spectrogram(x, fs, n_window=512, step_ms=1, f_min=250, f_max=8000, cut_off=0.0001):
    """
    Computes a spectrogram from an audio waveform according for the FALCON benchmark.

    Args:
        x (np.ndarray): Input audio signal.
        fs (int): Sampling frequency of the audio signal.
        n_window (int, optional): Number of samples per window segment. Defaults to 512.
        step_ms (int, optional): Step size in milliseconds between successive windows. Defaults to 1.
        f_min (int, optional): Minimum frequency to be included in the output. Defaults to 250 Hz.
        f_max (int, optional): Maximum frequency to be included in the output. Defaults to 8000 Hz.
        cut_off (float, optional): Threshold relative to the maximum of the spectrum, below which values are set to 1. Defaults to 0.0001.

    Returns:
        tuple:
            - np.ndarray: Array of time bins for the spectrogram.
            - np.ndarray: Array of frequencies between `f_min` and `f_max`.
            - np.ndarray: Spectrogram of the frequencies between `f_min` and `f_max`.
    """
    # Overlap: Window size minus smples in a millisec
    msec_samples = int(fs * 0.001)
    n_overlap = n_window - msec_samples * step_ms
    sigma = 1 / 200. * fs

    # Spectrogram computation
    f, t, Sxx = signal.spectrogram(x, fs,
                                   nperseg=n_window,
                                   noverlap=n_overlap,
                                   window=signal.windows.gaussian(n_window, sigma),
                                   scaling='spectrum')

    if cut_off > 0:
        Sxx[Sxx < np.max((Sxx) * cut_off)] = 1
    Sxx[f<f_min, :] = 1

    return t, f[(f>f_min) & (f<f_max)], Sxx[(f>f_min) & (f<f_max)]
    

def convert_datestr_to_datetime(collect_date):
    date_time = datetime.strptime(collect_date, "%Y.%m.%d").replace(
        tzinfo=gettz("America/Los_Angeles")
    )
    return date_time
    

def convert_to_NWB(
    bird,
    date,
    spike_matrix,
    spike_times,
    fs_neural,
    vocal_epochs,
    audio_times,
    fs_audio,
    split_label = '',
    save_dir = ''
): 

    file_id = f'{bird}_{date}_{split_label}'

    # Check if the lengths are equal
    print(len(spike_matrix), len(vocal_epochs))
    if len(spike_matrix) != len(vocal_epochs):
        raise ValueError("The lengths (num trials) of spike_matrix and audio_motifs are not equal.")

    # -------- NWB Metadata -------- #
    
    logger.info("Creating new NWBFile")
    nwbfile = NWBFile(
        session_description = "Freely-awake-singing Zebra finch with a Neuropixels probe implanted in RA.",
        identifier = file_id,
        session_start_time = convert_datestr_to_datetime(date),
        experiment_description = "Synchronous RA-neural and song data of a Zebra finch implanted with a Neuropixels probe during awake-singing.",
        file_create_date = datetime.now(tzlocal()),
        lab = "TNEL",
        institution = "UC San Diego",
        experimenter = "Dr. Pablo Tostado-Marcos and Dr. Ezequiel Arneodo",
    )

    subject = Subject(subject_id=f'Finch_{bird}_{split_label}', species='Zebra finch', sex='M', age='P90D/')
    nwbfile.subject = subject
    
    # -------- TRIAL INFO -------- #
    logger.info("Adding trial info")
    nwbfile.add_trial_column(name="spectrogram_times", description="Times associated with the spectrogram data values.")
    nwbfile.add_trial_column(name="spectrogram_frequencies", description="Frequency bands associated with the spectrogram data values.")
    nwbfile.add_trial_column(name="spectrogram_values", description="Spectrogram data values.")
    nwbfile.add_trial_column(name="spectrogram_eval_mask", description="Mask of timesteps and frequency bands considered during FALCON evaluation.")
    
    n_trials = len(vocal_epochs)
    
    for ix in range(n_trials): 
        # Compute spectrogram
        spec_t, spec_f, sxx = falcon_spectrogram(vocal_epochs[ix], fs_audio)
        sxx_eval_mask = np.tile([(spec_t >= 0.1) & (spec_t < 0.8)], (len(spec_f), 1)) # Eval mask [frequencies x timesteps]
        nwbfile.add_trial(
            start_time = spike_times[ix][0],
            stop_time = spike_times[ix][-1],
            spectrogram_times = spec_t,
            spectrogram_frequencies = spec_f,
            spectrogram_values = sxx,
            spectrogram_eval_mask = sxx_eval_mask
        )

    # -------- TIMESERIES -------- #
    # Concatenate trials (neural & audio)
    stacked_spikes = np.vstack(spike_matrix)
    t_spikes_continuous = np.concatenate(spike_times)
    stacked_vocal_epochs = np.concatenate(vocal_epochs)
    t_audio_continuous = np.concatenate(audio_times)
  
    # Tx Timeseries
    tx_timeseries = TimeSeries(
        name = "tx",
        description = f'Threshold crossings (Tx) extracted at original neural sampling rate - {fs_neural} Hz.',
        timestamps = t_spikes_continuous,
        data = stacked_spikes,
        unit = "int",
        )
    # Add the sampling rate as a custom attribute
    tx_timeseries.fields['sampling_rate'] = float(fs_neural)
    # Add the TimeSeries to the NWB file
    nwbfile.add_acquisition(tx_timeseries)
    
    # Amplitude Waveform Timeseries
    vocalizations = TimeSeries(
        name = "vocalizations",
        description = f'Amplitude waveform of vocal epochs time-aligned to Tx at original recorded audio sampling rate - {fs_audio} Hz.',
        timestamps = t_audio_continuous,
        data = stacked_vocal_epochs,
        unit = "int",
        )
    # Add the sampling rate as a custom attribute
    vocalizations.sampling_rate = float(fs_audio)
    # Add the TimeSeries to the NWB file
    nwbfile.add_acquisition(vocalizations)

    # -------- EVAL MASK -------- #
    # We will evaluate the reconstructions only on the duration of the motif, i.e. from 100ms to 800ms
    single_trial_audio_eval_mask = (audio_times[0] >= 0.1) & (audio_times[0] < 0.8)
    audio_eval_mask = np.tile(single_trial_audio_eval_mask, (audio_times.shape[0], 1))
    eval_mask_audio_continuous = np.concatenate(audio_eval_mask)
    
    # Spikes EvalMask Timeseries
    nwbfile.add_acquisition(
        TimeSeries(
            name = "eval_mask_audio",
            description = f"Mask of audio timestamps considered during FALCON evaluation.",
            timestamps = t_audio_continuous,
            data = eval_mask_audio_continuous,
            unit = "bool",
        )
    )

    # -------- SAVE FILE -------- #    
    nwb_path = save_dir + '/nwb_files'
    print(nwb_path)
    if not path.exists(nwb_path):
        os.makedirs(nwb_path, mode=0o755)
    save_fname = path.join(nwb_path, file_id + ".nwb")
    print(f"Saving NWB file to {save_fname}")
    
    with NWBHDF5IO(save_fname, "w") as io:
        io.write(nwbfile)

    return nwbfile
    

def load_nwb(nwb_filepath):

    with NWBHDF5IO(nwb_filepath, "r") as io:
        nwbfile = io.read()
    
        neural_array = np.array(nwbfile.get_acquisition('tx').data)
        spike_times = np.array(nwbfile.get_acquisition('tx').timestamps)
        
        audio_motifs = np.array(nwbfile.get_acquisition('vocalizations').data)
        audio_times = np.array(nwbfile.get_acquisition('vocalizations').timestamps)
        
        # Trial info
        trial_info = (
                    nwbfile.trials.to_dataframe()
                    .reset_index()
        )

    # Compute trialized neural array
    n_trials = len(trial_info)
    n_channels = neural_array.shape[-1]
    trial_length = round(trial_info['stop_time'][0]-trial_info['start_time'][0], 1) # Round to 1st decimal point
    print('Trial length: ', trial_length, ' seconds')
    
    neural_samples_per_trial = neural_array.shape[0] // n_trials # Number of samples per trial
    neural_array = neural_array.reshape(n_trials, neural_samples_per_trial, n_channels) # Trialized spike_matrix
    neural_array = neural_array.transpose(0,2,1) # Transpose to [Trials x Channels x Timestamps]
    
    fs_neural = neural_samples_per_trial/trial_length
    
    # Compute trialized audio array
    n_trials = len(trial_info)
    audio_samples_per_trial = audio_motifs.shape[0] // n_trials # Number of samples per trial
    audio_motifs = audio_motifs.reshape(n_trials, audio_samples_per_trial) # Trialized spike_matrix
    
    fs_audio = audio_samples_per_trial/trial_length

    return trial_info, neural_array, fs_neural, audio_motifs, fs_audio