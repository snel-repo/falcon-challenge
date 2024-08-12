r"""
    Load an EnSongdec model.
"""

import os
import json
from pathlib import Path
import numpy as np
from typing import List, Union, Optional
from scipy import signal

import torch

# Falcon Challenge
from falcon_challenge.interface import BCIDecoder
from falcon_challenge.config import FalconConfig
from preproc.b1_nwb_utils import load_nwb_b1

# Ensongdec Model: https://github.com/pabloslash/EnSongdec
from ensongdec.utils.evaluation_utils import load_model
import ensongdec.utils.train_utils as tu
from ensongdec.utils import encodec_utils as eu
from ensongdec.src.models.FFNNmodel import ffnn_predict


class EnSongdecDecoder(BCIDecoder):
    r"""
        Load an EnSongdec decoder for the FALCON challenge.
    """
    def __init__(self, 
                 task_config: FalconConfig, 
                 model_ckpt_paths: List[str],
                 model_cfg_paths: List[str],
                 dataset_handles: List[str],
                 batch_size: int=1
                ):
        r"""
            Loading EnSongdec requires both weights and model config. Weight loading through a checkpoint is standard.
            Model config is stored in .json config file.
        """
        super().__init__(task_config=task_config, batch_size=batch_size)
        self._task_config = task_config
        self.batch_size = batch_size
        
        # self.model_ckpt_paths = model_ckpt_paths
        # self.model_cfg_paths = model_cfg_paths

        # --------- CREATE MODEL-DATASET CORRESPONDENCE MAP --------- #
        assert len(model_ckpt_paths) == len(model_cfg_paths), f'Number of models loaded {len(model_ckpt_paths)} does not match the number of config paths loaded {len(model_cfg_paths)}'

        assert (len(model_ckpt_paths)==1) or (len(model_ckpt_paths)==len(dataset_handles)), f'Number of models loaded must be either a single one or one per dataset found: {len(dataset_handles)}'

        self.model_dataset_map = {}
        self.model_config_map = {}
        
        for i in range(len(dataset_handles)):

            # If only one loaded model, it'll be used to evaluate all datasets.
            if len(model_ckpt_paths) == 1:
                self.model_dataset_map[dataset_handles[i]] = model_ckpt_paths[0]
                self.model_config_map[dataset_handles[i]] = model_cfg_paths[0]
            # Otherwise, each model will be used to evaluate a dataset.    
            else:
                self.model_dataset_map[dataset_handles[i]] = model_ckpt_paths[i]
                self.model_config_map[dataset_handles[i]] = model_cfg_paths[i]
        
    def reset(self, dataset_tags: List[Path] = [""]):

        # Batch size is enforced to be 1, so dataset_tags must contain a single file_path.
        dataset_tag = dataset_tags[0]
        dataset_key = dataset_tag.stem
        print(f'Resetting decoder to dataset {dataset_key}')

        # --------- LOCATE EXPERIMENT METADATA --------- #

        model_ckpt_path = self.model_dataset_map[dataset_key]
        model_cfg_path = self.model_config_map[dataset_key]
        
        with open(model_cfg_path, 'rb') as file:
            print(f'Loading metadata from config file: {model_cfg_path}')
            experiment_metadata = json.load(file)

        model_layers = experiment_metadata['layers']
        learning_rate = experiment_metadata["learning_rate"]
        
        self.neural_mode = experiment_metadata['neural_mode']
        self.gaussian_smoothing_sigma = experiment_metadata["gaussian_smoothing_sigma"] 
        self.neural_history_ms = experiment_metadata["neural_history_ms"] 
        
        #--------- LOAD MODEL & OPTIMIZER STATE DICT --------- #
        print(f'Loading model: {model_ckpt_path}')
        self.ffnn_model, self.optimizer = load_model('', model_ckpt_path, model_layers, learning_rate)
        
        # --------- Load single NWB file metadata --------- #
        self.trial_info, self.neural_array, self.fs_neural, self.audio_motifs, self.fs_audio = load_nwb_b1(dataset_tag)
        # B1 task has different neural/behavioral sampling rates.
        tu.check_experiment_duration(self.neural_array, self.audio_motifs, self.fs_neural, self.fs_audio)

        # --------- INSTANTIATE ENCODEC --------- #
        self.encodec_model = eu.instantiate_encodec()
        audio_embeddings, audio_codes, scales = eu.encodec_encode_audio_array_2d(self.audio_motifs, self.fs_audio, self.encodec_model)
        
        self.target_embedding_len = audio_embeddings.shape[-1]
        self.audio_trial_len = self.audio_motifs.shape[-1]
        
        self.scale = scale = torch.mean(scales)

        return

    def on_done(self, dones: np.ndarray):
        pass

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        
        # --------- PROCESS NEURAL --------- #
    
        # Resample neural data to match audio embeddings
        original_samples = neural_observations.shape[-1]
        num_trials = neural_observations.shape[0]
        
        print(f'WARNING: Neural samples should be greater than embedding samples! Downsampling neural data from {original_samples} samples to match audio embedding samples {self.target_embedding_len}.')
        neural_array = tu.process_neural_data(neural_observations, self.neural_mode, self.gaussian_smoothing_sigma, original_samples, self.target_embedding_len)
        
        assert neural_array.shape[-1] == self.target_embedding_len, "Mismatch Error: The length of 'neural_array' does not match the length of 'audio_embeddings'."
        
        # bin_length = ((original_samples / self.fs_neural)*1000) / neural_array.shape[-1] # ms
        # decoder_history_bins = int(self.neural_history_ms // bin_length) # Must be minimum 1
        decoder_history_bins = 2 # Must be minimum 1
        
        print('Using {} bins of neural data history for decoding.'.format(decoder_history_bins))

        # --------- PREPARE EVAL DATALOADER --------- #
        dummy_audio_embeddings = np.zeros(neural_array.shape)
        
        _, test_loader = tu.prepare_dataloader(neural_array, 
                                               dummy_audio_embeddings, 
                                               self.batch_size, 
                                               decoder_history_bins, 
                                               max_temporal_shift_bins=0,
                                               noise_level=0,
                                               transform_probability=0, 
                                               shuffle_samples = False)
        
        # PREDICT AUDIO
        decoded_embeddings = ffnn_predict(self.ffnn_model, test_loader)
        decoded_embeddings = decoded_embeddings.permute(1, 0)
        decoded_embeddings = decoded_embeddings.reshape(decoded_embeddings.shape[0], num_trials, -1) # Embedding_dim x Trials x Samples
        
        decoded_embeddings = decoded_embeddings.to(self.scale.device)
        reconstructed_audio = np.array([eu.audio_from_embedding(decoded_embeddings[:,i,:], self.scale, self.encodec_model, self.fs_audio).squeeze(0).squeeze(0).detach() for i in range(num_trials)])

        # Pad the beginning of each row with zeros
        pad_length = self.audio_trial_len - reconstructed_audio.shape[-1]
        padded_reconstructed_audio = np.pad(reconstructed_audio, ((0, 0), (pad_length, 0)), mode='constant', constant_values=0)
        

        # --------- COMPUTE RECONSTRUCTED SPECTROGRAM --------- #
        reconstructed_sxx = np.array(self.compute_falcon_spectrogram(padded_reconstructed_audio[0], self.fs_audio)[-1])
        
        return reconstructed_sxx


    @staticmethod
    def compute_falcon_spectrogram(x, fs, n_window=512, step_ms=1, f_min=250, f_max=8000, cut_off=0.0001):
        """
        Computes a spectrogram from an audio waveform for the FALCON benchmark.
        To be consistent with the FALCON challenge, DO NOT MODIFY DEFAULTS.
    
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