import yaml, pickle, sys, time, os
import numpy as np

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder
from falcon_challenge.evaluator import DATASET_HELDINOUT_MAP

# make sure tf is running in graph mode 
import tensorflow as tf
tf.config.run_functions_eagerly(False)
# run on CPU to speed up inference 
tf.config.set_visible_devices([], 'GPU')

from lfads_tf2.subclasses.dimreduced.models import DimReducedLFADS
from lfads_tf2.tuples import LFADSInput

sys.path.insert(0, '/home/bkarpo2/bin/stability-benchmark/align_tf2')
for p in sys.path: 
    if 'nomad_dev' in p: 
        sys.path.remove(p)
from align_tf2.models import AlignLFADS

from decoder_demos.decoding_utils import generate_lagged_matrix

class NoMAD_Decoder(BCIDecoder):
    def __init__(self, task_config: FalconConfig, submission_dict: str):
        self._task_config = task_config
        self.batch_size = 1
        
        with open(submission_dict, 'r') as f:
            decoder_cfg = yaml.safe_load(f)

        self.track = self._task_config.task.name
        self.model_lookup = [x for x in decoder_cfg['submissions'] if x['track'] == self.track][0]
        rng = np.random.default_rng()

    def reset(self, dataset_tags):
        # called when the file changes
        # parse the tag into the right key 
        hashed_tag =  self._task_config.hash_dataset(dataset_tags[0])
        model_path = self.model_lookup[hashed_tag]['model']
        decoder_path = self.model_lookup[hashed_tag]['decoder']

        # load corresponding model for this file ID
        # if it's a heldin file, use LFADS, otherwise use NoMAD 
        if hashed_tag in DATASET_HELDINOUT_MAP[self.track]['held_in']: 
            self.model = DimReducedLFADS(model_dir=model_path)
        else: 
            self.model = AlignLFADS(align_dir=model_path).lfads_dayk

        self.seq_len = self.model.cfg.MODEL.SEQ_LEN
        self.model.cfg['MODEL']['SAMPLE_POSTERIORS'] = False 
        self.model.cfg['TRAIN']['EAGER_MODE'] = False
        self.ext_input = tf.zeros((1, self.seq_len, self.model.cfg.MODEL.EXT_INPUT_DIM), dtype=np.float32)
        self.dataset_name = tf.fill(1, '')
        self.behavior = tf.zeros((1, self.seq_len, 0), dtype=np.float32)

        # in any case, unpickle the decoder
        with open(decoder_path, 'rb') as f:
            decoder_info = pickle.load(f)
        self.decoder = decoder_info['decoder']
        self.history = decoder_info['history']

        self.lfads_input_buffer = np.zeros((self.seq_len, self._task_config.n_channels), dtype=np.float32)

    def predict(self, neural_observations: np.ndarray):
        self.lfads_input_buffer[0:-1, :] = self.lfads_input_buffer[1:]
        self.lfads_input_buffer[-1, :] = neural_observations
        
        # model inference 
        lfads_input = LFADSInput(enc_input=self.lfads_input_buffer[np.newaxis, :, :],
                                ext_input=self.ext_input,
                                dataset_name=self.dataset_name,
                                behavior=self.behavior)

        lfads_output = self.model.graph_call(lfads_input)
        gen_states = lfads_output.gen_states.numpy()
        # add lfads output to decoding buffer 
        decoding_buffer = gen_states[0, -(self.history + 1):, :]
        # apply decoder 
        lagged_input = generate_lagged_matrix(decoding_buffer, self.history)
        decoder_output = self.decoder.predict(lagged_input)
        # return prediction for that timestep 
        return decoder_output