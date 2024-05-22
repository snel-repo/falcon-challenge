import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os
from lfads_tf2.tuples import LoadableData, LFADSInput

def chop_and_infer(func,
                   data,
                   out_fields,
                   seq_len=30,
                   stride=1,
                   batch_size=64,
                   output_dim=None):
    """
    Chop data into sequences, run inference on those sequences, and merge
    the inferred result back into an array of continuous data. When merging
    overlapping sequences, only the non-overlapping sections of the incoming
    sequence are used.
    Parameters
    ----------
    func : callable
        Function to be used for inference. Must be of the form:
            prediction = func(data)
    data : `numpy.ndarray` of shape (n_samples, n_features)
        Data to be split up into sequences and passed into the model
    seq_len : int, optional
        Length of each sequence, by default 30
    stride : int, optional
        Step size (in samples) when shifting the sequence window, by default 1
    batch_size : int, optional
        Number of sequences to include in each batch, by default 64
    output_dim : int, optional
        Number of output features from func, by default None. If None,
        `output_dim` is set equal to the number of features in the input data.
    Returns
    -------
    output : numpy.ndarray of shape (n_samples, n_features)
        Inferred output from func
    Raises
    ------
    ValueError
        If `stride` is greater than `seq_len`
    """
    if stride > seq_len:
        raise ValueError(
            "Stride must be less then or equal to the sequence length")

    data_len, data_dim = data.shape[0], data.shape[1]
    output_dim = {k: data_dim for k in out_fields} if output_dim is None else output_dim

    batch = np.zeros((batch_size, seq_len, data_dim), dtype=data.dtype)
    output = {k: np.zeros((data_len, output_dim[k]), dtype=data.dtype) for k in out_fields}
    olap = seq_len - stride

    n_seqs = (data_len - seq_len) // stride + 1
    n_batches = np.ceil(n_seqs / batch_size).astype(int)

    i_seq = 0  # index of the current sequence
    for i_batch in range(n_batches):
        n_seqs_batch = 0  # number of sequences in this batch
        # chop
        start_ind_batch = i_seq * stride
        for i_seq_in_batch in range(batch_size):
            if i_seq < n_seqs:
                start_ind = i_seq * stride
                batch[i_seq_in_batch, :, :] = data[start_ind:start_ind +
                                                   seq_len]
                i_seq += 1
                n_seqs_batch += 1
        end_ind_batch = start_ind + seq_len
        # infer
        # batch_out = func(batch)[:n_seqs_batch]
        batch_out = {k:v[:n_seqs_batch] for k,v in func(batch).items()}
        n_samples = n_seqs_batch * stride

        # merge
        if start_ind_batch == 0:  # fill in the start of the sequence
             for k in out_fields: 
                output[k][:olap, :] = batch_out[k][0, :olap, :]

        out_idx_start = start_ind_batch + olap
        out_idx_end = end_ind_batch
        out_slice = np.s_[out_idx_start:out_idx_end]
        for k in out_fields: 
            output[k][out_slice, :] = batch_out[k][:, olap:, :].reshape(
                n_samples, output_dim[k])

    return output


def get_causal_model_output(model, 
                            binsize, 
                            input_data, 
                            out_fields, 
                            output_dim, 
                            stride=1, 
                            batch_size=64): 
    
    # make sure input data is appropriate level of precision 
    input_data = input_data.astype('float64')

    # pass means instead of samples 
    model.cfg['MODEL']['SAMPLE_POSTERIORS'] = False 
    
    seq_len = model.cfg.MODEL.SEQ_LEN
    data_dim = model.cfg.MODEL.DATA_DIM

    rng = np.random.default_rng()
    data = np.zeros((batch_size, seq_len, data_dim), dtype=np.float32)
    ext_input_dim = model.cfg.MODEL.EXT_INPUT_DIM
    ext_input = np.zeros((batch_size, seq_len, ext_input_dim), dtype=np.float32)
    dataset_name = np.full(shape=[1], fill_value='')
    behavior = np.zeros((batch_size, seq_len, 0), dtype=np.float32)

    # Chop the data into sequences and run inference
    # how much to shift the window for each sequence
    seq_len_non_fp = model.cfg.MODEL.SEQ_LEN

    def lfads_infer(data_in):            
        data[:, :seq_len_non_fp, :] = data_in # pad the data with zeros in the forward prediction bins
        # run inference
        lfads_input = LFADSInput(enc_input=data,
                                ext_input=ext_input,
                                dataset_name=dataset_name,
                                behavior=behavior)
        lfads_output = model.graph_call(lfads_input)

        out_dict = {}
        for out_field in out_fields:
            out_dict[out_field] = getattr(lfads_output, out_field)[:, :seq_len_non_fp, :].numpy()
        # leave out the forward prediction bins
        return out_dict

    model_output = chop_and_infer(
        lfads_infer,
        input_data,
        out_fields,
        stride=stride,
        seq_len=seq_len_non_fp,
        batch_size=batch_size,
        output_dim=output_dim
    )

    return model_output


def merge_data(bdf, data, lfads_online_fieldname, overwrite=True): 
    n_bins, n_chans = data.shape
    N = data.shape[1]
    column_inds = [f"chan_{i :04d}" for i in range(N)]
    midx = pd.MultiIndex.from_tuples(
        zip([lfads_online_fieldname] * N, column_inds))

    online_df = pd.DataFrame(data,
                            index=bdf.data.index,
                            columns=midx)

    if overwrite and lfads_online_fieldname in bdf.data:
        bdf.data.drop(columns=lfads_online_fieldname, inplace=True)

    bdf.data = bdf.data.join(online_df)
    return bdf