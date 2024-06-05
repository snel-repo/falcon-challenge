import numpy as np
from scipy.signal import butter, filtfilt
from scipy import signal
from scipy.signal.windows import gaussian
from sklearn.metrics import mean_squared_error


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut (float): Low frequency cutoff.
        highcut (float): High frequency cutoff.
        fs (float): Sampling frequency.
        order (int): Order of the filter. Default is 5.
    
    Returns:
        b, a (ndarray, ndarray): Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to data.
    
    Args:
        data (ndarray): Input data to be filtered.
        lowcut (float): Low frequency cutoff.
        highcut (float): High frequency cutoff.
        fs (float): Sampling frequency.
        order (int): Order of the filter. Default is 5.
    
    Returns:
        y (ndarray): Filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=1)
    return y

def detect_tx(data, fs, thresholds, refractory_th=1):
    """
    Detect spikes in the data based on threshold crossings and a refractory period.
    
    Args:
        data (ndarray): Input data from which to detect spikes.
        fs (float): Sampling frequency.
        thresholds (ndarray): Threshold values for each channel.
        refractory_th (float): Refractory period in milliseconds. Default is 1 ms.
    
    Returns:
        spikes (ndarray): Binary matrix indicating the presence of spikes.
    """
    spikes = np.zeros_like(data, dtype=float)
    
    for i in range(data.shape[0]):  # Iterate over channels
        channel_data = np.abs(data[i, :])
        crossings = (channel_data > thresholds[i]).astype(int)
        spike_times = np.where(crossings)[0]
        
        # Enforce refractory period (e.g., 1 ms)
        refractory_samples = int(refractory_th / 1000 * fs)
        valid_spike_idx = [spike_times[0]]
        for spike in spike_times[1:]:
            if spike - valid_spike_idx[-1] > refractory_samples:
                valid_spike_idx.append(spike)
        spikes[i, valid_spike_idx] = True
    return spikes

def get_window_traces(traces, channels, start_sample, end_sample) -> np.array:
    """
    Extract windowed traces from the data.
    
    Args:
        traces (ndarray): Input data traces.
        channels (list): List of channel indices to extract.
        start_sample (int): Start sample index.
        end_sample (int): End sample index.
    
    Returns:
        ndarray: Windowed traces.
    """
    return traces[start_sample:end_sample, channels].T

def extract_threshold_crossings(traces: np.array, channels: np.array, fs,
                                    start_sample, end_sample,
                                    th_cross, refractory_th=1,
                                    filt=False, lowcut=300, highcut=3000,
                                    clean_raster=False) -> np.array:
    """
    Get raster plots of threshold crossings from traces.
    
    Args:
        traces (ndarray): Input data traces [time x channels].
        channels (list): List of channel indices to process.
        fs (float): Sampling frequency.
        start_sample (int): Start sample index.
        end_sample (int): End sample index.
        th_cross (float): Threshold crossing value.
        refractory_th (float): Refractory period in milliseconds. Default is 1 ms.
        filt (bool): Whether to apply bandpass filtering. Default is False.
        lowcut (float): Low frequency cutoff for filtering. Default is 300.
        highcut (float): High frequency cutoff for filtering. Default is 3000.
        clean_raster (bool): Whether to clean the raster plot by removing noisy events. Default is False.
    
    Returns:
        ndarray: Raster plot of threshold crossings.
    """
    target_traces = get_window_traces(traces, channels, start_sample, end_sample)
    if filt:
        target_traces = bandpass_filter(target_traces, lowcut, highcut, fs)

    thresholds = th_cross * np.std(np.abs(target_traces), axis=1)  
    tx_matrix = detect_tx(target_traces, fs, thresholds, refractory_th=refractory_th)

    if clean_raster:
        tx_matrix = sh.clean_spikeRaster_noisyEvents2d(tx_matrix, verbose=verbose)  # Remove noisy (simultaneous) events
    
    return tx_matrix

def get_rasters_audio(audio, start_sample_list: list, end_sample_list: list) -> list:
    """
    Extract snippets of audio specified by start and end sample lists.
    
    Args:
        audio (ndarray): Input audio data.
        start_sample_list (list): List of start sample indices.
        end_sample_list (list): List of end sample indices.
    
    Returns:
        list: List of audio snippets.
    """
    audio_arr_list = [audio[start_sample_list[i] : end_sample_list[i]] 
                      for i in range(len(start_sample_list))]
    return np.array(audio_arr_list).squeeze().tolist()
