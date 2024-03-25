from pathlib import Path
from pynwb import NWBFile, NWBHDF5IO
from pynwb import TimeSeries
from pynwb.behavior import Position
from pynwb.behavior import BehavioralTimeSeries
from pynwb import ProcessingModule

r"""
    FALCON Consts
"""
BIN_SIZE_MS = 20
BIN_SIZE_S = BIN_SIZE_MS / 1000
FEW_SHOT_CALIBRATION_RATIO = 0.2
EVAL_RATIO = 0.4
EVAL_RATIO_HUMAN = 0.2
SMOKETEST_NUM = 2

def write_to_nwb(nwbfile: NWBFile, fn: Path):
    fn.parent.mkdir(exist_ok=True, parents=True)
    with NWBHDF5IO(str(fn), 'w') as io:
        io.write(nwbfile)

def create_multichannel_timeseries(data_name, chan_names, data, timestamps=None, unit=''):
    """Creates a BehavioralTimeSeries for multi-channel continuous data"""
    ts = BehavioralTimeSeries(name=data_name)
    for i, chan_name in enumerate(chan_names):
        ts.create_timeseries(name=chan_name,
                             data=data[:,i],
                             unit=unit,
                             timestamps=timestamps,
                             comments=f"columns=[{chan_name}]")

    return ts

def apply_filt_to_multi_timeseries(m_ts, filt_func, name, *args, timestamps=None, **kwargs):
    """apply filtering function to each TimeSeries in BehavioralTimeSeries"""
    ts_keys = list(m_ts.time_series.keys())
    filt_m_ts = BehavioralTimeSeries(name=name)
    for ts_key in ts_keys:
        ts = m_ts[ts_key]
        filt_data = filt_func(ts.data, *args, **kwargs)
        if timestamps is None:
            timestamps = ts.timestamps
        filt_m_ts.create_timeseries(name=ts_key,
                                    data=filt_data,
                                    unit=ts.unit,
                                    comments=ts.comments,
                                    timestamps=timestamps)

    return filt_m_ts