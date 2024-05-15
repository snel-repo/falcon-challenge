from __future__ import annotations

import enum
from typing import Union
from pathlib import Path
import datetime
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

H1_NEW_TO_OLD = {
    '19250101T111740': 'S0_set_1',
    '19250101T112404': 'S0_set_2',
    '19250108T110520': 'S1_set_1',
    '19250108T111022': 'S1_set_2',
    '19250108T111455': 'S1_set_3',
    '19250113T120811': 'S2_set_1',
    '19250113T121303': 'S2_set_2',
    '19250115T110633': 'S3_set_1',
    '19250115T111328': 'S3_set_2',
    '19250119T113543': 'S4_set_1',
    '19250119T114045': 'S4_set_2',
    '19250120T115044': 'S5_set_1',
    '19250120T115537': 'S5_set_2',
    '19250126T113454': 'S6_set_1',
    '19250126T114029': 'S6_set_2',
    '19250127T120333': 'S7_set_1',
    '19250127T120826': 'S7_set_2',
    '19250129T112555': 'S8_set_1',
    '19250129T113059': 'S8_set_2',
    '19250202T113958': 'S9_set_1',
    '19250202T114452': 'S9_set_2',
    '19250203T113515': 'S10_set_1',
    '19250203T114018': 'S10_set_2',
    '19250206T112219': 'S11_set_1',
    '19250206T112712': 'S11_set_2',
    '19250209T111826': 'S12_set_1',
    '19250209T112327': 'S12_set_2',
}

class FalconTask(enum.Enum):
    r"""
        Enumerates the tasks that the Falcon challenge supports.
    """
    h1 = "falcon_h1_7d"
    h2 = "falcon_h2_writing"
    m1 = "falcon_m1_finger"
    m2 = "falcon_m2_reach"

@dataclass
class FalconConfig:
    r"""
        User is responsible for copying this appropriately,
        since ultimately these values are to help user to inform decoder what outputs are expected.
    """
    # falcon_h1, falcon_h2_writing, falcon_m1_finger, falcon_m2_reach
    task: FalconTask = FalconTask.h1
    # n_channels: int = 176
    bin_size_ms: int = 20
    # dataset_handles: list[str] = field(default_factory=lambda: []) # Compute with evaluator.get_eval_handles

    @property
    def n_channels(self):
        if self.task == FalconTask.h1:
            return 176
        elif self.task == FalconTask.h2:
            return 192
        elif self.task == FalconTask.m1:
            return 64
        elif self.task == FalconTask.m2:
            return 96
        raise NotImplementedError(f"Task {self.task} not implemented.")

    @property
    def out_dim(self):
        if self.task == FalconTask.h1:
            return 7
        elif self.task == FalconTask.h2:
            return 28
        elif self.task == FalconTask.m1:
            return 16
        elif self.task == FalconTask.m2:
            return 2
        raise NotImplementedError(f"Task {self.task} not implemented.")
        
    def hash_dataset(self, handle: Union[str, Path]):
        r"""
            handle - path.stem of a datafile.
            Convenience function to help identify what "session" a datafile belongs to.. If multiple files per session in real-world time, this may _not_ uniquely identify runfile.
        """
        if isinstance(handle, Path):
            handle = handle.stem
        if self.task == FalconTask.h1:
            if 'sub-HumanPitt' in handle:
                date_hash = handle.split('_ses-')[-1]
                return H1_NEW_TO_OLD[date_hash]
            handle = handle.replace('-', '_')
            handle = '_'.join(handle.split('_')[:-1]) # exclude split annotation
            # print(handle)
            return handle
            # dandi-like atm but not quite determined; e.g. S0_set_1_calib
            # remove split and set information
            # pieces = handle.split('_')
            # for piece in pieces:
                # if piece[0].lower() == 's' and piece != 'set':
                    # return piece
            raise ValueError(f"Could not find session in {handle}.")
        elif self.task == FalconTask.h2:
            return handle.split('_')[1]
        elif self.task == FalconTask.m1: # return date
            # sub-MonkeyL-held-in-minival_ses-20120924_behavior+ecephys.nwb
            # or L_20120924_held_in_eval.nwb
            if 'behavior+ecephys' in handle:
                return handle.split('_')[-2].split('-')[-1]
            return handle.split('_')[1]
        elif self.task == FalconTask.m2:
            if 'behavior+ecephys' in handle: # public sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb
                run_str = handle.split('_')[1][-4:]
                date_str = ''.join(handle.split('_')[1].split('-')[1:4])
            # sub-MonkeyNRun1_20201019_held_in_eval.nwb 
            else:
                run_str = handle.split('_')[0][-4:]
                date_str = handle.split('_')[1][:8]
            return f'{run_str}_{date_str}'
            

cs = ConfigStore.instance()
cs.store(name="falcon_config", node=FalconConfig)