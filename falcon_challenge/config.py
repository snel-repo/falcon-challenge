from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


@dataclass
class FalconConfig:
    r"""
        Figuring this out...
    """
    # falcon_h1, falcon_h2_writing, falcon_m1_finger, falcon_m2_reach
    task: str = "falcon_h1_7d"
    n_channels: int = 192




cs = ConfigStore.instance()
cs.store(name="falcon_config", node=FalconConfig)