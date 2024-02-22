from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class DecodeConfig:
    r"""
        Figuring this out...
    """
    # stability_23_pitt_human_7dof, stability_23_chestek_nhp_finger, stability_23_rouse_reach, stability_23_human_handwriting
    task: str = "stability_23_human_7dof"
    n_channels: int = 192