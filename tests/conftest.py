"""This file prepares config fixtures for other tests."""

import pytest
from tests.helpers import helpers 
from pdb import set_trace as pb
from omegaconf import DictConfig, open_dict
from pathlib import Path

@pytest.fixture(scope="function")
def cfg_comfe_flowers(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=comfe", "model/dataset=torchvision_flowers", "model/networks=dinov2_vits_14"])
    return cfg

