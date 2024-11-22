import numpy as np
import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from pdb import set_trace as pb

from lightning.pytorch.callbacks import ModelCheckpoint

from main import train
from tests.helpers import helpers 

class Tests:
    @staticmethod
    def pretrain_eval_finetune_test(cfg, prev_results_train=None, prev_results_finetune=None, accelerator='gpu', check_keys_train=True, check_keys_finetune=False):
        cfg_train = cfg['pretrain']
        if 'finetune' in cfg:
            cfg_finetune = cfg['finetune']

        # =============================================
        """Run for 1 train, val and test step."""
        HydraConfig().set_config(cfg_train)
        with open_dict(cfg_train):
            cfg_train.model.trainer.accelerator = accelerator
        res_train = train(cfg_train)
        for callback in res_train[1]['trainer'].callbacks:
            if isinstance(callback, ModelCheckpoint):
                last_model_path = callback.last_model_path

        print(res_train[0])
        if prev_results_train is not None:
            assert helpers.test_dictionaries_are_equal(prev_results_train, res_train[0], check_keys=check_keys_train)
        # =============================================
        if 'finetune' in cfg:
            HydraConfig().set_config(cfg_finetune)
            with open_dict(cfg_finetune):
                cfg_finetune.model.trainer.accelerator = accelerator
                cfg_finetune.model.networks.networks.backbone.model_path = last_model_path
            res_finetune = train(cfg_finetune)
        # =============================================
            print(res_finetune[0])
            if prev_results_finetune is not None:
                assert helpers.test_dictionaries_are_equal(prev_results_finetune, res_finetune[0], check_keys=check_keys_finetune)

