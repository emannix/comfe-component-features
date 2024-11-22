import numpy as np
import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
from pdb import set_trace as pb

class Helpers:
    @staticmethod
    def test_dictionaries_are_equal(dict1, dict2, tolerance=1e-4, check_keys=True):
        if check_keys and dict1.keys() != dict2.keys():
            return False

        for key, value in dict1.items():
            if np.abs(value - dict2[key]) > tolerance:
                print(key)
                print(value)
                print(dict2[key])
                print(np.abs(value - dict2[key]))
                return False
        return True

    @staticmethod
    def load_configuration_file(tmp_path, override = [], quicker=False):
        with initialize(version_base="1.3", config_path="../../configs"):
            override_list = [*override, "debug=debug.yaml"]
            override_list = ['+'+x for x in override_list]
            cfg = compose(config_name="base.yaml", return_hydra_config=True, overrides=override_list) 
            with open_dict(cfg):
                cfg.model.trainer.deterministic = True
                cfg.paths.output_dir = str(tmp_path)
                cfg.paths.log_dir = str(tmp_path)
                cfg.ckpt_use = 'best'
                if quicker:
                    cfg.model.trainer.max_epochs = 3
                    cfg.model.trainer.check_val_every_n_epoch = 3
                    if hasattr(cfg.model, 'custom_callbacks'):
                        if hasattr(cfg.model.custom_callbacks, 'save_knn_pred'):
                            cfg.model.custom_callbacks.save_knn_pred.check_knn_every = 3
        return cfg