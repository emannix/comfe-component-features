from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from pdb import set_trace as pb
import os
import sys
import copy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import open_dict
from goo.utils.pylogger import get_pylogger
from goo.utils.instantiatiators import instantiate_callbacks, instantiate_loggers
from omegaconf import OmegaConf

def train(full_cfg):
    log = get_pylogger(__name__)
    cfg = full_cfg.model

    if hasattr(cfg.trainer, 'precision'):
        if cfg.trainer.precision == 64:
            torch.set_default_dtype(torch.float64)
    # if cfg.get("seed"):
    L.seed_everything(full_cfg.seed, workers=True)

    log.info("Logging hyperparameters!")
    log.info(str(cfg))
    # ===========================================================================
    log.info("Instantiating callbacks...")
    if 'custom_callbacks' in cfg:
        with open_dict(cfg.callbacks):
            cfg.callbacks.merge_with(cfg.custom_callbacks)
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"), log)
    # ===========================================================================
    if hasattr(cfg, 'dataset_override'):
        with open_dict(cfg):
            cfg.dataset_for_config = copy.deepcopy(cfg.dataset)
            cfg.dataset = cfg.dataset_override
    if hasattr(cfg, 'networks_override'):
        with open_dict(cfg):
            cfg.networks_for_config = copy.deepcopy(cfg.networks)
            cfg.networks = cfg.networks_override
    # ===========================================================================

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    with open_dict(cfg.dataset):
        cfg.dataset.merge_with(cfg.dataloaders)
        cfg.dataset.merge_with(cfg.augmentations)
        cfg.dataset.seed = full_cfg.seed

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    # ===========================================================================
    log.info(f"Instantiating method <{cfg.methods._target_}>")
    with open_dict(cfg.methods):
        cfg.methods.merge_with(cfg.optimizer)
        cfg.methods.merge_with(cfg.scheduler)
        cfg.methods.merge_with(cfg.networks)
        cfg.methods.merge_with(cfg.loss)
        # use dataloader to set the number of classes
        cfg.methods['num_classes'] = datamodule.num_classes
        # Can put something more general here

        if hasattr(cfg.methods.networks, 'backbone'):
            if cfg.methods.networks.backbone._target_ == 'goo.networks.self_supervised_wrapper.SelfSupervisedWrapper':
                cfg.methods.networks.backbone.num_classes = datamodule.num_classes
        if hasattr(cfg.methods.networks, 'head'):
            if cfg.methods.networks.head._target_ == 'goo.networks.load_module.LoadModule':
                cfg.methods.networks.head.num_classes = datamodule.num_classes
            if hasattr(cfg.methods.networks.head, 'seed'):
                cfg.methods.networks.head.seed = full_cfg.seed
        # ====================================
        if 'discriminator' in cfg.methods.networks:
            cfg.methods.networks.discriminator['output_dim'] = datamodule.num_classes
        if 'model' in cfg.methods.networks:
            cfg.methods.networks['model']['output_dim'] = datamodule.num_classes

        cfg.methods['seed'] = full_cfg.seed

    method: LightningModule = hydra.utils.instantiate(cfg.methods)
    # ===========================================================================
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"), log)
    # ===========================================================================
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    # ===========================================================================
    object_dict = {
        "cfg": full_cfg,
        "datamodule": datamodule,
        "method": method,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    # ===========================================================================
    if trainer.precision == '16-mixed':
        torch.set_float32_matmul_precision('medium')
    elif trainer.precision == 32:
        torch.set_float32_matmul_precision('high')
    elif trainer.precision == 64:
        torch.set_float32_matmul_precision('highest')
    # ===========================================================================

    if full_cfg.get("compile"):
        log.info("Compiling model!")
        method = torch.compile(method)

    if full_cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=method, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    log.info("Selecting checkpoint")
    if full_cfg.get("train"):
        if full_cfg.ckpt_use == 'last':
            ckpt_path = trainer.checkpoint_callback.last_model_path
        elif full_cfg.ckpt_use == 'best':
            ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Ckpt not found! Using current weights for testing...")
            ckpt_path = None
    else:
        ckpt_path = cfg.ckpt_path
    log.info(f"Ckpt path: {ckpt_path}")

    # from goo.methods import Supervised
    # res = Supervised.load_from_checkpoint(trainer.checkpoint_callback.last_model_path)

    if full_cfg.get("test"):
        log.info("Starting testing!")
        trainer.test(model=method, datamodule=datamodule, ckpt_path=ckpt_path)
    test_metrics = trainer.callback_metrics

    if full_cfg.get("predict"):
        log.info("Starting predicting!")
        method: LightningModule = hydra.utils.instantiate(cfg.methods)
        trainer.predict(model=method, datamodule=datamodule, ckpt_path=ckpt_path)

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="configs", config_name="base.yaml")
def main(full_cfg):
    train(full_cfg)


# ==========================================================

def get_jobarray_seed():
    # Job array management
    if os.environ.get('env_seed') is not None:
        seed = int(os.environ.get('env_seed'))
    else:
        seed = None
    return seed

if __name__ == "__main__":
    seed = get_jobarray_seed()
    if seed is not None:
        sys.argv.append('seed='+str(seed))
    main()
