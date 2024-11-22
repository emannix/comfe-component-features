# placeholder for supervised skeleton
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2457


import torch
from lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanMetric
from pdb import set_trace as pb
from typing import Dict, Any

from ..loss.cross_entropy_weighted import CrossEntropyWeighted
# from ..networks import SelfSupervisedWrapper

class ModelBase(LightningModule):
    def __init__(self, num_classes, networks, loss, optimizer, scheduler, scheduler_lightning, parameter_groups, seed=0):
        super(ModelBase, self).__init__()

        self.save_hyperparameters(logger=False)

        self.networks = networks
        if loss is not None:
            self.loss = loss()
        else:
            self.loss = None
        self.num_classes = num_classes

    def setup(self, stage):
        super(ModelBase, self).setup(stage)
        if isinstance(self.loss, CrossEntropyWeighted):
            dataset = self.trainer.datamodule.data_train
            self.loss.initialise_weights(dataset, self.num_classes)
        
        for key in self.networks.keys():
            if self.networks[key].func.__name__ == 'SelfSupervisedWrapper' or self.networks[key].func.__name__ == 'LoadModule':
                if not hasattr(self.networks[key], 'model_hparams'):
                    if 'module_target' in self.hparams.networks[key].keywords:
                        module_target = self.hparams.networks[key].keywords['module_target']
                    else:
                        module_target = key
                    self.hparams.networks[key].keywords['model_hparams'] =  \
                        getattr(self, module_target).hparams.model_hparams # update hparams
        
# ==============================================================
    # def on_save_checkpoint(
    #     self, checkpoint: Dict[str, Any]
    # ) -> None:
    #     pb()

    # def on_load_checkpoint(
    #     self, checkpoint: Dict[str, Any]
    # ) -> None:
    #     pb() # checkpoint['hyper_parameters']['networks']

    # def load_state_dict(self, checkpoint: Dict[str, Any], strict: bool=True):
    #     pb()
# ==============================================================

    def opt_parameters(self, named=True):
        if named:
            params = self.named_parameters()
        else:
            params = self.parameters()
        return params

    def configure_optimizers(self):
        # ======================================================================
        if 'optimizer' in self.hparams.optimizer.keywords and self.hparams.optimizer.keywords['optimizer'] is not None:
            parameters = self.hparams.parameter_groups(self, self.opt_parameters())
            _nested_optimizer = self.hparams.optimizer.keywords['optimizer']
            nested_optimizer = _nested_optimizer(params=parameters)
            self.hparams.optimizer.keywords['optimizer'] = nested_optimizer
            optimizer = self.hparams.optimizer()
            self.hparams.optimizer.keywords['optimizer'] = _nested_optimizer
        else:
            parameters = self.hparams.parameter_groups(self, self.opt_parameters())
            optimizer = self.hparams.optimizer(params=parameters)
        # ======================================================================
        # https://github.com/Lightning-AI/lightning/pull/11599
        training_batches = self.trainer.estimated_stepping_batches
        max_epochs = self.trainer.max_epochs
        if max_epochs == 0:
            max_epochs = 1
        if self.hparams.scheduler_lightning['interval'] == 'step':
            if 'T_max' in self.hparams.scheduler.keywords:
                self.hparams.scheduler.keywords['T_max'] *= training_batches
            if 'warmup_steps' in self.hparams.scheduler.keywords:
                self.hparams.scheduler.keywords['warmup_steps'] *= training_batches/max_epochs
        else:
            if 'T_max' in self.hparams.scheduler.keywords:
                self.hparams.scheduler.keywords['T_max'] *= max_epochs

        scheduler = self.hparams.scheduler(optimizer=optimizer)
        # ======================================================================
        configured = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }
        configured['lr_scheduler'].update(self.hparams.scheduler_lightning)
        return configured
