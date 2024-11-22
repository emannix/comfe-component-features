from typing import Any

import torch
from lightning import LightningModule
from .model_base import ModelBase
from pdb import set_trace as pb

from ..utils import visualize
from torchvision.utils import make_grid
import inspect

class ModelBaseS(ModelBase):

    def _training_step(self, batch: Any, batch_idx: int, prefix='train', **kwargs):
        if prefix == 'train':
            stage = 'fit'
        else:
            stage = 'eval'

        if False:
            visualize.show(make_grid(batch[0][0]))
            visualize.show(make_grid(batch[0][1]))
            pb()

        if 'dataloader_idx' in inspect.signature(self.model_step).parameters.keys():
            results_dict = self.model_step(batch, stage=stage, **kwargs)
        else:
            results_dict = self.model_step(batch, stage=stage)
        # update and log metrics
        device = self.device
        self.log(prefix+"/loss", results_dict['loss'].to(device), on_step=stage=='fit', on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss or backpropagation will fail
        return results_dict

    def training_step(self, batch: Any, batch_idx: int):
        return self._training_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        return self._training_step(batch, batch_idx, prefix='val', dataloader_idx=dataloader_idx)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        return self._training_step(batch, batch_idx, prefix='test', dataloader_idx=dataloader_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        return self._training_step(batch, batch_idx, prefix='predict', dataloader_idx=dataloader_idx)