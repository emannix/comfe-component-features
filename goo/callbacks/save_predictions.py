from lightning.pytorch.callbacks import Callback
from pdb import set_trace as pb

from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import os
import pandas as pd
import glob

import torch
import numpy as np
import torch.nn as nn

from torchmetrics import AUROC, AveragePrecision, StatScores
from torchmetrics import JaccardIndex 
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

import torch.distributed as dist
from torchmetrics.classification import MulticlassAccuracy

class PredictionCache(Metric):
    full_state_update=True
    
    def __init__(self):
        super().__init__()
    
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target = None):
        if isinstance(preds, list):
            if isinstance(preds[0], str):
                preds = np.array(preds, dtype=str)
            elif isinstance(preds[0], tuple):
                preds = np.array([x[0] for x in preds], dtype=int)
            else:
                preds = np.array(preds[0].cpu(), dtype=int)
            self.preds.append(preds)
        else:
            self.preds.append(preds.detach().cpu())
        if target is not None:
            self.target.append(target.detach().cpu())

    def reset(self):
        self.target = []
        self.preds = []
        
    def return_results(self):
        if (len(self.preds) > 0):
            if type(self.preds[0]) is np.ndarray:
                preds = np.concatenate(self.preds)
            else:
                preds = dim_zero_cat(self.preds).squeeze()
        else:
            preds = torch.tensor(0.0)
        if (len(self.target) > 0):
            target = dim_zero_cat(self.target).squeeze()
        else:
            target = torch.tensor(0.0)
        return(preds, target)
    
    def compute(self):
        pass


# ===============================================================================

class SavePredictions(Callback):

    def __init__(self, 
            metrics_calculate = ['accuracy', 'mean_class_accuracy', 'aucroc', 'average_precision'], 
            calculate_point='epoch', calculate_stages = ['train', 'val', 'test', 'predict'], 
            num_classes = None, save_csv=True):
        self.metrics_calculate = metrics_calculate
        self.calculate_point = calculate_point
        self.calculate_stages = calculate_stages
        self.num_classes = num_classes
        self.save_csv = save_csv

    def setup(self, trainer, pl_module, stage):
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes

        metrics = pl_module.metrics
        metrics_log = pl_module.metrics_log
        metrics_save = pl_module.metrics_save

        self.metrics_calculate_fnc = {}
        if 'accuracy' in self.metrics_calculate:
            self.metrics_calculate_fnc['accuracy'] = Accuracy(task="multiclass", num_classes=self.num_classes, average='micro')
        if 'mean_class_accuracy' in self.metrics_calculate:
            self.metrics_calculate_fnc['mean_class_accuracy'] = MulticlassAccuracy(num_classes=self.num_classes, average='macro')
        if 'aucroc' in self.metrics_calculate:
            self.metrics_calculate_fnc['aucroc'] = AUROC(task='multiclass', num_classes=self.num_classes, average='macro')
        if 'average_precision' in self.metrics_calculate:
            self.metrics_calculate_fnc['average_precision'] = AveragePrecision(task='multiclass', num_classes=self.num_classes, average=None)
        if 'stat_scores' in self.metrics_calculate:
            self.metrics_calculate_fnc['stat_scores'] = StatScores(task='multiclass', num_classes=self.num_classes, average=None)
        if 'mIoU' in self.metrics_calculate:
            self.metrics_calculate_fnc['mIoU'] = JaccardIndex(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=255)
        # ====================
        self.metrics = dict(zip(metrics, [PredictionCache() for x in metrics]))
        self.metrics_log = dict(zip(metrics, metrics_log))
        self.metrics_save = dict(zip(metrics, metrics_save))

        self.trainer = trainer
        self.pl_module = pl_module

    def update_metrics(self, results_dict):
        for key, cache in self.metrics.items():
            if key == 'y_hat':
                cache(results_dict[key], results_dict['label'])
            else:
                cache(results_dict[key], None)
    
    def save_predictions_as_csv(self, cache, trainer, col='', save_string='test'):
        print('saving results')
        logger = trainer.logger
        output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'predictions')
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file_base = 'epoch='+str(trainer.current_epoch)+'-step='+str(trainer.global_step)+'-'

        if (len(cache.shape)>1):
            columns = None
        else:
            columns = [col]
        
        if type(cache) is not np.ndarray:
            cache = cache.numpy()

        prediction_values = pd.DataFrame(cache, columns=columns)
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if self.save_csv:
            prediction_values.to_csv(
                output_folder + '/' + output_file_base + '_' + save_string +'_predictions'+str(rank)+'.csv', index=False)
    
    # ========================================================================
    def _on_log(
        self, stage='val', only_accuracy = False
    ):
        device = self.pl_module.device
        for key, cache in self.metrics.items():
            if key == 'y_hat':
                preds, target = cache.return_results()
                if preds.dim() == 0:
                    pass
                elif only_accuracy:
                    if 'accuracy' in self.metrics_calculate_fnc:
                        key_calc = 'accuracy'
                        metric_value = self.metrics_calculate_fnc[key_calc](preds, target)
                        self.log(stage+'/'+key_calc, metric_value.to(device), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
                    if 'mean_class_accuracy' in self.metrics_calculate_fnc:
                        key_calc = 'mean_class_accuracy'
                        metric_value = self.metrics_calculate_fnc[key_calc](preds, target)
                        self.log(stage+'/'+key_calc, metric_value.to(device), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
                    if 'mIoU' in self.metrics_calculate_fnc:
                        key_calc = 'mIoU'
                        metric_value = self.metrics_calculate_fnc[key_calc](preds, target)
                        self.log(stage+'/'+key_calc, metric_value.to(device), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
                else:
                    for key_calc, metric in self.metrics_calculate_fnc.items():
                        metric_value = metric(preds, target)
                        if len(metric_value.shape) == 0:
                            self.log(stage+'/'+key_calc, metric_value.to(device), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
                        else:
                            if key_calc == 'stat_scores':
                                state_names = ['tp', 'fp', 'tn', 'fn', 'support']
                                for i in range(metric_value.shape[0]):
                                    for j in range(len(state_names)):
                                        self.log(stage+'/'+key_calc+'_'+str(i)+'/'+state_names[j], float(metric_value[i, j]).to(device), prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
                            else:
                                for i in range(len(metric_value)):
                                    self.log(stage+'/'+key_calc+'_'+str(i), metric_value[i].to(device), prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            if self.metrics_log[key]:
                preds, _ = cache.return_results()
                self.log(stage+'/'+key, preds.mean().to(device), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            cache.reset()

    def _on_save(
        self, trainer, pl_module, stage='test'
    ):
        dataset = type(pl_module.trainer.datamodule).__name__

        for key, cache in self.metrics.items():
            if self.metrics_save[key]:
                preds, _ = cache.return_results()
                self.save_predictions_as_csv(preds, trainer, col=key, save_string=dataset+'_'+stage+'_'+key)

    # ============================================================================

    def on_validation_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if self.calculate_point == 'epoch' and 'val' in self.calculate_stages:
            self._on_log('val')

    def on_test_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if self.calculate_point == 'epoch' and 'test' in self.calculate_stages:
            self._on_save(trainer, pl_module, 'test')
            self._on_log('test')

    def on_predict_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if self.calculate_point == 'epoch' and 'predict' in self.calculate_stages:
            self._on_save(trainer, pl_module, 'predict')
            self._on_log('predict')
    # ============================================================================
    def on_train_batch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch, batch_idx
    ):
        if 'train' in self.calculate_stages:
            self.update_metrics(outputs)
            self._on_log('train', only_accuracy=True)

    def on_validation_batch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch, batch_idx, dataloader_idx=-1
    ):
        if 'val' in self.calculate_stages:
            self.update_metrics(outputs)
            if self.calculate_point == 'batch':
                self._on_log('val', only_accuracy=True)

    def on_test_batch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch, batch_idx, dataloader_idx=-1
    ):
        if 'test' in self.calculate_stages:
            self.update_metrics(outputs)
            if self.calculate_point == 'batch':
                self._on_log('test', only_accuracy=True)

    def on_predict_batch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch, batch_idx, dataloader_idx=-1
    ):
        if 'predict' in self.calculate_stages:
            self.update_metrics(outputs)
            if self.calculate_point == 'batch':
                self._on_log('predict', only_accuracy=True)