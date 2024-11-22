
from pdb import set_trace as pb
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist

from .helper import load_indices_from_csv, MySubset, random_indices
from sklearn.model_selection import KFold

# from lightly.data import LightlyDataset
# from lightly.data.multi_view_collate import MultiViewCollate
from ..lightly.dataset import LightlyDataset
from ..lightly.multi_view_collate import MultiViewCollate
from torch.utils.data.distributed import DistributedSampler

from functools import partial
import copy
import re

def dataloader_fix_shuffle(base_dataloader):
    if base_dataloader.keywords['sampler'] is not None:
        base_dataloader.keywords.pop('shuffle')
    elif base_dataloader.keywords['batch_sampler'] is not None:
        base_dataloader.keywords.pop('batch_size')
        base_dataloader.keywords.pop('shuffle')
        base_dataloader.keywords.pop('sampler')
        base_dataloader.keywords.pop('drop_last')
    return base_dataloader

def init_sampler(dataset, hparams, sampler, batch_scaler = 1.0):
    if sampler is not None:
        sampler = sampler(dataset = dataset, 
            seed=hparams.seed, batch_size=int(hparams.batch_size*batch_scaler))
    else:
        sampler = None
    return sampler

class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        aug_labelled = None,
        aug_unlabelled = None,
        aug_validation = None,
        aug_predict = None,
        aug_targets = None,
        train_labelled_batch_sampler=None,
        train_labelled_sampler=None,
        train_unlabelled_batch_sampler=None,
        train_unlabelled_sampler=None,
        val_batch_sampler=None,
        val_sampler=None,
        num_workers=0,
        batch_size=64,
        batch_size_unlabelled_scalar= 1.0,
        batch_size_validation_scalar= 1.0,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        # ============================
        seed=None,
        labelled_indices_csv=None,
        enable_dual_dataloaders=True,
        random_indices_params=None,
        kfolds=None,
        predict_dataloader_default = 'data_train',
        unlabelled_copies = 1,
        remove_labelled_from_unlabelled = False,
        dataloader_combined_mode = 'min_size',
        shuffle_predict_dataloader = False,
        seed_predict_dataloader = None
        ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if aug_labelled is not None:
            self.aug_labelled = aug_labelled()
        else:
            self.aug_labelled = aug_labelled

        if aug_unlabelled is not None:
            self.aug_unlabelled = aug_unlabelled()
        else:
            self.aug_unlabelled = aug_unlabelled

        if aug_validation is not None:
            self.aug_validation = aug_validation()
        else:
            self.aug_validation = aug_validation

        if aug_predict is not None:
            self.aug_predict = aug_predict()
        else:
            self.aug_predict = aug_predict

        self.aug_targets = aug_targets


    def setup(self, stage=None):
        if dist.is_initialized() and dist.get_world_size() > 1:
            self.batch_size = self.hparams.batch_size//dist.get_world_size()
        else:
            self.batch_size = self.hparams.batch_size

        self.train_labelled_batch_sampler = partial(init_sampler, 
                hparams=self.hparams,
                sampler=self.hparams.train_labelled_batch_sampler)
        self.train_labelled_sampler = partial(init_sampler, 
                hparams=self.hparams,
                sampler=self.hparams.train_labelled_sampler)
        self.train_unlabelled_batch_sampler = partial(init_sampler, 
                hparams=self.hparams,
                sampler=self.hparams.train_unlabelled_batch_sampler)
        self.train_unlabelled_sampler = partial(init_sampler, 
                hparams=self.hparams,
                sampler=self.hparams.train_unlabelled_sampler)
        self.val_batch_sampler = partial(init_sampler, 
                hparams=self.hparams,
                sampler=self.hparams.val_batch_sampler)
        self.val_sampler = partial(init_sampler, 
                hparams=self.hparams,
                sampler=self.hparams.val_sampler)

        if self.hparams.labelled_indices_csv is not None:
            if self.hparams.labelled_indices_csv == '':
                picked_indices = random_indices(self.data_train, self.hparams.random_indices_params, self.hparams.seed)

            else:
                picked_indices = load_indices_from_csv(re.sub('XXXXX', str(self.hparams.seed), self.hparams.labelled_indices_csv))
            
            if self.hparams.enable_dual_dataloaders:
                self.data_train_labelled = MySubset(self.data_train, picked_indices)
                if self.hparams.remove_labelled_from_unlabelled:
                    mask = torch.ones((len(self.data_train)), dtype=torch.bool)
                    mask[picked_indices] = False
                    self.data_train_unlabelled = MySubset(copy.deepcopy(self.data_train), torch.where(mask)[0])
                else:
                    self.data_train_unlabelled = copy.deepcopy(self.data_train)
            else:
                self.data_train = MySubset(self.data_train, picked_indices)

        if self.hparams.kfolds is not None:
            kf = KFold(n_splits=self.hparams.kfolds, random_state=0, shuffle=True) # fix random state
            indices = kf.split(np.arange(len(self.data_train)))
            indices = [[train, test] for train, test in indices]
            if self.data_val == self.data_test:
                self.data_val = MySubset(self.data_train, indices[self.hparams.seed][1])
                self.data_test = MySubset(self.data_train, indices[self.hparams.seed][1])
            else:
                self.data_val = MySubset(self.data_train, indices[self.hparams.seed][1])
            self.data_train = MySubset(self.data_train, indices[self.hparams.seed][0])
            print('kfold '+str(self.hparams.seed))
            print(np.unique(self.data_train.targets, return_counts=True))
            print(np.unique(self.data_val.targets, return_counts=True))

    def _train_dataloader(self):
        droplast = self.hparams.drop_last
        base_dataloader = partial(DataLoader,
            # dataset=dataset,
            # ============================================
            # batch_sampler=self.train_labelled_batch_sampler(dataset),
            # sampler=self.train_labelled_sampler(dataset),
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # ============================================
            # collate_fn=MultiViewCollate(),
            # ============================================
            drop_last=droplast,
            persistent_workers= True if self.hparams.num_workers > 0 else False,
            # persistent_workers= False,
            shuffle=self.hparams.shuffle,
        )
        return base_dataloader

    def train_dataloader(self, dataset=None, unlabelled_dataset=None, 
            base_dataloader=None, combined_mode_override = None):
        if combined_mode_override is not None:
            combined_mode = combined_mode_override
        else:
            combined_mode = self.hparams.dataloader_combined_mode

        if dataset is None:
            dataset = LightlyDataset.from_torch_dataset(self.data_train, self.aug_labelled, self.aug_targets)
        if unlabelled_dataset is None and hasattr(self, 'data_train_unlabelled'):
            unlabelled_dataset = LightlyDataset.from_torch_dataset(self.data_train_unlabelled, self.aug_unlabelled, self.aug_targets)
            dataset = LightlyDataset.from_torch_dataset(self.data_train_labelled, self.aug_labelled, self.aug_targets)
        if base_dataloader is None:
            base_dataloader = self._train_dataloader()

        if unlabelled_dataset is not None:
            dataset = dataset
            labelled_dataloader = copy.deepcopy(base_dataloader)
            labelled_dataloader.keywords['batch_sampler'] = self.train_labelled_batch_sampler(dataset)
            labelled_dataloader.keywords['sampler'] = self.train_labelled_sampler(dataset)
            labelled_dataloader = dataloader_fix_shuffle(labelled_dataloader)
            labelled_dataloader = labelled_dataloader(dataset)
            # ============================================================
            dataset = unlabelled_dataset
            unlabelled_dataloader = copy.deepcopy(base_dataloader)
            unlabelled_dataloader.keywords['batch_sampler'] = self.train_unlabelled_batch_sampler(dataset, batch_scaler=self.hparams.batch_size_unlabelled_scalar)
            unlabelled_dataloader.keywords['sampler'] = self.train_unlabelled_sampler(dataset, batch_scaler=self.hparams.batch_size_unlabelled_scalar)
            unlabelled_dataloader.keywords['batch_size'] = max(int(unlabelled_dataloader.keywords['batch_size']*self.hparams.batch_size_unlabelled_scalar), 1)
            unlabelled_dataloader = dataloader_fix_shuffle(unlabelled_dataloader)
            unlabelled_dataloader = unlabelled_dataloader(dataset)
            if self.hparams.unlabelled_copies == 1:
                base_dataloader = CombinedLoader({'labelled': labelled_dataloader, 
                        'unlabelled': unlabelled_dataloader}, mode=combined_mode)
            elif self.hparams.unlabelled_copies == 2:
                base_dataloader = CombinedLoader({'labelled': labelled_dataloader, 
                        'unlabelled1': copy.deepcopy(unlabelled_dataloader), 'unlabelled2': copy.deepcopy(unlabelled_dataloader)}, mode=combined_mode)
            # base_dataloader = {'labelled': labelled_dataloader, 'unlabelled': unlabelled_dataloader}
        else:
            base_dataloader.keywords['batch_sampler'] = self.train_labelled_batch_sampler(dataset)
            base_dataloader.keywords['sampler'] = self.train_labelled_sampler(dataset)
            base_dataloader = dataloader_fix_shuffle(base_dataloader)
            base_dataloader = base_dataloader(dataset)
        return base_dataloader

    def val_dataloader(self):
        dataset = LightlyDataset.from_torch_dataset(self.data_val, self.aug_validation, self.aug_targets)
        return DataLoader(
            dataset=dataset,
            # ============================================
            batch_sampler=self.val_batch_sampler(dataset, batch_scaler=self.hparams.batch_size_validation_scalar),
            sampler=self.val_sampler(dataset, batch_scaler=self.hparams.batch_size_validation_scalar),
            batch_size=max(int(self.batch_size*self.hparams.batch_size_validation_scalar), 1),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # ============================================
            # collate_fn=MultiViewCollate(),
            # ============================================
            drop_last=False,
            persistent_workers= False,
            shuffle=False,
        )

    def test_dataloader(self):
        dataset = LightlyDataset.from_torch_dataset(self.data_test, self.aug_validation, self.aug_targets)
        return DataLoader(
            dataset=dataset,
            # ============================================
            batch_sampler=self.val_batch_sampler(dataset, batch_scaler=self.hparams.batch_size_validation_scalar),
            sampler=self.val_sampler(dataset, batch_scaler=self.hparams.batch_size_validation_scalar),
            batch_size=max(int(self.batch_size*self.hparams.batch_size_validation_scalar), 1),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # ============================================
            # collate_fn=MultiViewCollate(),
            # ============================================
            drop_last=False,
            persistent_workers= False,
            shuffle=False,
        )

    def _predict_dataloader(self):
        base_dataloader = partial(DataLoader,
            # ============================================
            batch_size=max(int(self.batch_size*self.hparams.batch_size_validation_scalar), 1),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # ============================================
            # collate_fn=MultiViewCollate(),
            # ============================================
            drop_last=False,
            persistent_workers= False,
            shuffle=self.hparams.shuffle_predict_dataloader,
        )
        return base_dataloader

    def predict_dataloader(self, dataset=None, base_dataloader=None):
        if self.hparams.seed_predict_dataloader is not None:
            torch.manual_seed(self.hparams.seed_predict_dataloader)
        if dataset is None:
            dataset = copy.deepcopy(getattr(self, self.hparams.predict_dataloader_default))
            dataset = LightlyDataset.from_torch_dataset(dataset, self.aug_predict, self.aug_targets)
        if base_dataloader is None:
            base_dataloader = self._predict_dataloader()
        base_dataloader.keywords['batch_sampler'] = self.val_batch_sampler(dataset, batch_scaler=self.hparams.batch_size_validation_scalar)
        base_dataloader.keywords['sampler'] = self.val_sampler(dataset, batch_scaler=self.hparams.batch_size_validation_scalar)
        if base_dataloader.keywords['batch_sampler'] is None and base_dataloader.keywords['sampler'] is None:
            if dist.is_available():
                if dist.is_initialized():
                    base_dataloader.keywords['sampler'] = DistributedSampler(dataset)
        base_dataloader = dataloader_fix_shuffle(base_dataloader)
        base_dataloader = base_dataloader(dataset)
        return base_dataloader

    @property
    def num_classes(self):
        if not hasattr(self, 'data_val'):
            self.setup()
        if hasattr(self.data_val, 'classes'):
            return len(self.data_val.classes)
        if hasattr(self.data_val, 'target_labels'):
            return self.data_val.target_labels.shape[0]
        else:
            return np.unique(self.data_val.targets).shape[0]
