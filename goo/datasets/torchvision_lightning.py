
from lightning import LightningDataModule
from .base_lightning_module import BaseDataModule

from torchvision import datasets
from pdb import set_trace as pb

class TorchvisionDataModule(BaseDataModule):
    def __init__(
        self,
        TorchvisionDataset = datasets.CIFAR10,
        dataset_params = {},
        split_style = 'bool',
        target_attribute = 'targets',
        data_dir = "data/",
        pre_target_transform=None, **kwargs):
        super().__init__(**kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.pre_target_transform = pre_target_transform
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        if not hasattr(self, 'data_val'):
            test = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, download=True, **self.hparams.dataset_params)
            print(test.classes)
            return len(test.classes)
        else:
            print(self.data_val.classes)
            return len(self.data_val.classes)

    def setup(self, stage = None):
        
        if self.hparams.split_style == 'bool':
            self.data_train = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, train=True, download=True, **self.hparams.dataset_params)
            self.data_val = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, train=False, download=True, **self.hparams.dataset_params)
            self.data_test = self.data_val
        elif self.hparams.split_style == 'trainval':
            self.data_train = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='train', download=True, **self.hparams.dataset_params)
            self.data_val = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='val', download=True, **self.hparams.dataset_params)
            self.data_test = self.data_val
        elif self.hparams.split_style == 'traintest':
            self.data_train = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='train', download=True, **self.hparams.dataset_params)
            self.data_val = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='test', download=True, **self.hparams.dataset_params)
            self.data_test = self.data_val
        elif self.hparams.split_style == 'trainvaltest':
            self.data_train = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='train', download=True, **self.hparams.dataset_params)
            self.data_val = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='val', download=True, **self.hparams.dataset_params)
            self.data_test = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='test', download=True, **self.hparams.dataset_params)
        elif self.hparams.split_style == 'trainvalnovaltest':
            self.data_train = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='trainval', download=True, **self.hparams.dataset_params)
            self.data_val = self.hparams.TorchvisionDataset(root=self.hparams.data_dir, split='test', download=True, **self.hparams.dataset_params)
            self.data_test = self.data_val

        if self.hparams.target_attribute == '_samples':
            pb()
        # =======================================================
        setattr(self.data_train, 'targets', getattr(self.data_train, self.hparams.target_attribute))
        setattr(self.data_val, 'targets', getattr(self.data_val, self.hparams.target_attribute))
        setattr(self.data_test, 'targets', getattr(self.data_test, self.hparams.target_attribute))
        
        if self.pre_target_transform is not None:
            self.data_train.targets = self.pre_target_transform(self.data_train.targets)
            self.data_val.targets = self.pre_target_transform(self.data_val.targets)
            self.data_test.targets = self.pre_target_transform(self.data_test.targets)
        # =======================================================
        super().setup(stage)


