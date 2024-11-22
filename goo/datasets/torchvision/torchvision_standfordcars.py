import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import os
import PIL.Image
from ..helper import set_base_dir

import shutil
from torchvision.datasets import StanfordCars 
from torchvision.datasets.utils import download_url, extract_archive
from pdb import set_trace as pb

class MyStandfordCars(StanfordCars):

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        self.download_root = root
        self.extract_root = set_base_dir(self.download_root)
        super().__init__(self.extract_root, 
            split=split,
            transform=transform, 
            target_transform=target_transform,
            download=download)
        self.targets = [x[1] for x in self._samples]

    def _check_exists(self) -> bool:
        return Path(self.download_root, self.filename).exists()

    # https://github.com/pytorch/vision/issues/7545
    def download(self) -> None:
        if self._split == 'train':
            self._MD5 = 'c3b158d763b6e2245038c8ad08e45376'
            self._URL = 'https://ai.stanford.edu/~jkrause/car196/cars_train.tgz'
        else:
            self._MD5 = '4ce7ebf6a94d07f1952d94dd34c4d501'
            self._URL = 'https://ai.stanford.edu/~jkrause/car196/cars_test.tgz'
        # https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset?resource=download

        self.filename = os.path.basename(self._URL)

        # devkit_url = "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
        devkit_url = 'https://github.com/pytorch/vision/files/11644847/car_devkit.tgz'
        devkit_md5 = "c3b158d763b6e2245038c8ad08e45376"
        self.devkit_filename = os.path.basename(devkit_url)

        test_url = 'https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat'
        # test_url = "https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input?select=cars_test_annos_withlabels+\%281\%29.mat"
        test_md5 = 'b0a2b23655a3edd16d84508592a98d10'
        self.test_filename = os.path.basename(test_url)

        if not self._check_exists():
            download_url(devkit_url, self.download_root, self.devkit_filename, devkit_md5)
            download_url(self._URL, self.download_root, self.filename, self._MD5)
            if not self._split == 'train':
                download_url(test_url, self.download_root, self.test_filename, test_md5)
        self._extract()

    def _check_exists_extract(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False
        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

    def _extract(self) -> None:
        if self._check_exists_extract():
            return
        archive = os.path.join(self.download_root, self.filename)
        extract_archive(archive, self._base_folder, False)
        archive = os.path.join(self.download_root, self.devkit_filename)
        extract_archive(archive, self._base_folder, False)
        if not self._split == 'train':
            archive = os.path.join(self.download_root, self.test_filename)
            shutil.copyfile(archive, self._annotations_mat_path)