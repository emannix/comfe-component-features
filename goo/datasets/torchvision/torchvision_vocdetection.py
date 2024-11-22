import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import os
import PIL.Image
from ..helper import set_base_dir

from sklearn.preprocessing import MultiLabelBinarizer

from torchvision.datasets import VOCDetection 
from torchvision.datasets.voc import DATASET_YEAR_DICT 
from torchvision.datasets.utils import download_url, extract_archive, verify_str_arg
from pdb import set_trace as pb
import numpy as np

from PIL import Image

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

class MyVOCDetection(VOCDetection):

    def __init__(
        self,
        root: str,
        year: str = '2007',
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        image_set=split
        # ==========================================
        self.download_root = root
        self.extract_root = set_base_dir(self.download_root)
        self.root = self.extract_root
        # ==========================================
        self.year = verify_str_arg(year, "year", valid_values=[str(yr) for yr in range(2007, 2013)])

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]
        self.key = key

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)
        self.voc_root = voc_root
        # ==========================================
        if download:
            self._download()
        # ==========================================
        super().__init__(self.extract_root,
            year=year,
            image_set=image_set,
            transform=transform, 
            target_transform=target_transform,
            transforms=transforms,
            download=False)

        self.targets = [self.parse_voc_xml(ET_parse(x).getroot()) for x in self.annotations]
        self.targets = [ np.unique([obj['name'] for obj in x['annotation']['object'] ]) for x in self.targets ]
        self.classes = np.unique(np.concatenate(self.targets))

        mlb = MultiLabelBinarizer()
        self.targets = mlb.fit_transform(self.targets)
        self.classes =  mlb.classes_

    def _check_exists(self) -> bool:
        return Path(self.download_root, self.filename).exists()

    def _download(self) -> None:
        if not self._check_exists():
            download_url(self.url, self.download_root, self.filename, self.md5)
        self._extract()

    def _check_exists_extract(self) -> bool:
        if self.key == '2007':
            return os.path.isdir(self.voc_root) and Path(self.voc_root, 'ImageSets/Main/val.txt').exists()
        elif self.key == '2007-test':
            return os.path.isdir(self.voc_root) and Path(self.voc_root, 'ImageSets/Main/test.txt').exists()
        else:
            pb()

    def _extract(self) -> None:
        if self._check_exists_extract():
            return
        archive = os.path.join(self.download_root, self.filename)
        extract_archive(archive, self.extract_root, False)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.targets[index]
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target