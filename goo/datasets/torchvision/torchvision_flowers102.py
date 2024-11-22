import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import os
import PIL.Image
from ..helper import set_base_dir
import shutil

from torchvision.datasets import Flowers102 
from torchvision.datasets.utils import download_url, extract_archive, check_integrity
from pdb import set_trace as pb

class MyFlowers102(Flowers102):

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
        self.targets = self._labels
        self.classes = np.unique(self.targets)

    def _check_integrity(self) -> bool:
        return Path(self.download_root, self._file_dict['image'][0]).exists()

    def download(self) -> None:
        if not self._check_integrity():
            download_url(
                f"{self._download_url_prefix}{self._file_dict['image'][0]}",
                str(self.download_root),
                self._file_dict['image'][0],
                md5=self._file_dict["image"][1],
            )
            for id in ["label", "setid"]:
                filename, md5 = self._file_dict[id]
                download_url(self._download_url_prefix + filename, str(self.download_root), md5=md5)
        self._extract()

    def _check_exists_extract(self) -> bool:
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def _extract(self) -> None:
        if self._check_exists_extract():
            return
        for key, file in enumerate(self._file_dict):
            archive = os.path.join(self.download_root, self._file_dict[file][0])
            # if not os.path.exists(self._base_folder):
            #     os.mkdir(self._base_folder)
            if file == 'image':
                extract_archive(archive, self._base_folder, False)
            else:
                shutil.copy(archive, self._base_folder)
