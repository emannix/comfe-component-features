import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import os
import PIL.Image
from ..helper import set_base_dir

from torchvision.datasets import DTD 
from torchvision.datasets.utils import download_url, extract_archive
from pdb import set_trace as pb

class MyDTD(DTD):

    def __init__(
        self,
        root: str,
        split: str = "train",
        partition: int = 1,
        grab_all = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.download_root = root
        self.extract_root = set_base_dir(self.download_root)
        self.filename = os.path.basename(self._URL)
        super().__init__(self.extract_root,
            split=split,
            partition=partition,
            transform=transform,
            target_transform=target_transform,
            download=download)

        if grab_all:
            self._image_files = []
            classes = []
            with open(self._meta_folder / f"labels_joint_anno.txt") as file:
                for line in file:
                    cls, name = line.strip().split("/")
                    name = name.split(" ")[0]
                    self._image_files.append(self._images_folder.joinpath(cls, name))
                    classes.append(cls)

            self.classes = sorted(set(classes))
            self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
            self._labels = [self.class_to_idx[cls] for cls in classes]

        self.targets = self._labels

    def _check_exists(self) -> bool:
        return Path(self.download_root, self.filename).exists()

    def _download(self) -> None:
        if not self._check_exists():
            download_url(self._URL, self.download_root, self.filename, self._MD5)
        self._extract()

    def _check_exists_extract(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _extract(self) -> None:
        if self._check_exists_extract():
            return
        archive = os.path.join(self.download_root, self.filename)
        extract_archive(archive, self._base_folder, False)
