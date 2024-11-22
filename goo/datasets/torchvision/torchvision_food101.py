import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import os
import PIL.Image
from ..helper import set_base_dir

from torchvision.datasets import Food101 
from torchvision.datasets.utils import download_url, extract_archive
from pdb import set_trace as pb

class MyFood101(Food101):

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
        self.filename = os.path.basename(self._URL)
        super().__init__(self.extract_root, 
            split=split,
            transform=transform, 
            target_transform=target_transform,
            download=download)
        self.targets = self._labels

    def _check_exists(self) -> bool:
        return Path(self.download_root, self.filename).exists()

    def _download(self) -> None:
        if not self._check_exists():
            download_url(self._URL, self.download_root, self.filename, self._MD5)
        self._extract()

    def _check_exists_extract(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _extract(self) -> None:
        if self._check_exists_extract():
            return
        archive = os.path.join(self.download_root, self.filename)
        extract_archive(archive, self.extract_root, False)
