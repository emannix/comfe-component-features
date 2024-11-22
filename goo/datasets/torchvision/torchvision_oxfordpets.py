import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import os
import PIL.Image
from ..helper import set_base_dir

from torchvision.datasets import OxfordIIITPet 
from torchvision.datasets.utils import download_url, extract_archive
from pdb import set_trace as pb

from PIL import Image

class MyOxfordIIITPet(OxfordIIITPet):

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.download_root = root
        self.extract_root = set_base_dir(self.download_root)
        super().__init__(self.extract_root,
            split=split,
            target_types=target_types,
            transform=transform,
            target_transform=target_transform,
            download=download)
        self.targets = self._labels

    def _check_exists(self) -> bool:
        return Path(self.download_root, self.filenames[0]).exists() and Path(self.download_root, self.filenames[1]).exists() 

    def _download(self) -> None:
        self.filenames = [os.path.basename(url) for url, _ in self._RESOURCES]
        if not self._check_exists():
            i = 0
            for url, md5 in self._RESOURCES:
                download_url(url, self.download_root, self.filenames[i], md5=md5)
                i += 1

        self._extract()

    def _check_exists_extract(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _extract(self) -> None:
        if self._check_exists_extract():
            return
        for filename in self.filenames:
            archive = os.path.join(self.download_root, filename)
            extract_archive(archive, self._base_folder, False)
# ===================

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        # if self.transforms:
        #     image, target = self.transforms(image, target)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
