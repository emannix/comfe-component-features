import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import os
import PIL.Image
from ..helper import set_base_dir

from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from typing import Any, Callable, Dict, List, Optional, Tuple
import collections

from sklearn.preprocessing import MultiLabelBinarizer

from torchvision.datasets import VOCSegmentation, VOCDetection
from torchvision.datasets.voc import DATASET_YEAR_DICT 
from torchvision.datasets.utils import download_url, extract_archive, verify_str_arg
from pdb import set_trace as pb
import numpy as np

from PIL import Image
import torchvision
from torchvision import tv_tensors

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

class MyVOCSegmentation(VOCSegmentation):
    def __init__(
        self,
        root: str,
        year: str = '2012',
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

        self.target_files = self.targets

        target_dir = os.path.join(voc_root, 'Annotations')
        file_names = [os.path.splitext(os.path.basename(x))[0] for x in self.targets] 
        self.target_class_files = [os.path.join(target_dir, x + '.xml') for x in file_names]

        self.targets = [self.parse_voc_xml(ET_parse(x).getroot()) for x in self.target_class_files]
        self.targets = [ np.unique([obj['name'] for obj in x['annotation']['object'] ]) for x in self.targets ]

        self.classes = np.unique(np.concatenate(self.targets + [np.array(['background'])]))

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
            return os.path.isdir(self.voc_root) and Path(self.voc_root, 'ImageSets/Main/val.txt').exists()

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
        img = tv_tensors.Image(img)
        target = self.targets[index]
        target_seg = np.array(Image.open(self.target_files[index]))
        # target_seg[target_seg == 255] = 0
        # target_seg = torchvision.datapoints.Mask(target_seg)
        target_seg = tv_tensors.Mask(target_seg)

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.transform is not None:
            res = self.transform([img, target_seg])
        if len(res) == 2:
            img = res[0]
            target = res[1]
            # ============================
            # temporary target downsample
            # from torchvision.transforms import v2
            # transforms = v2.Compose([
            #         v2.Resize(16, interpolation= torchvision.transforms.InterpolationMode.NEAREST)
            #     ])
            # target_use = transforms(target)
            # target_use = [x.reshape(-1) for x in target_use]
            # torchvision.transforms.ToPILImage()(target).show()
            target_use = target
            # ============================
            if len(target_use) == 1:
                target_use = target_use[0]
            return img, target_use
        else:
            return res

# ==========================================================================

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict