# https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/ssl.py

import numpy as np
from torchvision.transforms import v2
import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform
import torch

from pdb import set_trace as pb

class ConditionalTransform(Transform):
    def __init__(self, transform, object_type):
        super().__init__()
        self.transform = transform
        self.object_type = object_type

    def __call__(self, img):
        img_list = [isinstance(x, self.object_type) for x in img]
        img_true = [x for x in img if isinstance(x, self.object_type)]
        img_true_auged = self.transform(img_true)
        counter = 0
        for i in np.argwhere(img_list):
            img[i[0]] = img_true_auged[counter]
            counter += 1
        return img

class toTensorv2(Transform):
    def __init__(self):
        super().__init__()
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __call__(self, img):
        return self.transform(img)
    