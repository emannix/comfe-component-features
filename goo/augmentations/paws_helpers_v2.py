
import torch

import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from pdb import set_trace as pb

from torchvision.transforms.v2.functional import to_pil_image, to_image
from torchvision.transforms import v2

class MyColorJitter(object):
    def __init__(self, s=1.0, p=0.8, align_transform=True):
        color_jitter = v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = v2.RandomApply([color_jitter], p=p)
        self.function = rnd_color_jitter
        self.align_transform = align_transform

    def __call__(self, img):
        if self.align_transform:
            img = self.function(img)
        else:
            for i in range(len(img)):
                img[i] = self.function(img[i])
        return img

class MyRandomGrayscale(object):
    def __init__(self, p=0.8, align_transform=True):
        self.function = v2.RandomGrayscale(p=p)
        self.align_transform = align_transform

    def __call__(self, img):
        if self.align_transform:
            img = self.function(img)
        else:
            for i in range(len(img)):
                img[i] = self.function(img[i])
        return img

class Solarize(object):
    def __init__(self, p=0.2, align_transform=True):
        self.prob = p
        self.align_transform = align_transform

    def __call__(self, img):
        if self.align_transform:
            if torch.bernoulli(torch.tensor(self.prob)) == 0:
                return img
            v = torch.rand(1) * 256
            for i in range(len(img)):
                img_pil = to_pil_image(img[i])
                img_pil = ImageOps.solarize(img_pil, v)
                img[i] = to_image(img_pil)
        else:
            for i in range(len(img)):
                if torch.bernoulli(torch.tensor(self.prob)) == 0:
                    pass
                else:
                    v = torch.rand(1) * 256
                    img_pil = to_pil_image(img[i])
                    img_pil = ImageOps.solarize(img_pil, v)
                    img[i] = to_image(img_pil)
        return img


class Equalize(object):
    def __init__(self, p=0.2, align_transform=True):
        self.prob = p
        self.align_transform = align_transform

    def __call__(self, img):
        if self.align_transform:
            if torch.bernoulli(torch.tensor(self.prob)) == 0:
                return img
            for i in range(len(img)):
                img_pil = to_pil_image(img[i])
                img_pil = ImageOps.equalize(img_pil)
                img[i] = to_image(img_pil)
        else:
            for i in range(len(img)):
                if torch.bernoulli(torch.tensor(self.prob)) == 0:
                    pass
                else:
                    img_pil = to_pil_image(img[i])
                    img_pil = ImageOps.equalize(img_pil)
                    img[i] = to_image(img_pil)
        return img


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2., align_transform=True):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.align_transform = align_transform

    def __call__(self, img):
        if self.align_transform:
            if torch.bernoulli(torch.tensor(self.prob)) == 0:
                return img
            radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
            for i in range(len(img)):
                img_pil = to_pil_image(img[i])
                img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
                img[i] = to_image(img_pil)
        else:
            for i in range(len(img)):
                if torch.bernoulli(torch.tensor(self.prob)) == 0:
                    pass
                radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
                img_pil = to_pil_image(img[i])
                img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
                img[i] = to_image(img_pil)
        return img




