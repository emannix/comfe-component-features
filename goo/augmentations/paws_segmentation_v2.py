from PIL import ImageFilter, ImageOps
import random
from torchvision.transforms import v2
from PIL import Image
from ..lightly.multi_view_transform_v2 import MultiViewTransform
import numpy as np
import torch

from .paws_helpers_v2 import Solarize, Equalize, GaussianBlur, MyColorJitter, MyRandomGrayscale
from .torchvision_v2_helpers import ConditionalTransform, toTensorv2
from torchvision import tv_tensors
import torchvision

from pdb import set_trace as pb

def get_color_distortion(s=1.0, grayscale = False, solarize = True, equalize = True, align_transform=True):
    # s is the strength of color distortion.
    rnd_color_jitter = MyColorJitter(s=s, p=0.8, align_transform=align_transform)
    rnd_gray = MyRandomGrayscale(p=0.2, align_transform=align_transform)

    base_distort = [rnd_color_jitter]
    if grayscale:
        base_distort.append(rnd_gray)
    if solarize:
        base_distort.append(Solarize(p=0.2, align_transform=align_transform))
    if equalize:
        base_distort.append(Equalize(p=0.2, align_transform=align_transform))

    color_distort = v2.Compose(base_distort)
    return color_distort

class DataAugmentationPAWS(MultiViewTransform):
    def __init__(self, 
        global_crops_scale=(0.75, 1.0), 
        local_crops_scale=(0.3, 0.75), 
        global_crops_number=2, 
        local_crops_number=8, 
        global_crop_size=32,
        local_crop_size=18,
        color_distortion_s = 0.5,
        color_distortion_grayscale = False,
        color_distortion_solarize = True,
        color_distortion_equalize = True,
        blur_prob = 0.0,
        normalize = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
        static_seed=None, masks=True, **kwargs):
        self.static_seed = static_seed
        align_transform = False
        self.masks = masks

        global_transform = [
             v2.RandomResizedCrop(size=global_crop_size, scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BILINEAR),
             v2.RandomHorizontalFlip(),
             ConditionalTransform(get_color_distortion(color_distortion_s, color_distortion_grayscale, 
                    color_distortion_solarize, color_distortion_equalize, align_transform=align_transform), tv_tensors.Image),
             ConditionalTransform(GaussianBlur(p=blur_prob, align_transform=align_transform), tv_tensors.Image),
             ConditionalTransform(toTensorv2(), tv_tensors.Image),
        ]

        local_tranform = [
             v2.RandomResizedCrop(size=local_crop_size, scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BILINEAR),
             v2.RandomHorizontalFlip(),
             ConditionalTransform(get_color_distortion(color_distortion_s, color_distortion_grayscale, 
                    color_distortion_solarize, color_distortion_equalize, align_transform=align_transform), tv_tensors.Image),
             ConditionalTransform(GaussianBlur(p=blur_prob, align_transform=align_transform), tv_tensors.Image),
             ConditionalTransform(toTensorv2(), tv_tensors.Image),
        ]

        if normalize:
            global_transform += [ConditionalTransform(v2.Normalize(mean=normalize["mean"], std=normalize["std"]), tv_tensors.Image)]
            local_tranform += [ConditionalTransform(v2.Normalize(mean=normalize["mean"], std=normalize["std"]), tv_tensors.Image)]

        global_transform = v2.Compose(global_transform)
        local_tranform = v2.Compose(local_tranform)

        transform_list=[global_transform, local_tranform]
        copies_list = [global_crops_number, local_crops_number]

        super().__init__(transforms=transform_list, copies=copies_list)

    def __call__(self, image):
        if self.static_seed is not None:
            torch.manual_seed(self.static_seed)
            np.random.seed(self.static_seed)
            random.seed(self.static_seed)
            torch.cuda.manual_seed_all(self.static_seed)

        if not self.masks:
            image = tv_tensors.Image(image)

        transform_list = []
        for i in range(len(self.transforms)):
            # check to see how augmentations behave when showing multiple identical images
            if self.copies[i] > 0:
                image_list = [image for x in range(self.copies[i])]
                if isinstance(image_list[0], list):
                    image_list = sum(image_list, [])
                transform_list += [self.transforms[i](image_list)]
                # torchvision.transforms.functional.to_pil_image(transform_list[0][0]).show()

        transform_list = sum(transform_list, [])
        if self.masks:
            images = transform_list[0::2]
            masks = transform_list[1::2]
            return [images, masks]
        else:
            return transform_list
