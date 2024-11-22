from torchvision import transforms as T
from pdb import set_trace as pb

from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

imagenet_normalize = IMAGENET_NORMALIZE

class CenterCropAugment(MultiViewTransform):
    def __init__(
        self,
        crop_size = 224, resize=256, stretch=False, normalize=imagenet_normalize, views=1, **kwargs
    ):
        if stretch:
            resize_use = (resize, resize)
        else:
            resize_use = resize

        transform =  [
                T.Resize(size=resize_use,
                    interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(size=crop_size) if crop_size is not None else None,
                T.ToTensor()
            ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        transform = T.Compose([x for x in transform if x is not None])

        super().__init__(transforms=[transform for x in range(views)])
