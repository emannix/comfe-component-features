from typing import List, Sequence, Union

from PIL.Image import Image
from torch import Tensor
from torchvision import transforms as T

from pdb import set_trace as pb


class MultiViewTransform:
    """Transforms an image into multiple views.

    Args:
        transforms:
            A sequence of transforms. Every transform creates a new view.

    """
    def __init__(self, transforms: Sequence[T.Compose], copies: List[int], static_seed=None):
        self.transforms = transforms
        self.copies = copies
        self.static_seed = static_seed

    def __call__(self, image):
        if self.static_seed is not None:
            torch.manual_seed(self.static_seed)
            np.random.seed(self.static_seed)
            random.seed(self.static_seed)
            torch.cuda.manual_seed_all(self.static_seed)

        transform_list = []
        for i in range(len(self.transforms)):
            # check to see how augmentations behave when showing multiple identical images
            if self.copies[i] > 0:
                image_list = [image for x in range(self.copies[i])]
                image_list = sum(image_list, [])
                transform_list += [self.transforms[i](image_list)]

        transform_list = sum(transform_list, [])
        images = transform_list[0::2]
        masks = transform_list[1::2]

        return [images, masks]