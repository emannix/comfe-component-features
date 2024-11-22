import numpy as np
import torchvision.transforms.functional as F

import importlib
matplotlib_present = importlib.util.find_spec("matplotlib")
if matplotlib_present is not None:
    import matplotlib.pyplot as plt

    plt.rcParams["savefig.bbox"] = 'tight'

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show(block=False)

import torch
import numpy as np
import matplotlib.colors as mcolors
import torchvision.transforms as T
from pdb import set_trace as pb
from PIL import Image

def visualize_mask(img):
    integer_image = img.cpu().clone().detach()
    unique_vals, inverse_indices = torch.unique(integer_image, return_inverse=True)
    integer_image = inverse_indices

    # Determine the number of unique classes
    num_classes = torch.unique(integer_image).numel()
    num_classes = integer_image.max().int()+1

    # Generate color palette in HSV and convert to RGB
    hues = np.linspace(0, 1, num_classes, endpoint=False)
    palette_hsv = [(hue, 1, 1) for hue in hues]  # Full saturation and value
    palette_rgb = [mcolors.hsv_to_rgb(color) for color in palette_hsv]
    palette_rgb = (np.array(palette_rgb) * 255).astype(np.uint8)

    # Map the integer image to an RGB image using the palette
    rgb_image = np.array(palette_rgb[integer_image.numpy()])

    # Convert the RGB image to a PIL image
    to_pil_image = T.ToPILImage()
    pil_image = to_pil_image(torch.ByteTensor(rgb_image).permute(2, 0, 1))
    return pil_image

def visualize_image(img):
    raw_image = img.cpu().clone().detach()
    to_pil_image = T.ToPILImage()
    return to_pil_image(raw_image)

def overlay_mask(img, mask):
    img1 = visualize_mask(mask)
    if not isinstance(img, Image.Image):
        img2 = T.ToPILImage()(img)
    else:
        img2 = img
    img2.putalpha(127)
    img3 = Image.alpha_composite(img1.convert('RGBA'), img2.convert('RGBA'))
    return img3

# ===========================================================

def visualize_mask_fixed(img, classes= [0,1,2,3,4,5], binary=False, background=False):
    integer_image = img.cpu().clone().detach()

    # Generate color palette in HSV and convert to RGB
    hues = np.linspace(0, 1, len(classes), endpoint=False)
    palette_hsv = [(hue, 1, 1) for hue in hues]  # Full saturation and value
    palette_rgb = [mcolors.hsv_to_rgb(color) for color in palette_hsv]
    palette_rgb = (np.array(palette_rgb) * 255).astype(np.uint8)

    # Map the integer image to an RGB image using the palette
    rgb_image = np.array(palette_rgb[integer_image.numpy()])
    if binary:
        alpha_array = (integer_image == integer_image.min()).float()*127
        rgb_image = np.concatenate([rgb_image, alpha_array[:,:,None]], axis=2)
    if background:
        alpha_array = (integer_image != classes[-1]).float()*127
        rgb_image = np.concatenate([rgb_image, alpha_array[:,:,None]], axis=2)

    # Convert the RGB image to a PIL image
    to_pil_image = T.ToPILImage()
    pil_image = to_pil_image(torch.ByteTensor(rgb_image).permute(2, 0, 1))
    return pil_image

def overlay_mask_fixed(img, mask, classes, binary=False, background=False):
    img1 = visualize_mask_fixed(mask, classes, binary, background)
    if not isinstance(img, Image.Image):
        img2 = T.ToPILImage()(img)
    else:
        img2 = img
    if not binary and not background:
        img1.putalpha(127)
    img3 = Image.alpha_composite(img2.convert('RGBA'), img1.convert('RGBA'))
    return img3