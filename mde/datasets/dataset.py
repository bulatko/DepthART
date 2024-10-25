import torch
import logging
import numpy as np
from PIL import Image
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class _Dataset:
    def __init__(self,
                 root,
                 domain="depth",
                 size=None) -> None:
        self.root = root
        self.domain = domain

        if size is not None:
            self.size = tuple(size)
        else:
            self.size = None

    @classmethod
    def read_image(cls, path, size=None):
        image = _Dataset.resize_image(Image.open(path), size)
        image = np.array(image).astype(np.float32) / 255.
        image = torch.tensor(image).permute(2, 0, 1)
        return image
    
    @classmethod
    def read_target(cls, path, size=None, scale=1.0):
        target = None
        if path.lower().endswith(".png") or path.lower().endswith(".jpg"):
            target = np.array(Image.open(path)).astype(np.float32)
        if path.lower().endswith(".npy"):
            target = np.load(path).astype(np.float32)
        if path.lower().endswith(".pth"):
            target = torch.load(path).float()

        if target is None:
            logger.error(f"read_target() got unknown extension.")
        else:
            target = torch.tensor(target) / scale

        if len(target.shape) == 2:
            target = target[None]
        
        return _Dataset.resize_target(target, size)

    @classmethod
    def resize_image(cls, image, size):
        if size is None:
            return image
        
        if isinstance(image, torch.Tensor):
            assert image.ndim == 3, "Image tensor has invalid shape."
            return F.interpolate(input=image[None],
                                 size=size,
                                 mode="bicubic")[0]
        try:
            return image.resize(size=reversed(size), 
                                resample=Image.BICUBIC)
        except:
            raise TypeError("Image has invalid type.")

        
    @classmethod
    def resize_target(cls, target, size):
        if size is None:
            return target
        
        if isinstance(target, torch.Tensor):
            assert target.ndim == 3, "Target tensor has invalid shape."
            return F.interpolate(input=target[None], 
                                 size=size, 
                                 mode="nearest")[0]
        
        raise TypeError("Target has invalid type.")
    


        
