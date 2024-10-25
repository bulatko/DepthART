import os
import cv2
import h5py
import scipy
import torch
import numpy as np
from PIL import Image
from adict import adict
from torch.nn import functional as F

from mde.datasets.dataset import _Dataset
import scipy.interpolate as interpolate
import albumentations as A
import pandas as pd


rules = {'image': 'image', 'depth': 'image', 'mask': 'image'}





def inpaint_depth(depth, mask):
    depth, mask = depth[0], mask[0]
    
    H, W = depth.shape
    
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    points = np.vstack((np.ravel(X[~mask]), np.ravel(Y[~mask]))).T
    values = np.ravel(depth[~mask])
    interpolant = scipy.interpolate.NearestNDInterpolator(points, values)

    depth_filled = interpolant(np.ravel(X), np.ravel(Y)).reshape(X.shape)
    return depth_filled[None]

def inpaint_image(image, mask):
    image = (image * 255.).to(torch.uint8).numpy()
    mask  = (mask[0] * 255).to(torch.uint8).numpy()
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def focal_from_fov(fov, size):
    return size / (2.0 * np.tan(fov / 2.0))


def distance_to_depth(dist, fovx):
    height, width = dist.shape
    focal = focal_from_fov(fovx, size=width)

    uu = torch.linspace((-0.5 * width) + 0.5, 
                        (0.5 * width) - 0.5, 
                        width)
    uu = uu.view(1, -1).repeat(height, 1)
    vv = torch.linspace((-0.5 * height) + 0.5, 
                        (0.5 * height) - 0.5, 
                        height)
    vv = vv.view(-1, 1).repeat(1, width)
    ww = torch.ones(height, width, dtype=torch.float32) * focal

    grid = torch.stack([uu, vv, ww], dim=2)
    return dist * focal / grid.norm(dim=2)



class HypersimDataset(_Dataset):
    def __init__(self, 
                 root,
                 partition='val',
                 domain="depth", 
                 size=None) -> None:
        super().__init__(root=root, domain=domain, size=size)
        #self.samples = torch.load(f"{root}/samples_{partition}.pth")
        self.samples = pd.read_csv(os.path.join(self.root, f"final_{partition}_split.csv"))
        self.root = root
        self.name = "HYPERSIM"
        self.near = 1e-4
        self.far = 20
        

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, item):
        sample = self.samples.iloc[item]

        #image_path = os.path.join(self.root, "data", sample[0])
        image_path = os.path.join(
            self.root,
            sample['images']
        )
        image = _Dataset.read_image(image_path, size=self.size)
        image = inpaint_image(image=image.permute(1, 2, 0), 
                              mask=torch.isnan(image))
        image = torch.tensor(image).permute(2, 0, 1) / 255.
        
        #distance_path = os.path.join(self.root, "data", sample[1])
        distance_path = os.path.join(
            self.root,
            sample['depth']
        )
        distance = np.asarray(h5py.File(distance_path, "r")["dataset"])

        depth = distance_to_depth(distance, float(sample["settings_camera_fov"]))[None]
        depth = _Dataset.resize_target(depth, self.size)
        depth = inpaint_depth(depth=depth, 
                              mask=torch.isnan(depth))
        depth = torch.tensor(depth)
        
        mask = torch.ones_like(depth).bool()
        
        metadata = adict(image_path=image_path,
                         distance_path=distance_path,
                         domain=self.domain,
                         name='hypersim',
                         range=(self.near, self.far))
        
        
        return adict(image=image.clamp(0, 1.0), 
                     target=depth,
                     mask=mask,
                     metadata=metadata)
    