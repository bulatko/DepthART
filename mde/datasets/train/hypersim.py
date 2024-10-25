
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





def depth_inpaint(depth, mask):
    depth, mask = depth[0], mask[0]
    
    H, W = depth.shape
    
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    points = np.vstack((np.ravel(X[~mask]), np.ravel(Y[~mask]))).T
    values = np.ravel(depth[~mask])
    interpolant = scipy.interpolate.NearestNDInterpolator(points, values)
    depth_filled = interpolant(np.ravel(X), np.ravel(Y)).reshape(X.shape)
    return torch.from_numpy(depth_filled)[None]

def image_inpaint(image, mask):
    if len(mask.shape) == 3:
        mask = mask[0]
    if image.shape[0] == 3:
        image = image.permute(1,2,0)
    if image.max() <= 1:
        image = image * 255
    
    image = image.numpy().astype('uint8')
    mask = (mask * 255).numpy().astype('uint8')
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return torch.from_numpy(inpainted / 255).permute(2, 0, 1)


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
    def __init__(self, root, 
                       height=768, 
                       width=1024, 
                       partition="train", 
                       augmentations=None, 
                       image_augmentations=None) -> None:
        self.root = root
        self.name = "HYPERSIM"
        self.size = (height, width)
        self.near = 1e-4
        self.far = 20

        assert partition in ["train", "val", "test"]
        print(os.path.join(self.root, f"final_{partition}_split.csv"))
        self.samples = pd.read_csv(os.path.join(self.root, f"final_{partition}_split.csv"))
        self.domain = 'depth'
       
        self.partition = partition
        self.augmentations_image = image_augmentations
        self.augmentations = augmentations


    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, item):
        sample = self.samples.iloc[item]
        image_path = os.path.join(
            self.root,
            sample['images']
        )
        distance_path = os.path.join(
            self.root,
            sample['depth']
        )
        image = _Dataset.read_image(image_path, size=self.size)

        distance = np.asarray(h5py.File(distance_path, "r")["dataset"])
        distance = torch.tensor(distance).float()

        depth = distance_to_depth(distance, float(sample["settings_camera_fov"]))
        depth = _Dataset.resize_target(depth[None], self.size)
        mask_depth = torch.isnan(depth)
        
        if mask_depth.sum() != 0:
            if mask_depth.sum() < 1000:
            
                depth = depth_inpaint(depth, mask_depth)
            else:
                raise ValueError("Dataset should not contain samples with more then 1000 nans")
                
        depth = torch.tensor(depth[0])

        if self.augmentations is not None and self.partition == 'train':
            res = self.augmentations(image=image.permute(1,2,0).numpy(), depth=depth[0].numpy())#, mask=mask.numpy()[0])
            image = torch.from_numpy(res['image'])
            depth = torch.from_numpy(res['depth'])
            if self.augmentations_image is not None:
                image = torch.from_numpy(self.augmentations_image(image=image.numpy())['image'])
            image = image.permute(2, 0, 1)
            assert False
        

        if depth.shape[0] != self.size[0] or depth.shape[1] != self.size[1]:
            depth = F.interpolate(depth[None, None], size=self.size, mode="nearest")[0][0]
            image = F.interpolate(image[None], size=self.size, mode="bilinear")[0]
                
        mask_image = torch.isnan(image)
        if mask_image.sum() != 0:
            if mask_image.sum() < 1000:
                image = image_inpaint(image, mask_image)
            else:
                raise ValueError("Dataset should not contain samples with more then 1000 nans")
            
        depth = depth[None]
        mask = torch.ones_like(depth).bool()
        assert depth.isnan().sum() == 0 and depth.isinf().sum() == 0
        
        
        metadata = adict(image_path=image_path,
                         distance_path=distance_path,
                         domain=self.domain,
                         name='hypersim',
                         range=(self.near, self.far))

        return adict(image=image.clamp(0, 1.0), target=depth, metadata=metadata, mask=mask, num_nans_depth=mask_depth.sum())

