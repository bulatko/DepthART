import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from adict import adict


from mde.datasets.dataset import _Dataset

class NYUv2Dataset(_Dataset):
    def __init__(self, 
                 root,
                 size=None,
                 near=1e-8,
                 far=10.0,
                 partition="test"):
        super().__init__(root, domain="depth", size=size)
        self.images, _ = pickle.load(open(f"{root}/data/nyuv2_{partition}.pkl", "rb"))
        self.depths = pickle.load(open(f"{root}/data/raw_depth_{partition}.pkl", "rb"))

        self.name = "NYUv2"
        self.near = near
        self.far = far
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        image = self.images[item].transpose(2,1,0)[8:-8,8:-8]
        image = Image.fromarray(image)
        image = _Dataset.resize_image(image, size=self.size)
        image = np.array(image).astype(np.float32) / 255.
        image = torch.tensor(image).permute(2, 0, 1)
        
        depth = self.depths[item].transpose(1,0)[8:-8, 8:-8]
        depth = torch.tensor(depth)[None].float()
        depth = _Dataset.resize_target(depth, size=self.size)
        
        mask = (depth > self.near) & (depth <= self.far)

        metadata = adict(id=item,
                         domain=self.domain,
                         name=self.name,
                         range=(self.near, self.far))

        return adict(image=image.clamp(0, 1.0), 
                     target=depth.float(), 
                     mask=mask,
                     metadata=metadata)
    

class NYUv2RawDataset(_Dataset):
    # intinsics of the camera
    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02
    cy = 2.5373616633400465e+02

    def __init__(self, 
                 root, 
                 partition="test_0_01",
                 domain="depth",
                 size=None,
                 crop_border=8,
                 far=10.0,
                 near=1e-4,
                 ) -> None:
        super().__init__(root=root, domain=domain, size=size)
        self.samples = torch.load(f"{root}/samples_{partition}.pth")
        self.crop = crop_border
        self.far = far
        self.near = near

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, item):
        sample = self.samples[item]

        image_path = os.path.join(self.root, "data", sample[0])
        image = torch.tensor(torch.load(image_path)["image"])
        image = image[self.crop:-self.crop, self.crop:-self.crop]
        image = image.float() / 255.
        image = image.permute(2,0,1)

        depth_path = os.path.join(self.root, "data", sample[1])
        depth = torch.tensor(torch.load(depth_path)["depth"].astype(np.float32))
        depth = depth[self.crop:-self.crop, self.crop:-self.crop]
        depth = depth[None, ...]
        depth = depth.float() / (2.0 ** 16 - 1.0) * 10.0


        image = _Dataset.resize_image(image, self.size)
        depth = _Dataset.resize_target(depth, self.size)
        
        mask = (depth > self.near) & (depth <= self.far)

        metadata = adict(image_path=image,
                         depth_path=depth_path,
                         domain=self.domain,
                         name="NYUv2Raw",
                         range=(self.near, self.far))
        return adict(image=image.clamp(0, 1.0), 
                     target=depth, 
                     mask=mask,
                     metadata=metadata)
    
