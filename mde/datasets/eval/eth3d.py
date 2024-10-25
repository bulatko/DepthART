import os
import torch
from adict import adict

from mde.datasets.dataset import _Dataset


class ETH3DDataset(_Dataset):
    def __init__(self, 
                 root, 
                 size=None,
                 near=1e-4,
                 far=72.0, 
                 partition="test"):
        super().__init__(root, domain="depth", size=size)
        self.samples = sorted(torch.load(f"{root}/samples_{partition}.pth"))
        # clip space
        self.name = "ETH3D"
        self.near = near
        self.far = far

    def __len__(self):
        return  len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]

        image_path = os.path.join(self.root, "data", sample[0])
        image = _Dataset.read_image(image_path, size=self.size)
        
        depth_path = os.path.join(self.root, "data", sample[1])
        depth = _Dataset.read_target(depth_path, size=self.size, scale=1000.0)    
        
        # valid pixels
        mask = (depth > self.near) & (depth <= self.far)
        
        metadata = adict(image_path=image_path,
                         depth_path=depth_path,
                         domain=self.domain,
                         name=self.name,
                         range=(self.near, self.far))
        #print(image.shape, depth.shape)
        return adict(image=image.clamp(0, 1.0), 
                     target=depth, 
                     mask=mask,
                     metadata=metadata)