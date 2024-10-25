import os
import h5py
import torch
import numpy as np
from PIL import Image
from adict import adict

from mde.datasets.dataset import _Dataset


class TUMDataset(_Dataset):
    def __init__(self, 
                 root, 
                 size=None,
                 near=1e-2,
                 far=10.0,
                 partition="test"):
        super().__init__(root, domain="depth", size=size)
        self.samples = sorted(torch.load(f"{root}/samples_{partition}.pth"))
        # clip space
        self.name = "TUM"
        self.near = near
        self.far = far
       
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, item):
        
        path = os.path.join(self.root, "data", self.samples[item])
        file = h5py.File(path, 'r')
        image = np.array(file.get('image'))
        image = Image.fromarray(image)
        image = _Dataset.resize_image(image, self.size)
        image = np.array(image).astype(np.float32) / 255.
        image = torch.tensor(image).permute(2, 0, 1)

        depth = file.get('depth')
        depth = np.array(depth).astype(np.float32)
        depth = torch.tensor(depth)[None]
        depth = _Dataset.resize_target(depth, self.size)        
        
        mask = (depth > self.near) & (depth <= self.far)

        metadata = adict(file=self.samples[item],
                         domain=self.domain,
                         name=self.name,
                         range=(self.near, self.far))

        return adict(image=image.clamp(0, 1.0), 
                     target=depth.float(), 
                     mask=mask,
                     metadata=metadata)