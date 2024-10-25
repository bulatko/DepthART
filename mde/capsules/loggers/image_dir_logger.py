import numpy as np
import sys

from rocket.core.capsule import Capsule, Attributes
import rocket
from accelerate import Accelerator
import cv2
import os
import matplotlib.pyplot as plt



class ImageDirLogger(rocket.Capsule):
    def __init__(self, 
                 accelerator: Accelerator = None, 
                 priority: int = 1000, save_every=5, save_dir='outputs/var_self/', mode='train') -> None:  # higher priority then optimizer
        super().__init__(accelerator=accelerator, 
                         statefull=False,
                         priority=priority)
        self.step = 0
        self.save_every = save_every
        self.mode = mode
        self.save_dir = save_dir


    def launch(self, attrs: Attributes = None):
        # if no attributes provided, nothing to do
        if attrs is None:
            return
        
        # if no batch provided, nothing to do
        if attrs.batch is None:
            return
        
       
        pred_depth = attrs.batch.predict_vis.cpu()
        real_depth = attrs.batch.target_vis.cpu()

        os.makedirs(os.path.join(self.save_dir, attrs.batch.metadata.name[0], "pred"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, attrs.batch.metadata.name[0], "real"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, attrs.batch.metadata.name[0], "image"), exist_ok=True)

        
        if self._accelerator.sync_gradients and self.step % self.save_every == 0:
            plt.imsave(os.path.join(self.save_dir, attrs.batch.metadata.name[0], "image", f"image_{self.step}.png"), attrs.batch.image[0].permute(1,2,0).cpu().numpy().astype(np.float64))

            plt.imsave(os.path.join(self.save_dir, attrs.batch.metadata.name[0], "pred", f"image_{self.step}.png"), pred_depth[0].mean(axis=0).cpu())
            plt.imsave(os.path.join(self.save_dir, attrs.batch.metadata.name[0], "real", f"image_{self.step}.png"), real_depth[0].mean(axis=0).cpu())
       
        self.step += 1 