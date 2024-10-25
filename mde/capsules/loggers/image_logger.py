import numpy as np
import sys

from rocket.core.capsule import Capsule, Attributes
import rocket
from accelerate import Accelerator


class ImageLogger(rocket.Capsule):
    def __init__(self, 
                 accelerator: Accelerator = None, 
                 priority: int = 1000, save_every=10, mode='train') -> None:  # higher priority then optimizer
        super().__init__(accelerator=accelerator, 
                         statefull=False,
                         priority=priority)
        self.step = 0
        self.save_every = save_every
        self.mode = mode


    def launch(self, attrs: Attributes = None):
        # if no attributes provided, nothing to do
        if attrs is None:
            return
        
        # if no batch provided, nothing to do
        if attrs.batch is None:
            return
        
       
        pred_depth = attrs.batch.predict_vis.cpu()
        real_depth = attrs.batch.target_vis.cpu()

        # if pred_depth.min() < 0:
        #     pred_depth = pred_depth * 2 - 1
        # if real_depth.min() < 0:
        #     real_depth = real_depth * 2 - 1
        
        if self._accelerator.sync_gradients and self.step % self.save_every == 0:
            # send log into trackers and reset
            if attrs.tracker is not None:
                attrs.tracker.images.update({f'{self.mode}_depth_pred': pred_depth,
                                             f'{self.mode}_depth_real': real_depth,
                                             f'{self.mode}_image': attrs.batch.image.cpu()})

        self.step += 1 