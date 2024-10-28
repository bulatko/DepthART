import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn

from logging import getLogger, DEBUG
import safetensors

logger = getLogger(__name__)
logger.setLevel(DEBUG)



class VARFull(nn.Module):
    def __init__(self, model, resume=None, visualize=True, de=True):
        '''
        Adapting model for depth_estimation tool
        
        '''
        super().__init__()

        # init models
        self.model = model

        self.domain = 'depth'
        self.visualize = visualize
        self.normalizer = self.model.normalizer
        self.model.de = de



    @torch.no_grad()
    def forward(self, batch):
        
        outputs = self.model(batch).predict
        prediction = torch.nn.functional.interpolate(
            outputs,
            size=(batch.image.shape[-2], batch.image.shape[-1]),
            mode="bicubic",
            align_corners=False,
        )

        batch.metadata.predict_domain = [self.domain] * prediction.shape[0]
        
        batch.predict = prediction
        
        if self.visualize:
            if self.normalizer is not None:
                batch.predict_vis = self.normalizer(prediction.detach().clone()).clip(0, 1)
                batch.target_vis = self.normalizer(batch.target.detach().clone()).clip(0, 1)
            else:
                batch.predict_vis = prediction.detach()
                batch.target_vis = batch.target.detach()
        return batch
        