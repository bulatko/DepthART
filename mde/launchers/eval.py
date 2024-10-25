import torch
import torch.nn.functional as F
import rocket
from rocket.core.capsule import Attributes

from mde.capsules.aligner import *
from mde.capsules.loggers.image_dir_logger import ImageDirLogger


class PredictionUpsampler(rocket.Capsule):
    def __init__(self):
        super().__init__()

    def launch(self, attrs=None):
        if attrs is None or attrs.batch is None:
            return

        b, _, h, w = attrs.batch.target.shape
        attrs.batch.predict = F.interpolate(
            attrs.batch.predict,
            size=(h, w),
            mode='bilinear',
            align_corners=True
        )


def make_eval(datasets, model, metrics, accelerator, aligner=None, **data_kwargs):
    capsules = list()
    
    for dataset in datasets:
        meter_capsules = [PredictionUpsampler()]
        if aligner is not None:
            meter_capsules += [aligner]
        meter_capsules += metrics
        meter = rocket.Meter(capsules=meter_capsules,
                             keys=["target", "predict", "mask"])
        
        capsules.append(
            rocket.Looper(capsules=[
                rocket.Dataset(dataset=dataset, 
                            device_placement=True,
                            **data_kwargs),
                rocket.Module(module=model),
                ImageDirLogger(save_every=1, mode='val'),
                meter,
                rocket.Tracker()
            ], grad_enabled=False, tag=dataset.name)
        )
    
    return rocket.Launcher(capsules=capsules, accelerator=accelerator)