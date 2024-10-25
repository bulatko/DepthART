import torch
import rocket
from functools import partial

from mde.capsules.aligner.aligners_fn import *


class Aligner(rocket.Capsule, torch.nn.Module):
    def __init__(self, align_fn):
        super().__init__()
        self.align_fn = align_fn


    @torch.no_grad()
    def forward(self, predict, target, mask=None):
        if mask is None:
            mask = torch.ones_like(predict).bool()
        
        mask = mask.bool()

        assert predict.ndim == 4, "Alignment expects 4 dim predict tensor"
        assert target.ndim == 4, "Alignment expects 4 dim target tensor"
        assert mask.ndim == 4, "Alignment expects 4 dim mask tensor"

        predicts = torch.split(predict, 1)
        targets = torch.split(target, 1)
        masks = torch.split(mask, 1)

        aligned_predicts = list()
        for predict, target, mask in zip(predicts, targets, masks):
            aligned_predicts.append(
                self.align_fn(predict, target, mask)
            )
        aligned_predict = torch.cat(aligned_predicts, dim=0)

        return aligned_predict
    

    def launch(self, attrs=None):
        if attrs is None or attrs.batch is None:
            return

        target = attrs.batch.target
        predict = attrs.batch.predict
        mask = attrs.batch.mask

        aligned_predict = self.forward(predict, target, mask)
        
        attrs.batch.predict = aligned_predict


class AlignerSSIL2(Aligner):
    def __init__(self):
        super().__init__(align_fn=align_ssi_l2)

class AlignerSSIL1(Aligner):
    def __init__(self):
        super().__init__(align_fn=align_ssi_l1)

class AlignerSIL2(Aligner):
    def __init__(self):
        super().__init__(align_fn=align_si_l2)

class AlignerSIL1(Aligner):
    def __init__(self):
        super().__init__(align_fn=align_si_l1)

class AlignerSIL1Log(Aligner):
    def __init__(self):
        super().__init__(align_fn=partial(align_si_l1log, log_domain=False))