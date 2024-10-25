"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class VQVAE(nn.Module):
    def __init__(
        self, 
        ddconfig,
        quantconfig,
        vocab_size=4096,     
        quant_conv_ks=3,        # quant conv kernel size
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        ckpt_path=None,
        ignore_keys={}
    ):
        super().__init__()

        self.Cvae = ddconfig['z_channels']
        self.vocab_size = vocab_size
        self.v_patch_nums = v_patch_nums
        
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize = VectorQuantizer2(vocab_size=vocab_size, 
                                         Cvae=self.Cvae, 
                                         v_patch_nums=v_patch_nums, 
                                         **quantconfig
                                    )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    
    def forward(self, inp, ret_dict=False):   
        h_BChw, vq_loss, h_dict = self.encode(inp, return_dict=ret_dict)
        if ret_dict:
            dec = [self.decode(f_hat) for f_hat in h_dict['fhat']]
        else:
            dec = self.decode(h_BChw)
        return dec, vq_loss

    def decode(self, x: torch.Tensor):
        quant = self.post_quant_conv(x)
        dec = self.decoder(quant).clamp_(-1, 1) # do we really need clamp?
        return dec


    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def encode(self, x: torch.Tensor, return_dict=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        h, vq_loss, fhat_or_idx = self.quantize(h, return_dict=return_dict) # h: BxCxhxw
        return h, vq_loss, fhat_or_idx
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape=False, last_one=False, v_patch_nums=None) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h=ms_h, all_to_max_scale=same_shape, last_one=last_one, v_patch_nums=v_patch_nums)
    
    def embed_to_img(self, ms_h: List[torch.Tensor], all_to_max_scale: bool, last_one=False, v_patch_nums=None) -> Union[List[torch.Tensor], torch.Tensor]:
        
        x = self.quantize.embed_to_fhat(ms_h, all_to_max_scale=all_to_max_scale, last_one=last_one, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decode(x) #.clamp_(-1, 1)
        else:
            return [self.decode(f_hat) for f_hat in x]
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=True) #["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        # need to find out what is it??
        if sd['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:   # load a pretrained VAE, but using a new num of scales
            assert False
            sd['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        self.load_state_dict(sd, strict=True)
        print(f"VAE Restored from {path}")