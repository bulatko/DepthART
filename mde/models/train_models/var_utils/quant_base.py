from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

from models.basic_vae import ResidualLayers


# this file only defines the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, 
        vocab_size, 
        Cvae, 
        using_znorm, 
        beta: float = 0.25,
        v_patch_nums=None, 
        resiconfig=None,
        
    ):
        '''
        Input:
            - vocab_size: size of vocab 
            - Cvae: dim of embedding
            - using_znorm: whether to normalize when computing the nearest neighbors
            - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
            - resiconfig:
        
        '''
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        
        # create convolutional residual layers
        self.quant_resi = ResidualLayers(Cvae, **resiconfig).quant_resi
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
        self.create_phi_dx()
        
    def create_phi_dx(self, ):
        self.phi_idx = []
        SN = len(self.v_patch_nums)
        for si in np.arange(len(self.v_patch_nums)):
            self.phi_idx.append(si/(SN-1))
        
    def forward(self, f_BChw: torch.Tensor, return_dict=False, v_patch_nums=None) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        if v_patch_nums is None:
            v_patch_nums = self.v_patch_nums
        
        B, C, H, W = f_BChw.shape
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        
        f_rest = f_BChw.detach().clone()
        f_hat = torch.zeros_like(f_rest)
        
        mean_vq_loss: torch.Tensor = 0.0
        vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
        
        SN = len(self.v_patch_nums)

        if return_dict:
            f_hat_or_idx_Bl = {'fhat': [],
                               'h_BChw': [],
                               'frest': [],
                               'idx': []
                               }
        else:
            f_hat_or_idx_Bl = None

        for si, pn in enumerate(self.v_patch_nums): # from small to large
            # find the nearest embedding
            if self.using_znorm:
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                rest_NC = F.normalize(rest_NC, dim=-1)
                idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
                d_no_grad = torch.sum(rest_NC ** 2, dim=1, keepdim=True) + \
                            torch.sum(self.embedding.weight**2, dim=1, keepdim=False) - 2 * \
                            torch.matmul(rest_NC, self.embedding.weight.t())
                
                # find closest encodings
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, pn, pn)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN - 1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat = f_hat + h_BChw
            if return_dict:
                f_hat_or_idx_Bl['h_BChw'].append(h_BChw.clone())
                f_hat_or_idx_Bl['fhat'].append(f_hat.clone())
                f_hat_or_idx_Bl['frest'].append(f_rest.clone())
                f_hat_or_idx_Bl['idx'].append(idx_N.reshape(B, pn*pn))

            
            f_rest -= h_BChw
                
            # calc loss
            mean_vq_loss += F.mse_loss(f_hat.detach(), f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_BChw.detach())
        
        mean_vq_loss *= 1. / SN
        f_hat = (f_hat.data - f_BChw.detach()).add_(f_BChw)
    

        return f_hat, mean_vq_loss, f_hat_or_idx_Bl
    
    def embed_to_fhat(self, ms_h_BChw, all_to_max_scale=True, last_one=False, v_patch_nums=None) -> Union[List[torch.Tensor], torch.Tensor]:
        if v_patch_nums is None:
            v_patch_nums = np.arange(len(self.v_patch_nums))
        
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si in v_patch_nums: # from small to large
                si = int(si)
                h_BChw = ms_h_BChw[si]
                pn = self.v_patch_nums[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[self.phi_idx[si]](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training (where we'll interpolate every token map to the max scale), so it may cause some training-inference inconsistency
            # WARNING: this should only be used for experimental visualization
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si in v_patch_nums: # from small to large
                si = int(si)
                h_BChw = ms_h_BChw[si]
                pn = self.v_patch_nums[si] # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[self.phi_idx[si]](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                #print(f_hat.shape)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        with torch.cuda.amp.autocast(enabled=False):
            f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
            pn_next: int = self.v_patch_nums[0]
            for si in range(SN - 1):
                if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
                h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
                f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
                pn_next = self.v_patch_nums[si + 1]
                next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]), mode='area')
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat

