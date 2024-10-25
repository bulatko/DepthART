import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

# import dist
from .basic_var import AdaLNBeforeHead, SharedAdaLin, AdaLNSelfAttn, AdaLNCrossAttn
from .helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from .vqvae import VQVAE, VectorQuantizer2

from .utils import *
from torch.nn import functional as F
import safetensors
from .depth_var import DepthVAR

import scipy
class DepthART(DepthVAR):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, 
        norm_eps=1e-6, 
        shared_aln=False, 
        cond_drop_rate=0.1,
        depth=16, 
        embed_dim=1024,  
        drop_path_rate=0.066666666667,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        BlockConfig=None,
        ckpt_path=None,
        ignore_keys={},
        flash_if_available=True, 
        fused_if_available=True, 
        normalizer=None,
        rec_loss=False,
    ):
        super().__init__(vae_local=vae_local,
                         num_classes=num_classes,
                         norm_eps=norm_eps,
                         shared_aln=shared_aln,
                         cond_drop_rate=cond_drop_rate,
                         depth=depth,
                         embed_dim=embed_dim,
                         drop_path_rate=drop_path_rate,
                         patch_nums=patch_nums,
                         BlockConfig=BlockConfig,
                         ckpt_path=ckpt_path,
                         ignore_keys=ignore_keys,
                         flash_if_available=flash_if_available,
                         fused_if_available=fused_if_available,
                         normalizer=normalizer,
                         )

        del self.attn_bias_for_masking
        self.rec_loss = rec_loss

    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, image,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """ 
        # Set labels
        top_k=900
        top_p=0.90
        num_samples=5



        # Set labels
        label_B = torch.full((B,), fill_value=self.num_classes, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(label_B)
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.vae_proxy.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        _, image_var = self.prepare_to_var(image, None)
        
        next_token_map = torch.cat([image_var, next_token_map], axis=1)
        
        for b in self.blocks: b.attn.kv_caching(True)
        
        for si, pn in enumerate(self.patch_nums):
            
            x = next_token_map
            for i, b in enumerate(self.blocks):
                x = b(x=x, cond_BD=cond_BD_or_gss, context=None, attn_bias=None)
            
            logits_BlV = self.get_logits(x[:, -(pn * pn):], cond_BD)
            cur_L += pn * pn
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=None, top_k=top_k, top_p=top_p, num_samples=num_samples)
            h_BChw = self.vae_quant_proxy.embedding(idx_Bl) 
            h_BChw = h_BChw.mean(axis=-2)
            
                
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.vae_proxy.Cvae, pn, pn)
            
            f_hat, next_token_map = self.vae_quant_proxy.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != len(self.patch_nums) - 1:
                next_token_map = next_token_map.view(B, self.vae_proxy.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                
                    
        for b in self.blocks: b.attn.kv_caching(False)
        

        return self.vae_proxy.decode(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

    def forward_train(self, batch):  
        
        image = batch.image

        B = len(image)
        depth = self.normalizer(batch.target).clip(-1, 1).repeat(1, 3, 1, 1).float()
        batch.target_vis = depth[:, :1] / 2 + 0.5


        label_B = torch.full((B,), fill_value=self.num_classes, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(label_B)
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.vae_proxy.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        _, image_var = self.prepare_to_var(image, None)
        
        h = self.vae_proxy.encoder(depth)
        f_hat_target = self.vae_proxy.quant_conv(h)
        next_token_map = torch.cat((image_var, next_token_map), dim=1)
        

        preds = []
        targets = []
        for b in self.blocks: b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            x = next_token_map

            for i, b in enumerate(self.blocks):
                x = b(x=x, cond_BD=cond_BD_or_gss, context=None, attn_bias=None)

            logits_BlV = self.get_logits(x[:, -(pn * pn):], cond_BD)
            if self.rec_loss and si == len(self.patch_nums) - 1:
                h_BChw = F.gumbel_softmax(logits_BlV, tau=1, hard=False)
                h_BChw = h_BChw @ self.vae_quant_proxy.embedding.weight[None]
            else:
                idx_Bl = logits_BlV.argmax(-1) # sample_with_top_k_top_p_(logits_BlV, rng=None, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                h_BChw = self.vae_quant_proxy.embedding(idx_Bl)   # B, l, Cvae
                
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.vae_proxy.Cvae, pn, pn)
            cur_L += pn * pn
            f_hat, new_next_token_map, target = self.vae_quant_proxy.get_next_train_input(si, len(self.patch_nums), f_hat, h_BChw, f_hat_target)
            if self.rec_loss and si == len(self.patch_nums) - 1:
                pass
            else:
                preds.append(logits_BlV)
                targets.append(target.view(B, -1))
            if si != len(self.patch_nums) - 1:   # prepare for next stage
                new_next_token_map = new_next_token_map.view(B, self.vae_proxy.Cvae, -1).transpose(1, 2)
                new_next_token_map = self.word_embed(new_next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = new_next_token_map.detach()
                    

        preds = torch.cat(preds, 1)
        targets = torch.cat(targets, 1)
        if self.rec_loss:
            batch.f_dif = ((f_hat - f_hat_target) ** 2).mean() * 10
        
        batch.target = targets
        batch.pred = preds
        batch.predict_vis = (self.vae_proxy.decode(f_hat) / 2 + 0.5).mean(1, keepdim=True)
        batch.vocab_size = self.vae_proxy.vocab_size
        
            
        return batch   # de-normalize, from [-1, 1] to [0, 1]

