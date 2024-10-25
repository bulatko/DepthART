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
import sys
from torch.nn import functional as F
import safetensors

from copy import deepcopy



from pprint import pformat
from typing import Optional, Tuple, Union, List, Dict




def filter_params(model, nowd_keys=()) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        if not 'vae_proxy' in name and not 'ae_quant_proxy' in name:
            name = name.replace('_fsdp_wrapped_module.', '')
            if not para.requires_grad:
                names_no_grad.append(name)
                continue  # frozen weights
            count += 1
            numel += para.numel()
            names.append(name)
            paras.append(para)
            
            if para.ndim == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
                cur_wd_sc, group_name = 0., 'ND'
            else:
                cur_wd_sc, group_name = 1., 'D'
            cur_lr_sc = 1.
            if group_name not in para_groups:
                para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
                #para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups[group_name]['params'].append(para)
            #para_groups_dbg[group_name]['params'].append(name)
        else:
            para.requires_grad = False
    
    
    assert len(names_no_grad) == 0, f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n'
    return list(para_groups.values())

    
class DepthVAR(nn.Module):
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
    ):
        super().__init__()

        assert vae_local.v_patch_nums == patch_nums
        assert embed_dim % BlockConfig['num_heads'] == 0
        
        self.cond_drop_rate = cond_drop_rate
        self.embed_dim = embed_dim
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2

        
        self.vae_proxy = vae_local
        self.vae_quant_proxy = vae_local.quantize

        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.vae_proxy.Cvae, embed_dim)
        
        # 2. class embedding
        init_std = math.sqrt(1 / embed_dim / 3)
        self.num_classes = num_classes
        self.class_emb, self.pos_start = class_embedding(num_classes, init_std, embed_dim, self.first_l)

        # 3. absolute position embedding
        self.lvl_embed, self.pos_1LC = level_ebmedding(self.patch_nums, embed_dim, init_std)
    
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*embed_dim)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=embed_dim, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=embed_dim, norm_layer=norm_layer, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                flash_if_available=flash_if_available, 
                fused_if_available=fused_if_available,
                **BlockConfig
            ) 
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L

        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(self.L, self.L)
        

        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(embed_dim, embed_dim, norm_layer=norm_layer)
        self.head = nn.Linear(embed_dim, self.vae_proxy.vocab_size)

        # normalizer for depth estimation
        self.normalizer = normalizer
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        top_attn_bias_for_masking = torch.cat([torch.zeros_like(attn_bias_for_masking), torch.zeros_like(attn_bias_for_masking) - torch.inf], dim=1)
        top_attn_bias_for_masking[:, self.L] = 0.
        bot_attn_bias_for_masking = torch.cat([torch.zeros_like(attn_bias_for_masking), attn_bias_for_masking], dim=1)
        attn_bias_for_masking = torch.cat([top_attn_bias_for_masking, bot_attn_bias_for_masking], dim=0)

        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        

    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, image
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

            ratio = si / (len(self.patch_nums) - 1)
            
            x = next_token_map
            for i, b in enumerate(self.blocks):
                x = b(x=x, cond_BD=cond_BD_or_gss, context=None, attn_bias=None)
            
            logits_BlV = self.get_logits(x[:, -(pn * pn):], cond_BD)
            cur_L += pn * pn
            
            idx_Bl = logits_BlV.argmax(-1)
            h_BChw = self.vae_quant_proxy.embedding(idx_Bl)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.vae_proxy.Cvae, pn, pn)
            
            f_hat, next_token_map = self.vae_quant_proxy.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != len(self.patch_nums) - 1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.vae_proxy.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                
                    
        for b in self.blocks: b.attn.kv_caching(False)

        return self.vae_proxy.decode(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    

    def get_parameters(self, ):
        para_groups = filter_params(self, nowd_keys={
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
        })

        
        return sum([a['params'] for a in para_groups], [])

    def prepare_to_var(self, image, sos):
        '''
        Prepare image and depth to var input
        Input:
            - image: torch.Tensor: BxCXHxW - depth or image input
            - sos: encoded class
            - ed: 

        return:
        
        '''
        B = image.shape[0]
        
        if image.shape[1] == 1:
            data_type = 'depth'
            image = image.repeat(1, 3, 1, 1)
        else:
            data_type = 'image'

        h_dict = self.encode_idx(image)

        if data_type == 'depth':
            real_idx_Bl = self.vae_proxy.img_to_idxBl(image.float())
            target = torch.cat(real_idx_Bl, dim=1)
            input_var = self.vae_quant_proxy.idxBl_to_var_input(real_idx_Bl)
            input_var = torch.cat((sos, self.word_embed(input_var.float())), dim=1)
        else:
            input_var = []
            
            for i, h_hat in enumerate(h_dict['fhat']):
                input_var.append(F.interpolate(h_hat, size=self.vae_proxy.v_patch_nums[i], mode='bicubic').contiguous().view(B, h_hat.shape[1], -1).transpose(1, 2))


            input_var = torch.cat(input_var, dim=1)
            input_var = self.word_embed(input_var.float())
            target = None

        input_var += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC # lvl: BLC;  pos: 1LC
        
        return target, input_var

    
    def encode_idx(self, image):
        '''
        Encode image to idx and different scale vectors
        Input:
            - image: torch.tensor, Bx3xHxW
        
        return:
            h_dict:
                'idx': List[torch.Tensor] - 
                'fhat': List[torch.Tensor]
        
        '''
        
        if image.min() >= 0:
            image = image * 2 - 1

        B = image.shape[0]
        
        assert image.min() >= -1 and image.max() <= 1 

        _, _, h_dict = self.vae_proxy.encode(image.float(), return_dict=True)

        return h_dict
    

    def forward(self, batch): 
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_infer(batch)

    def forward_train(self, batch):  
        """
        train step
        Input:
            - batch dict:
                - image: torch.Tensor, Bx3xHxW
                - depth: torch.Tensor, Bx1xHxW
                - mask: torch.Tensor, Bx1xHxW - bool mask, True - depth exists
                - metadata: dict



        :return: 
            - batch:

        """
        

        image = batch.image

        B = len(image)
        depth = self.normalizer(batch.target).clip(-1, 1)
        batch.target_vis = depth.repeat(1, 3, 1, 1) / 2 + 0.5


        label_B = torch.full((B, 1), fill_value=self.num_classes, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(label_B)
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        _, image_var = self.prepare_to_var(image, None)
        target, depth_var = self.prepare_to_var(depth, sos)
        depth_var = torch.cat([image_var, depth_var], dim=1)
        x = depth_var
        for i, b in enumerate(self.blocks):
            x = b(x=x, cond_BD=cond_BD_or_gss, context=None, attn_bias=self.attn_bias_for_masking)
        x = x[:, self.L:]
        preds = self.get_logits(x, cond_BD)
        batch.target = target
        batch.pred = preds
        batch.predict_vis = (self.logits_to_image(preds) / 2 + 0.5).mean(1, keepdim=True)
        batch.vocab_size = self.vae_proxy.vocab_size
            
        return batch
    
    def forward_infer(self, batch)  -> torch.Tensor:
        img = batch.image
        recon_B3HW = self.autoregressive_infer_cfg(B=len(img), image=img)
        depth = recon_B3HW.mean(1, keepdim=True) # Bx1xHxW

        batch.predict = depth
        batch.predict_vis = recon_B3HW
        batch.target_vis = self.normalizer(batch.target.clone()).clip(-1, 1) / 2 + 0.5

        if 'depth' in batch:
            batch.target = batch.depth.float()
        else:
            batch.target = batch.target.float()

        return batch
        

    def logits_to_image(self, logits: torch.Tensor, sizes=None, top_k=5, top_p=0.99):
        '''
        Encode predicted logits to image
        Input:
            logits: Bx?x?
            top_k:
            top_p

        return: 
            image: torch.Tensor
        
        '''
        if sizes is None:
            sizes = self.vae_proxy.v_patch_nums 
        if logits.shape[-1] == self.vae_proxy.vocab_size:
            idx_Bl = logits.argmax(-1)#sample_with_top_k_top_p_(logits, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        else:
            idx_Bl = logits

        idx_Bl_list = []
        summ = 0
        for i in range(len(sizes)):
            idx_Bl_list.append(idx_Bl[:, summ: summ + sizes[i]**2])
            summ += sizes[i] ** 2

        return self.vae_proxy.idxBl_to_img(idx_Bl_list, last_one=True)
    
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        if path[-3:] != 'pth':
            safetensors.torch.load_model(self, path, strict=False)
        else:
            sd = torch.load(path, map_location="cpu", weights_only=True) #["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            missing_keys, _ = self.load_state_dict(sd, strict=False)
            # print('missing_keys', missing_keys)

        print(f"Restored from {path}")




