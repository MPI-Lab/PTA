import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import models
from collections import Counter
    


class PTA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device
        
    def forward(self, x):
        for _ in range(self.steps):
            outputs, fea = self.forward_and_adapt(x)
        
        return outputs, fea

       
    @torch.enable_grad() 
    def forward_and_adapt(self, x):

        with autocast():
            # forward
            outputs, a, v, attn, fea = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)

        entropys = softmax_entropy(outputs)
        pred = outputs.argmax(1)

        conf_weights = entropys.softmax(dim=-1).max(dim=-1)[0]

        # calculate frequency 
        class_counts = torch.unique(pred, return_counts=True)
        unique_classes = class_counts[0]    
        counts = class_counts[1]
        pred_slash = torch.tensor([1 / len(unique_classes) for _ in range(len(unique_classes))]).cuda()
        observe = counts / counts.sum()
        
        Z = pred_slash - observe
        Z_tan = torch.tanh(Z) # optional
        # assign weights for each sample
        Z_ = torch.zeros_like(pred, dtype=torch.float).cuda()
        for i, cls in enumerate(unique_classes):
            class_mask = (pred == cls)
            Z_[class_mask] = Z_tan[i]
        
        if (Z_> 0).any().item() and (Z_ < 0).any().item():
           
            # 
            positive_indices = torch.where(Z_>0)[0]
            # 
            negative_indices = torch.where(Z_<0)[0]

            # rank weight for pos
            rank_conf = quantile_rank(conf_weights[positive_indices])
            rank_bias = quantile_rank(Z_[positive_indices],True)
            total_weights = rank_bias * rank_conf
           
            # attn regularization all
            pos = attn[positive_indices]
            neg = attn[negative_indices]
            attn[:,:,:512,:512] = attn[:,:,:512,:512]  # Audio-Audio
            attn[:,:,512:,512:] = attn[:,:,512:,512:]  # Video-Video
            attn[:,:,:512,512:] = attn[:,:,:512,512:]  # Audio-Video
            attn[:,:,512:,:512] = attn[:,:,512:,:512]  # Video-Audio
            # only aa
            # pos_aa = attn[positive_indices,:,:512,:512]
            # neg_aa = attn[negative_indices,:,:512,:512]
            # # only vv
            # pos_vv = attn[positive_indices,:,512:,512:]
            # neg_vv = attn[negative_indices,:,512:,512:]
            # # only av
            # pos_av = attn[positive_indices,:,:512,512:]
            # neg_av = attn[negative_indices,:,:512,512:]
            # # only va
            # pos_va = attn[positive_indices,:,512:,:512]
            # neg_va = attn[negative_indices,:,512:,:512]
            

            # loss_attn = mmd_rbf_single_kernel(pos_aa, neg_aa) + mmd_rbf_single_kernel(pos_va, neg_va)
            loss_attn = mmd_rbf_single_kernel(pos,neg)
            loss_ent = (total_weights.detach() * entropys[positive_indices]).mean(0)
          
            loss_total = loss_ent + 0.5 * loss_ent + 1.0 * loss_attn 

        try:
            self.optimizer.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        except RuntimeError:
            return outputs, fea
    
        with torch.no_grad():
            with autocast():
            # forward
                outputs2,  a, v, attn, fea = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
            
            
            return outputs2, fea
    






def mmd_rbf_single_kernel(source, target, sigma=None, block_size=16):
    """
    """
    device = source.device
    total = torch.cat([source, target], dim=0)
    
    N = total.size(0)
    
    b1 = source.size(0)
    
    #(N, C*H*W)
    total_flat = total.view(N, -1)
    
    
    # 
    l2_distance = torch.zeros(N, N, device=device)
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            i_end = min(i + block_size, N)
            j_end = min(j + block_size, N)
            block_i = total_flat[i:i_end]
            block_j = total_flat[j:j_end]
            l2_distance[i:i_end, j:j_end] = torch.cdist(block_i, block_j, p=2) ** 2

    # ========================================================
    # ========================================================
    if sigma is None:
        # median of pairwise distances
        with torch.no_grad():
            triu_indices = torch.triu_indices(N, N, offset=1)  # 
            median_sq_dist = torch.median(l2_distance[triu_indices[0], triu_indices[1]])
            sigma = torch.sqrt(0.5 * median_sq_dist)  #
           
    # ========================================================
    # ========================================================
    kernel = torch.exp(-l2_distance / (sigma ** 2))
    
    # 
    XX = kernel[:b1, :b1]
    YY = kernel[b1:, b1:]
    XY = kernel[:b1, b1:]
    
    # MMD
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd   


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)



def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_fusion_qkv = []
    names_fusion_qkv = []

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")

    return params_fusion_qkv, names_fusion_qkv

def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            m.requires_grad_(True)

    return model




def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def quantile_rank(x, descending=False):
    """
    """
    n = x.size(0)
    
    # 
    sorted_x, sorted_indices = torch.sort(x, descending=descending)
    
    # 
    ranks = torch.arange(1, n+1, dtype=torch.float32, device=x.device)

    # 
    diff = torch.cat([
        torch.tensor([True], device=x.device),
        sorted_x[1:] != sorted_x[:-1],
        torch.tensor([True], device=x.device)
    ])
    unique_indices = torch.where(diff)[0]
    
    # 
    for i in range(len(unique_indices) - 1):
        start = unique_indices[i]
        end = unique_indices[i + 1]
        if start == end:
            continue
        segment_ranks = ranks[start:end]
        ranks[start:end] = segment_ranks.mean()
    
    # 
    restored_ranks = torch.empty_like(ranks)
    restored_ranks[sorted_indices] = ranks
    
    # 
    normalized_ranks = restored_ranks / n
    
    return normalized_ranks



