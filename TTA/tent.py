import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math
import numpy as np
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F


class TENT(nn.Module):
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


    @torch.enable_grad() 
    def forward_and_adapt(self, x):
        
        with autocast():
            outputs, a, v, attn, fea = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
        
    
        entropys = softmax_entropy(outputs)
        loss = entropys.mean(0)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return outputs
    
    
    
    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        return outputs



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
    params = []
    names = []

    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            for np, p in m.named_parameters():
                
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")

    
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)

    
    
    return model
