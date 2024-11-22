import torch.nn as nn

def activate_requires_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
        