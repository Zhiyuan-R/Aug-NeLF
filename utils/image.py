import torch
import numpy as np

def to8b(x):
    return (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)

def img2mse(x, y, reduction='mean'):
    diff = torch.mean((x - y) ** 2, -1)
    if reduction == 'mean':
        return torch.mean(diff)
    elif reduction == 'sum':
        return torch.sum(diff)
    elif reduction == 'none':
        return diff
    
def mse2psnr(x):
    if isinstance(x, float):
        x = torch.tensor([x])
    return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
    