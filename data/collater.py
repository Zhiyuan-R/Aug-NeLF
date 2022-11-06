import os, sys
import numpy as np
import torch

# Pre-defined collaters

class RayBatchCollater:
    def __init__(self):
        pass

    def __call__(self, xs):
        batch_rays = torch.stack([torch.as_tensor(x['rays']) for x in xs], 0)
        batch_rays = torch.transpose(batch_rays, 0, 1)
        
        batch_rgbs = None
        if 'target_s' in xs[0]:
            batch_rgbs = torch.stack([torch.as_tensor(x['target_s']) for x in xs], 0)

        batch_cam_ids = None
        if 'cam_id' in xs[0]:
            batch_cam_ids = torch.stack([torch.as_tensor(x['cam_id']) for x in xs], 0)
            return batch_rays, batch_rgbs, batch_cam_ids

        return batch_rays, batch_rgbs