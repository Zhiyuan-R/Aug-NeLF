import os, sys
import math, time, random

import imageio
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from utils.image import to8b, img2mse, mse2psnr

def eval_one_view(model, batch, near_far, device, **render_kewargs):
    
    model.eval()
    
    near, far = near_far
    with torch.no_grad():
        batch_rays = batch['rays'].to(device)
        
        # run nelf
        