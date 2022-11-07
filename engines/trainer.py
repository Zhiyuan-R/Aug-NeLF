import os, sys, copy
import math, time, random

import numpy as np
import imageio
import json
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import to8b, img2mse, mse2psnr

def train_one_epoch(model, optimizer, scheduler,
                    train_loader, test_set, exhibit_set,
                    summary_writer,
                    global_step,
                    max_steps,
                    run_dir,
                    device):
    
    near, far = train_loader.dataset.near_far()
    
    start_step = global_step
    epoch = global_step // len(train_loader) + 1
    time0 = time.time()
    
    for (batch_rays, gt) in train_loader:
        model.train()
        
        # counter accumulate
        global_step += 1
        
        # make sure on cuda for data end
        batch_rays, gt = batch_rays.to(device), gt.to(device)
        
        ##### Core Optimization Loop #####
        ret_dict = model(batch_rays, (near, far))