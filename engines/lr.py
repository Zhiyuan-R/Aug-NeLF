import torch
import numpy as np

class LRScheduler:

    def __init__(self, optimizer, init_lr, decay_rate, decay_steps):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
        self.optimizer = optimizer
        
    def step(self, step):
        for lr, param_group in zip(self.init_lr, self.optimizer.param_groups):
            new_lrate = lr * (self.decay_rate ** (step / self.decay_steps))
            param_group['lr'] = new_lrate