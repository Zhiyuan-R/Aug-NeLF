import torch.nn as nn

class Adv_NeLFNet(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.final = nn.Linear(10, 5)
        
    def forward(self):
        pass