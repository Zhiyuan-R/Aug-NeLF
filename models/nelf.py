import torch.nn as nn

class NeLFNet(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.final = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.final(x)