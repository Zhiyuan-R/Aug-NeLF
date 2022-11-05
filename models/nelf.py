import torch.nn as nn

class NeLFNet(nn.Module):

    def __init(self):
        
        super().__init()
        self.final = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.final(x)