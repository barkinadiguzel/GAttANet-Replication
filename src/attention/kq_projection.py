import torch
import torch.nn as nn

class KQProjectionConv(nn.Module):
    # Eq. (1)
    def __init__(self, in_channels, d):
        super().__init__()
        self.key = nn.Conv2d(in_channels, d, kernel_size=1, bias=False)
        self.query = nn.Conv2d(in_channels, d, kernel_size=1, bias=False)

    def forward(self, x):
        k = self.key(x)   
        q = self.query(x)  
        return k, q


class KQProjectionLinear(nn.Module):
    # Eq. (2)
    def __init__(self, in_features, d):
        super().__init__()
        self.key = nn.Linear(in_features, d, bias=False)
        self.query = nn.Linear(in_features, d, bias=False)

    def forward(self, x):
        k = self.key(x)    
        q = self.query(x)  
        return k, q
