import torch.nn as nn

class GainControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(nn.zeros(1))  

    def forward(self, x, gatta):
        return x * (1 + self.alpha * gatta)
