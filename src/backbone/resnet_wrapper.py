import torch.nn as nn

class ResNetWrapper(nn.Module):
    def __init__(self, backbone, hook_layers):
        super().__init__()
        self.backbone = backbone
        self.hook_layers = hook_layers
        self.features = {}

        for name, layer in self.backbone.named_modules():
            if name in hook_layers:
                layer.register_forward_hook(self.save_hook(name))

        for p in self.backbone.parameters():
            p.requires_grad = False

    def save_hook(self, name):
        def hook(_, __, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        out = self.backbone(x)
        return out, self.features
