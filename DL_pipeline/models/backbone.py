# models/backbone.py
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        resnet = models.resnet18(weights="DEFAULT")
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Conv2d(512, out_dim, 1)

    def forward(self, x):
        return self.proj(self.encoder(x))
