# models/homography_net.py
import torch.nn as nn
import torch

class HomographyNet(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 8)
        )

    def forward(self, fs, fd):
        fs = self.pool(fs).flatten(1)
        fd = self.pool(fd).flatten(1)
        return self.fc(torch.cat([fs, fd], dim=1))
