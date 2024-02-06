import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import matplotlib.pyplot as plt

class RWMAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print('In RWMAB')
        x_ = self.layer1(x)
        x__ = self.layer2(x_)

        x = x__ * x_ + x

        return x


class ShortResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layers = nn.ModuleList([RWMAB(in_channels) for _ in range(16)])

    def forward(self, x):
        # print('In Short Residual Block')
        x_ = x.clone()

        for layer in self.layers:
            x_ = layer(x_)

        return x_ + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, blocks=8, scale = 4):
        super().__init__()

        # Add the noise part here

        self.conv = nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1)

        self.short_blocks = nn.ModuleList(
            [ShortResidualBlock(64) for _ in range(blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),  # Remove if output is 2x the input
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, (1, 1), stride=1, padding=0),  # Change 64 -> 256
            nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # print('In Generator')
        x = self.conv(x)
        x_ = x.clone()

        for layer in self.short_blocks:
            x_ = layer(x_)

        x = torch.cat([self.conv2(x_), x], dim=1)

        x = self.conv3(x)

        return x