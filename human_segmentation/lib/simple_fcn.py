import torch
from torch import nn


class FCNBlock(nn.Module):
    def __init__(self, input_size, *sizes):
        super(FCNBlock, self).__init__()
        layers = []
        for size in sizes:
            l = nn.Conv2d(input_size, size, 3, padding=1)
            layers.append(l)
            layers.append(nn.BatchNorm2d(size))
            layers.append(nn.ReLU())
            input_size = size
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden
            
        
class SimpleFCN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleFCN, self).__init__()
        
        layers = [
            FCNBlock(3, 16, 16, 16),
            nn.Conv2d(16, 16, 7, dilation=3, padding=9),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            FCNBlock(16, 16, 16, 16),
            nn.Conv2d(16, 16, 7, dilation=3, padding=9),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            FCNBlock(16, 16, 16, 16),
            nn.Conv2d(16, 16, 7, dilation=3, padding=9),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 7, dilation=3, padding=9),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            FCNBlock(16, 16, 16),
            nn.Conv2d(16, n_classes, 3, padding=1)
        ]
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden