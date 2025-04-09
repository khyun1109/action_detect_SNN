import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer

class SimpleSNN(nn.Module):
    def __init__(self, in_channels=12, window_size=20, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.lif1 = neuron.LIFNode()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        x: (B, C=12, W, T)
        """
        x = self.conv(x)
        x = self.lif1(x)
        x = self.pool(x)  # shape: (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        out = self.fc(x)  # (B, num_classes)
        return out
