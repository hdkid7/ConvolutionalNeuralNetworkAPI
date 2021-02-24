import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # First hidden conv layer
        t = F.max_pool2d(F.relu(self.conv1(t)), kernel_size=2, stride=2)

        # Second hidden conv layer
        t = F.max_pool2d(F.relu(self.conv2(t)), kernel_size=2, stride=2)

        # First hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = F.relu(self.fc1(t))

        # second hidden linear layer
        t = F.relu(self.fc2(t))

        # output layer

        t = self.out(t)

        return t
