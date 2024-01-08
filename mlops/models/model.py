import torch
import torch.nn.functional as F
from torch import nn


class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, 32, 3, padding=1)  # 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 14x14
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7

        self.fc = nn.Linear(64 * 7 * 7, out_features)

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')

        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.fc(x), dim=1)

        return x
