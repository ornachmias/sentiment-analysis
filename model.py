import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.2)
        )

        self.fc1 = nn.Linear(in_features=21632, out_features=num_classes)

    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)

        out = out.view(-1, 21632)
        out = self.fc1(out)
        probas = F.log_softmax(out, dim=1)

        return out, probas
