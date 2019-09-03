import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        logits = self.vgg16.classifier(x.view(-1, 25088))
        probas = F.softmax(logits, dim=1)
        return logits, probas
