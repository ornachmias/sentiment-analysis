import torch
from torch import nn

from torchvision.models import densenet121


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        # get the pretrained densenet model
        self.densenet = densenet121(pretrained=True)

        # replace the classifier with a fully connected embedding layer
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=2)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, images):
        densenet_outputs = self.densenet(images)
        soft_max = self.soft_max(densenet_outputs)
        return soft_max
