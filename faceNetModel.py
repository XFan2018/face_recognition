import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet34
import torchvision.models.densenet


class faceNet(nn.Module):
    def __init__(self, embedding_size, pretrained=True):
        super(faceNet, self).__init__()

        self.model = resnet34(pretrained)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(2048 * 3 * 3, self.embedding_size)

    def l2_norm(self, input):
        dim = input.size()
        square_result = torch.pow(input, 2)
        sum_result = torch.sum(square_result, 1).add_(1e-10)
        norm = torch.sqrt(sum_result)
        result = torch.div(input, norm.view(-1,1).expand_as(input)).view(dim)
        return result

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        embeddings = self.l2_norm(x)
        alpha = 10  # from paper https://arxiv.org/pdf/1703.09507.pdf
        embeddings = embeddings * alpha

        return embeddings



