import numpy as np
import torch
import os
import torchvision
import torch.nn as nn
from torchvision.models import resnet34
import torchvision.models.densenet
from detect_faces import detect_faces


class faceNet(nn.Module):
    def __init__(self, embedding_size, pretrained=True):
        super(faceNet, self).__init__()

        self.model = resnet34(pretrained)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(8192, self.embedding_size)

    def l2_norm(self, input):
        dim = input.size()
        square_result = torch.pow(input, 2)
        sum_result = torch.sum(square_result, 1).add_(1e-10)
        norm = torch.sqrt(sum_result)
        result = torch.div(input, norm.view(-1, 1).expand_as(input)).view(dim)
        return result

    def forward(self, x):
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


def main():
    confidence_threshold = 0.9
    prototxt = '/model/deploy.prototxt.txt'
    caffemodel = '/model/res10_300x300_ssd_iter_140000.caffemodel'

    base_dir = os.path.dirname(__file__)
    prototxt = os.path.join(base_dir + prototxt)
    caffemodel = os.path.join(base_dir + caffemodel)
    if not os.path.exists("detected_face"):
        print("create new dir")
        os.mkdir("detected_face")
    dataset = os.listdir(base_dir + "/faceDataset")
    dataset_path = base_dir + "/faceDataset"
    extracted_faces = detect_faces(dataset_path, dataset, prototxt, caffemodel, confidence_threshold)
    extracted_faces = np.asarray(extracted_faces)
    extracted_faces = torch.from_numpy(extracted_faces)
    net = faceNet(128)
    print(net(extracted_faces))


if __name__ == "__main__":
    main()
