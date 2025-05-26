import torch.nn as nn
from torchvision.models import resnet50

class AttributeRecognitionModel(nn.Module):
    def __init__(self):
        super(AttributeRecognitionModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 8)

    def forward(self, x):
        return self.resnet(x)