import torch
import torch.nn.functional as F
import resnest.torch as resnest_torch
from efficientnet_pytorch import EfficientNet


def get_encoder(name):
    """
    Loads a pretrained model.
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.
    Args:
        name (str): Name of the model to load
        num_classes (int, optional): Number of classes to use. Defaults to 1.
    Raises:
        NotImplementedError: Unknown model name.
    Returns:
        torch model: Pretrained model
    """
    if "resnest" in name:
        model = getattr(resnest_torch, name)(pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif "resnext" in name or "resnet" in name or "densenet" in name:
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif "efficientnet" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        raise NotImplementedError

    if "efficientnet" in name:
        model.nb_ft = model._fc.in_features
    elif "densenet" in name:
        model.nb_ft = model.classifier.in_features
        model.extract_features = lambda x: extract_features_densenet(model, x)
    else:
        model.nb_ft = model.fc.in_features
        model.extract_features = lambda x: extract_features_resnet(model, x)

    model.name = name
    return model


def extract_features_resnet(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)

    return x1, x2, x3, x4


def extract_features_densenet(self, x):
    x = self.features(x)
    x = F.relu(x, inplace=True)   # remove ?
    return x
