import timm
import torch
import torch.nn.functional as F
import resnest.torch as resnest_torch


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
    elif "efficientnetv2" in name:
        model = getattr(timm.models, name)(
            pretrained=True,
            drop_path_rate=0.2,
        )
    else:
        raise NotImplementedError

    if "efficientnetv2" in name:
        # model.nb_ft = model.classifier.in_features
        model.nb_ft = model.blocks[6][-1].conv_pwl.out_channels
        model.nb_ft_int = model.blocks[4][-1].conv_pwl.out_channels
        model.extract_features = lambda x: extract_features_efficientnet(model, x)
    elif "densenet" in name:
        model.nb_ft = model.classifier.in_features
        model.nb_ft_int = model.nb_ft // 2
        model.extract_features = lambda x: extract_features_densenet(model, x)
    else:
        model.nb_ft = model.fc.in_features
        model.nb_ft_int = model.nb_ft // 2
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


def extract_features_efficientnet(self, x):
    x = self.conv_stem(x)
    x = self.bn1(x)
    x = self.act1(x)

    features = []
    for i, b in enumerate(self.blocks):
        x = b(x)
        if i in [1, 2, 4, 6]:
            # print(x.size())
            features.append(x)

    return features
