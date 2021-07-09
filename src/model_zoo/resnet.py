import torch
import torch.nn as nn
import numpy as np


class MixStyle(nn.Module):
    """
    MixStyle : Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Constructor

        Args:
            p (float, optional): Probability of using MixStyle. Defaults to 0.5.
            alpha (float, optional): Parameter of the Beta distribution. Defaults to 0.3.
            eps ([type], optional): Scaling parameter to avoid numerical issues. Defaults to 1e-6.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [BS x C x H x W]): Hidden features.

        Returns:
            torch tensor [BS x C x H x W]: Mixed features.
        """
        if not self.training:
            return x

        if np.random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


def add_ft_extractor(resnet, use_mixstyle=False):
    """
    Adds a function to extract features from a ResNet backbone.

    Args:
        resnet (torch ResNet): ResNet model.
        use_mixstyle (bool, optional): Whether to apply mixstyle. Defaults to False.
    """
    resnet.extract_fts = lambda x: extract_features_resnet(resnet, x, use_mixstyle=use_mixstyle)
    if use_mixstyle:
        resnet.mixstyle = MixStyle(p=0.5, alpha=0.1)


def extract_features_resnet(resnet, x, use_mixstyle=False):
    """
    Function to extract features with a ResNet model.
    Supports mixstyle.

    Args:
        resnet (ResNet model): Model to add the function to.
        x (torch tensor [batch_size x 3 x w x h]): Image.
        use_mixstyle (bool, optional): Whether to apply mixstyle. Defaults to False.

    Returns:
        torch tensor [batch_size x 3 x w x h]: Feature maps.
    """
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)

    x = resnet.layer1(x)
    if use_mixstyle:
        x = resnet.mixstyle(x)

    x = resnet.layer2(x)
    if use_mixstyle:
        x = resnet.mixstyle(x)

    x = resnet.layer3(x)
    if use_mixstyle:
        x = resnet.mixstyle(x)

    x = resnet.layer4(x)

    return x


def forward_with_aux(resnet, x):
    """
    Forward function for a ResNet model with an auxiliary classifier

    Args:
        resnet (ResNet model): Model to add the function to.
        x (torch tensor): Input image.

    Returns:
        torch tensor: Predictions.
        torch tensor: Auxiliary predictions.
    """
    fts = resnet.extract_fts(x)
    fts = resnet.avgpool(fts)
    fts = torch.flatten(fts, 1)

    y = resnet.fc(fts)

    if resnet.num_classes_aux > 0:
        y_aux = resnet.fc_aux(fts)
        return y, y_aux

    return y, 0


def add_aux_classifier(resnet, num_classes_aux=0):
    """
    Modifies the forward function of a ResNet model to include an auxiliary classifier.

    Args:
        resnet (torch ResNet): ResNet model.
        num_classes_aux (int, optional): [description]. Defaults to 0.
    """

    resnet.num_classes_aux = num_classes_aux
    resnet.fc_aux = nn.Linear(resnet.nb_ft, num_classes_aux)

    add_ft_extractor(resnet)
    resnet.forward = lambda x: forward_with_aux(resnet, x)
