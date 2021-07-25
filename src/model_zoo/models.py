import torch
import torch.nn as nn

from model_zoo.encoders import get_encoder


def get_model(name, num_classes=1, reduce_stride=False):
    """
    Loads a pretrained model.
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and encoder.
    Args:
        name (str): Name of the model to load
        num_classes (int, optional): Number of classes to use. Defaults to 1.
    Raises:
        NotImplementedError: Unknown model name.
    Returns:
        torch model: Pretrained model
    """

    encoder = get_encoder(name)

    model = CovidModel(encoder, num_classes=num_classes, reduce_stride=reduce_stride)

    return model


class CovidModel(nn.Module):
    def __init__(self, encoder, num_classes=1, reduce_stride=False):
        """
        Constructor.

        Args:
            encoder (nn.Module): encoder to build the model from.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.nb_ft = encoder.nb_ft

        if reduce_stride:
            if "resnext" in self.encoder.name:
                self.encoder.layer4[0].conv2.stride = (1, 1)
                self.encoder.layer4[0].downsample[0].stride = (1, 1)
            elif "resnet" in self.encoder.name:
                self.encoder.layer4[0].conv1.stride = (1, 1)
                self.encoder.layer4[0].downsample[0].stride = (1, 1)
            else:
                raise NotImplementedError

        self.mask_head_3 = self.get_mask_head(self.encoder.nb_ft_int)
        self.mask_head_4 = self.get_mask_head(self.nb_ft)

        self.key_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2d(self.nb_ft, self.nb_ft, kernel_size=3, padding=1)

        self.logits_img = nn.Linear(self.nb_ft, 1)
        self.logits_study = nn.Linear(self.nb_ft, num_classes)

    @staticmethod
    def get_mask_head(nb_ft):
        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def attention_mechanism(self, fts, masks):
        """
        Performs the forward pass of the attention mechanism.

        Args:
            fts (torch tensor [batch_size x nb_ft x h x w]): Feature maps.
            masks (list of torch tensor [batch_size x 1 x h x w]): Masks.

        Returns:
            torch tensor (batch_size x nb_ft): Pooled features.
        """
        bs, c, h, w = fts.size()

        weights = []
        for mask in masks:
            mask = self.key_conv(mask)
            weights.append(torch.softmax(mask.view(bs, -1, h * w), -1).transpose(1, 2))

        att = torch.cat(weights, -1).sum(-1, keepdims=True)

        fts = self.value_conv(fts)
        fts = fts.view(bs, c, h * w)

        return torch.bmm(fts, att).view(-1, self.nb_ft)

    def forward(self, x):
        # F.interpolate(truth_mask, size=(48,48), mode='bilinear', align_corners=False)

        x1, x2, x3, x4 = self.encoder.extract_features(x)

        mask_3 = self.mask_head_3(x3)
        mask_4 = self.mask_head_4(x4)
        masks = [mask_3, mask_4]

        # pooled = self.attention_mechanism(x4, masks)
        pooled = x4.mean(-1).mean(-1)

        logits_img = self.logits_img(pooled)
        logits_study = self.logits_study(pooled)

        return logits_study, logits_img, masks
