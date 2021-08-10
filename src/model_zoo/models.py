import torch.nn as nn

from model_zoo.encoders import get_encoder


def get_model(name, num_classes=1, use_unet=False, pretrained=False):
    """
    Loads the model.

    Args:
        name (str): Name of the encoder.
        num_classes (int, optional): Number of classes to use. Defaults to 1.

    Returns:
        torch model: Model.
    """
    encoder = get_encoder(name)
    model = CovidModel(encoder, num_classes=num_classes)
    return model


class CovidModel(nn.Module):
    def __init__(self, encoder, num_classes=1):
        """
        Constructor.

        Args:
            encoder (nn.Module): encoder to build the model from.
            num_classes (int, optional): Number of classes to use. Defaults to 1.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.nb_ft = encoder.nb_ft
        self.mean = encoder.mean
        self.std = encoder.std

        self.mask_head_3 = self.get_mask_head(self.encoder.nb_fts[2])
        self.mask_head_4 = self.get_mask_head(self.nb_ft)

        self.logits_img = nn.Linear(self.nb_ft, 1)
        self.logits_study = nn.Linear(self.nb_ft, num_classes)

    @staticmethod
    def get_mask_head(nb_ft):
        """
        Returns a segmentation head.

        Args:
            nb_ft (int): Number of input features.

        Returns:
            nn.Sequential: Segmentation head.
        """
        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        """
        Usual torch forward function

        Args:
            x (torch tensor [BS x 3 x H x W]): Input image

        Returns:
            torch tensor [BS x NUM_CLASSES]: Study logits.
            torch tensor [BS x 1]: Image logits.
            list or torch tensors : Masks.
        """
        x1, x2, x3, x4 = self.encoder.extract_features(x)

        mask_3 = self.mask_head_3(x3)
        mask_4 = self.mask_head_4(x4)
        masks = [mask_3, mask_4]

        pooled = x4.mean(-1).mean(-1)

        logits_img = self.logits_img(pooled)
        logits_study = self.logits_study(pooled)

        return logits_study, logits_img, masks
