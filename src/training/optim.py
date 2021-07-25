import torch
import torch.nn as nn
import torch.nn.functional as F

from params import DEVICE

LOSSES = ["CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss"]
ONE_HOT = torch.eye(10).to(DEVICE)


def define_loss(name, weight=None, device="cuda"):
    """
    Defines the loss function associated to the name.
    Supports loss from torch.nn, see the LOSSES list.

    Args:
        name (str): Loss name.
        weight (list or None, optional): Weights for the loss. Defaults to None.
        device (str, optional): Device for torch. Defaults to "cuda".

    Raises:
        NotImplementedError: Specified loss name is not supported.

    Returns:
        torch loss: Loss function
    """
    if weight is not None:
        weight = torch.FloatTensor(weight).to(device)
    if name in LOSSES:
        loss = getattr(torch.nn, name)(weight=weight, reduction="none")
    else:
        raise NotImplementedError

    return loss


def prepare_for_loss(preds, truths, loss, device="cuda"):
    """
    Reformats predictions to fit a loss function.

    Args:
        preds (torch tensor): List of predictions.
        truths (torch tensor): List of truths.
        loss (str): Name of the loss function.
        device (str, optional): Device for torch. Defaults to "cuda".
    Raises:
        NotImplementedError: Specified loss name is not supported.

    Returns:
        torch tensor: Reformated predictions.
        torch tensor: Reformated truths.
    """
    truths = [y.to(device) if y is not None else None for y in truths]

    if loss == "BCEWithLogitsLoss":
        num_classes = preds[0].size(-1)

        if num_classes == 1:  # Binary
            preds = [y.view(-1) if y is not None else None for y in preds]

        elif len(truths[0].size()) == 1:  # Multiclass
            truths = [ONE_HOT[y.long()][:, :num_classes] if y is not None else None for y in truths]

    elif loss == "CrossEntropyLoss":
        truths = [y.long() if y is not None else None for y in truths]
    else:
        raise NotImplementedError
    return preds, truths


def define_optimizer(name, params, lr=1e-3):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.

    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters.
        lr (float, optional): Learning rate. Defaults to 1e-3.

    Raises:
        NotImplementedError: Specified optimizer name is not supported.

    Returns:
        torch optimizer: Optimizer.
    """
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr)
    except AttributeError:
        raise NotImplementedError

    return optimizer


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.35, beta=0.65, gamma=2):
        # larger betas weight recall more than precision  - # alpha=0.3, beta=0.7
        bs = inputs.size(0)

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(bs, -1)
        targets = targets.view(bs, -1)

        tp = (inputs * targets).sum(-1)
        fp = ((1 - targets) * inputs).sum(-1)
        fn = (targets * (1 - inputs)).sum(-1)

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return (1 - tversky) ** gamma


class CovidLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.focal_tversky = FocalTverskyLoss()

        self.w_bce = 0.75
        self.w_seg_loss = 0.5
        self.w_study = 2
        self.w_img = 1

    def compute_seg_loss(self, preds, truth):
        losses = []
        truth = truth.unsqueeze(1)

        for pred in preds:
            truth = F.interpolate(
                truth, size=pred.size()[-2:], mode='bilinear', align_corners=False
            )
            loss = self.w_bce * self.bce(pred, truth).mean((1, 2, 3))
            loss += (1 - self.w_bce) * self.focal_tversky(pred, truth)
            losses.append(loss)

        return torch.stack(losses, -1).sum(-1)

    def __call__(self, pred_study, pred_img, preds_mask, y_study, y_img, y_mask, apply_mix=False):
        if apply_mix:
            raise NotImplementedError

        seg_loss = self.compute_seg_loss(preds_mask, y_mask)

        study_loss = self.w_study * self.ce(pred_study, y_study.long())
        img_loss = self.w_img * self.bce(pred_img.view(y_img.size()), y_img)

        return self.w_seg_loss * seg_loss + (1 - self.w_seg_loss) * (img_loss + study_loss)
