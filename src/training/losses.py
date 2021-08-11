import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (n_class - 1)

        loss = -targets * F.log_softmax(inputs, dim=1)
        loss = loss.sum(-1)

        return loss


class CovidLoss(nn.Module):
    """
    Loss for the problem :
    - Mask loss : FocalTversky + BCE
    - Study loss : CE
    - Img loss : BCE
    """
    def __init__(self, config):
        """
        Constructor

        Args:
            config (dict): Loss config.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = SmoothCrossEntropyLoss()
        self.focal_tversky = FocalTverskyLoss()

        self.w_bce = config["w_bce"]
        self.w_seg_loss = config["w_seg_loss"]
        self.w_study = config["w_study"]
        self.w_img = config["w_img"]
        self.seg_loss_multiplier = config["seg_loss_multiplier"]

    def compute_seg_loss(self, preds, truth):
        """
        Computes the auxiliary segmentation loss.

        Args:
            preds (list of torch tensors [BS x h_i x w_i]): Predicted masks.
            truth (torch tensor [BS x H x W]): Ground truth mask.

        Returns:
            torch tensor [BS]: Loss value.
        """
        losses = []
        truth = truth.unsqueeze(1)

        for pred in preds:
            truth = F.interpolate(
                truth, size=pred.size()[-2:], mode='bilinear', align_corners=False
            )
            loss = self.w_bce * self.bce(pred, truth).mean((1, 2, 3))
            loss += (1 - self.w_bce) * self.focal_tversky(pred, truth)
            losses.append(loss)

        loss = self.seg_loss_multiplier * torch.stack(losses, -1).mean(-1)

        return loss

    def compute_study_loss(self, pred, truth, mix_lambda=1):
        """
        Computes the study loss. Handles mixup / cutmix.

        Args:
            preds (list of torch tensors or torch tensor [BS x num_classes]): Predictions.
            truth (torch tensor [BS x num_classes]): Ground truth.
            mix_lambda (float, optional): Mix coefficitnet. Defaults to 1.

        Returns:
            torch tensor [BS]: Loss value.
        """
        if isinstance(truth, list):
            return self.w_study * (
                mix_lambda * self.ce(pred, truth[0].long()) +
                (1 - mix_lambda) * self.ce(pred, truth[1].long())
            )
        else:
            return self.w_study * self.ce(pred, truth.long())

    def __call__(
        self, pred_study, pred_img, preds_mask, y_study, y_img, y_mask, mix_lambda=1
    ):
        """
        Computes the overall loss.

        Args:
            pred_study (list of torch tensors or torch tensor [BS x num_classes]): Study preds.
            pred_img (torch tensor [BS x num_classes]): Opacity predictions.
            preds_mask (list of torch tensors [BS x h_i x w_i]): Predicted masks.
            y_study (torch tensor [BS x num_classes]): Study ground truth.
            y_img (torch tensor [BS x num_classes]): Opacity ground truth.
            y_mask (torch tensor [BS x H x W]): Ground truth mask.
            mix_lambda (float, optional): Mix coefficitnet. Defaults to 1.

        Returns:
            torch tensor [BS]: Loss value.
        """
        seg_loss = self.compute_seg_loss(preds_mask, y_mask)

        study_loss = self.compute_study_loss(pred_study, y_study, mix_lambda=mix_lambda)
        img_loss = self.w_img * self.bce(pred_img.view(y_img.size()), y_img)

        return self.w_seg_loss * seg_loss + (1 - self.w_seg_loss) * (img_loss + study_loss)
