import torch
import numpy as np
import torch.nn as nn
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


def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242.
    This is used for scheduling the mean teacher loss.

    Args:
        current (int): Point to evaluate the function at.
        rampup_length (int): Number of steps to reach 1.

    Returns:
        float: Current value.
    """

    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class ConsistencyLoss(nn.Module):
    """
    Mean teacher consistency loss from https://arxiv.org/abs/1703.01780.
    """
    def __init__(self, config, activation, steps_per_epoch):
        """
        Constructor

        Args:
            config (dict): Mean teacher parameters.
            activation (str): Activation to apply. Support sigmoid and softmax.
            steps_per_epoch (int): Number of training steps per epochs.
        """
        super().__init__()
        self.config = config
        self.activation = activation
        self.steps_per_epoch = steps_per_epoch
        self.mse = nn.MSELoss(reduction="none")

    def get_weight(self, step):
        """
        Returns the weighting of the loss.

        Args:
            step (int): Current training step

        Returns:
            float: Weighting.
        """
        return self.config['weight'] * sigmoid_rampup(
            step, self.config['rampup_epochs'] * self.steps_per_epoch
        )

    def forward(self, student_pred, teacher_pred, step):
        """
        Comptues the loss function.

        Args:
            student_pred (torch tensor [BS x C or BS]): Predictions of the student model.
            teacher_pred (torch tensor [BS x C or BS]): Predictions of the teacher model.
            step (int): Current training step.

        Returns:
            torch tensor [BS]: Loss value.
        """
        teacher_pred = teacher_pred.view(student_pred.size())
        weight = self.get_weight(step)

        if self.activation == "sigmoid":
            student_pred = torch.sigmoid(student_pred)
            teacher_pred = torch.sigmoid(teacher_pred)
        elif self.activation == "softmax":
            student_pred = torch.softmax(student_pred, -1)
            teacher_pred = torch.softmax(teacher_pred, -1)

        loss = self.mse(student_pred, teacher_pred)

        if len(loss.size()) == 2:
            loss = loss.mean(-1)

        return loss * weight
