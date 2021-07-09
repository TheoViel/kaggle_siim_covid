# From https://github.com/facebookresearch/mixup-cifar10

import torch
import numpy as np


def mixup_data(x, y, y_aux=None, alpha=0.4, device="cuda"):
    """
    TODO : OUTDATED
    Applies mixup to a sample

    Args:
        x (torch tensor [batch_size x input_size]): Input batch.
        y (torch tensor [batch_size x num_classes]): Labels.
        alpha (float, optional): Parameter of the beta distribution. Defaults to 0.4.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        torch tensor [batch_size x input_size]: Mixed input.
        torch tensor [batch_size x num_classes]: Labels of the original batch.
        torch tensor [batch_size x num_classes]: Labels of the shuffle batch.
        float: Probability samples by the beta distribution.
    """

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    if y_aux is not None:
        y_aux_a, y_aux_b = y_aux, y_aux[index]
    else:
        y_aux_a, y_aux_b = None, None

    return mixed_x, y_a, y_b, y_aux_a, y_aux_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Not used.
    Updates a loss function to use mixup.

    Args:
        criterion (function): Loss function
        pred (torch tensor [batch_size x num_classes]): Prediction
        y_a (torch tensor [batch_size x num_classes]): Labels of the original batch.
        y_b (torch tensor [batch_size x num_classes]): Labels of the shuffle batch.
        lam (float): Value sampled by the beta distribution

    Returns:
        torch tensor: Loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """
    Retuns the coordinate of a random rectangle in the image for cutmix.

    Args:
        size (torch tensor [batch_size x c x W x H): Input size.
        lam (int): Lambda sampled by the beta distribution. Controls the size of the squares.

    Returns:
        int: 4 coordinates of the rectangle.
        int: Proportion of the unmasked image.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return bbx1, bby1, bbx2, bby2, lam


def cutmix_data(x, x_teach, y, y_aux=None, w=None, alpha=1.0, device="cuda"):
    """
    Applies cutmix to a sample.

    Args:
        x (torch tensor [batch_size x input_size]): Input batch.
        x_teach (torch tensor [batch_size x input_size]): Teacher input batch.
        y (torch tensor [batch_size x num_classes]): Labels.
        y_aux (torch tensor [batch_size x num_classes_aux], optional): Aux labels. Defaults to None.
        w (torch tensor [batch_size], optional): Sample weights. Defaults to None.
        alpha (float, optional): Parameter of the beta distribution. Defaults to 1.0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        torch tensor [batch_size x input_size]: Mixed input.
        torch tensor [batch_size x input_size]: Mixed teacher input.
        torch tensor [batch_size x num_classes]: Labels of the original batch.
        torch tensor [batch_size x num_classes]: Labels of the shuffle batch.
        torch tensor [batch_size x num_classes_aux] or None: Aux labels of the original batch.
        torch tensor [batch_size x num_classes_aux] or None: Aux labels of the shuffle batch.
        torch tensor [batch_size] or None: Sample weights of the original batch.
        torch tensor [batch_size] or None: Sample weights of the shuffle batch.
        float: Probability samples by the beta distribution.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).to(device)

    bbx1, bby1, bbx2, bby2, lam = rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    mixed_x_teach = x_teach.clone()

    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    mixed_x_teach[:, :, bbx1:bbx2, bby1:bby2] = x_teach[index, :, bbx1:bbx2, bby1:bby2]

    y_a, y_b = y, y[index]

    if y_aux is not None:
        y_aux_a, y_aux_b = y_aux, y_aux[index]
    else:
        y_aux_a, y_aux_b = None, None

    if w is not None:
        w_a, w_b = w, w[index]
    else:
        w_a, w_b = None, None

    return mixed_x, mixed_x_teach, y_a, y_b, y_aux_a, y_aux_b, w_a, w_b, lam
