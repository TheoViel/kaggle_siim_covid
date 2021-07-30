import torch
import torch.nn as nn


def define_optimizer(model, config):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.

    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.

    Raises:
        NotImplementedError: Specified optimizer name is not supported.

    Returns:
        torch optimizer: Optimizer
    """
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(pg0, lr=config.lr, betas=(config.momentum, 0.999))
    else:
        raise NotImplementedError

    optimizer.add_param_group({'params': pg1, 'weight_decay': config.weight_decay})
    optimizer.add_param_group({'params': pg2})

    del pg0, pg1, pg2
    return optimizer
