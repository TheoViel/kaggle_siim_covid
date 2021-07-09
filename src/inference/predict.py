import torch
import numpy as np
from torch.utils.data import DataLoader

from params import NUM_WORKERS


def predict(model, dataset, activation="sigmoid", batch_size=64, num_classes=1, device="cuda"):
    """
    Usual torch predict function. Supports sigmoid and softmax activations.
    Args:
        model (torch model): Model to predict with.
        dataset (ColorBCCDataset): Dataset to predict on.
        activation (str, optional): Name of the activation function to use. Defaults to "sigmoid".
        batch_size (int, optional): Batch size. Defaults to 64.
        num_classes (int, optional): Number of classes. Defaults to 1.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()
    preds = np.empty((0, num_classes))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y_pred = model(x)[0].detach()

            if activation == "sigmoid":
                y_pred = torch.sigmoid(y_pred)
            elif activation == "softmax":
                y_pred = torch.softmax(y_pred, -1)

            preds = np.concatenate([preds, y_pred.cpu().numpy()])

    return preds


def get_attention_map(model, dataset, batch_size=16, device="cuda"):
    """
    Computes the attention maps .

    Args:
        model (torch model): Model to use, expected to be a ConvAttentionResNet.
        dataset (ColorBCCDataset): Dataset to get the maps from on.
        batch_size (int, optional): [description]. Defaults to 16.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x h x w]: Attention maps.
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    atts, preds = [], []
    for batch in loader:
        x = batch[0].to(device)
        att, pred, _ = model(x, return_att=True)
        preds.append(pred.sigmoid().detach().cpu().numpy())
        atts.append(att.detach().cpu().numpy())

    return np.vstack(atts), np.vstack(preds)