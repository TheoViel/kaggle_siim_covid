import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from params import NUM_WORKERS


def predict(
    model, dataset, batch_size=64, num_classes=1, flip_tta=False, scale_tta=False, device="cuda"
):
    """
    Usual torch predict function. Supports sigmoid and softmax activations.
    Args:
        model (torch model): Model to predict with.
        dataset (ColorBCCDataset): Dataset to predict on.
        batch_size (int, optional): Batch size. Defaults to 64.
        num_classes (int, optional): Number of classes. Defaults to 1.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()
    all_preds_study = np.empty((0, num_classes))
    all_preds_img = np.empty((0))

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    with torch.no_grad():
        preds_study = []
        preds_img = []
        for batch in val_loader:
            x = batch[0].to(device)

            pred_study, pred_img, _ = model(x)
            preds_study.append(torch.softmax(pred_study, -1).detach().cpu().numpy())
            preds_img.append(torch.sigmoid(pred_img).view(-1).detach().cpu().numpy())

            if flip_tta:
                x = x.flip([-1])
                pred_study, pred_img, _ = model(x)
                preds_study.append(torch.softmax(pred_study, -1).detach().cpu().numpy())
                preds_img.append(torch.sigmoid(pred_img).view(-1).detach().cpu().numpy())

            if scale_tta:
                x = F.interpolate(
                    x, size=x.size()[-2:] + 128, mode='bilinear', align_corners=False
                )
                pred_study, pred_img, _ = model(x)
                preds_study.append(torch.softmax(pred_study, -1).detach().cpu().numpy())
                preds_img.append(torch.sigmoid(pred_img).view(-1).detach().cpu().numpy())

            if scale_tta and flip_tta:
                x = x.flip([-1])
                pred_study, pred_img, _ = model(x)
                preds_study.append(torch.softmax(pred_study, -1).detach().cpu().numpy())
                preds_img.append(torch.sigmoid(pred_img).view(-1).detach().cpu().numpy())

            all_preds_study = np.concatenate([all_preds_study, np.mean(preds_study, 0)])
            all_preds_img = np.concatenate([all_preds_img, np.mean(preds_img, 0)])

    return all_preds_study, all_preds_img
