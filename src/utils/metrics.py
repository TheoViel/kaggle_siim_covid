import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from params import CLASSES, NUM_CLASSES


def per_class_average_precision_score(pred, truth, num_classes=1, average=True):
    """
    Computes the per class average precision.

    Args:
        pred (np array [N x NUM_CLASSES]): Predictions.
        truth (np array [N]): Ground truth
        num_classes (int, optional): Number of classes. Defaults to 1.
        average (bool, optional): Whether to average results. Defaults to True.

    Returns:
        float or list of float: Metric value or per class metric value.
    """
    if num_classes == 1:
        return average_precision_score(truth.flatten(), pred.flatten())

    if len(truth.shape) > 1:
        truth = truth.argmax(-1)

    scores = [average_precision_score(truth == i, pred[:, i]) for i in range(pred.shape[1])]

    if average:
        return np.mean(scores)
    else:
        return np.array(scores)


def study_level_map(pred, truth, studies, agg=np.mean):
    """
    Computes the study level metric.

    Args:
        pred (np array [N x NUM_CLASSES]): Predictions.
        truth (np array [N x NUM_CLASSES] or [N]): Ground truth
        studies (list of str): Studies for aggregation.
        agg (fct, optional): How to aggreagate predictions. Defaults to np.mean.

    Returns:
        float: Metric value.
    """
    df = pd.DataFrame({"study": studies})

    if len(truth.shape) > 1:
        truth = truth.argmax(-1)
    df["truth"] = truth

    pred_cols = [c + "_pred" for c in CLASSES]
    for i, c in enumerate(pred_cols):
        df[c] = pred[:, i]

    df_study = df.groupby('study').agg(agg)
    return per_class_average_precision_score(
        df_study[pred_cols].values, df_study["truth"].values, num_classes=NUM_CLASSES
    ) * 2/3
