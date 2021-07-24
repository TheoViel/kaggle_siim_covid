import numpy as np
from sklearn.metrics import average_precision_score


def per_class_average_precision_score(pred, truth, num_classes=1, average=True):
    if num_classes == 1:
        return average_precision_score(truth.flatten(), pred.flatten())

    if len(truth.shape) > 1:
        truth = truth.argmax(-1)

    scores = [average_precision_score(truth == i, pred[:, i]) for i in range(pred.shape[1])]

    if average:
        return np.mean(scores)
    else:
        return np.array(scores)
