import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_score(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    # (xc_1, yc_1, w1, h1) = bbox1
    # (xc_2, yc_2, w2, h2) = bbox2

    # (x0_1, y0_1, x1_1, y1_1) = (
    #     xc_1 - w1 / 2,
    #     yc_1 - h1 / 2,
    #     xc_1 + w1 / 2,
    #     yc_1 + h1 / 2,
    # )
    # (x0_2, y0_2, x1_2, y1_2) = (
    #     xc_2 - w2 / 2,
    #     yc_2 - h2 / 2,
    #     xc_2 + w2 / 2,
    #     yc_2 + h2 / 2,
    # )

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def precision_calc(gt_boxes, pred_boxes, threshold=0.25, return_assignment=False):
    cost_matrix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            iou = iou_score(box1, box2)

            if iou < threshold:
                continue

            else:
                cost_matrix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    if return_assignment:
        return cost_matrix, row_ind, col_ind

    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1

    return tp, fp, fn


def compute_metrics(preds, truths):
    """
    Computes metrics.

    Args:
        preds (List of Boxes): Predictions.
        truths (List of Boxes): Truths.
    Returns:
        dict: Metrics
    """
    ftp, ffp, ffn = [], [], []
    for pred, truth in zip(preds, truths):
        tp, fp, fn = precision_calc(truth['pascal_voc'].copy(), pred['pascal_voc'].copy())
        ftp.append(tp)
        ffp.append(fp)
        ffn.append(fn)

    tp = np.sum(ftp)
    fp = np.sum(ffp)
    fn = np.sum(ffn)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
