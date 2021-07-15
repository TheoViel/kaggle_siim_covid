# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from util.metrics import precision_calc


def plot_sample(img, boxes, bbox_format="yolo"):
    plt.figure(figsize=(15, 5))
    plt.imshow(img, cmap="gray")
    plt.axis(False)

    for box in boxes:
        if bbox_format == "yolo":
            h, w, _ = img.shape
            rect = Rectangle(
                ((box[0] - box[2] / 2) * w, (box[1] - box[3] / 2) * h), box[2] * w, box[3] * h,
                linewidth=2, edgecolor='salmon', facecolor='none'
            )
        elif bbox_format == "pascal_voc":
            rect = Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='salmon', facecolor='none'
            )
        else:
            raise NotImplementedError()

        plt.gca().add_patch(rect)


def plot_predictions(img, truths, preds, figsize=(15, 6)):
    cost_matrix, row_ind, col_ind = precision_calc(
        truths["pascal_voc"].copy(), preds["pascal_voc"].copy(), return_assignment=True
    )

    matched_truths = []
    matched_preds = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 0:
            matched_truths.append(i)
            matched_preds.append(j)

    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    plt.axis(False)

    for i, box in enumerate(truths["pascal_voc"]):
        col = "dodgerblue" if i in matched_truths else "firebrick"

        rect = Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=col, facecolor='none'
        )
        plt.gca().add_patch(rect)

    for i, box in enumerate(preds["pascal_voc"]):
        col = "skyblue" if i in matched_preds else "tomato"

        rect = Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=col, facecolor='none', linestyle="--"
        )
        plt.gca().add_patch(rect)

    tp_gt = Patch(edgecolor='dodgerblue', facecolor='none', label='TP (truth)')
    tp_pred = Patch(edgecolor='skyblue', facecolor='none', label='TP (pred)', linestyle="--")
    fn = Patch(edgecolor='firebrick', facecolor='none', label='FN')
    fp = Patch(edgecolor='tomato', facecolor='none', label='FP', linestyle="--")
    plt.legend(handles=[tp_gt, tp_pred, fp, fn], bbox_to_anchor=(1.15, 1.02))

    plt.title(
        f'TPs : {len(matched_truths)} - '
        f'FPs : {len(preds)  - len(matched_preds)} - '
        f'FNs : {len(truths) - len(matched_truths)}'
    )
    plt.show()
