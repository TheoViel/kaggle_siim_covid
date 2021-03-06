# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_sample(img, boxes, bbox_format="yolo", axis=False):
    """
    Plots an image an its bounding boxes.
    Supports boxes of formats yolo, coco, pascal_voc and albumentations.

    Args:
        img (np array [H x W (x 3)]): Image.
        boxes (list of np array): Boxes.
        bbox_format (str, optional): Bounding box format. Defaults to "yolo".
        axis (bool, optional): Whether to display axes. Defaults to False.

    Raises:
        NotImplementedError: bbox_format is not supported.
    """
    # plt.figure(figsize=(9, 9))
    plt.imshow(img, cmap="gray")
    plt.axis(axis)

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
        elif bbox_format == "coco":
            rect = Rectangle(
                (box[0], box[1]), box[2], box[3],
                linewidth=2, edgecolor='salmon', facecolor='none'
            )
        elif bbox_format == "albu":
            h, w, _ = img.shaoe
            rect = Rectangle(
                (box[0] * h, box[1] * w), box[2] * h, box[3] * w,
                linewidth=2, edgecolor='salmon', facecolor='none'
            )
        else:
            raise NotImplementedError()

        plt.gca().add_patch(rect)
