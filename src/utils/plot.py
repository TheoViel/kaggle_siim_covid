import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_sample(img, boxes, normalized=True):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.axis(False)

    h, w, _ = img.shape
    if normalized and len(boxes):
        boxes = np.array(boxes)
        boxes[:, 0] *= w
        boxes[:, 1] *= h
        boxes[:, 2] *= w
        boxes[:, 3] *= h

    for box in boxes:
        rect = Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor='salmon', facecolor='none'
        )

        # rect = Rectangle(
        #     (box[0] - box[2] / 2, box[1] - box[3] / 2), box[2], box[3],
        #     linewidth=2, edgecolor='salmon', facecolor='none'
        # )
        plt.gca().add_patch(rect)
