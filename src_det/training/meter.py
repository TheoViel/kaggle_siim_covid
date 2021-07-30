import numpy as np

from util.metrics import compute_metrics
from util.plot import plot_predictions
from util.boxes import Boxes


class DetectionMeter:
    """
    Meter to handle predictions & metrics.
    """
    def __init__(self, pred_format="coco", truth_format="yolo"):
        """
        Constructor

        TODO : Update
        """
        self.truth_format = truth_format
        self.pred_format = pred_format
        self.reset()

    def update(self, y_batch, preds, shapes, images):
        """
        Update ground truths and predictions.

        Args:
            y_batch (list of np arrays): Truths.
            preds (list of np tensors): Predictions.
            shapes (list of np arrays): Image shapes.

        Raises:
            NotImplementedError: Mode not implemented.
        """
        n, c, h, w = shapes  # TODO : verif h & w

        self.truths += [Boxes(box, (h, w), bbox_format=self.truth_format) for box in y_batch]
        self.images += list((images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8))

        for pred in preds:
            pred = pred.cpu().numpy()
            pred, confidences = pred[:, :4], pred[:, 4]

            self.preds.append(Boxes(pred, (h, w), bbox_format=self.pred_format))
            self.confidences.append(confidences)

    def compute(self, iou_threshold=0.2):
        """
        Computes the metrics.

        Returns:
            dict: Metrics dictionary.
        """
        self.fusion(iou_threshold=iou_threshold)
        return compute_metrics(self.preds, self.truths)

    def reset(self):
        """
        Resets everything.
        """
        self.preds = []
        self.confidences = []
        self.truths = []
        self.images = []

        self.metrics = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }

    def plot(self, n_samples=1):
        """
        Visualize results.

        Args:
            n_samples (int, optional): Number of samples to plot. Defaults to 1.
        """
        if n_samples > 0:
            indices = np.random.choice(len(self.images), n_samples, replace=False)
        elif n_samples == -1:
            indices = range(len(self.images))

        for idx in indices:
            plot_predictions(
                self.images[idx], self.truths[idx], self.preds[idx]
            )
