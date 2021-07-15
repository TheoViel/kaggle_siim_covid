import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from util.metrics import compute_metrics
from util.plot import plot_predictions
from util.boxes import Boxes


class DetectionMeter:
    """
    Meter to handle predictions & metrics.
    """
    def __init__(self, stacks, pred_format="coco", truth_format="yolo"):
        """
        Constructor

        TODO : Update
        """
        self.stacks = stacks
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

            # print(pred)
            self.preds.append(Boxes(pred, (h, w), bbox_format=self.pred_format))
            self.confidences.append(confidences)

    def fusion(self, iou_threshold=0.2):
        self.truths_fusion = []
        self.pred_fusion = []
        self.confidences_fusion = []
        self.images_fusion = []

        for stack in np.unique(self.stacks):
            idxs = np.arange(len(self.stacks))[self.stacks == stack]

            pred_stack = [self.preds[i]["albu"].copy() for i in idxs]
            confidences_stack = [self.confidences[i] for i in idxs]
            # pred_stack = [yolo_to_pascal(pred.copy()) for pred in pred_stack]
            labels = [[0] * len(p) for p in pred_stack]

            pred_wbf, confidences_wbf, _ = weighted_boxes_fusion(
                pred_stack, confidences_stack, labels, iou_thr=iou_threshold
            )

            pred_wbf = Boxes(
                pred_wbf,
                shape=(self.preds[idxs[0]].h, self.preds[idxs[0]].w),
                bbox_format="albu"
            )

            self.truths_fusion.append(self.truths[idxs[0]])
            self.images_fusion.append(self.images[idxs[len(idxs) // 2]])
            self.pred_fusion.append(pred_wbf)
            self.confidences_fusion.append(confidences_wbf)

        return self.truths_fusion, self.pred_fusion, self.confidences_fusion

    def compute(self, iou_threshold=0.2):
        """
        Computes the metrics.

        Returns:
            dict: Metrics dictionary.
        """
        self.fusion(iou_threshold=iou_threshold)
        return compute_metrics(self.pred_fusion, self.truths_fusion)

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

    def plot(self, n_samples=1, fused=True):
        """
        Visualize results.

        Args:
            n_samples (int, optional): Number of samples to plot. Defaults to 1.
        """
        if fused:
            if n_samples > 0:
                indices = np.random.choice(len(self.images_fusion), n_samples, replace=False)
            elif n_samples == -1:
                indices = range(len(self.images_fusion))

            for idx in indices:
                plot_predictions(
                    self.images_fusion[idx], self.truths_fusion[idx], self.pred_fusion[idx]
                )

        else:
            if n_samples > 0:
                indices = np.random.choice(len(self.images), n_samples, replace=False)
            elif n_samples == -1:
                indices = range(len(self.images))

            for idx in indices:
                plot_predictions(
                    self.images[idx], self.truths[idx], self.preds[idx]
                )
