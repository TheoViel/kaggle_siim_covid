import sys
import torch
import torch.nn as nn

from params import SIZE, IMG_SIZE, MEAN, STD

sys.path.append("../yolov5/")
from utils.general import non_max_suppression  # noqa
from utils.loss import ComputeLoss  # noqa

try:
    sys.path.append("../efficientdet-pytorch")
    from effdet.factory import create_model  # noqa
    from effdet.bench import _post_process, _batch_detection  # noqa
    from effdet.anchors import Anchors, AnchorLabeler  # noqa
    from effdet.loss import DetectionLoss  # noqa
except ModuleNotFoundError:
    pass


def define_model(config):
    if "yolo" in config.selected_model:
        model = torch.hub.load(
            "../yolov5",
            config.selected_model,
            source="local",
            classes=config.num_classes,
        )
        model = YoloWrapper(model.model, config)

        model.mean = 0
        model.std = 1

    elif "efficientdet" in config.selected_model:
        model = create_model(
            config.selected_model,
            bench_task="",
            num_classes=config.num_classes,
            image_size=SIZE,
            pretrained=True,
        )

        model = EffDetWrapper(model, config)

        model.mean = MEAN
        model.std = STD

    else:
        raise NotImplementedError

    model.zero_grad()
    model.eval()
    return model


class EffDetWrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = model.config

        self.conf_thresh = config.conf_thresh
        self.iou_thresh = config.iou_thresh

        self.anchors = Anchors.from_config(self.config)
        self.anchor_labeler = AnchorLabeler(
            self.anchors, self.config.num_classes, match_threshold=0.5
        )

        self.loss_fn = DetectionLoss(self.config)

    def forward(self, x):
        if self.training:
            return self.model(x)

        else:
            class_out, box_out = self.model(x)

            class_out, box_out, indices, classes = _post_process(
                class_out,
                box_out,
                num_levels=self.config.num_levels,
                num_classes=self.config.num_classes,
                max_detection_points=self.config.max_detection_points,
            )

            pred_boxes = _batch_detection(
                x.shape[0],
                class_out,
                box_out,
                self.anchors.boxes,
                indices,
                classes,
                max_det_per_image=self.config.max_det_per_image,
                soft_nms=self.config.soft_nms,
                iou_threshold=self.iou_thresh,
            )
            return self.filter_predictions(pred_boxes)

    def filter_predictions(self, preds):
        filtered_preds = []
        for i in range(len(preds)):
            confidences = preds[i, :, 4]
            filtered_preds.append(preds[i][confidences > self.conf_thresh])
        return filtered_preds

    def compute_loss(self, x, target):
        class_out, box_out = x
        device = box_out[0].device

        target = [t.to(device) for t in target]
        classes = [torch.ones(t.size(0), device=device) for t in target]
        # TODO : Verif that classes start at 1
        (
            cls_targets,
            box_targets,
            num_positives,
        ) = self.anchor_labeler.batch_label_anchors(target, classes)

        loss, class_loss, box_loss = self.loss_fn(
            class_out, box_out, cls_targets, box_targets, num_positives
        )
        return loss


class YoloWrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        self.update_params()

        self.conf_thresh = config.conf_thresh
        self.iou_thresh = config.iou_thresh

        self.loss_fn = ComputeLoss(model, config)

    def forward(self, x):
        if self.training:
            return self.model(x)

        else:
            pred_boxes, _ = self.model(x)

            pred_boxes = non_max_suppression(
                pred_boxes, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
            )

            return pred_boxes

    def compute_loss(self, x, target):
        return self.loss_fn(x, target.to(x[0].device))[0]

    def update_params(self):
        nl = self.model.model[-1].nl
        self.config.box *= 3. / nl  # scale to layers
        self.config.cls *= 1 / 80. * 3. / nl  # scale to classes and layers
        self.config.obj *= (IMG_SIZE / 640) ** 2 * 3. / nl  # scale to image size and layers

        self. model.nc = 1  # attach number of classes to model
        self.model.config = self.config  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.class_weights = torch.ones(1).to(self.config.device) * 1  # attach class weights
