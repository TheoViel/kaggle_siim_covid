import json
import numpy as np

from ensemble_boxes import weighted_boxes_fusion


def treat_boxes(boxes_string):
    """
    Treats the string of box dict in the coco format.

    Args:
        boxes_string (str): String to treat.

    Returns:
        list of lists: Treated boxes string.
    """
    try:
        boxes_string = boxes_string.replace("'", "\"")
    except AttributeError:
        return []

    boxes_dic = json.loads(boxes_string)

    boxes = []
    for b in boxes_dic:
        boxes.append([b['x'], b['y'], b['width'], b['height']])
    return boxes


def pascal_to_yolo(boxes, h=None, w=None):
    """
    x0, y0, x1, y1 -> Normalized xc, yc, w, h

    Args:
        boxes (np array): Boxes in the pascal format.
        h (int, optional): Image height. Defaults to None.
        w (int, optional): Image width. Defaults to None.

    Returns:
        np array: Boxes in the yolo format.
    """
    if not len(boxes):
        return boxes

    if h is not None and w is not None:
        boxes = boxes.astype(float)
        boxes[:, 0] = np.clip(boxes[:, 0] / w, 0, 1)
        boxes[:, 1] = np.clip(boxes[:, 1] / h, 0, 1)
        boxes[:, 2] = np.clip(boxes[:, 2] / w, 0, 1)
        boxes[:, 3] = np.clip(boxes[:, 3] / h, 0, 1)

    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 2] - boxes[:, 0]
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 3] - boxes[:, 1]

    return boxes


def pascal_to_albu(boxes, h, w):
    """
    x0, y0, x1, y1 -> Normalized x0, y0, x1, y1.
    Args:
        boxes (np array): Boxes in the pascal format.
        h (int): Image height.
        w (int,): Image width.

    Returns:
        np array: Boxes in the albu format.
    """
    if not len(boxes):
        return boxes

    boxes = boxes.astype(float)
    boxes[:, 0] = boxes[:, 0] / w
    boxes[:, 1] = boxes[:, 1] / h
    boxes[:, 2] = boxes[:, 2] / w
    boxes[:, 3] = boxes[:, 3] / h

    boxes = np.clip(boxes, 0, 1)

    return boxes


def albu_to_pascal(boxes, h, w):
    """
    Normalized x0, y0, x1, y1 -> x0, y0, x1, y1.
    Args:
        boxes (np array): Boxes in the albu format.
        h (int): Image height.
        w (int): Image width.

    Returns:
        np array: Boxes in the pascal format.
    """
    if not len(boxes):
        return boxes

    boxes = np.clip(boxes, 0, 1).astype(float)

    boxes[:, 0] = boxes[:, 0] * w
    boxes[:, 1] = boxes[:, 1] * h
    boxes[:, 2] = boxes[:, 2] * w
    boxes[:, 3] = boxes[:, 3] * h

    return boxes.astype(int)


def pascal_to_coco(boxes):
    """
    x0, y0, x1, y1 -> x0, y0, w, h
    Args:
        boxes (np array): Boxes in the pascal format.

    Returns:
        np array: Boxes in the yolo format.
    """
    if not len(boxes):
        return boxes
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    return boxes


def coco_to_pascal(boxes):
    """
    x0, y0, w, h -> x0, y0, x1, y1
    Args:
        boxes (np array): Boxes in the coco format.

    Returns:
        np array: Boxes in the pascal format.
    """
    if not len(boxes):
        return boxes
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    return boxes


def yolo_to_pascal(boxes, h=None, w=None):
    """
    Normalized xc, yc, w, h -> x0, y0, x1, y1

    Args:
        boxes (np array): Boxes in the yolo format
        h (int, optional): Image height. Defaults to None.
        w (int, optional): Image width. Defaults to None.

    Returns:
        np array: Boxes in the pascal format.
    """
    if not len(boxes):
        return boxes

    boxes[:, 0], boxes[:, 2] = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 0] + boxes[:, 2] / 2
    boxes[:, 1], boxes[:, 3] = boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 1] + boxes[:, 3] / 2

    if h is not None and w is not None:
        boxes[:, 0] = boxes[:, 0] * w
        boxes[:, 1] = boxes[:, 1] * h
        boxes[:, 2] = boxes[:, 2] * w
        boxes[:, 3] = boxes[:, 3] * h

        boxes = boxes.astype(int)

    return boxes


def expand_boxes(boxes, r=1):
    """
    Expands boxes. Handled in the coco format which is perhaps to the easiest.

    Args:
        boxes (Boxes): Boxes.
        r (int, optional): Exansion ratio. Defaults to 1.

    Returns:
        Boxes: Expanded boxes.
    """
    shape = boxes.shape
    boxes = boxes["yolo"]
    boxes[:, 2] = np.clip(boxes[:, 2] * r, 0, 0.75)
    boxes[:, 3] = np.clip(boxes[:, 3] * r, 0, 0.75)

    for b in boxes:  # shift boxes out of bounds
        if b[0] - b[2] / 2 < 0:
            b[2] += b[0] - b[2] / 2
            b[0] = b[2] / 2

        if b[0] + b[2] / 2 > 1:
            b[2] -= b[0] + b[2] / 2 - 1
            b[0] = 1 - (b[2] / 2)

        if b[1] - b[3] / 2 < 0:
            b[3] += b[1] - b[3] / 2
            b[1] = b[3] / 2

        if b[1] + b[3] / 2 > 1:
            b[3] -= b[1] + b[3] / 2 - 1
            b[1] = 1 - (b[3] / 2)

    return Boxes(boxes, shape, bbox_format="yolo")


def box_fusion(boxes, iou_threshold=0.5, return_once=True):
    """
    Fuses boxes using wbf.

    Args:
        boxes (list of N Boxes): Boxes to fuse.
        iou_threshold (float, optional): IoU threshold for wbf. Defaults to 0.5.
        return_once (bool, optional): Whether to return the result once or x N. Defaults to True.

    Returns:
        Boxes or list of Boxes: Fused boxes.
    """
    boxes_albu = [box["albu"].copy() for box in boxes]

    confidences = [[1] * len(p) for p in boxes_albu]
    labels = [[0] * len(p) for p in boxes_albu]

    pred_wbf, confidences_wbf, _ = weighted_boxes_fusion(
        boxes_albu, confidences, labels, iou_thr=iou_threshold
    )

    if return_once:
        return Boxes(pred_wbf, boxes[0].shape, bbox_format="albu")
    else:
        return [Boxes(pred_wbf, boxes[0].shape, bbox_format="albu") for _ in range(len(boxes))]


def merge_boxes(boxes, transpositions):
    """
    Merges boxes and handles the transposed ones.

    Args:
        boxes (list of Boxes): Boxes to merge.
        transpositions (list of bool): Whether to transpose the boxes.

    Returns:
        List of boxes: Merged boxes.
    """
    for box, tran in zip(boxes, transpositions):
        if tran:
            box.hflip()

    fused_boxes = box_fusion(boxes, return_once=False)

    for box, tran in zip(fused_boxes, transpositions):
        if tran:
            box.hflip()

    return fused_boxes


class Boxes:
    """
    Class to handle different format of bounding boxes easily.
    Supports boxes of formats yolo, coco, pascal_voc and albumentations.
    """
    def __init__(self, data, shape, bbox_format="yolo"):
        """
        Constructor.

        Args:
            data (np array [N x 4]): Boxe coordinates.
            shape (np array [2 or 3]): Associated image shape.
            bbox_format (str, optional): Boxes format. Defaults to "yolo".

        Raises:
            NotImplementedError: bbox_format not supported.
        """
        h, w = shape[:2]
        self.shape = shape[:2]
        self.h = h
        self.w = w

        if bbox_format == "yolo":
            self.boxes_yolo = data
            self.boxes_pascal = yolo_to_pascal(self.boxes_yolo.copy(), h, w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), h, w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
        elif bbox_format == "pascal_voc":
            self.boxes_pascal = data
            self.boxes_yolo = pascal_to_yolo(self.boxes_pascal.copy(), h, w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), h, w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
        elif bbox_format == "albu":
            self.boxes_albu = data
            self.boxes_pascal = albu_to_pascal(self.boxes_albu.copy(), h, w)
            self.boxes_yolo = pascal_to_yolo(self.boxes_pascal.copy(), h, w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
        elif bbox_format == "coco":
            self.boxes_coco = data
            self.boxes_pascal = coco_to_pascal(self.boxes_coco.copy())
            self.boxes_yolo = pascal_to_yolo(self.boxes_pascal.copy(), h, w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), h, w)
        else:
            raise NotImplementedError

    def __getitem__(self, bbox_format):
        """
        Item accessor.

        Args:
            bbox_format (str): Format to return the boxes as.

        Raises:
            NotImplementedError: bbox_format not supported.

        Returns:
            np array: Boxes.
        """
        if bbox_format == "yolo":
            return self.boxes_yolo
        elif bbox_format == "pascal_voc":
            return self.boxes_pascal
        elif bbox_format == "albu":
            return self.boxes_albu
        elif bbox_format == "coco":
            return self.boxes_coco
        else:
            print(bbox_format)
            raise NotImplementedError

    def __len__(self):
        return len(self.boxes_yolo)

    def resize(self, shape):
        """
        Resize boxes.

        Args:
            shape (np array [2 or 3]): New shape.
        """
        self.shape = shape
        self.h, self.w = shape[0], shape[1]
        self.boxes_pascal = yolo_to_pascal(self.boxes_yolo.copy(), self.h, self.w)
        self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())

    def fill(self, img, value=1):
        """
        Fills an image with the boxes.

        Args:
            img (np array): Image to fill.
            value (int, optional): Value to fill with. Defaults to 1.
        """
        for box in self.boxes_pascal:
            img[box[1]: box[3], box[0]: box[2]] = value

    def hflip(self):
        """
        Flips the boxes horizontally.
        """
        if len(self.boxes_yolo):
            self.boxes_yolo[:, 0] = 1 - self.boxes_yolo[:, 0]

            self.boxes_pascal = yolo_to_pascal(self.boxes_yolo.copy(), self.h, self.w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), self.h, self.w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
