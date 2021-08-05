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
    boxes[:, 2] = np.clip(boxes[:, 2] * r, 0, 1)
    boxes[:, 3] = np.clip(boxes[:, 3] * r, 0, 1)

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
    """
    def __init__(self, data, shape, bbox_format="yolo"):
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
        self.shape = shape
        self.h, self.w = shape[0], shape[1]
        self.boxes_pascal = yolo_to_pascal(self.boxes_yolo.copy(), self.h, self.w)
        self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())

    def fill(self, img, value=1):
        for box in self.boxes_pascal:
            img[box[1]: box[3], box[0]: box[2]] = value

    def crop(self, img, idx=0):
        box = self.boxes_pascal[idx]
        return img[box[1]: box[3], box[0]: box[2]]

    def expand(self, r, max_w=1):
        boxes = self["yolo"]
        boxes[:, 2] = np.clip(boxes[:, 2] * r, 0, max_w)
        boxes[:, 3] = np.clip(boxes[:, 3] * r, 0, max_w)

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

        self.boxes_yolo = boxes
        self.boxes_pascal = yolo_to_pascal(self.boxes_yolo.copy(), self.h, self.w)
        self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), self.h, self.w)
        self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())

    def hflip(self):
        if len(self.boxes_yolo):
            self.boxes_yolo[:, 0] = 1 - self.boxes_yolo[:, 0]

            self.boxes_pascal = yolo_to_pascal(self.boxes_yolo.copy(), self.h, self.w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), self.h, self.w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())

    def clip(self):
        if len(self.boxes_pascal):
            self.boxes_pascal[:, 0] = np.clip(self.boxes_pascal[:, 0], 0, self.w - 1)
            self.boxes_pascal[:, 1] = np.clip(self.boxes_pascal[:, 1], 0, self.h - 1)
            self.boxes_pascal[:, 2] = np.clip(self.boxes_pascal[:, 2], 0, self.w - 1)
            self.boxes_pascal[:, 3] = np.clip(self.boxes_pascal[:, 3], 0, self.h - 1)

            self.boxes_yolo = pascal_to_yolo(self.boxes_pascal.copy(), self.h, self.w)
            self.boxes_albu = pascal_to_albu(self.boxes_pascal.copy(), self.h, self.w)
            self.boxes_coco = pascal_to_coco(self.boxes_pascal.copy())
