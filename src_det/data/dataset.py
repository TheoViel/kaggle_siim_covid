import cv2
import numpy as np
from torch.utils.data import Dataset

from util.boxes import Boxes


class LungDataset(Dataset):
    def __init__(
        self,
        boxes_dic_cxr,
        boxes_dic_siim,
        transforms=None,
        train=False,
        bbox_format="yolo"
    ):
        """
        Constructor

        Args:
            df (pandas dataframe): Metadata.
            root_dir (str): Directory with all the images. Defaults to "".
            img_name (str, optional): Column corresponding to the image. Defaults to "img_name".
        """
        self.train = train
        self.transforms = transforms
        self.bbox_format = bbox_format

        self.img_names = list(boxes_dic_cxr.keys()) + list(boxes_dic_siim.keys())
        self.boxes = list(boxes_dic_cxr.values()) + list(boxes_dic_siim.values())
        self.formats = ['coco'] * len(boxes_dic_cxr) + ['yolo'] * len(boxes_dic_siim)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            torch tensor [C x H x W]: Image.
            torch tensor [NUM_CLASSES]: Label.
        """
        image = cv2.imread(self.img_names[idx])

        h, w, _ = image.shape
        boxes = Boxes(
            np.array(self.boxes[idx]),
            (h, w),
            bbox_format=self.formats[idx]
        )[self.bbox_format]

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes, class_labels=[0] * len(boxes)
            )
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"])

        return image, boxes, (h, w)
