import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from params import CLASSES, SIZE
from utils.boxes import Boxes


class CovidDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir="",
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
        self.df = df
        self.train = train
        self.root_dir = root_dir
        self.transforms = transforms
        self.bbox_format = bbox_format

        self.img_names = df["save_name"].values
        self.targets = df[CLASSES].values
        self.studies = df["study_id"].values

        self.get_boxes()

    def get_boxes(self):
        self.original_shapes = self.df["shape"].values
        self.crop_starts = self.df['crop_starts'].values
        self.boxes = []
        for boxes, orig_shape, starts in self.df[['boxes', 'shape', 'crop_starts']].values:
            boxes = np.array(boxes).astype(float)

            if len(boxes):
                boxes[:, 0] -= starts[0]
                boxes[:, 1] -= starts[1]

            boxes = Boxes(boxes, orig_shape, bbox_format="coco")
            boxes.resize((SIZE, SIZE))
            self.boxes.append(boxes)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            torch tensor [C x H x W]: Image.
            torch tensor [NUM_CLASSES]: Label.
        """
        image = cv2.imread(self.root_dir + self.img_names[idx])
        boxes = self.boxes[idx][self.bbox_format]

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes, class_labels=[0] * len(boxes)
            )
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"])

        y = torch.tensor(self.targets[idx], dtype=torch.float)
        # y = np.array(self.targets[idx])

        return image, y, boxes
