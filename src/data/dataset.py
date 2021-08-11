import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from params import CLASSES, SIZE
from utils.boxes import Boxes


class CovidClsDataset(Dataset):
    """
    Dataset for classification.
    """
    def __init__(
        self,
        df,
        transforms=None,
    ):
        """
        Constructor

        Args:
            df (pandas dataframe): Metadata.
            transforms (albu transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.transforms = transforms

        self.img_names = df["save_name"].values
        self.study_targets = df[CLASSES].values.argmax(-1)
        self.img_targets = df["img_target"].values
        self.studies = df["study_id"].values

        self.root_dirs = df['root'].values

        self.get_boxes()

    def get_boxes(self):
        """
        Retrieves opacity boxes.
        """
        self.original_shapes = self.df["shape"].values
        self.crop_starts = self.df['crop_starts'].values
        self.boxes = []
        for boxes, orig_shape, starts in self.df[['boxes', 'shape_crop', 'crop_starts']].values:
            boxes = np.array(boxes).astype(float)
            try:
                _ = len(boxes)
            except TypeError:  # nan
                boxes = np.array([])

            if len(boxes):
                boxes[:, 0] -= starts[1]
                boxes[:, 1] -= starts[0]

            boxes = Boxes(boxes, orig_shape, bbox_format="coco")
            boxes.resize((SIZE, SIZE))
            self.boxes.append(boxes)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch tensor [C x H x W]: Image.
            torch tensor [H x W]: Mask.
            torch tensor [NUM_CLASSES]: Study label.
            torch tensor [1]: Image label.
        """
        image = cv2.imread(self.root_dirs[idx] + self.img_names[idx])

        mask = np.zeros(image.shape[:-1])
        self.boxes[idx].fill(mask)

        if self.transforms:
            transformed = self.transforms(
                image=image, mask=mask
            )
            image = transformed["image"]
            mask = transformed["mask"]

        y_study = torch.tensor(self.study_targets[idx], dtype=torch.float)
        y_img = torch.tensor(self.img_targets[idx], dtype=torch.float)

        return image, mask, y_study, y_img


class CovidInfDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir="",
        transforms=None,
    ):
        """
        Constructor

        Args:
            df (pandas dataframe): Metadata.
            root_dir (str): Directory with all the images. Defaults to "".
            transforms (albu transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms

        self.img_names = df["save_name"].values

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

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image
