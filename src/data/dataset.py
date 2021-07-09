import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from params import CLASSES


class CovidDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir="",
        transforms=None,
        train=False,
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

        self.img_names = df["image_id"].values
        self.targets = df[CLASSES].values
        self.studies = df["study_id"].values

        self.original_shapes = df[["dim1", "dim0"]].values
        self.boxes = df['boxes']

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
        image = cv2.imread(self.root_dir + self.img_names[idx] + ".png")

        boxes = np.array(self.boxes[idx])
        if len(boxes):
            boxes[:, 0] /= self.original_shapes[idx][0]
            boxes[:, 1] /= self.original_shapes[idx][1]
            boxes[:, 2] /= self.original_shapes[idx][0]
            boxes[:, 3] /= self.original_shapes[idx][1]

        if self.transforms:
            image = self.transforms(image=image)["image"]

        y = torch.tensor(self.targets[idx], dtype=torch.float)

        return image, y, boxes
