import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from params import CLASSES, SIZE
from utils.boxes import Boxes
from data.extraction import auto_windowing


class CovidClsDataset(Dataset):
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

        self.img_names = df["save_name"].values
        self.study_targets = df[CLASSES].values.argmax(-1)
        self.img_targets = df["img_target"].values
        self.studies = df["study_id"].values

        self.get_boxes()
        self.get_boxes_lungs()

    def get_boxes(self):
        self.original_shapes = self.df["shape"].values
        self.crop_starts = self.df['crop_starts'].values
        self.boxes = []
        for boxes, orig_shape, starts in self.df[
            ['boxes', 'shape_crop', 'crop_starts']
        ].values:
            boxes = np.array(boxes).astype(float)

            if len(boxes):
                boxes[:, 0] -= starts[1]
                boxes[:, 1] -= starts[0]

            boxes = Boxes(boxes, orig_shape, bbox_format="coco")
            boxes.resize((SIZE, SIZE))
            self.boxes.append(boxes)

    # def get_boxes_lungs(self):
    #     self.original_shapes = self.df["shape"].values
    #     self.crop_starts = self.df['crop_starts'].values
    #     self.boxes_lungs = []
    #     for boxes, orig_shape, starts in self.df[
    #         ['boxes_lung', 'shape_crop', 'crop_starts']
    #     ].values:
    #         boxes = np.array([boxes]).astype(float)

    #         if len(boxes):
    #             boxes[:, 0] -= starts[1]
    #             boxes[:, 1] -= starts[0]
    #             boxes[:, 2] -= starts[1]
    #             boxes[:, 3] -= starts[0]

    #         boxes = Boxes(boxes, orig_shape, bbox_format="pascal_voc")
    #         boxes.resize((SIZE, SIZE))
    #         boxes.expand(1.1)
    #         self.boxes_lungs.append(boxes)

    def get_boxes_lungs(self):
        self.boxes_lungs = []
        for boxes in self.df['boxes_lung']:
            x_start = boxes[:, 0].min()
            y_start = boxes[:, 1].min()
            x_end = boxes[:, 2].max()
            y_end = boxes[:, 3].max()

            boxes = Boxes(
                np.array([[x_start, y_start, x_end, y_end]]), (512, 512), bbox_format="albu"
            )
            self.boxes_lungs.append(boxes)

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

        mask = np.zeros(image.shape[:-1])
        self.boxes[idx].fill(mask)

        lungs = self.boxes_lungs[idx]
        lungs.expand(1.1)
        lungs.clip()

        image = lungs.crop(image)
        mask = lungs.crop(mask)

        image, _ = auto_windowing(image)

        if self.transforms:
            transformed = self.transforms(
                image=image, mask=mask
            )
            image = transformed["image"]
            mask = transformed["mask"]

        y_study = torch.tensor(self.study_targets[idx], dtype=torch.float)
        y_img = torch.tensor(self.img_targets[idx], dtype=torch.float)

        return image, mask, y_study, y_img


class CovidDetDataset(Dataset):
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
        self.study_targets = df[CLASSES].values.argmax(-1)
        self.img_targets = df["img_target"].values
        self.studies = df["study_id"].values

        self.get_boxes()
        self.get_boxes_lungs()

    def get_boxes(self):
        self.original_shapes = self.df["shape"].values
        self.crop_starts = self.df['crop_starts'].values
        self.boxes = []
        for boxes, orig_shape, starts in self.df[['boxes', 'shape_crop', 'crop_starts']].values:
            boxes = np.array(boxes).astype(float)

            if len(boxes):
                boxes[:, 0] -= starts[1]
                boxes[:, 1] -= starts[0]

            boxes = Boxes(boxes, orig_shape, bbox_format="coco")
            boxes.resize((SIZE, SIZE))
            self.boxes.append(boxes)

    # def get_boxes_lungs(self):
    #     self.original_shapes = self.df["shape"].values
    #     self.crop_starts = self.df['crop_starts'].values
    #     self.boxes_lungs = []
    #     for boxes, orig_shape, starts in self.df[
    #         ['boxes_lung', 'shape_crop', 'crop_starts']
    #     ].values:
    #         boxes = np.array(boxes).astype(float)

    #         if len(boxes):
    #             boxes[:, 0] -= starts[1]
    #             boxes[:, 1] -= starts[0]
    #             boxes[:, 2] -= starts[1]
    #             boxes[:, 3] -= starts[0]

    #         boxes = Boxes(boxes, orig_shape, bbox_format="pascal_voc")
    #         boxes.resize((SIZE, SIZE))
    #         self.boxes_lungs.append(boxes)

    def get_boxes_lungs(self):
        self.boxes_lungs = []
        for boxes in self.df['boxes_lung']:
            x_start = boxes[:, 0].min()
            y_start = boxes[:, 1].min()
            x_end = boxes[:, 2].max()
            y_end = boxes[:, 3].max()

            boxes = Boxes(
                np.array([[x_start, y_start, x_end, y_end]]), (512, 512), bbox_format="albu"
            )
            self.boxes_lungs.append(boxes)

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
        boxes = self.boxes_lungs[idx]
        b = self.boxes_lungs[idx]['pascal_voc'].tolist()
        boxes.expand(1.1)
        boxes.clip()
        b += self.boxes_lungs[idx]['pascal_voc'].tolist()

        boxes = Boxes(np.array(b), boxes.shape, bbox_format="pascal_voc")

        mask = np.zeros(image.shape[:-1])

        boxes.fill(mask)
        boxes = boxes[self.bbox_format]

        if self.transforms:
            transformed = self.transforms(
                image=image, mask=mask, bboxes=boxes, class_labels=[0] * len(boxes)
            )
            image = transformed["image"]
            mask = transformed["mask"]
            boxes = np.array(transformed["bboxes"])

        y_study = torch.tensor(self.study_targets[idx], dtype=torch.float)
        y_img = torch.tensor(self.img_targets[idx], dtype=torch.float)

        return image, mask, y_study, y_img, boxes


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
            img_name (str, optional): Column corresponding to the image. Defaults to "img_name".
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
