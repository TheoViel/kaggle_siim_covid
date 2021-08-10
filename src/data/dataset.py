import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from params import CLASSES, SIZE
from utils.boxes import Boxes


class CovidClsDataset(Dataset):
    def __init__(
        self,
        df,
        df_extra=None,
        transforms=None,
        train=True,
        extra_prop=0.5,
        fold=0,
        iter_per_epochs=None
    ):
        """
        Constructor

        Args:
            df (pandas dataframe): Metadata.
            img_name (str, optional): Column corresponding to the image. Defaults to "img_name".
        """
        self.df = df
        self.df_extra = df_extra
        self.train = train
        self.transforms = transforms
        self.extra_prop = extra_prop
        self.train = train
        self.fold = fold
        self.iter_per_epochs = iter_per_epochs

        self.img_names = df["save_name"].values
        self.study_targets = df[CLASSES].values.argmax(-1)
        self.img_targets = df["img_target"].values
        self.studies = df["study_id"].values
        self.unique_studies = np.unique(self.studies)

        self.is_pl = df['is_pl'].values
        self.root_dirs = df['root'].values

        if df_extra is not None:
            self.study_targets_ext = df_extra[[f'pl_{c}_{fold}' for c in CLASSES]].values
            self.study_targets_ext /= self.study_targets_ext.max(0, keepdims=True)

            self.img_targets_ext = df_extra[f'pl_img_{fold}'].values
            self.img_targets_ext /= self.img_targets_ext.max()

        self.get_boxes()

    def get_boxes(self):
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
        return len(self.df) if self.iter_per_epochs is None else self.iter_per_epochs

    def getitem_extra(self):
        idx = np.random.randint(len(self.df_extra))

        image = cv2.imread(self.df_extra['path'][idx])

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y_study = torch.tensor(self.study_targets_ext[idx], dtype=torch.float)
        y_img = torch.tensor(self.img_targets_ext[idx], dtype=torch.float)
        y_aux = torch.tensor(self.df_extra['target'][idx], dtype=torch.float)
        is_pl = torch.tensor(1, dtype=torch.float)

        mask = torch.zeros(image.shape[1:])

        return image, mask, y_study, y_img, y_aux, is_pl

    def getitem(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            torch tensor [C x H x W]: Image.
            torch tensor [NUM_CLASSES]: Label.
        """
        if self.train:  # sample according to study
            study = np.random.choice(self.unique_studies)
            idx = np.random.choice(self.df[self.df['study_id'] == study].index)

        image = cv2.imread(self.root_dirs[idx] + self.img_names[idx])

        mask = np.zeros(image.shape[:-1])
        self.boxes[idx].fill(mask)

        if self.transforms:
            transformed = self.transforms(
                image=image, mask=mask
            )
            image = transformed["image"]
            mask = transformed["mask"]

        y_study = np.zeros(len(CLASSES))
        y_study[self.study_targets[idx]] = 1
        y_study = torch.tensor(y_study, dtype=torch.float)
        y_img = torch.tensor(self.img_targets[idx], dtype=torch.float)
        y_aux = torch.tensor(0, dtype=torch.float)
        is_pl = torch.tensor(self.is_pl[idx], dtype=torch.float)

        return image, mask, y_study, y_img, y_aux, is_pl

    def __getitem__(self, idx):
        if np.random.random() < self.extra_prop and self.train:
            return self.getitem_extra()
        else:
            return self.getitem(idx)


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
        boxes = self.boxes[idx]

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
