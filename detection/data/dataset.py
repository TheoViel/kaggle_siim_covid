import cv2
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset

from util.saving import load_stack
from util.boxes import Boxes, yolo_to_pascal
from params import ORIG_SIZE


class FollicleDataset(Dataset):
    """
    Segmentation dataset
    """

    def __init__(self, df, transforms=None, mosaic_proba=0, root="", bbox_format="yolo"):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata.
            transforms (albu transforms, optional): Augmentations. Defaults to None.
            mosaic_proba (int, optional): Probability to apply mosaicing with. Defaults to 0.
            root (str, optional): Folder containing images. Defaults to "".
            bbox_format (str, optional): Boxes format. Defaults to "yolo".
        """
        self.df = df
        self.root = root
        self.transforms = transforms
        self.mosaic_proba = mosaic_proba

        self.img_paths = df["path"].values
        self.stacks = df["stack"].values

        self.bbox_format = bbox_format
        self.boxes = [Boxes(vol, ORIG_SIZE, "volume") for vol in df["volumes"].values]

        # Mosaicing
        bbox_params = albu.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.35
        )
        self.pre_mosaic_aug = albu.Compose(
            [albu.RandomCrop(int(ORIG_SIZE[0] * 0.75), int(ORIG_SIZE[1] * 0.75))],
            bbox_params=bbox_params,
        )
        self.post_mosaic_aug = albu.Compose(
            [albu.RandomCrop(ORIG_SIZE[0], ORIG_SIZE[1])],
            bbox_params=bbox_params,
        )

    def getitem_mosaic(self, idx):
        images, new_boxes = [], []
        indices = [idx] + np.random.choice(
            len(self), 3
        ).tolist()  # 3 additional image indices

        # Sample and random crop
        for i, index in enumerate(indices):
            image = cv2.imread(self.root + self.img_paths[index])
            boxes = np.array(self.boxes[index]["yolo"])

            transformed = self.pre_mosaic_aug(
                image=image, bboxes=boxes, class_labels=[0] * len(boxes)
            )

            images.append(transformed["image"])
            new_boxes.append(np.array(transformed["bboxes"]) / 2)

        # New image
        new_img = np.concatenate(
            [
                np.concatenate([images[0], images[2]], 0),
                np.concatenate([images[1], images[3]], 0),
            ],
            1,
        )

        # New boxes
        if len(new_boxes[1]):
            new_boxes[1][:, 0] += 0.5
        if len(new_boxes[2]):
            new_boxes[2][:, 1] += 0.5
        if len(new_boxes[3]):
            new_boxes[3][:, 0] += 0.5
            new_boxes[3][:, 1] += 0.5

        new_boxes = np.concatenate([b for b in new_boxes if len(b)])
        if len(new_boxes):
            new_boxes[:, 0] = np.clip(new_boxes[:, 0], new_boxes[:, 2] / 2, 1 - new_boxes[:, 2] / 2)
            new_boxes[:, 1] = np.clip(new_boxes[:, 1], new_boxes[:, 3] / 2, 1 - new_boxes[:, 3] / 2)

        # Back to desired size
        transformed = self.post_mosaic_aug(
            image=new_img, bboxes=new_boxes, class_labels=[0] * len(new_boxes)
        )

        # Process boxes
        new_boxes = np.array(transformed["bboxes"])
        if len(new_boxes):
            new_boxes[:, 0] = np.clip(new_boxes[:, 0], new_boxes[:, 2] / 2, 1 - new_boxes[:, 2] / 2)
            new_boxes[:, 1] = np.clip(new_boxes[:, 1], new_boxes[:, 3] / 2, 1 - new_boxes[:, 3] / 2)

        if self.bbox_format == "pascal_voc":
            new_boxes = yolo_to_pascal(
                new_boxes, transformed["image"].shape[0], transformed["image"].shape[1]
            )
        elif self.bbox_format != "yolo":
            raise NotImplementedError

        return transformed["image"], new_boxes

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if np.random.random() < self.mosaic_proba:
            image, boxes = self.getitem_mosaic(idx)
        else:
            image = cv2.imread(self.root + self.img_paths[idx])
            boxes = self.boxes[idx][self.bbox_format]

        shape = image.shape

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes, class_labels=[0] * len(boxes)
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]

        return image, np.array(boxes), shape


class InferenceDataset(Dataset):
    """
    Follicle dataset for inference.
    """

    def __init__(self, stack_path, seg_path, step=5, transforms=None):
        """
        Constructor

        Args:
            images (list of np arrays): Images.
            transforms (albumentations transforms): Transforms.
        """
        stack = load_stack(stack_path)
        seg = load_stack(seg_path)

        surf_height = int(np.percentile((seg == 30).argmax(1).flatten(), 90)) + 10
        jde_height = int(np.percentile((seg == 10).argmax(1).flatten(), 75))

        self.images = [stack[:, i] for i in range(surf_height, jde_height, step)]
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            torch tensor [C x H x W]: Image.
            0: placeholder.
        """
        image = np.dstack([self.images[idx]] * 3)
        shape = image.shape

        if self.transforms:
            image = self.transforms(image=image, bboxes=[], class_labels=[])["image"]

        return image, [], shape
