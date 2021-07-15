import albumentations as albu
from albumentations import pytorch as AT

from params import MEAN, STD, SIZE


def blur_transforms(p=0.5):
    """
    Applies MotionBlur, GaussianBlur or RandomFog random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(blur_limit=5, always_apply=True),
            albu.GaussianBlur(blur_limit=(1, 5), always_apply=True),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.RandomGamma(gamma_limit=(70, 130), always_apply=True),
            albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        ],
        p=p,
    )


def OCT_preprocess(
    augment=True, visualize=False, mean=MEAN, std=STD, size=SIZE, bbox_format="yolo"
):
    """
    Returns transformations for the OCT images.
    This version ensures masks keep a meaningful shape.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.
        mean (np array [3], optional): Mean for normalization. Defaults to MEAN.
        std (np array [3], optional): Standard deviation for normalization. Defaults to STD.
        std (int, optional): Image will be resized to (size, size * 3). Defaults to SIZE.
        bbox_format (str, optional): Bounding box format. Defaults to "yolo".

    Returns:
        albumentation transforms: transforms.
    """
    if visualize:
        normalizer = albu.Compose(
            [
                AT.transforms.ToTensorV2(),
            ],
            p=1,
        )
    else:
        normalizer = albu.Compose(
            [
                albu.Normalize(mean=0, std=1),
                AT.transforms.ToTensorV2(),
            ],
            p=1,
        )

    if augment:
        return albu.Compose(
            [
                # albu.ShiftScaleRotate(
                #     scale_limit=0.1, shift_limit=0.1, rotate_limit=45, p=0.5
                # ),
                albu.Resize(size[0], size[1]),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                color_transforms(p=0.5),
                blur_transforms(p=0.5),
                normalizer,
            ],
            bbox_params=albu.BboxParams(
                format=bbox_format, label_fields=["class_labels"], min_visibility=0.5
            ),
        )
    else:
        return albu.Compose(
            [
                albu.Resize(size[0], size[1]),
                normalizer,
            ],
            bbox_params=albu.BboxParams(format=bbox_format, label_fields=["class_labels"]),
        )
