import albumentations as albu
from albumentations import pytorch as AT

from params import IMG_SIZE


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
            albu.MotionBlur(blur_limit=10, always_apply=True),
            albu.GaussianBlur(blur_limit=(1, 11), always_apply=True),
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
            albu.RandomGamma(gamma_limit=(60, 150), always_apply=True),
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ],
        p=p,
    )


def get_transfos_lung(
    augment=True, mean=None, std=None, bbox_format="yolo"
):
    """
    Returns transformations for detection.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        mean (np array [3], optional): Mean for normalization. Defaults to None.
        std (np array [3], optional): Standard deviation for normalization. Defaults to None.
        bbox_format (str, optional): Bounding box format. Defaults to "yolo".

    Returns:
        albumentation transforms: transforms.
    """
    bbox_params = albu.BboxParams(
        format=bbox_format, label_fields=["class_labels"], min_visibility=0.1
    )
    if mean is None or std is None:
        normalizer = albu.Compose(
            [
                AT.transforms.ToTensorV2(),
            ],
            p=1,
        )
    else:
        normalizer = albu.Compose(
            [
                albu.Normalize(mean=mean, std=std),
                AT.transforms.ToTensorV2(),
            ],
            p=1,
        )

    if augment:
        return albu.Compose(
            [
                albu.Resize(IMG_SIZE, IMG_SIZE),
                albu.ShiftScaleRotate(
                    scale_limit=(-0.2, 0.1), shift_limit=0.1, rotate_limit=20, p=1,
                ),
                albu.HorizontalFlip(p=0.5),
                color_transforms(p=0.75),
                blur_transforms(p=0.75),
                normalizer,
            ],
            bbox_params=bbox_params
        )
    else:
        return albu.Compose(
            [
                albu.Resize(IMG_SIZE, IMG_SIZE),
                normalizer,
            ],
            bbox_params=bbox_params
        )
