import cv2
import albumentations as albu
from albumentations import pytorch as AT

from params import MEAN, STD


def blur_transforms(p=0.5, blur_limit=5):
    """
    Applies MotionBlur or GaussianBlur random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.
        blur_limit (int, optional): Blur intensity limit. Defaults to 5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(always_apply=True),
            albu.GaussianBlur(always_apply=True),
        ],
        p=p,
    )


def noise_transforms(p=0.5):
    """
    Applies GaussNoise or RandomFog random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.GaussNoise(var_limit=(1.0, 50.0), always_apply=True),
            albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.25, always_apply=True),
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
            albu.RandomGamma(gamma_limit=(80, 120), always_apply=True),
            albu.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, always_apply=True
            ),
        ],
        p=p,
    )


def distortion_transforms(p=0.5):
    """
    Applies ElasticTransform with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.ElasticTransform(
                alpha=1,
                sigma=5,
                alpha_affine=10,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        ],
        p=p,
    )


def dropout_transforms(p=0.5):
    """
    Applies CoarseDropout with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.CoarseDropout(
                max_holes=32,
                max_height=32,
                max_width=32,
                min_holes=16,
                min_height=8,
                min_width=8,
                fill_value=0,
                always_apply=True,
            ),
        ],
        p=p,
    )


def get_tranfos_inference(mean, std):
    """
    Returns transformations for inference.

    Returns:
        albumentation transforms: transforms.
    """

    return albu.Compose(
        [
            albu.Normalize(mean=mean, std=std),
            AT.transforms.ToTensorV2(),
        ],
        p=1,
    )


def get_transfos(
    augment=True, visualize=False, mean=MEAN, std=STD, bbox_format="yolo"
):
    """
    Returns transformations for the OCT images.
    This version ensures masks keep a meaningful shape.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.
        mean (np array [3], optional): Mean for normalization. Defaults to MEAN.
        std (np array [3], optional): Standard deviation for normalization. Defaults to STD.
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
                albu.Normalize(mean=mean, std=std),
                AT.transforms.ToTensorV2(),
            ],
            p=1,
        )

    if augment:
        return albu.Compose(
            [
                albu.ShiftScaleRotate(
                    scale_limit=0.1, shift_limit=0, rotate_limit=20, p=0.5
                ),
                albu.HorizontalFlip(p=0.5),
                color_transforms(p=0.5),
                blur_transforms(p=0.5),
                # dropout_transforms(p=0.1),
                normalizer,
            ],
            bbox_params=albu.BboxParams(
                format=bbox_format, label_fields=["class_labels"], min_visibility=0.1
            ),
        )
    else:
        return albu.Compose(
            [
                normalizer,
            ],
            bbox_params=albu.BboxParams(format=bbox_format, label_fields=["class_labels"]),
        )
