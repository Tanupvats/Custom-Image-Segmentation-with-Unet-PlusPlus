
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(cfg) -> A.Compose:
    image_size = cfg.data.image_size
    a = cfg.augment
    return A.Compose([
        # Resize-with-aspect-ratio first using LongestMaxSize, then PadIfNeeded
        # would preserve aspect. For simplicity and consistency with the original
        # we use plain Resize. Switch to the pad-preserving variant if your
        # dataset has very non-square images.
        A.Resize(image_size, image_size, interpolation=1),  # 1=INTER_LINEAR for image
        A.HorizontalFlip(p=a.horizontal_flip_p),
        A.ShiftScaleRotate(
            shift_limit=a.shift_limit,
            scale_limit=a.scale_limit,
            rotate_limit=a.rotate_limit,
            border_mode=0,
            p=a.shift_scale_rotate_p,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=a.brightness_contrast_p
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=a.hue_saturation_p
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=a.gauss_noise_p),
        A.MotionBlur(blur_limit=5, p=a.motion_blur_p),
        A.CoarseDropout(
            max_holes=a.cutout_max_holes,
            max_height=a.cutout_max_size,
            max_width=a.cutout_max_size,
            fill_value=0,
            mask_fill_value=cfg.data.ignore_index,  # don't supervise on dropout
            p=a.cutout_p,
        ),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def build_val_transform(cfg) -> A.Compose:
    image_size = cfg.data.image_size
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=1),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])
