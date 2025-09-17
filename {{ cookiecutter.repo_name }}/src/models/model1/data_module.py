import math
import os

import lightning as L
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from models.swrd_model.dataset import WeldDataset


class ResizeWithAspectRatioAndPad:
    """Resize image to target width while maintaining aspect ratio, then pad height to target."""

    def __init__(self, target_width, target_height):
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, img, mask=None):
        # Get original dimensions
        h, w = img.shape[-2:]

        # Calculate new height maintaining aspect ratio
        new_h = int(h * (self.target_width / w))

        # Resize to target width and calculated height
        resize_transform = v2.Resize(size=(new_h, self.target_width), antialias=True)
        img = resize_transform(img)

        # Calculate padding needed for height
        pad_bottom = max(0, self.target_height - new_h)

        # Pad to target height
        if pad_bottom > 0:
            pad_transform = v2.Pad(padding=[0, 0, 0, pad_bottom], fill=0)
            img = pad_transform(img)

        # Apply same transforms to mask if provided
        if mask is not None:
            mask = resize_transform(mask)
            if pad_bottom > 0:
                mask = pad_transform(mask)
            return img, mask

        return img


class RandomResizedCropWithDifferentInterpolation:
    """RandomResizedCrop with different interpolation modes for image and mask."""

    def __init__(
        self,
        size,
        image_interpolation=v2.InterpolationMode.BICUBIC,
        mask_interpolation=v2.InterpolationMode.NEAREST,
    ):
        self.size = size
        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation

    def __call__(self, image, mask):
        # Get random crop parameters
        h, w = image.shape[-2:]
        area = h * w

        # Calculate scale range
        scale_min, scale_max = 0.08, 1.0  # default from torchvision
        target_area = area * torch.empty(1).uniform_(scale_min, scale_max).item()

        # Calculate aspect ratio range (0.75 to 1.33)
        aspect_ratio = torch.empty(1).uniform_(0.75, 1.33).item()

        # Calculate crop dimensions
        crop_h = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_w = int(round(math.sqrt(target_area / aspect_ratio)))

        # Ensure crop is within image bounds
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        # Get random crop position
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()

        # Crop image and mask
        image = v2.functional.crop(image, top, left, crop_h, crop_w)
        mask = v2.functional.crop(mask, top, left, crop_h, crop_w)

        # Resize with different interpolation modes
        image = v2.functional.resize(
            image, self.size, interpolation=self.image_interpolation, antialias=True
        )
        mask = v2.functional.resize(
            mask, self.size, interpolation=self.mask_interpolation, antialias=False
        )

        return image, mask


class RandomErasingForImageAndMask:
    """Apply joint RandomErasing with different values to image and mask."""

    def __init__(
        self,
        image_value=1,
        mask_value=0,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        inplace=False,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.image_value = image_value
        self.mask_value = mask_value
        self.inplace = inplace

    def __call__(self, image, mask):
        if torch.rand(1) > self.p:
            return image, mask

        # Get image dimensions
        _, height, width = image.shape

        # Calculate area and aspect ratio (same logic as v2.RandomErasing)
        area = height * width
        log_ratio = torch.log(torch.tensor(self.ratio))

        # Sample target area and aspect ratio
        target_area = (
            area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        )
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        # Calculate dimensions
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        # Clamp to valid ranges
        h = max(1, min(h, height))
        w = max(1, min(w, width))

        # Sample position within valid bounds
        top = torch.randint(0, height - h + 1, size=(1,)).item()
        left = torch.randint(0, width - w + 1, size=(1,)).item()

        # Apply the SAME erasing to both image and mask
        if not self.inplace:
            image = image.clone()
            mask = mask.clone()

        # Set the same region with different values
        image[:, top : top + h, left : left + w] = self.image_value
        mask[:, top : top + h, left : left + w] = self.mask_value

        return image, mask


class WeldDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Get all file paths recursively from the given data dir
        root_path = (
            os.environ.get("SM_CHANNEL_TRAINING")
            if os.environ.get("SM_CHANNEL_TRAINING") is not None
            else self.config.data_dir
        )

        train_paths = [
            os.path.join(root_path, "train_crop_weld_images_rotated", file)
            for file in os.listdir(
                os.path.join(root_path, "train_crop_weld_images_rotated")
            )
            if file.endswith(".tif")
        ]

        val_paths = [
            os.path.join(root_path, "val_crop_weld_images_rotated", file)
            for file in os.listdir(
                os.path.join(root_path, "val_crop_weld_images_rotated")
            )
            if file.endswith(".tif")
        ]

        image_conversion = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]

        augmentations = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(
                brightness=self.config.color_jitter.brightness,
                contrast=self.config.color_jitter.contrast,
                saturation=self.config.color_jitter.saturation,
            ),
            v2.RandomChoice(
                [
                    v2.RandomAdjustSharpness(
                        sharpness_factor=self.config.random_adjust_sharpness.sharpness_decreasing_factor,
                    ),
                    v2.RandomAdjustSharpness(
                        sharpness_factor=self.config.random_adjust_sharpness.sharpness_increasing_factor,
                    ),
                ]
            ),
            RandomResizedCropWithDifferentInterpolation(
                size=(self.config.resize.height, self.config.resize.width),
                image_interpolation=v2.InterpolationMode.BICUBIC,
                mask_interpolation=v2.InterpolationMode.NEAREST_EXACT,
            ),
            RandomErasingForImageAndMask(image_value=1, mask_value=0)
            if self.config.random_erasing
            else None,
        ]

        resize = [
            v2.Resize(size=(self.config.resize.height, self.config.resize.width)),
        ]

        normalization = [
            v2.Normalize(mean=[0.449], std=[0.226]),
        ]

        # Filter out None values
        augmentations = [aug for aug in augmentations if aug is not None]

        self.train_dataset = WeldDataset(
            train_paths,
            conversion_transforms=v2.Compose(image_conversion),
            post_transforms=v2.Compose(augmentations + normalization),
        )
        self.val_dataset = WeldDataset(
            val_paths,
            conversion_transforms=v2.Compose(image_conversion),
            post_transforms=v2.Compose(resize + normalization),
        )

        self.test_dataset = WeldDataset(
            val_paths,
            conversion_transforms=v2.Compose(image_conversion),
            post_transforms=v2.Compose(resize + normalization),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
