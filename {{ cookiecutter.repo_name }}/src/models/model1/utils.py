import json
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import tifffile as tif
import torch
from sklearn.model_selection import train_test_split

from models.swrd_model.enums import (
    ModelArchs,
    label_to_color_mapping,
    symbol_to_label_mapping,
    unicode_to_label_mapping,
)


def get_model(arch, encoder, num_classes):
    if arch == ModelArchs.UNET:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
        )
    elif arch == ModelArchs.SEGFORMER:
        model = smp.Segformer(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Model arch {arch} not supported (yet)")
    return model


def get_optimizer(optimizer, model_parameters, lr):
    if optimizer == "adamw":
        return torch.optim.AdamW(model_parameters, lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported (yet)")


def get_loss_fn(loss_fn, num_classes):
    if loss_fn == "dice":
        return smp.losses.DiceLoss(
            mode="multiclass" if num_classes > 1 else "binary", from_logits=True
        )
    else:
        raise ValueError(f"Loss function {loss_fn} not supported (yet)")


def split_data(paths, split_ratio):
    # split the data into train and val
    train_paths, val_paths = train_test_split(paths, test_size=split_ratio)
    return train_paths, val_paths


def create_masks_from_polygon_annotations(
    img_dir_path, img_output_dir_path, mask_output_dir_path
):
    img_paths = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(img_dir_path)
        for file in files
        if file.endswith(".tif")
    ]

    json_paths = [
        path.replace("crop_weld_images", "crop_weld_jsons").replace(".tif", ".json")
        for path in img_paths
    ]

    for img_path, json_path in zip(img_paths, json_paths):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        with open(json_path, "r") as f:
            labels = json.load(f)

        annotations = labels["shapes"]
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for anno in annotations:
            if "label" not in anno.keys() or anno["label"] is None:
                continue

            label = anno["label"]
            if not label.startswith("\\u"):
                if label not in unicode_to_label_mapping.keys():
                    continue
                label = unicode_to_label_mapping[label]
            else:
                if label not in symbol_to_label_mapping.keys():
                    continue
                label = symbol_to_label_mapping[label]

            color = label_to_color_mapping[label]
            points = np.array(anno["points"])

            mask = cv2.fillPoly(mask, [points], color)
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(os.path.join(img_output_dir_path, img_path.split("/")[-1]), img)
        cv2.imwrite(
            os.path.join(mask_output_dir_path, img_path.split("/")[-1]).replace(
                ".tif", ".png"
            ),
            mask,
        )


def create_val_crops(
    img_dir_path: str,
    mask_dir_path: str,
    size: int,
):
    image_paths = [
        os.path.join(img_dir_path, file)
        for file in os.listdir(img_dir_path)
        if file.endswith(".tif")
    ]

    mask_paths = [
        image_path.replace("val_crop_weld_images_rotated", "gt_masks").replace(
            ".tif", ".png"
        )
        for image_path in image_paths
    ]

    for image_path, mask_path in zip(image_paths, mask_paths):
        # img loading
        img = tif.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        assert img.shape == mask.shape

        # crop into patches of width 512 and original height
        img_patches = []
        mask_patches = []
        num_patches = img.shape[1] // size
        for i in range(num_patches):
            if i == num_patches - 1:
                img_patch = img[:, i * size :]
                mask_patch = mask[:, i * size :]
            else:
                img_patch = img[:, i * size : (i + 1) * size]
                mask_patch = mask[:, i * size : (i + 1) * size]
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)

        for i, patch in enumerate(img_patches):
            # dry run paths:
            tif.imwrite(
                f"{os.path.join(img_dir_path, image_path.split('/')[-1].replace('.tif', ''))}_patch_{i}.tif",
                patch,
            )
            cv2.imwrite(
                f"{os.path.join(mask_dir_path, mask_path.split('/')[-1].replace('.png', ''))}_patch_{i}.png",
                mask_patches[i],
            )


def create_random_train_crops(img_dir_path: str, mask_dir_path: str):
    pass


if __name__ == "__main__":
    create_val_crops(
        "/media/luke/Extreme SSD/SWRD_Data/crop_weld_data/custom/val_crop_weld_images_rotated",
        "/media/luke/Extreme SSD/SWRD_Data/crop_weld_data/custom/gt_masks",
        512,
    )
