import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from torchmetrics.classification import JaccardIndex, Precision, Recall

import wandb
from models.swrd_model.enums import LabelNames
from models.swrd_model.utils import (
    get_loss_fn,
    get_model,
    get_optimizer,
)


class WeldModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_model(
            self.config.arch,
            self.config.encoder,
            self.config.num_classes,
        )
        self.loss_fn = get_loss_fn(self.config.loss_fn, self.config.num_classes)
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.ious = []

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "iou_per_class": JaccardIndex(
                    task="multiclass",
                    num_classes=self.config.num_classes,
                    ignore_index=None,
                    average="none",
                ),
                "iou_overall": JaccardIndex(
                    task="multiclass",
                    num_classes=self.config.num_classes,
                    ignore_index=None,
                    average="macro",
                ),
                "precision_per_class": Precision(
                    task="multiclass",
                    num_classes=self.config.num_classes,
                    ignore_index=None,
                    average="none",
                ),
                "recall_per_class": Recall(
                    task="multiclass",
                    num_classes=self.config.num_classes,
                    ignore_index=None,
                    average="none",
                ),
                "precision_overall": Precision(
                    task="multiclass",
                    num_classes=self.config.num_classes,
                    ignore_index=None,
                    average="macro",
                ),
                "recall_overall": Recall(
                    task="multiclass",
                    num_classes=self.config.num_classes,
                    ignore_index=None,
                    average="macro",
                ),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="val_")

        # Track which classes have been logged this epoch
        self.logged_classes_this_epoch = set()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.config.optimizer, self.model.parameters(), self.config.lr
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        imgs, masks = train_batch
        preds = self.model(imgs)
        loss = self.loss_fn(preds, masks)

        # Update metrics
        self.train_loss(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        # Compute metrics
        batch_values = self.train_metrics(preds, masks)

        # Log overall metrics
        self.log(
            "train_iou_overall",
            batch_values["train_iou_overall"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_precision_overall",
            batch_values["train_precision_overall"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_recall_overall",
            batch_values["train_recall_overall"],
            on_step=True,
            on_epoch=True,
        )

        # Log per-class metrics
        class_names = ["background"] + [label.value for label in LabelNames]
        for i in range(1, self.config.num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            self.log(
                f"train_iou_{class_name}",
                batch_values["train_iou_per_class"][i],
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"train_precision_{class_name}",
                batch_values["train_precision_per_class"][i],
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"train_recall_{class_name}",
                batch_values["train_recall_per_class"][i],
                on_step=True,
                on_epoch=True,
            )

        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, masks = val_batch
        preds = self.model(imgs)
        loss = self.loss_fn(preds, masks)

        # Update metrics
        self.val_loss(loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        # Compute metrics
        batch_values = self.valid_metrics(preds, masks)

        # Log overall metrics
        self.log(
            "val_iou_overall",
            batch_values["val_iou_overall"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "val_precision_overall",
            batch_values["val_precision_overall"],
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "val_recall_overall",
            batch_values["val_recall_overall"],
            on_step=True,
            on_epoch=True,
        )

        # Log per-class metrics
        class_names = ["background"] + [label.value for label in LabelNames]
        for i in range(1, self.config.num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            self.log(
                f"val_iou_{class_name}",
                batch_values["val_iou_per_class"][i],
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"val_precision_{class_name}",
                batch_values["val_precision_per_class"][i],
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"val_recall_{class_name}",
                batch_values["val_recall_per_class"][i],
                on_step=True,
                on_epoch=True,
            )

        # Log example images
        if len(self.logged_classes_this_epoch) < self.config.num_classes - 1:
            self._log_example_images(
                imgs,
                masks,
                preds,
                class_names,
                batch_idx,
            )

        return loss

    def test_step(self, test_batch, batch_idx):
        imgs, masks, (original_height, original_width) = test_batch
        preds = self.model(imgs)

        # Get original image (grayscale)
        original_img = imgs.squeeze().cpu().numpy()

        # Process masks and predictions for visualization
        preds_vis = torch.sigmoid(preds).squeeze().cpu().numpy()
        preds_vis = preds_vis[1:, :, :]  # Remove background class
        preds_vis = (preds_vis > 0.5).astype(int)

        # Create class-wise predictions (allow multiple classes per pixel)
        # We'll handle overlaps by prioritizing higher class numbers (later classes overwrite earlier ones)
        preds_class = torch.zeros_like(masks.squeeze())
        for i in range(preds_vis.shape[0]):
            preds_class[preds_vis[i] == 1] = i + 1  # +1 because class 0 is background

        # Calculate IoU metrics
        # 1. Binary IoU (defect vs no defect)
        masks_binary = torch.where(masks > 0, 1, 0).squeeze()
        preds_binary = torch.any(torch.tensor(preds_vis), dim=0).int()

        iou_binary = torchmetrics.functional.jaccard_index(
            preds_binary, masks_binary, "binary"
        )

        # 2. Per-class IoU
        iou_per_class = torchmetrics.functional.jaccard_index(
            preds_class,
            masks.squeeze(),
            task="multiclass",
            num_classes=self.config.num_classes,
            ignore_index=None,
            average="none",
        )

        iou_per_class = iou_per_class[1:]

        # Determine layout based on image aspect ratio
        img_height, img_width = original_img.shape
        aspect_ratio = img_width / img_height

        new_height = original_height.item() * 2

        fig, axes = plt.subplots(3, 1, figsize=(16, 8), dpi=300)

        # Define class names for legend
        class_names = ["background"] + [label.value for label in LabelNames]

        # Original image
        axes[0].imshow(
            cv2.resize(original_img, (original_width.item(), new_height)),
            cmap="gray",
        )
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Ground truth overlaid on original image
        axes[1].imshow(
            cv2.resize(original_img, (original_width.item(), new_height)),
            cmap="gray",
            alpha=0.9,
        )
        # Create masked array to make background transparent
        masks_display = cv2.resize(
            masks.squeeze().numpy().astype(np.uint8),
            (original_width.item(), new_height),
        )
        masks_display = np.ma.masked_where(masks_display == 0, masks_display)
        axes[1].imshow(
            masks_display,
            alpha=0.6,
            cmap="tab10",
            vmin=0,
            vmax=9,
        )
        axes[1].set_title("Ground Truth Overlay")
        axes[1].axis("off")

        # Predictions overlaid on original image
        axes[2].imshow(
            cv2.resize(original_img, (original_width.item(), new_height)),
            cmap="gray",
            alpha=0.9,
        )
        # Create masked array to make background transparent
        preds_display = cv2.resize(
            preds_class.numpy().astype(np.uint8),
            (original_width.item(), new_height),
        )
        preds_display = np.ma.masked_where(preds_display == 0, preds_display)
        axes[2].imshow(
            preds_display,
            alpha=0.6,
            cmap="tab10",
            vmin=0,
            vmax=9,
        )
        axes[2].set_title("Predictions Overlay")
        axes[2].axis("off")

        # Create custom legend with class names
        legend_elements = []
        for i in range(
            1, min(len(class_names), 10)
        ):  # Skip background, limit to 9 classes
            if i < len(class_names):
                legend_elements.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=plt.cm.tab10(i / 9),
                        alpha=0.7,
                        label=class_names[i],
                    )
                )

        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                fontsize=8,
                ncol=len(legend_elements),  # Horizontal layout
                frameon=True,
            )

        # Add IoU metrics as text
        iou_text = f"Binary IoU: {iou_binary.item():.3f}\n"
        iou_text += "Per-class IoU:\n"

        # Debug: print all class IoUs including zeros
        for i, iou_val in enumerate(iou_per_class):
            class_name = [label.value for label in LabelNames][i]
            iou_text += f"  {class_name}: {iou_val.item():.3f}\n"

        # Make IoU text horizontal (one line with separators)
        iou_text_horizontal = f"Binary IoU: {iou_binary.item():.3f}  |  "
        iou_parts = []
        for i, iou_val in enumerate(iou_per_class):
            class_name = [label.value for label in LabelNames][i]
            iou_parts.append(f"{class_name}: {iou_val.item():.3f}")
        iou_text_horizontal += "  |  ".join(iou_parts)

        fig.text(
            0.5,
            0.08,
            iou_text_horizontal,
            fontsize=7,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        # Adjust subplot spacing with room for horizontal legend (top) and text (bottom)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15, hspace=0.15)

        if batch_idx < 150:
            plt.savefig(
                f"/home/luke/deeplify/repos/weld_defect_detection/src/models/swrd_model/debug_images/{batch_idx}.png",
                bbox_inches="tight",
                pad_inches=0.2,
                dpi=150,  # Reduce DPI to save memory
            )

        # CRITICAL: Always close the figure to prevent memory leaks
        plt.close(fig)
        plt.clf()

        # Store metrics
        self.ious.append(iou_binary.item())
        if not hasattr(self, "ious_per_class"):
            self.ious_per_class = [[] for _ in range(self.config.num_classes)]

        for i, iou_val in enumerate(iou_per_class):
            self.ious_per_class[i].append(iou_val.item())

        return iou_binary

    def on_train_epoch_end(self):
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Log validation metrics at the end of each epoch."""
        self.val_loss.reset()
        self.valid_metrics.reset()

        # Reset image logging counter for next epoch
        self.logged_classes_this_epoch.clear()

    def _log_example_images(self, imgs, masks, preds, class_names, batch_idx):
        """Log example images with ground truth and predictions to wandb, one per class per epoch."""
        # Iterate through each image in the batch
        for i in range(imgs.shape[0]):
            # Get the ground truth mask for this image
            gt_mask = masks[i].cpu()

            # Find unique classes present in the ground truth
            unique_classes = torch.unique(gt_mask).tolist()
            # Remove background class (0) if present
            if 0 in unique_classes:
                unique_classes.remove(0)

            if unique_classes == []:
                continue

            # Check each class in this image
            for class_idx in unique_classes:
                # Skip if we've already logged this class this epoch
                if class_idx in self.logged_classes_this_epoch:
                    continue

                # Get the image, ground truth mask, and prediction
                img = imgs[i].cpu()
                pred_masks = preds[i].cpu()

                # Get class name for logging
                class_name = class_names[class_idx]

                # Create visualization masks
                gt_mask_for_class = (gt_mask == class_idx).long()
                pred_mask_for_class = pred_masks[
                    class_idx
                ]  # Get probability for this class
                pred_mask_for_class = torch.where(pred_mask_for_class > 0.5, 1, 0)

                # Normalize image to [0, 1] for visualization
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

                # Create side-by-side comparison plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

                # Plot ground truth
                ax1.imshow(img_norm.permute(1, 2, 0), cmap="gray")
                ax1.imshow(gt_mask_for_class, alpha=0.5, cmap="Reds")
                ax1.set_title(
                    f"Ground Truth - {class_name.title()}", fontsize=14, pad=20
                )
                ax1.axis("off")

                # Plot prediction
                ax2.imshow(img_norm.permute(1, 2, 0), cmap="gray")
                ax2.imshow(pred_mask_for_class, alpha=0.5, cmap="Blues")
                ax2.set_title(f"Prediction - {class_name.title()}", fontsize=14, pad=20)
                ax2.axis("off")

                plt.tight_layout(pad=2.0)

                # Convert matplotlib figure to wandb image
                comparison_image = wandb.Image(fig)
                plt.close(fig)  # Close the figure to free memory

                # Log to wandb
                wandb.log(
                    {
                        f"val_examples/class_{class_name}_sample_{i}_{batch_idx}": comparison_image
                    }
                )

                # Mark this class as logged
                self.logged_classes_this_epoch.add(class_idx)

                # Break out of the class loop since we found a new class to log
                break

    def on_test_epoch_end(self):
        if hasattr(self, "ious") and len(self.ious) > 0:
            print(f"Binary IoU: {np.mean(self.ious):.3f}")
            self.ious = []

        if hasattr(self, "ious_per_class"):
            print("Per-class IoU:")
            for i, class_ious in enumerate(self.ious_per_class):
                if len(class_ious) > 0:
                    if i - 1 < len(LabelNames):
                        class_name = [label.value for label in LabelNames][i - 1]
                    else:
                        class_name = f"class_{i}"
                    print(f"  {class_name}: {np.mean(class_ious):.3f}")
            self.ious_per_class = [[] for _ in range(self.config.num_classes)]
