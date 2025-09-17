import argparse
import os

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from models.swrd_model.data_module import WeldDataModule
from models.swrd_model.enums import LabelNames
from models.swrd_model.lightning_model import WeldModel


def main(args):
    seed_everything(42)
    config = OmegaConf.load(args.config_file)
    model = WeldModel(config)
    data_module = WeldDataModule(config)
    # Initialize WandB logger with more detailed configuration
    wandb_logger = WandbLogger(
        project="swrd-model",
        name=f"{config.arch}-{config.encoder}-({config.resize.width}x{config.resize.height}"
        + "-random_erasing"
        if config.random_erasing
        else "",
        save_dir="lightning_logs",
    )

    # Log config to WandB
    wandb_logger.log_hyperparams(config)

    chkpt_path = (
        "/opt/ml/checkpoints"
        if os.environ.get("SM_TRAINING_ENV") is not None
        else "./checkpoints"
    )

    # Create checkpoint callbacks for different metrics
    checkpoint_callbacks = []

    # Main validation loss checkpoint (lower is better)
    val_loss_checkpoint = ModelCheckpoint(
        dirpath=chkpt_path,
        filename="best_val_loss_{epoch:02d}",
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    checkpoint_callbacks.append(val_loss_checkpoint)

    # Overall IoU checkpoint (higher is better)
    iou_checkpoint = ModelCheckpoint(
        dirpath=chkpt_path,
        filename="best_val_iou_{epoch:02d}",
        monitor="val_iou_overall_epoch",
        mode="max",
        save_top_k=1,
    )
    checkpoint_callbacks.append(iou_checkpoint)

    # Overall precision checkpoint (higher is better)
    precision_checkpoint = ModelCheckpoint(
        dirpath=chkpt_path,
        filename="best_val_precision_{epoch:02d}",
        monitor="val_precision_overall_epoch",
        mode="max",
        save_top_k=1,
    )
    checkpoint_callbacks.append(precision_checkpoint)

    # Overall recall checkpoint (higher is better)
    recall_checkpoint = ModelCheckpoint(
        dirpath=chkpt_path,
        filename="best_val_recall_{epoch:02d}",
        monitor="val_recall_overall_epoch",
        mode="max",
        save_top_k=1,
    )
    checkpoint_callbacks.append(recall_checkpoint)

    # Per-class IoU checkpoints (higher is better)
    class_names = ["background"] + [label.value for label in LabelNames]
    for i in range(1, config.num_classes):
        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
        class_iou_checkpoint = ModelCheckpoint(
            dirpath=chkpt_path,
            filename=f"best_val_iou_{class_name}_{{epoch:02d}}",
            monitor=f"val_iou_{class_name}_epoch",
            mode="max",
            save_top_k=1,
        )
        checkpoint_callbacks.append(class_iou_checkpoint)

        class_precision_checkpoint = ModelCheckpoint(
            dirpath=chkpt_path,
            filename=f"best_val_precision_{class_name}_{{epoch:02d}}",
            monitor=f"val_precision_{class_name}_epoch",
            mode="max",
            save_top_k=1,
        )
        checkpoint_callbacks.append(class_precision_checkpoint)

        class_recall_checkpoint = ModelCheckpoint(
            dirpath=chkpt_path,
            filename=f"best_val_recall_{class_name}_{{epoch:02d}}",
            monitor=f"val_recall_{class_name}_epoch",
            mode="max",
            save_top_k=1,
        )
        checkpoint_callbacks.append(class_recall_checkpoint)

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=10,
        logger=wandb_logger,
        default_root_dir=chkpt_path,
        callbacks=checkpoint_callbacks,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="./models/swrd_model/configs/config.yaml"
    )
    main(parser.parse_args())
