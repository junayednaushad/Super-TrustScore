import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from dataset import HAM10kDataModule, EyePACSDataModule
from model import LitSwin, LitResNet, LitEfficientNet
from utils import get_HAM10k_class_weights, get_EyePACS_class_weights
import sys
import argparse
import yaml


if __name__ == "__main__":
    # load config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Config file (YAML) with experiment hyperparameters and settings",
    )
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])

    # log on Weights & Biases
    logger = WandbLogger(project=config["project"], name=config["run_name"])

    # check if GPU available
    if torch.cuda.is_available():
        trainer_kwargs = {"accelerator": "gpu", "devices": 1, "precision": "16-mixed"}
    else:
        print("GPU NOT AVAILABLE!!!")
        trainer_kwargs = {"accelerator": "cpu"}

    # create DataModule and Trainer
    if config["dataset"] == "HAM10k":
        data_module = HAM10kDataModule(config)
        class_weights = get_HAM10k_class_weights(config)
        trainer = pl.Trainer(
            logger=logger,
            min_epochs=1,
            max_epochs=config["epochs"],
            callbacks=[
                LearningRateMonitor(),
                EarlyStopping(
                    monitor="val_balanced_accuracy",
                    mode="max",
                    min_delta=0.01,
                    patience=10,
                ),
                # ModelCheckpoint(
                #     dirpath=config["model_dir"], every_n_epochs=1, save_top_k=-1
                # ),
                ModelCheckpoint(
                    dirpath=config["model_dir"],
                    filename="{epoch}-{val_balanced_accuracy:.2f}",
                    monitor="val_balanced_accuracy",
                    mode="max",
                ),
            ],
            # limit_train_batches=10,
            # limit_val_batches=5,
            # limit_test_batches=10,
            **trainer_kwargs,
        )
    elif config["dataset"] == "EyePACS" and config["num_classes"] > 2:
        data_module = EyePACSDataModule(config)
        class_weights = get_EyePACS_class_weights(config)
        # class_weights = torch.ones(config["num_classes"]).float()
        trainer = pl.Trainer(
            logger=logger,
            min_epochs=1,
            max_epochs=config["epochs"],
            callbacks=[
                LearningRateMonitor(),
                EarlyStopping(
                    monitor="val_kappa",
                    mode="max",
                    min_delta=0.01,
                    patience=6,
                ),
                ModelCheckpoint(
                    dirpath=config["model_dir"],
                    filename="{epoch}-{val_kappa:.2f}",
                    monitor="val_kappa",
                    mode="max",
                    save_last=True,
                ),
            ],
            **trainer_kwargs,
        )
    elif config["dataset"] == "EyePACS" and config["num_classes"] == 2:
        data_module = EyePACSDataModule(config)
        class_weights = get_EyePACS_class_weights(config)
        trainer = pl.Trainer(
            logger=logger,
            min_epochs=1,
            max_epochs=config["epochs"],
            callbacks=[
                LearningRateMonitor(),
                EarlyStopping(
                    monitor="val_AUROC",
                    mode="max",
                    min_delta=0.01,
                    patience=5,
                ),
                ModelCheckpoint(
                    dirpath=config["model_dir"],
                    filename="{epoch}-{val_AUROC:.2f}",
                    monitor="val_AUROC",
                    mode="max",
                    save_last=True,
                ),
            ],
            **trainer_kwargs,
        )
    else:
        print("Please specify a dataset that is supported (HAM10k, EyePACS)")
        sys.exit()

    # define model
    if config["model"] == "Swin":
        model = LitSwin(
            dataset=config["dataset"],
            num_classes=config["num_classes"],
            pretrained=config["pretrained"],
            class_weights=class_weights,
            learning_rate=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    elif "resnet" in config["model"]:
        model = LitResNet(
            variant=config["model"],
            dataset=config["dataset"],
            num_classes=config["num_classes"],
            pretrained=config["pretrained"],
            class_weights=class_weights,
            learning_rate=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
    else:
        model = LitEfficientNet(
            variant=config["model"],
            dataset=config["dataset"],
            num_classes=config["num_classes"],
            pretrained=["pretrained"],
            class_weights=class_weights,
            learning_rate=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )

    # train and validate
    trainer.fit(model, data_module)

    # test
    trainer.test(model, data_module)
