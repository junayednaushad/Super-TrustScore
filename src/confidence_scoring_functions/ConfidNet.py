import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
import numpy as np
import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.getcwd()))
from dataset import ConfidNetDataModule
from model import LitConfidNet

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

    torch.manual_seed(0)

    # log on Weights & Biases
    if config["train"]:
        logger = WandbLogger(project=config["project"], name=config["run_name"])

    # create DataModule
    data_module = ConfidNetDataModule(config)

    # check if GPU available
    if torch.cuda.is_available():
        trainer_kwargs = {"accelerator": "gpu", "devices": 1, "precision": "16-mixed"}
    else:
        print("GPU NOT AVAILABLE!!!")
        trainer_kwargs = {"accelerator": "cpu"}

    # define trainer
    if config["train"]:
        trainer = pl.Trainer(
            logger=logger,
            min_epochs=1,
            max_epochs=config["epochs"],
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(
                    dirpath=config["model_dir"],
                    filename="{epoch}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                ),
            ],
            **trainer_kwargs
        )
    else:
        trainer = pl.Trainer(**trainer_kwargs)

    if config["train"]:
        # define model
        model = LitConfidNet(
            num_features=config["num_features"],
            num_hidden_units=config["num_hidden_units"],
            learning_rate=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # train and validate
        trainer.fit(model, data_module)
    else:
        model = LitConfidNet.load_from_checkpoint(config["ckpt"])
        data_module.setup()

    # add predictions to inference results
    inference_results = np.load(config["inference_file"], allow_pickle=True).item()

    # get predictions
    if not config["SD"]:
        val_dataloader = data_module.val_dataloader()
        val_predictions = trainer.predict(model, val_dataloader)
        val_predictions = torch.cat(val_predictions).cpu().numpy()
        inference_results["val"]["TCP_hat"] = val_predictions

    test_dataloader = data_module.test_dataloader()
    test_predictions = trainer.predict(model, test_dataloader)
    test_predictions = torch.cat(test_predictions).cpu().numpy()
    inference_results["test"]["TCP_hat"] = test_predictions

    np.save(config["inference_file"], inference_results)
