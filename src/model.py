import torch
from torch import nn, optim
import lightning.pytorch as pl
import torchmetrics
from torchvision.models import (
    swin_s,
    Swin_S_Weights,
    resnet34,
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b1,
)
import numpy as np

features = {}


def get_intermediate_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


class LitSwin(pl.LightningModule):
    def __init__(
        self,
        dataset,
        num_classes,
        pretrained,
        class_weights,
        learning_rate,
        weight_decay,
        momentum=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.class_weights = class_weights
        self.lr = learning_rate
        self.wd = weight_decay
        self.momentum = momentum

        # define model
        if self.pretrained:
            self.model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        else:
            self.model = swin_s()
        # change num of outputs
        if self.num_classes > 2:
            self.model.head = torch.nn.Linear(
                in_features=768, out_features=self.num_classes, bias=True
            )
        else:
            self.model.head = torch.nn.Linear(
                in_features=768, out_features=1, bias=True
            )

        # loss and metrics
        if self.num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        if self.dataset == "HAM10k":
            self.val_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes, average="macro"
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.val_metric = torchmetrics.CohenKappa(
                task="multiclass", num_classes=self.num_classes, weights="quadratic"
            )
        else:
            self.val_metric = torchmetrics.AUROC(task="binary")

        if self.num_classes > 2:
            self.accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
        else:
            self.accuracy = torchmetrics.Accuracy(task="binary")

        # save predictions and targets to compute metrics at end of epoch
        self.training_step_preds = []
        self.training_step_targets = []
        self.validation_step_preds = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            y = y.float()
        loss = self.loss_fn(logits, y)

        if self.num_classes > 2:
            self.training_step_preds.extend(logits.argmax(dim=1).detach().cpu())
        else:
            # preds are acutally probs
            self.training_step_preds.extend(
                nn.functional.sigmoid(logits).detach().cpu()
            )
        self.training_step_targets.extend(y.cpu())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            y = y.float()
        loss = self.loss_fn(logits, y)

        if self.num_classes > 2:
            self.validation_step_preds.extend(logits.argmax(dim=1).cpu())
        else:
            self.validation_step_preds.extend(
                nn.functional.sigmoid(logits).detach().cpu()
            )
        self.validation_step_targets.extend(y.cpu())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            probs = nn.functional.sigmoid(logits)
            preds = torch.round(probs)
            val_metric = self.val_metric(probs, y)
        else:
            preds = logits.argmax(dim=1)
            val_metric = self.val_metric(preds, y)
        accuracy = self.accuracy(preds, y)

        if self.dataset == "HAM10k":
            self.log_dict(
                {"test_balanced_accuracy": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log_dict(
                {"test_kappa": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log_dict(
                {"test_AUROC": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def predict_step(self, batch, mcdropout):
        x, y, filenames = batch
        if mcdropout:
            self.train()
        else:
            self.eval()
        with torch.no_grad():
            logits = self(x)
            if self.num_classes == 2:
                probs = list(nn.functional.sigmoid(torch.squeeze(logits)).cpu().numpy())
                preds = list(np.round(probs))
            else:
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = list(probs.argmax(axis=1))
            embs = features["embedding"].squeeze().cpu().numpy()
        return embs, preds, probs

    def on_train_epoch_end(self):
        self.val_metric.to(torch.device("cpu"))
        val_metric = self.val_metric(
            torch.tensor(self.training_step_preds),
            torch.tensor(self.training_step_targets),
        )
        if self.dataset == "HAM10k":
            self.log(
                "train_balanced_accuracy",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log(
                "train_kappa",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log(
                "train_AUROC",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.training_step_preds.clear()
        self.training_step_targets.clear()

    def on_validation_epoch_end(self):
        self.val_metric.to(torch.device("cpu"))
        val_metric = self.val_metric(
            torch.tensor(self.validation_step_preds),
            torch.tensor(self.validation_step_targets),
        )
        if self.dataset == "HAM10k":
            self.log(
                "val_balanced_accuracy",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log(
                "val_kappa",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log(
                "val_AUROC",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.validation_step_preds.clear()
        self.validation_step_targets.clear()

    def configure_optimizers(self):
        if self.dataset == "HAM10k":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.wd,
                momentum=self.momentum,
            )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=3, threshold=0.01
        )

        if self.dataset == "HAM10k":
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_balanced_accuracy",
            }
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_kappa",
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_AUROC",
            }


class LitResNet(pl.LightningModule):
    def __init__(
        self,
        variant,
        dataset,
        num_classes,
        pretrained,
        class_weights,
        learning_rate,
        weight_decay,
        momentum=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.variant = variant
        self.dataset = dataset
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.class_weights = class_weights
        self.lr = learning_rate
        self.wd = weight_decay
        self.momentum = momentum

        # define model
        if self.variant == "resnet50":
            if self.pretrained:
                self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                self.model = resnet50()
            # change num of outputs
            if self.num_classes > 2:
                self.model.fc = torch.nn.Linear(
                    in_features=2048, out_features=self.num_classes, bias=True
                )
            else:
                self.model.fc = torch.nn.Linear(
                    in_features=2048, out_features=1, bias=True
                )
        elif self.variant == "resnet34":
            if self.pretrained:
                self.model = resnet34(weights="DEFAULT")
            else:
                self.model = resnet34()

            if self.num_classes > 2:
                self.model.fc = torch.nn.Linear(
                    in_features=512, out_features=self.num_classes, bias=True
                )
            else:
                self.model.fc = torch.nn.Linear(
                    in_features=512, out_features=1, bias=True
                )

        # loss and metrics
        if self.num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        if self.dataset == "HAM10k":
            self.val_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes, average="macro"
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.val_metric = torchmetrics.CohenKappa(
                task="multiclass", num_classes=self.num_classes, weights="quadratic"
            )
        else:
            self.val_metric = torchmetrics.AUROC(task="binary")

        if self.num_classes > 2:
            self.accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
        else:
            self.accuracy = torchmetrics.Accuracy(task="binary")

        # save predictions and targets to compute metrics at end of epoch
        self.training_step_preds = []
        self.training_step_targets = []
        self.validation_step_preds = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            y = y.float()
        loss = self.loss_fn(logits, y)

        if self.num_classes > 2:
            self.training_step_preds.extend(logits.argmax(dim=1).detach().cpu())
        else:
            # preds are acutally probs
            self.training_step_preds.extend(
                nn.functional.sigmoid(logits).detach().cpu()
            )
        self.training_step_targets.extend(y.cpu())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            y = y.float()
        loss = self.loss_fn(logits, y)

        if self.num_classes > 2:
            self.validation_step_preds.extend(logits.argmax(dim=1).cpu())
        else:
            self.validation_step_preds.extend(
                nn.functional.sigmoid(logits).detach().cpu()
            )
        self.validation_step_targets.extend(y.cpu())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            probs = nn.functional.sigmoid(logits)
            preds = torch.round(probs)
            val_metric = self.val_metric(probs, y)
        else:
            preds = logits.argmax(dim=1)
            val_metric = self.val_metric(preds, y)
        accuracy = self.accuracy(preds, y)

        if self.dataset == "HAM10k":
            self.log_dict(
                {"test_balanced_accuracy": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log_dict(
                {"test_kappa": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log_dict(
                {"test_AUROC": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def predict_step(self, batch, mcdropout):
        x, y, filenames = batch
        if mcdropout:
            self.train()
        else:
            self.eval()
        with torch.no_grad():
            logits = self(x)
            if self.num_classes == 2:
                probs = list(nn.functional.sigmoid(torch.squeeze(logits)).cpu().numpy())
                preds = list(np.round(probs))
            else:
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = list(probs.argmax(axis=1))
            embs = features["embedding"].squeeze().cpu().numpy()
        return embs, preds, probs

    def on_train_epoch_end(self):
        self.val_metric.to(torch.device("cpu"))
        val_metric = self.val_metric(
            torch.tensor(self.training_step_preds),
            torch.tensor(self.training_step_targets),
        )
        if self.dataset == "HAM10k":
            self.log(
                "train_balanced_accuracy",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log(
                "train_kappa",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log(
                "train_AUROC",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.training_step_preds.clear()
        self.training_step_targets.clear()

    def on_validation_epoch_end(self):
        self.val_metric.to(torch.device("cpu"))
        val_metric = self.val_metric(
            torch.tensor(self.validation_step_preds),
            torch.tensor(self.validation_step_targets),
        )
        if self.dataset == "HAM10k":
            self.log(
                "val_balanced_accuracy",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log(
                "val_kappa",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log(
                "val_AUROC",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.validation_step_preds.clear()
        self.validation_step_targets.clear()

    def configure_optimizers(self):
        if self.dataset == "HAM10k":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.wd,
                momentum=self.momentum,
            )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=3, threshold=0.01
        )

        if self.dataset == "HAM10k":
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_balanced_accuracy",
            }
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_kappa",
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_AUROC",
            }


class LitEfficientNet(pl.LightningModule):
    def __init__(
        self,
        variant,
        dataset,
        num_classes,
        pretrained,
        class_weights,
        learning_rate,
        weight_decay,
        momentum=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.variant = variant
        self.dataset = dataset
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.class_weights = class_weights
        self.lr = learning_rate
        self.wd = weight_decay
        self.momentum = momentum

        # define model
        if self.variant == "efficientnet_b0":
            if self.pretrained:
                self.model = efficientnet_b0(weights="DEFAULT")
            else:
                self.model = efficientnet_b0()
        elif self.variant == "efficientnet_b1":
            if self.pretrained:
                self.model = efficientnet_b1(weights="DEFAULT")
            else:
                self.model = efficientnet_b1()

        # change num of outputs
        if self.num_classes > 2:
            self.model.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=self.num_classes, bias=True
            )
        else:
            self.model.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=1, bias=True
            )

        # loss and metrics
        if self.num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        if self.dataset == "HAM10k":
            self.val_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes, average="macro"
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.val_metric = torchmetrics.CohenKappa(
                task="multiclass", num_classes=self.num_classes, weights="quadratic"
            )
        else:
            self.val_metric = torchmetrics.AUROC(task="binary")

        if self.num_classes > 2:
            self.accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
        else:
            self.accuracy = torchmetrics.Accuracy(task="binary")

        # save predictions and targets to compute metrics at end of epoch
        self.training_step_preds = []
        self.training_step_targets = []
        self.validation_step_preds = []
        self.validation_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            y = y.float()
        loss = self.loss_fn(logits, y)

        if self.num_classes > 2:
            self.training_step_preds.extend(logits.argmax(dim=1).detach().cpu())
        else:
            # preds are acutally probs
            self.training_step_preds.extend(
                nn.functional.sigmoid(logits).detach().cpu()
            )
        self.training_step_targets.extend(y.cpu())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            y = y.float()
        loss = self.loss_fn(logits, y)

        if self.num_classes > 2:
            self.validation_step_preds.extend(logits.argmax(dim=1).cpu())
        else:
            self.validation_step_preds.extend(
                nn.functional.sigmoid(logits).detach().cpu()
            )
        self.validation_step_targets.extend(y.cpu())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.num_classes == 2:
            logits = torch.squeeze(logits)
            probs = nn.functional.sigmoid(logits)
            preds = torch.round(probs)
            val_metric = self.val_metric(probs, y)
        else:
            preds = logits.argmax(dim=1)
            val_metric = self.val_metric(preds, y)
        accuracy = self.accuracy(preds, y)

        if self.dataset == "HAM10k":
            self.log_dict(
                {"test_balanced_accuracy": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log_dict(
                {"test_kappa": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log_dict(
                {"test_AUROC": val_metric, "accuracy": accuracy},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def predict_step(self, batch, mcdropout):
        x, y, filenames = batch
        if mcdropout:
            self.train()
        else:
            self.eval()
        with torch.no_grad():
            logits = self(x)
            if self.num_classes == 2:
                probs = list(nn.functional.sigmoid(torch.squeeze(logits)).cpu().numpy())
                preds = list(np.round(probs))
            else:
                probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = list(probs.argmax(axis=1))
            embs = features["embedding"].squeeze().cpu().numpy()
        return embs, preds, probs

    def on_train_epoch_end(self):
        self.val_metric.to(torch.device("cpu"))
        val_metric = self.val_metric(
            torch.tensor(self.training_step_preds),
            torch.tensor(self.training_step_targets),
        )
        if self.dataset == "HAM10k":
            self.log(
                "train_balanced_accuracy",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log(
                "train_kappa",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log(
                "train_AUROC",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.training_step_preds.clear()
        self.training_step_targets.clear()

    def on_validation_epoch_end(self):
        self.val_metric.to(torch.device("cpu"))
        val_metric = self.val_metric(
            torch.tensor(self.validation_step_preds),
            torch.tensor(self.validation_step_targets),
        )
        if self.dataset == "HAM10k":
            self.log(
                "val_balanced_accuracy",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            self.log(
                "val_kappa",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log(
                "val_AUROC",
                val_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.validation_step_preds.clear()
        self.validation_step_targets.clear()

    def configure_optimizers(self):
        if self.dataset == "HAM10k":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.wd,
                momentum=self.momentum,
            )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=3, threshold=0.01
        )

        if self.dataset == "HAM10k":
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_balanced_accuracy",
            }
        elif self.dataset == "EyePACS" and self.num_classes > 2:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_kappa",
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_AUROC",
            }


class LitConfidNet(pl.LightningModule):
    def __init__(self, num_features, num_hidden_units, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.lr = learning_rate
        self.wd = weight_decay

        self.net = nn.Sequential(
            nn.Linear(self.num_features, self.num_hidden_units),
            nn.ReLU(),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            nn.ReLU(),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            nn.ReLU(),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            nn.ReLU(),
            nn.Linear(self.num_hidden_units, 1),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=5,
            threshold=0.01,
            threshold_mode="rel",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
