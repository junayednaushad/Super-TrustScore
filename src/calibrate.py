import torch
from torch import nn, optim
from torch.nn import functional as F
import yaml
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys
from dataset import (
    HAM10kDataModule,
    ISIC2019DataModule,
    PAD_UFES_20_DataModule,
    EyePACSDataModule,
    MessidorDataModule,
    APTOS_2019_DataModule,
)
from model import LitSwin


def acc(results_df, csf, bins):
    bin_width = 1 / bins
    conf_range = np.arange(0, 1, bin_width)
    accuracies = []
    bin_counts = []

    for conf_min in conf_range:
        results_in_range = results_df.loc[
            (results_df[csf] >= conf_min) & (results_df[csf] < conf_min + bin_width)
        ]
        bin_counts.append(len(results_in_range))
        if len(results_in_range) == 0:
            exp_acc = 0
        else:
            exp_acc = (
                results_in_range["model_pred"] == results_in_range["label"]
            ).sum() / len(results_in_range)
        accuracies.append(exp_acc)
    return np.array(accuracies), np.array(bin_counts)


def conf(results_df, csf, bins):
    bin_width = 1 / bins
    conf_range = np.arange(0, 1, bin_width)
    confs = []
    bin_counts = []

    for conf_min in conf_range:
        results_in_range = results_df.loc[
            (results_df[csf] >= conf_min) & (results_df[csf] < conf_min + bin_width)
        ]
        bin_counts.append(len(results_in_range))
        if len(results_in_range) == 0:
            avg_conf = 0
        else:
            avg_conf = results_in_range[csf].mean()
        confs.append(avg_conf)
    return np.array(confs), np.array(bin_counts)


def ece(accs, confs, bin_counts):
    return ((bin_counts / bin_counts.sum()) * np.abs(accs - confs)).sum()


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, device):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(valid_loader):
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(
            "Before temperature - NLL: %.3f, ECE: %.3f"
            % (before_temperature_nll, before_temperature_ece)
        )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        print("Optimal temperature: %.3f" % self.temperature.item())
        print(
            "After temperature - NLL: %.3f, ECE: %.3f"
            % (after_temperature_nll, after_temperature_ece)
        )


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def get_calibrated_confidence(scaled_model, dataloader, device):
    softmax = nn.Softmax(dim=1)
    calibrated_confs = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = scaled_model(inputs)
            probs = softmax(outputs)
            confs, _ = torch.max(probs, 1)
            calibrated_confs += confs.cpu().tolist()
    return calibrated_confs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Config file (YAML) with experiment hyperparameters and settings",
    )
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)

    if config["calibrate_model"]:
        if config["dataset"] == "HAM10k":
            data_module = HAM10kDataModule(config)
        elif config["dataset"] == "ISIC2019":
            data_module = ISIC2019DataModule(config)
        elif config["dataset"] == "PAD-UFES-20":
            data_module = PAD_UFES_20_DataModule(config)
        elif config["dataset"] == "EyePACS":
            data_module = EyePACSDataModule(config)
        elif config["dataset"] == "Messidor-2":
            data_module = MessidorDataModule(config)
        elif config["dataset"] == "APTOS_2019":
            data_module = APTOS_2019_DataModule(config)
        else:
            print("Please specify a dataset that is supported (i.e. HAM10k, ISIC2019)")
            sys.exit()
        data_module.setup()

        if not config["SD"]:
            val_dataloader = data_module.val_dataloader()
        test_dataloader = data_module.test_dataloader()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE: ", device)

        for idx, (model_ckpt, inference_file) in enumerate(
            zip(config["model_ckpts"], config["inference_files"])
        ):
            model = LitSwin.load_from_checkpoint(model_ckpt)
            scaled_model = ModelWithTemperature(model)
            if not config["load_temp"]:
                print(f"Calibrating model at {model_ckpt}")
                scaled_model.set_temperature(val_dataloader, device)
                torch.save(
                    scaled_model.temperature,
                    os.path.join(os.path.dirname(model_ckpt), "temperature.pt"),
                )
            else:
                temp = torch.load(
                    os.path.join(os.path.dirname(model_ckpt), "temperature.pt"),
                    map_location=device,
                )
                scaled_model.temperature = temp
                print(f"Loading calibrated model with temperature {temp.item():.3f}")
            scaled_model.to(device)

            df = np.load(inference_file, allow_pickle=True).item()
            if not config["SD"]:
                df["val"]["Calibrated Softmax"] = get_calibrated_confidence(
                    scaled_model, val_dataloader, device
                )
            df["test"]["Calibrated Softmax"] = get_calibrated_confidence(
                scaled_model, test_dataloader, device
            )
            np.save(inference_file, df)

    uncalibrated_ece = []
    calibrated_ece = []
    for inference_file in config["inference_files"]:
        df = np.load(inference_file, allow_pickle=True).item()

        accs, bin_counts = acc(df["test"], "Softmax", config["bins"])
        confs, _ = conf(df["test"], "Softmax", config["bins"])
        uncalibrated_ece.append(ece(accs, confs, bin_counts))

        accs, bin_counts = acc(df["test"], "Calibrated Softmax", config["bins"])
        confs, _ = conf(df["test"], "Calibrated Softmax", config["bins"])
        calibrated_ece.append(ece(accs, confs, bin_counts))

    print(f"{config['dataset']}")
    print(f"ECE before calibration: {np.mean(uncalibrated_ece):.2%}")
    print(f"ECE after calibration: {np.mean(calibrated_ece):.2%}")
