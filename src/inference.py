import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from dataset import (
    HAM10kDataModule,
    ISIC2019DataModule,
    PAD_UFES_20_DataModule,
    EyePACSDataModule,
    MessidorDataModule,
)
from model import LitSwin, get_intermediate_features


def get_inference_dataframe(model, dataloader, device, config, test):
    """Returns dataframe with |filename|label|embedding|model_pred|probs|MCDropout_probs|"""
    filenames = []
    labels = []
    embeddings = []
    preds = []
    probs = []
    if test and config["MCDropout"]:
        mcd_probs = []

    for batch in tqdm(dataloader):
        x, y, f = batch
        x = x.to(device)
        embs, y_hat, y_probs = lit_swin.predict_step(batch=(x, y, f), mcdropout=False)

        filenames += f
        labels += list(y.cpu().numpy())
        embeddings.append(embs)
        preds += y_hat
        probs.append(y_probs)

        if test and config["MCDropout"]:
            dropout_probs = []
            for _ in range(config["num_inferences"]):
                _, _, y_probs = lit_swin.predict_step(batch=(x, y, f), mcdropout=True)
                dropout_probs.append(y_probs)
            mcd_probs.append(np.stack(dropout_probs, axis=0).mean(axis=0))

    embeddings = np.vstack(embeddings)
    embs = []
    for emb in embeddings:
        embs.append(emb.reshape(1, -1))

    probs = np.vstack(probs)
    softmax_probs = []
    for prob in probs:
        softmax_probs.append(prob.reshape(1, -1))

    df = pd.DataFrame(
        {
            "filename": filenames,
            "label": labels,
            "embedding": embs,
            "model_pred": preds,
            "probs": softmax_probs,
        }
    )

    if test and config["MCDropout"]:
        mcd_probs = np.vstack(mcd_probs)
        mcd_msr = []
        for prob in mcd_probs:
            mcd_msr.append(prob.reshape(1, -1))
        df["mcd_probs"] = mcd_msr

    return df


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
    else:
        print("Please specify a dataset that is supported (i.e. HAM10k, ISIC2019)")
        sys.exit()
    data_module.setup()
    if not config["only_test"]:
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)

    test_scores = []
    accs = []
    ckpt_path = config["ckpt_path"]
    save_path = config["save_path"]

    print(f"Getting inference results for {ckpt_path}")
    lit_swin = LitSwin.load_from_checkpoint(ckpt_path)
    lit_swin.model.avgpool.register_forward_hook(get_intermediate_features("embedding"))
    if not config["only_test"]:
        df_train = get_inference_dataframe(
            model=lit_swin,
            dataloader=train_dataloader,
            device=device,
            config=config,
            test=False,
        )
        df_val = get_inference_dataframe(
            model=lit_swin,
            dataloader=val_dataloader,
            device=device,
            config=config,
            test=False,
        )
    df_test = get_inference_dataframe(
        model=lit_swin,
        dataloader=test_dataloader,
        device=device,
        config=config,
        test=True,
    )

    # save dataframes
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if not config["only_test"]:
        np.save(
            save_path,
            {"train": df_train, "val": df_val, "test": df_test},
        )
    else:
        np.save(
            save_path,
            {"test": df_test},
        )
