import os
from PIL import Image
import torch
import torchvision.transforms as T  # necessary for eval(self.config["train_transforms"])
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import pandas as pd
import numpy as np


class HAM10kDataset(Dataset):
    def __init__(self, df, ids, image_dir, transform=None, return_filename=False):
        self.df = df
        self.ids = ids
        self.image_dir = image_dir
        self.transform = transform
        self.return_filename = return_filename

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.df.loc[self.df["image"] == image_id, "label"].values[0]
        if self.return_filename:
            return image, torch.tensor(label), image_id
        else:
            return image, torch.tensor(label)

    def __len__(self):
        return len(self.ids)


class HAM10kDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]

    def setup(self, stage=None):
        print("Creating HAM10k datasets")
        train_df = pd.read_csv(os.path.join(self.data_dir, "train_df.csv"))
        df_test = pd.read_csv(os.path.join(self.data_dir, "test_df.csv"))
        train_image_dir = os.path.join(self.data_dir, "preprocessed_train_images")
        test_image_dir = os.path.join(self.data_dir, "preprocessed_test_images")
        split = np.load(
            os.path.join(self.data_dir, "train_val_test_split.npy"), allow_pickle=True
        ).item()
        train_ids = split["train"]
        val_ids = split["val"]
        test_ids = split["test"]
        print("Number of samples")
        print("Train:", len(train_ids))
        print("Val:", len(val_ids))
        print("Test:", len(test_ids))

        train_transforms = eval(self.config["train_transforms"])
        test_transforms = eval(self.config["test_transforms"])
        self.train_dataset = HAM10kDataset(
            train_df,
            train_ids,
            train_image_dir,
            train_transforms,
            self.config["return_filename"],
        )
        self.val_dataset = HAM10kDataset(
            train_df,
            val_ids,
            train_image_dir,
            test_transforms,
            self.config["return_filename"],
        )
        self.test_dataset = HAM10kDataset(
            df_test,
            test_ids,
            test_image_dir,
            test_transforms,
            self.config["return_filename"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class ISIC2019Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None, return_filename=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.return_filename = return_filename

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["image"]
        image_path = os.path.join(self.image_dir, filename + ".jpg")
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx]["label"]
        if self.return_filename:
            return image, torch.tensor(label), filename
        else:
            return image, torch.tensor(label)

    def __len__(self):
        return len(self.df)


class ISIC2019DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]

    def setup(self):
        # ISIC 2019 is only used for testing/inference
        print("Creating ISIC_2019 dataset")
        image_dir = os.path.join(self.data_dir, "preprocessed_images")
        df = pd.read_csv(os.path.join(self.data_dir, "labels.csv"))
        print("Number of samples:", len(df))

        test_transforms = eval(self.config["test_transforms"])
        self.test_dataset = ISIC2019Dataset(
            df, image_dir, test_transforms, self.config["return_filename"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class PAD_UFES_20_Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None, return_filename=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.return_filename = return_filename

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["img_id"].split(".")[0]
        image_path = os.path.join(self.image_dir, filename + ".jpg")
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx]["label"]
        if self.return_filename:
            return image, torch.tensor(label), filename
        else:
            return image, torch.tensor(label)

    def __len__(self):
        return len(self.df)


class PAD_UFES_20_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]

    def setup(self):
        print("Creating PAD-UFES-20 dataset")
        image_dir = os.path.join(self.data_dir, "preprocessed_images")
        df = pd.read_csv(os.path.join(self.data_dir, "labels.csv"))
        print("Number of samples:", len(df))

        test_transforms = eval(self.config["test_transforms"])
        self.test_dataset = PAD_UFES_20_Dataset(
            df, image_dir, test_transforms, self.config["return_filename"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class EyePACSDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, return_filename=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.return_filename = return_filename

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image_id = data["image"]
        image_path = os.path.join(self.image_dir, image_id + ".jpeg")
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = data["label"]
        if self.return_filename:
            return image, torch.tensor(label), image_id
        else:
            return image, torch.tensor(label)

    def __len__(self):
        return len(self.df)


class EyePACSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]

    def setup(self, stage=None):
        print("Creating EyePACS datasets")
        df_train = pd.read_csv(os.path.join(self.data_dir, "trainLabels.csv"))
        if self.num_classes == 2:
            df_train["label"] = (df_train["level"] > 1).astype(int)
        else:
            df_train = df_train.rename(columns={"level": "label"})

        df_val_test = pd.read_csv(
            os.path.join(self.data_dir, "retinopathy_solution.csv")
        )
        if self.num_classes == 2:
            df_val_test["label"] = (df_val_test["level"] > 1).astype(int)
        else:
            df_val_test = df_val_test.rename(columns={"level": "label"})
        df_val = df_val_test.loc[df_val_test["Usage"] == "Public"]
        df_test = df_val_test.loc[df_val_test["Usage"] == "Private"]

        print("Number of samples")
        print("Train:", len(df_train))
        print("Val:", len(df_val))
        print("Test:", len(df_test))

        train_image_dir = os.path.join(self.data_dir, "preprocessed_train_images")
        val_test_image_dir = os.path.join(self.data_dir, "preprocessed_test_images")
        train_transforms = eval(self.config["train_transforms"])
        test_transforms = eval(self.config["test_transforms"])
        self.train_dataset = EyePACSDataset(
            df_train, train_image_dir, train_transforms, self.config["return_filename"]
        )
        self.val_dataset = EyePACSDataset(
            df_val, val_test_image_dir, test_transforms, self.config["return_filename"]
        )
        self.test_dataset = EyePACSDataset(
            df_test, val_test_image_dir, test_transforms, self.config["return_filename"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class MessidorDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, return_filename=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.return_filename = return_filename

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image_id = data["image_id"]
        if image_id.split(".")[-1] == "png":
            image_path = os.path.join(self.image_dir, image_id)
        else:
            image_path = os.path.join(self.image_dir, image_id.replace("jpg", "JPG"))
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = data["adjudicated_dr_grade"]
        if self.return_filename:
            return image, torch.tensor(label), image_id
        else:
            return image, torch.tensor(label)

    def __len__(self):
        return len(self.df)


class MessidorDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]

    def setup(self, stage=None):
        print("Creating Messidor-2 datasets")
        image_dir = os.path.join(self.data_dir, "preprocessed_images")
        df = pd.read_csv(os.path.join(self.data_dir, "messidor_data.csv"))
        df.dropna(inplace=True)
        print("Number of samples:", len(df))
        if self.num_classes == 2:
            df["adjudicated_dr_grade"] = (df["adjudicated_dr_grade"] > 1).astype(int)
        test_transforms = eval(self.config["test_transforms"])
        self.test_dataset = MessidorDataset(
            df, image_dir, test_transforms, self.config["return_filename"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class APTOS_2019_Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None, return_filename=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.return_filename = return_filename

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image_id = data["id_code"]
        image_path = os.path.join(self.image_dir, image_id + ".png")
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = data["diagnosis"]
        if self.return_filename:
            return image, torch.tensor(label), image_id
        else:
            return image, torch.tensor(label)

    def __len__(self):
        return len(self.df)


class APTOS_2019_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]

    def setup(self, stage=None):
        print("Creating APTOS 2019 datasets")
        image_dir = os.path.join(self.data_dir, "preprocessed_train_images")
        df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        if self.num_classes == 2:
            df["diagnosis"] = (df["diagnosis"] > 1).astype(int)
        print("Number of samples:", len(df))

        test_transforms = eval(self.config["test_transforms"])
        self.test_dataset = APTOS_2019_Dataset(
            df, image_dir, test_transforms, self.config["return_filename"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class ConfidNetDataset(Dataset):
    def __init__(self, inference_df):
        self.inference_df = inference_df

    def __getitem__(self, idx):
        data = self.inference_df.iloc[idx]
        emb = torch.from_numpy(data["embedding"]).squeeze()
        lbl = int(data["label"])
        TCP = torch.tensor(data["probs"].squeeze()[lbl])
        return emb, TCP

    def __len__(self):
        return len(self.inference_df)


class ConfidNetDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]
        self.SD = config["SD"]

    def setup(self, stage=None):
        inference_results = np.load(
            self.config["inference_file"], allow_pickle=True
        ).item()
        if not self.SD:
            self.train_dataset = ConfidNetDataset(inference_results["train"])
            self.val_dataset = ConfidNetDataset(inference_results["val"])
        self.test_dataset = ConfidNetDataset(inference_results["test"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
