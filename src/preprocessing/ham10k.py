import random
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.utils import apply_cc

random.seed(0)
np.random.seed(0)


def get_labels(df):
    """
    The metadata contains one-hot labels and this function converts them into integer labels

    Parameters
    ----------
    df : pandas.DataFrame
        Contains metadata with columns corresponding to each diagnosis (i.e., ['BKL', 'DF',...,'VASC'])

    Returns
    -------
    list of int
        List of labels
    """
    labels = []
    for i in range(len(df)):
        sample = df.iloc[i]
        if sample["AKIEC"] == 1:
            labels.append(0)
        elif sample["BCC"] == 1:
            labels.append(1)
        elif sample["BKL"] == 1:
            labels.append(2)
        elif sample["DF"] == 1:
            labels.append(3)
        elif sample["MEL"] == 1:
            labels.append(4)
        elif sample["NV"] == 1:
            labels.append(5)
        else:
            labels.append(6)
    return labels


if __name__ == "__main__":
    # preprocess images
    img_train_paths = glob("../../data/HAM10k/ISIC2018_Task3_Training_Input/*.jpg")
    output_train_path = "../../data/HAM10k/preprocessed_train_images"
    img_test_paths = glob("../../data/HAM10k/ISIC2018_Task3_Test_Input/*.jpg")
    output_test_path = "../../data/HAM10k/preprocessed_test_images"

    apply_cc(img_train_paths, output_train_path, True)
    apply_cc(img_test_paths, output_test_path, True)

    # split data
    train_gt_df = pd.read_csv(
        "../../data/HAM10k/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
    )
    test_gt_df = pd.read_csv(
        "../../data/HAM10k/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"
    )
    lesion_df = pd.read_csv(
        "../../data/HAM10k/ISIC2018_Task3_Training_LesionGroupings.csv"
    )
    train_lesions, val_lesions = train_test_split(
        np.unique(lesion_df["lesion_id"]), test_size=0.1, random_state=0
    )
    train_ids = []
    for lesion in train_lesions:
        train_ids += list(
            lesion_df.loc[lesion_df["lesion_id"] == lesion, "image"].values
        )
    val_ids = []
    for lesion in val_lesions:
        val_ids += list(lesion_df.loc[lesion_df["lesion_id"] == lesion, "image"].values)
    test_ids = test_gt_df["image"].values
    np.save(
        "../../data/HAM10k/train_val_test_split.npy",
        {"train": train_ids, "val": val_ids, "test": test_ids},
    )
    print("Number of samples")
    print("Train:", len(train_ids))
    print("Val:", len(val_ids))
    print("Test:", len(test_ids))

    # save dataframes with labels
    train_gt_df["label"] = get_labels(train_gt_df)
    train_gt_df.to_csv("../../data/HAM10k/train_df.csv", index=False)
    test_gt_df["label"] = get_labels(test_gt_df)
    test_gt_df.to_csv("../../data/HAM10k/test_df.csv", index=False)
