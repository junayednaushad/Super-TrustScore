import random
import numpy as np
import os
import pandas as pd
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
        if sample["AK"] == 1:
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
    df = pd.read_csv("../../data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv")
    # remove squamous cell carcinoma examples
    test_df = df.loc[df["SCC"] == 0]
    # remove unknown examples
    test_df = test_df.loc[test_df["UNK"] == 0]
    # remove HAM10k examples
    ham10k = np.load(
        "../../data/HAM10k/train_val_test_split.npy", allow_pickle=True
    ).item()
    ham10k_ids = list(ham10k["train"]) + list(ham10k["val"]) + list(ham10k["test"])
    test_df = test_df.loc[~test_df["image"].isin(ham10k_ids)]
    # should be left with 14885 images

    # save dataframe with labels
    test_df["label"] = get_labels(test_df)
    test_df.to_csv("../../data/ISIC_2019/labels.csv", index=False)

    # get image files
    img_dir = "../../data/ISIC_2019/ISIC_2019_Training_Input"
    img_ids = test_df["image"].values
    filenames = [f + ".jpg" for f in img_ids]
    img_paths = [os.path.join(img_dir, f) for f in filenames]
    output_path = "../../data/ISIC_2019/preprocessed_images"

    # preprocess images
    apply_cc(img_paths, output_path, True)
