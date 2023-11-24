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
    df = pd.read_csv("../../data/PAD-UFES-20/metadata.csv")
    # remove squamous cell carcinoma examples, should be left with 2106 images
    test_df = df.loc[df["diagnostic"] != "SCC"]

    # save dataframe with labels
    test_df["label"] = test_df["diagnostic"].map(
        {"NEV": 5, "BCC": 1, "ACK": 0, "SEK": 2, "MEL": 4}
    )
    test_df.to_csv("../../data/PAD-UFES-20/labels.csv", index=False)

    # get image files
    img_dir = "../../data/PAD-UFES-20/images"
    img_ids = test_df["img_id"].values
    img_paths = [os.path.join(img_dir, f) for f in img_ids]
    output_path = "../../data/PAD-UFES-20/preprocessed_images"

    # preprocess images
    apply_cc(img_paths, output_path, True)
