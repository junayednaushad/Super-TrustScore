import pandas as pd
import os
import sys
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    cohen_kappa_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm


def get_HAM10k_class_weights(config):
    """Returns class weights for HAM10k, giving higher weight to classes with fewer data points"""

    data_dir = config["data_dir"]
    train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
    split = np.load(
        os.path.join(data_dir, "train_val_test_split.npy"), allow_pickle=True
    ).item()
    train_ids = split["train"]
    _, train_counts = np.unique(
        train_df.loc[train_df["image"].isin(train_ids)]["label"].values,
        return_counts=True,
    )
    class_weights = torch.tensor(
        [np.max(train_counts) / count for count in train_counts]
    ).float()
    return class_weights


def get_EyePACS_class_weights(exp_config):
    """Returns class weights for EyePACS, giving higher weight to classes with fewer data points"""

    data_dir = exp_config["data_dir"]
    train_df = pd.read_csv(os.path.join(data_dir, "trainLabels.csv"))
    class_weights = (
        train_df["level"].value_counts().max()
        / train_df["level"].value_counts().sort_index()
    ).values
    return torch.tensor(class_weights).float()


def add_column_of_label_names(df, config):
    if (
        config["dataset"] == "HAM10k"
        or config["dataset"] == "PAD-UFES-20"
        or config["dataset"] == "ISIC2019"
    ):
        label_to_name = {
            0: "actinic keratosis",
            1: "basal cell carcinoma",
            2: "benign keratosis",
            3: "dermatofibroma",
            4: "melanoma",
            5: "nevus",
            6: "vascular lesion",
        }
    else:
        print("Please specify a dataset that is supported")
        sys.exit()
    df["label_name"] = df["label"].map(label_to_name)
    return df


def get_color_palette(config):
    if (
        config["dataset"] == "HAM10k"
        or config["dataset"] == "PAD-UFES-20"
        or config["dataset"] == "ISIC2019"
    ):
        labels = [
            "actinic keratosis",
            "basal cell carcinoma",
            "benign keratosis",
            "dermatofibroma",
            "melanoma",
            "nevus",
            "vascular lesion",
        ]
    else:
        print("Please specify a dataset that is supported")
        sys.exit()

    colors = sns.color_palette("hls", len(labels))
    palette = {}
    for idx, label in enumerate(labels):
        palette[label] = colors[idx]
    return palette


def get_softmax_scores(df):
    if len(df["label"].unique()) > 2:
        probs = df["probs"].values
        probs = np.vstack(probs)
        return probs.max(axis=1)
    else:

        def model_conf(row):
            if row["model_pred"] == 0:
                return 1 - row["probs"]
            else:
                return row["probs"]

        return df.apply(model_conf, axis=1).values


# def get_mcd_scores(df):
#     if len(df["label"].unique()) > 2:
#         mcd_probs = df["mcd_probs"].values
#         mcd_probs = np.vstack(mcd_probs)
#         model_preds = df["model_pred"].values
#         return mcd_probs[np.arange(len(mcd_probs)), model_preds]
#     else:

#         def model_conf(row):
#             if row["model_pred"] == 0:
#                 return 1 - row["mcd_probs"]
#             else:
#                 return row["mcd_probs"]

#         return df.apply(model_conf, axis=1).values


def get_mcd_scores(df):
    if len(df["label"].unique()) > 2:
        mcd_probs = df["mcd_probs"].values
        mcd_probs = np.vstack(mcd_probs)
        mcd_preds = np.argmax(mcd_probs, axis=1)
        mcd_confs = np.max(mcd_probs, axis=1)
    else:
        mcd_preds = np.round(df["mcd_probs"])
        mcd_confs = (mcd_preds * df["mcd_probs"]) + (
            (1 - mcd_preds) * (1 - df["mcd_probs"])
        )
    return mcd_preds, mcd_confs


# def get_ensemble_scores(dfs, model_preds):
#     probs = []
#     for df in dfs:
#         probs.append(np.vstack(df["probs"].values))
#     probs = np.stack(probs, axis=0).mean(axis=0)
#     return probs[np.arange(len(probs)), model_preds]


def get_ensemble_scores(dfs, is_binary):
    probs = []
    if is_binary:
        for df in dfs:
            probs.append(df["probs"].values)
        probs = np.stack(probs, axis=0).mean(axis=0)
        model_preds = np.round(probs)
        confs = []
        for prob, pred in zip(probs, model_preds):
            if pred == 1:
                confs.append(prob)
            else:
                confs.append(1 - prob)
    else:
        for df in dfs:
            probs.append(np.vstack(df["probs"].values))
        probs = np.stack(probs, axis=0).mean(axis=0)
        model_preds = np.argmax(probs, axis=1)
        confs = probs[np.arange(len(probs)), model_preds]
    return model_preds, confs


def get_classification_performance(df, clf_metrics):
    preds = df["model_pred"].values
    labels = df["label"].values

    scores = []
    for metric in clf_metrics:
        if metric == "Balanced Accuracy":
            scores.append(balanced_accuracy_score(labels, preds))
        if metric == "Accuracy":
            scores.append(accuracy_score(labels, preds))
        if metric == "AUROC" and len(np.unique(labels)) == 2:
            probs = df["probs"].values
            scores.append(roc_auc_score(labels, probs))
        if metric == "Kappa":
            scores.append(cohen_kappa_score(labels, preds, weights="quadratic"))
    return scores


def silhouette_score(test_emb, test_label, train_embs, train_labels):
    intra_cluster = train_embs[train_labels == test_label]
    i = np.mean(np.linalg.norm(test_emb - intra_cluster, ord=2, axis=1))

    inter_cluster_distances = []
    for c in np.unique(train_labels):
        if c != test_label:
            inter_cluster = train_embs[train_labels == c]
            inter_cluster_distances.append(
                np.mean(np.linalg.norm(test_emb - inter_cluster, ord=2, axis=1))
            )
    n = np.min(inter_cluster_distances)

    return (n - i) / max(n, i)


def get_embedding_quality(
    df_train, df_test, reduce_dim=False, n_components=None, norm=None
):
    train_embs = np.vstack(df_train["embedding"].values)
    train_labels = df_train["label"].values
    test_embs = np.vstack(df_test["embedding"].values)
    test_labels = df_test["label"].values

    if reduce_dim:
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(train_embs)
        train_embs = pca.transform(train_embs)
        test_embs = pca.transform(test_embs)
        if norm:
            train_embs = normalize(train_embs, norm="l2", axis=1)
            test_embs = normalize(test_embs, norm="l2", axis=1)

    s_scores = []
    print("Getting embedding quality")
    for test_emb, test_label in tqdm(
        zip(test_embs, test_labels), total=len(test_labels)
    ):
        test_emb = test_emb.reshape(1, -1)
        s_scores.append(
            silhouette_score(test_emb, test_label, train_embs, train_labels)
        )
    return np.mean(s_scores)


def fpr_at_tpr(y_true, y_score, rate):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tpr[tpr >= rate] = True
    return fpr[np.where(tpr == 1)[0][0]]


def get_misclassification_results(results_df, csf):
    df = results_df.copy(deep=True)
    df["correct"] = df["model_pred"] == df["label"]
    df["incorrect"] = df["model_pred"] != df["label"]
    auc = roc_auc_score(df["correct"], df[csf])
    fpr85 = fpr_at_tpr(df["correct"], df[csf], 0.85)
    fpr95 = fpr_at_tpr(df["correct"], df[csf], 0.95)
    aupr = average_precision_score(df["incorrect"], -1 * df[csf])
    return [auc, fpr85, fpr95, aupr]


def get_per_class_misclassification_results(df, csf):
    aucs = []
    fpr85s = []
    fpr95s = []
    auprs = []
    for lbl in np.unique(df["label"]):
        df_lbl = df.loc[df["label"] == lbl].copy(deep=True)
        df_lbl["correct"] = df_lbl["model_pred"] == df_lbl["label"]
        df_lbl["incorrect"] = df_lbl["model_pred"] != df_lbl["label"]
        try:
            auc = roc_auc_score(df_lbl["correct"], df_lbl[csf])
        except ValueError:
            auc = 1
        aucs.append(auc)

        aupr = average_precision_score(df_lbl["incorrect"], -1 * df_lbl[csf])
        auprs.append(aupr)

        fpr85 = fpr_at_tpr(df_lbl["correct"], df_lbl[csf], 0.85)
        if np.isnan(fpr85):
            fpr85 = 0
        fpr85s.append(fpr85)
        fpr95 = fpr_at_tpr(df_lbl["correct"], df_lbl[csf], 0.95)
        if np.isnan(fpr95):
            fpr95 = 0
        fpr95s.append(fpr95)
    return aucs, fpr85s, fpr95s, auprs


def RC_curve(residuals, confidence):
    coverages = []
    risks = []
    n = len(residuals)
    idx_sorted = np.argsort(confidence)
    cov = n
    error_sum = sum(residuals[idx_sorted])
    coverages.append(cov / n),
    risks.append(error_sum / n)
    weights = []
    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        cov = cov - 1
        error_sum = error_sum - residuals[idx_sorted[i]]
        selective_risk = error_sum / (n - 1 - i)
        tmp_weight += 1
        if i == 0 or confidence[idx_sorted[i]] != confidence[idx_sorted[i - 1]]:
            coverages.append(cov / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    # add a well-defined final point to the RC-curve.
    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / n)

    # aurc is computed as a weighted average over risk scores analogously to the average precision score.
    aurc = sum([a * w for a, w in zip(risks, weights)])

    # compute e-aurc
    err = np.mean(residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    e_aurc = aurc - kappa_star_aurc

    curve = (coverages, risks)
    return curve, aurc, e_aurc
