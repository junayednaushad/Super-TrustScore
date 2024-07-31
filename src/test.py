import numpy as np
import argparse
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    cohen_kappa_score,
)
import warnings


warnings.filterwarnings("ignore", module="sklearn")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Config file (YAML) with inference file names and performance metrics to calculate",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    iid_inference_files = config["iid_inference_files"]
    if config["SD"]:
        sd_inference_files = config["sd_inference_files"]

    scores = []
    for idx, f in enumerate(iid_inference_files):
        model_results = []
        df = np.load(f, allow_pickle=True).item()
        X_train = np.vstack(df["train"]["embedding"].values)
        y_train = df["train"]["label"].values

        if config["SD"]:
            df_test = np.load(sd_inference_files[idx], allow_pickle=True).item()["test"]
        else:
            df_test = df["test"]
        X_test = np.vstack(df_test["embedding"].values)
        y_test = df_test["label"].values

        if config["reduce_dim"]:
            clf = make_pipeline(
                PCA(n_components=config["dim"], random_state=0),
                LogisticRegression(
                    random_state=0, class_weight=config["class_weight"], C=config["C"]
                ),
            )
        else:
            clf = LogisticRegression(
                random_state=0, class_weight=config["class_weight"], C=config["C"]
            )
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)

        for metric in config["clf_metrics"]:
            if metric == "Accuracy":
                model_results.append(accuracy_score(y_test, y_hat))
            elif metric == "Balanced Accuracy":
                model_results.append(balanced_accuracy_score(y_test, y_hat))

        scores.append(model_results)

    scores = np.array(scores).mean(axis=0)
    for metric, score in zip(config["clf_metrics"], scores):
        print(f"{metric}: {score:.2f}")
