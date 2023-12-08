import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from confidence_scoring_functions.TrustScore import get_trustscores
from confidence_scoring_functions.Mahalanobis import get_mahalanobis_scores
from confidence_scoring_functions.SuperTrustScore import STS
from confidence_scoring_functions.Euclidean import get_euclidean_scores
from tabulate import tabulate
from utils import (
    get_softmax_scores,
    get_mcd_scores,
    get_ensemble_scores,
    RC_curve,
    get_classification_performance,
    get_misclassification_results,
    get_embedding_quality,
)

plt.style.use("seaborn-v0_8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Config file (YAML) with embedding file names and uncertainty metrics to calculate",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if (
        "DeepEnsemble" in config["confidence_scoring_functions"]
        and len(config["iid_inference_files"]) < 2
    ):
        print(
            "Must have at least 2 inference result files in order to benchmark DeepEnsemble"
        )
        sys.exit()

    iid_result_paths = [
        os.path.join(config["iid_inference_results_dir"], f)
        for f in config["iid_inference_files"]
    ]
    if config["SD"]:
        sd_result_paths = [
            os.path.join(config["sd_inference_results_dir"], f)
            for f in config["sd_inference_files"]
        ]
        assert len(iid_result_paths) == len(
            sd_result_paths
        ), "Each SD inference file should have a corresponding IID inference file"

    csf_metrics = []
    clf_scores = []
    s_scores = []
    risk_at_coverage = []
    misclf_metrics_df = []
    for idx, iid_result_path in enumerate(iid_result_paths):
        df = np.load(iid_result_path, allow_pickle=True).item()
        df_train = df["train"]
        df_val = df["val"]
        if config["SD"]:
            df_sd = np.load(sd_result_paths[idx], allow_pickle=True).item()
            df_test = df_sd["test"]
        else:
            df_test = df["test"]
        df_test_residuals = (df_test["label"] != df_test["model_pred"]).values.astype(
            int
        )

        if config["get_classification_performance"]:
            clf_scores.append(
                get_classification_performance(df_test, config["clf_metrics"])
            )

        if config["get_clustering_metrics"]:
            s_score = get_embedding_quality(
                df_train,
                df_test,
                config["clustering_reduce_dim"],
                config["clustering_n_components"],
                config["clustering_norm"],
            )
            s_scores.append(s_score)

        csfs = []
        errors = []
        misclf_metrics = []
        plt.figure(figsize=(10, 4))
        for j, csf in enumerate(config["confidence_scoring_functions"]):
            if csf == "Softmax" and config["get_scores"][j]:
                df_test[csf] = get_softmax_scores(df_test)
            elif csf == "MCDropout" and config["get_scores"][j]:
                df_test[csf] = get_mcd_scores(df_test)
            elif csf == "DeepEnsemble" and config["get_scores"][j]:
                if idx == 0:
                    if config["SD"]:
                        idx_remove = len(sd_result_paths) - 1
                    else:
                        idx_remove = len(iid_result_paths) - 1
                else:
                    idx_remove = idx - 1
                if config["SD"]:
                    df_paths = [
                        sd_result_paths[i]
                        for i in np.delete(np.arange(len(sd_result_paths)), idx_remove)
                    ]
                else:
                    df_paths = [
                        iid_result_paths[i]
                        for i in np.delete(np.arange(len(iid_result_paths)), idx_remove)
                    ]
                dfs = []
                for df_path in df_paths:
                    dfs.append(np.load(df_path, allow_pickle=True).item()["test"])
                df_test[csf] = get_ensemble_scores(dfs)
            elif csf == "TrustScore" and config["get_scores"][j]:
                df_test[csf] = get_trustscores(
                    df_train,
                    df_test,
                    config["ts_reduce_dim"],
                    config["ts_n_components"],
                    config["ts_norm"],
                    None,
                    0,
                    config["ts_filtering"],
                    config["ts_num_workers"],
                )
            elif csf == "Mahalanobis" and config["get_scores"][j]:
                df_test[csf] = get_mahalanobis_scores(
                    df_train,
                    df_test,
                    norm=config["mahal_norm"],
                    reduce_dim=config["mahal_reduce_dim"],
                    n_components=config["mahal_n_components"],
                )
            elif csf == "Euclidean" and config["get_scores"][j]:
                df_test[csf] = get_euclidean_scores(
                    df_train,
                    df_test,
                    norm=config["euc_norm"],
                    reduce_dim=config["euc_reduce_dim"],
                    n_components=config["euc_n_components"],
                )
            elif csf == "ConfidNet" and config["get_scores"][j]:
                df_test[csf] = df_test["TCP_hat"]
            elif csf == "Local" and config["get_scores"][j]:
                sts = STS(
                    df_train=df_train,
                    reduce_dim=config["sts_reduce_dim"],
                    n_components=config["sts_n_components"],
                    local_distance_metric=config["local_distance_metric"],
                    filter_training=config["knn_filtering"],
                    local_conf=True,
                    global_conf=False,
                )
                sts.set_k(
                    df_val,
                    config["min_k"],
                    config["max_k"],
                    config["k_step"],
                    config["N_samples"],
                    config["eps"],
                )
                local_confs = sts.compute_conf(df_test)
                df_test[csf] = local_confs
            elif csf == "Global" and config["get_scores"][j]:
                sts = STS(
                    df_train=df_train,
                    reduce_dim=config["sts_reduce_dim"],
                    n_components=config["sts_n_components"],
                    global_distance_metric=config["global_distance_metric"],
                    filter_training=config["knn_filtering"],
                    local_conf=False,
                    global_conf=True,
                )
                global_confs = sts.compute_conf(df_test)
                df_test[csf] = global_confs
            elif csf == "Super-TrustScore" and config["get_scores"][j]:
                sts = STS(
                    df_train=df_train,
                    reduce_dim=config["sts_reduce_dim"],
                    n_components=config["sts_n_components"],
                    local_distance_metric=config["local_distance_metric"],
                    global_distance_metric=config["global_distance_metric"],
                    filter_training=config["knn_filtering"],
                    local_conf=True,
                    global_conf=True,
                )
                sts.set_k(
                    df_val,
                    config["min_k"],
                    config["max_k"],
                    config["k_step"],
                    config["N_samples"],
                    config["eps"],
                )
                local_confs, global_confs = sts.compute_conf(df_test)
                df_test["Local"] = local_confs
                df_test["Global"] = global_confs
                df_test[csf] = local_confs + global_confs

            curve, aurc, _ = RC_curve(df_test_residuals, df_test[csf].values)
            coverages, risks = curve
            errors.append(
                risks[np.where(np.array(coverages) <= config["coverage"])[0][0]]
            )
            csfs.append(aurc * 1000)
            if config["plot_rc"]:
                plt.plot(coverages, risks, label=csf)

            misclf_metrics.append(get_misclassification_results(df_test, csf))

        if True in config["get_scores"]:
            if config["SD"]:
                np.save(sd_result_paths[idx], df_sd)
            else:
                np.save(iid_result_path, df)

        csf_metrics.append(np.array(csfs))
        risk_at_coverage.append(np.array(errors))
        if config["plot_rc"]:
            plot_path = os.path.join(config["plot_dir"], f"RC_plot_{idx}.png")
            if not os.path.exists(os.path.dirname(plot_path)):
                os.makedirs(os.path.dirname(plot_path))
            plt.title(config["plot_title"], fontsize=18)
            plt.xlabel("Coverage")
            plt.ylabel("Risk (1 - Accuracy)")
            plt.legend()
            plt.savefig(plot_path)
            plt.close()

        misclf_metrics_df.append(np.vstack(misclf_metrics))

    if config["get_classification_performance"]:
        print("\nClassification Performance")
        clf_scores = np.mean(clf_scores, axis=0)
        for clf_metric, clf_score in zip(config["clf_metrics"], clf_scores):
            print("{}: {:.3f}".format(clf_metric, clf_score))

    if config["get_clustering_metrics"]:
        print("\nClustering metrics")
        print(
            "Silhouette Score: {:.4} +/- {:.4}".format(
                np.mean(s_scores), np.std(s_scores)
            )
        )

    print("\nFailure Detection Metrics\n")
    aurcs = np.stack(csf_metrics, axis=0).mean(axis=0).reshape(-1, 1)
    risk_at_coverage = np.stack(risk_at_coverage, axis=0).mean(axis=0).reshape(-1, 1)
    failure_det_results = np.stack(misclf_metrics_df, axis=0).mean(axis=0)
    table = np.hstack([aurcs, risk_at_coverage, failure_det_results])
    print(
        tabulate(
            table,
            headers=[
                "CSF",
                "AURC",
                f"Risk@{config['coverage']}",
                "AUROC",
                "FPR@85TPR",
                "FPR@95TPR",
                "AUPR_misclf",
            ],
            showindex=config["confidence_scoring_functions"],
            floatfmt=".3f",
        )
    )
