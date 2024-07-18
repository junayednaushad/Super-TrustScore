import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
import argparse
import yaml
import os


def get_mahalanobis_scores(
    df_train,
    df_test,
    norm,
    tied_covariance,
    relative,
    reduce_dim,
    n_components=None,
    batch_size=1,
):
    mahalanobis = Mahalanobis(
        norm, tied_covariance, relative, reduce_dim, n_components, batch_size
    )
    mahalanobis.fit(df_train)
    distances = mahalanobis.predict(df_test)
    if not relative:
        model_preds = df_test["model_pred"].values.astype(int)
        scores = -1 * distances[np.arange(len(model_preds)), model_preds]
    else:
        scores = -1 * np.min(distances, axis=1)
    return scores


class Mahalanobis:
    def __init__(
        self,
        norm,
        tied_covariance,
        relative,
        reduce_dim,
        n_componenets=None,
        batch_size=1,
    ):
        """
        Initialize particular version of Mahalanobis distance to use

        Parameters
        ----------
        norm : bool
            If True, normalize embeddings to be unit vectors.
        tied_covariance : bool
            If True, the class-conditioned distributions will share the same covariance matrix.
            If False, separate covariance matrices will be used for each class.
        relative : bool
            If True, computes relative Mahalanobis distance as defined in
            'A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection'
        reduce_dim : bool
            If True, apply PCA to embeddings
        n_components : int or float
            If n_components > 1 then it is the number of features (with highest variability) to retain.
            If 0 < n_components < 1 then the number of features required to retain that percentage of explained
            variability will be kept
        batch_size : int
            Batch size for computing mahalnobis distance
        """
        self.reduce_dim = reduce_dim
        self.n_components = n_componenets
        if self.reduce_dim:
            self.pca = make_pipeline(
                StandardScaler(), PCA(n_components=self.n_components, random_state=0)
            )
        self.norm = norm
        self.tied_covariance = tied_covariance
        self.relative = relative
        self.batch_size = batch_size

    def fit(self, df_train):
        self.labels = df_train["label"].values

        train_embs = np.vstack(df_train["embedding"].values)
        if self.reduce_dim:
            self.pca.fit(train_embs)
            train_embs = self.pca.transform(train_embs)
        if self.norm:
            train_embs = normalize(train_embs, norm="l2", axis=1)

        self.centroids = self._get_centroids(train_embs)
        self.covariances = self._get_covariances(train_embs)

        if self.relative:
            self.mean_0 = np.mean(train_embs, axis=0)
            self.cov_0 = np.cov(train_embs, rowvar=False)

    def predict(self, df_test):
        test_embs = np.vstack(df_test["embedding"].values)
        if self.reduce_dim:
            test_embs = self.pca.transform(test_embs)
        if self.norm:
            test_embs = normalize(test_embs, norm="l2", axis=1)

        preds = []
        start_idx = 0
        end_idx = self.batch_size
        for start_idx in tqdm(range(0, test_embs.shape[0], self.batch_size)):
            end_idx = min(test_embs.shape[0], start_idx + self.batch_size)
            preds.append(self._mahalanobis_distance(test_embs[start_idx:end_idx]))
        return np.vstack(preds)

    def _get_centroids(self, embs):
        centroids = {}
        for label in np.sort(np.unique(self.labels)):
            centroids[label] = np.mean(embs[self.labels == label], axis=0)
        return centroids

    def _get_covariances(self, embs):
        if self.tied_covariance:
            return np.cov(embs, rowvar=False)
        else:
            covariances = {}
            for label in np.sort(np.unique(self.labels)):
                covariances[label] = np.cov(embs[self.labels == label], rowvar=False)
                if not np.all(np.linalg.eigvals(covariances[label]) > 0):
                    covariances[label] += np.eye(covariances[label].shape[0]) * 1e-13
                    assert np.all(
                        np.linalg.eigvals(covariances[label]) > 0
                    ), "Covariance matrix has numerical error so need to add larger positive value"

            return covariances

    def _mahalanobis_distance(self, embs):
        distances = []
        for label in np.sort(np.unique(self.labels)):
            centroid = self.centroids[label].reshape(1, -1)
            if self.tied_covariance:
                cov = self.covariances
            else:
                cov = self.covariances[label]
            distances.append(
                np.sqrt(
                    (
                        (embs - centroid) @ np.linalg.inv(cov) @ (embs - centroid).T
                    ).diagonal()
                )
            )
        distances = np.vstack(distances).T
        if not self.relative:
            return distances
        else:
            distance_0 = np.sqrt(
                (
                    (embs - self.mean_0)
                    @ np.linalg.inv(self.cov_0)
                    @ (embs - self.mean_0).T
                ).diagonal()
            )
            return distances - distance_0.reshape(-1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file (YAML)")
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)

    iid_file_paths = [
        os.path.join(config["iid_inference_results_dir"], f)
        for f in config["iid_inference_files"]
    ]
    if config["SD"]:
        sd_file_paths = [
            os.path.join(config["sd_inference_results_dir"], f)
            for f in config["sd_inference_files"]
        ]
    min_mahal_distances = []
    gt_mahal_distances = []
    accs = []
    for idx, path in enumerate(iid_file_paths):
        iid_df = np.load(path, allow_pickle=True).item()
        iid_df_train = iid_df["train"]

        mahalanobis = Mahalanobis(
            config["mahal_norm"],
            config["tied_covariance"],
            config["relative"],
            config["mahal_reduce_dim"],
            config["mahal_n_components"],
            config["mahal_batch_size"],
        )
        mahalanobis.fit(iid_df_train)
        if not config["SD"]:
            df_test = iid_df["test"]
        else:
            df_test = np.load(sd_file_paths[idx], allow_pickle=True).item()["test"]

        distances = np.vstack(mahalanobis.predict(df_test))
        min_distance = np.min(distances, axis=1)
        min_mahal_distances.append(np.mean(min_distance))

        labels = df_test["label"].values
        gt_distance = distances[np.arange(distances.shape[0]), labels]
        gt_mahal_distances.append(np.mean(gt_distance))

        accuracy = np.sum(gt_distance <= min_distance) / distances.shape[0]
        accs.append(accuracy)

    print()
    print(config["dataset"])
    print(
        f"Mean Mahalanobis distance from nearest class: {np.mean(min_mahal_distances):.2f}"
    )
    print(
        f"Mean Mahalanobis distance from ground truth class: {np.mean(gt_mahal_distances):.2f}"
    )
    print(f"Accuracy: {np.mean(accs):.2f}")
