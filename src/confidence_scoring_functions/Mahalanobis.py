import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm


def get_mahalanobis_scores(df_train, df_test, norm, reduce_dim, n_components=None):
    mahalanobis = Mahalanobis(norm, reduce_dim, n_components)
    mahalanobis.fit(df_train)
    distances = mahalanobis.predict(df_test)
    model_preds = df_test["model_pred"].values
    scores = []
    for dist, pred in zip(distances, model_preds):
        dist = dist.flatten()
        mahal_pred = dist[pred]
        scores.append(-1 * mahal_pred)
    return scores


class Mahalanobis:
    def __init__(self, distance_metric, reduce_dim, n_componenets=None):
        self.reduce_dim = reduce_dim
        self.n_components = n_componenets
        if self.reduce_dim:
            self.pca = PCA(n_components=self.n_components, random_state=0)
        self.distance_metric = distance_metric

    def fit(self, df_train):
        self.labels = df_train["label"].values

        train_embs = np.vstack(df_train["embedding"].values)
        if self.reduce_dim:
            self.pca.fit(train_embs)
            train_embs = self.pca.transform(train_embs)
        if self.distance_metric == "cosine":
            train_embs = normalize(train_embs, norm="l2", axis=1)

        self.centroids = self._get_centroids(train_embs)
        self.covariances = self._get_covariances(train_embs)

    def predict(self, df_test):
        test_embs = np.vstack(df_test["embedding"].values)
        if self.reduce_dim:
            test_embs = self.pca.transform(test_embs)
        if self.distance_metric == "cosine":
            test_embs = normalize(test_embs, norm="l2", axis=1)

        preds = []
        for test_emb in tqdm(test_embs):
            preds.append(self._mahalanobis_distance(test_emb))
        return preds

    def _get_centroids(self, embs):
        centroids = {}
        for label in np.sort(np.unique(self.labels)):
            centroids[label] = np.mean(embs[self.labels == label], axis=0)
        return centroids

    def _get_covariances(self, embs):
        covariances = {}
        for label in np.sort(np.unique(self.labels)):
            covariances[label] = np.cov(embs[self.labels == label], rowvar=False)
            if not np.all(np.linalg.eigvals(covariances[label]) > 0):
                covariances[label] += np.eye(covariances[label].shape[0]) * 1e-15
                assert np.all(
                    np.linalg.eigvals(covariances[label]) > 0
                ), "Covariance matrix has numerical error so need to add larger positive value"

        return covariances

    def _mahalanobis_distance(self, emb):
        emb = emb.reshape(1, -1)
        distances = []
        for label in np.sort(np.unique(self.labels)):
            centroid = self.centroids[label].reshape(1, -1)
            cov = self.covariances[label]
            distances.append(
                np.sqrt((emb - centroid) @ np.linalg.inv(cov) @ (emb - centroid).T)
            )
        return np.array(distances).reshape(1, -1)
