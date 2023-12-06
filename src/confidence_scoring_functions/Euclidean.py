import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def get_euclidean_scores(df_train, df_test, norm, reduce_dim, n_components=None):
    euclidean = Euclidean(norm, reduce_dim, n_components)
    euclidean.fit(df_train)
    distances = euclidean.predict(df_test)
    model_preds = df_test["model_pred"].values
    scores = []
    for dist, pred in zip(distances, model_preds):
        dist = dist.flatten()
        scores.append(-1 * dist[pred])
    return scores


class Euclidean:
    def __init__(self, norm, reduce_dim, n_componenets=None):
        self.reduce_dim = reduce_dim
        self.n_components = n_componenets
        if self.reduce_dim:
            self.pca = PCA(n_components=self.n_components, random_state=0)
        self.norm = norm

    def fit(self, df_train):
        self.labels = df_train["label"].values

        train_embs = np.vstack(df_train["embedding"].values)
        if self.reduce_dim:
            self.pca.fit(train_embs)
            train_embs = self.pca.transform(train_embs)
        if self.norm:
            train_embs = normalize(train_embs, norm="l2", axis=1)

        self.centroids = self._get_centroids(train_embs)

    def predict(self, df_test):
        test_embs = np.vstack(df_test["embedding"].values)
        if self.reduce_dim:
            test_embs = self.pca.transform(test_embs)
        if self.norm:
            test_embs = normalize(test_embs, norm="l2", axis=1)

        preds = []
        for centroid in self.centroids:
            preds.append(
                np.sqrt(np.sum((test_embs - centroid) ** 2, axis=1)).reshape(-1, 1)
            )
        return np.hstack(preds)

    def _get_centroids(self, embs):
        centroids = []
        for label in np.sort(np.unique(self.labels)):
            centroids.append(np.mean(embs[self.labels == label], axis=0))
        return centroids
