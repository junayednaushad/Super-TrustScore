import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.getcwd()))
from utils import RC_curve

plt.style.use("seaborn-v0_8-dark")


def get_sts_scores(
    df_train,
    df_val,
    df_test,
    reduce_local_dim,
    reduce_global_dim,
    n_components,
    tied_covariance,
    local_distance_metric,
    global_norm,
    global_batch_size,
    rmd,
    filter_training,
    local_conf,
    global_conf,
    min_k,
    max_k,
    k_step,
    N_samples,
    eps,
):
    sts = STS(
        df_train=df_train,
        reduce_local_dim=reduce_local_dim,
        reduce_global_dim=reduce_global_dim,
        n_components=n_components,
        tied_covariance=tied_covariance,
        local_distance_metric=local_distance_metric,
        global_norm=global_norm,
        global_batch_size=global_batch_size,
        rmd=rmd,
        filter_training=filter_training,
        local_conf=local_conf,
        global_conf=global_conf,
    )
    sts.set_k(df_val, min_k, max_k, k_step, N_samples, eps)
    if local_conf and global_conf:
        local_confs, global_confs = sts.compute_conf(df_test)
        return local_confs, global_confs
    else:
        return sts.compute_conf(df_test)


class STS:
    def __init__(
        self,
        df_train,
        k=None,
        reduce_local_dim=True,
        reduce_global_dim=True,
        n_components=0.9,
        local_distance_metric="l2",
        tied_covariance=False,
        global_norm=False,
        global_batch_size=1,
        rmd=False,
        filter_training=False,
        local_conf=True,
        global_conf=True,
    ):
        """
        Initializes Super-TrustScore with training data

        Parameters
        ----------
        df_train : pandas.DataFrame
            DataFrame containing training data with following columns: 'label', 'model_pred', and 'embedding'
            where 'embedding' column contains embeddings as row vectors (1, D)
        k : int
            Number of nearest neighbors for local confidence (can be initialized to None)
        reduce_local_dim : bool
            Determines if embedding dimensions should be reduced prior to computing Local confidence
        reduce_global_dim : bool
            Determines if embedding dimensions should be reduced prior to computing Global confidence
        n_components : int or float
            If n_components > 1 then it is the number of features (with highest variability) to retain.
            If 0 < n_components < 1 then the number of features required to retain that percentage of explained
            variability will be kept
        local_distance_metric : String
            Distance metric (e.g., cosine, l2) used to compute nearest neighbors for the Local confidence score
        tied_covariance : bool
            True if a tied covariance matrix is used to compute the Mahalanobis distances for the Global confidence score
            False if a different covariance matrix is used for each class conditional distribution
        global_norm : bool
            If True, normalize embeddings prior to computing Mahalanobis distances for Global confidence score
        global_batch_size : int
            Batch size used to compute Mahalanobis distances
        rmd : bool
            If True, compute relative Mahalanobis distance
        filter_training : bool
            Determines if outliers in training data should be removed
        local_conf : bool
            Determines if local confidence score should be computed
        global_conf : bool
            Determines if global confidence score should be computed
        """
        self.df_train = df_train
        self.k = k
        self.classes = np.sort(np.unique(self.df_train["label"].values))
        self.reduce_local_dim = reduce_local_dim
        self.reduce_global_dim = reduce_global_dim
        self.n_components = n_components
        self.local_distance_metric = local_distance_metric
        self.tied_covariance = tied_covariance
        self.global_norm = global_norm
        self.global_batch_size = global_batch_size
        self.rmd = rmd
        self.filter = filter_training
        self.local_conf = local_conf
        self.global_conf = global_conf

        self.train_embs = np.vstack(self.df_train["embedding"].values)

        if self.filter:
            print("Removing outlier training examples")
            unfiltered_idxs = []
            for lbl in tqdm(self.df_train["label"].unique()):
                lbl_df = df_train.loc[df_train["label"] == lbl]
                idxs = lbl_df.index.values
                embs = self.train_embs[idxs]
                centroid = np.mean(embs, axis=0)
                covariance = np.cov(embs, rowvar=False)
                if not np.all(np.linalg.eigvals(covariance) > 0):
                    covariance += np.eye(covariance.shape[0]) * 1e-15
                    assert np.all(
                        np.linalg.eigvals(covariance) > 0
                    ), "Covariance matrix has numerical error so need to add larger positive value"
                distances = []
                for emb in embs:
                    emb = emb.reshape(1, -1)
                    centroid = centroid.reshape(1, -1)
                    distances.append(
                        np.sqrt(
                            (emb - centroid)
                            @ np.linalg.inv(covariance)
                            @ (emb - centroid).T
                        )
                    )
                cutoff = np.percentile(distances, 97)
                unfiltered_idxs += list(idxs[np.where(distances < cutoff)[0]])
            self.df_train = self.df_train.iloc[unfiltered_idxs]
            self.train_embs = self.train_embs[unfiltered_idxs]

        if self.reduce_local_dim or self.reduce_global_dim:
            print("Reducing embedding dimensions")
            print("Original embedding size:", self.train_embs[0].shape[0])
            self.pca = make_pipeline(
                StandardScaler(), PCA(n_components=self.n_components, random_state=0)
            )
            self.pca.fit(self.train_embs)

        if self.local_conf:
            self.local_train_embs = self.train_embs.copy()
            if self.reduce_local_dim:
                self.local_train_embs = self.pca.transform(self.local_train_embs)
                print("Local embedding size:", self.local_train_embs.shape[1])

        if self.global_conf:
            self.labels = self.df_train["label"].values
            self.global_train_embs = self.train_embs.copy()

            if self.reduce_global_dim:
                self.global_train_embs = self.pca.transform(self.global_train_embs)
                print("Global embedding size:", self.global_train_embs.shape[1])
            if self.global_norm:
                self.global_train_embs = normalize(
                    self.global_train_embs, norm="l2", axis=1
                )
            self.centroids = self._get_centroids(self.global_train_embs)
            self.covariances = self._get_covariances(self.global_train_embs)
            if self.rmd:
                self.mean_0 = np.mean(self.global_train_embs, axis=0)
                self.cov_0 = np.cov(self.global_train_embs, rowvar=False)

    def set_k(self, df_val, min_k, max_k, k_step, N_samples, eps):
        """
        Sets k to the best value found using the validation data
        """
        print("Setting k to best value according to scores on validation data")

        k_values, aurcs, stats = self.search_k_values(
            df_val, min_k, max_k, k_step, N_samples
        )
        best_score = 1000
        for idx, score in enumerate(aurcs):
            if score < best_score + eps:
                best_idx = idx
                best_score = score
                best_k = (idx * k_step) + min_k
        self.k = best_k
        print("Best k is", self.k)

        if self.local_conf and self.global_conf:
            self.local_stats = stats[best_idx][0]
            self.global_stats = stats[best_idx][1]

    def plot_k_values(self, df_val, min_k, max_k, k_step, N_samples, title):
        """
        Create plot using scores for different k values
        """
        k_values, aurcs, _ = self.search_k_values(
            df_val, min_k, max_k, k_step, N_samples
        )
        plt.plot(k_values, aurcs)
        plt.ylabel("AURC x 10\u00b3")
        plt.xlabel("k")
        plt.title(title)
        plt.show()

    def search_k_values(self, df_val, min_k, max_k, k_step, N_samples):
        """
        Returns AURC scores for given range of k values

        Parameters
        ----------
        df_val : pandas.DataFrame
            DataFrame containing validation data with following columns: 'label', 'model_pred', and 'embedding'
        min_k : int
            Value of k to start search from
        max_k : int
            Maximum value of k
        k_step : int
            Value to increase k by after each iteration
        N_samples : int
            Number of samples to use from the validation data

        Returns
        -------
        k_values : numpy.ndarray
            Range of k values searched
        aurcs : list of float
            AURC scores
        stats :  list containg means and stddevs of local and global confidence scores
        """
        if len(df_val) > N_samples:
            _, df_val = train_test_split(
                df_val, test_size=N_samples, random_state=0, stratify=df_val["label"]
            )
        val_preds = df_val["model_pred"].values.astype(int)
        df_val_residuals = (df_val["label"] != df_val["model_pred"]).values.astype(int)
        val_embs = np.vstack(df_val["embedding"].values)

        if self.reduce_local_dim:
            local_val_embs = self.pca.transform(val_embs)
            distances, indexes = self.get_nn_dists(local_val_embs, max_k)
        else:
            distances, indexes = self.get_nn_dists(val_embs, max_k)

        if self.global_conf:
            mahal_scores = self.global_predict(val_embs)

        print("Searching k values")
        k_values = np.arange(min_k, max_k + 1, k_step)
        aurcs = []
        stats = []
        for k in tqdm(k_values):
            local_confs = []
            global_confs = []
            for i, pred in enumerate(val_preds):
                if self.local_conf:
                    pred_dist = np.mean(distances[pred][i])
                    other_dists = []
                    for c in np.delete(self.classes, pred):
                        other_dists.append(distances[c][i])
                    other_dist = np.mean(np.sort(np.concatenate(other_dists))[:k])
                    local_confs.append(other_dist / pred_dist)
                if self.global_conf:
                    m_pred = mahal_scores[i].flatten()[pred]
                    m_other = np.delete(mahal_scores[i].flatten(), pred).min()
                    global_confs.append(m_other / m_pred)
                    # global_confs.append(m_other - m_pred)
                    # global_confs.append(-1 * m_pred)
                    # global_confs.append(1/m_pred)

            if self.local_conf and self.global_conf:
                local_confs = np.array(local_confs)
                # local_stats = [np.min(local_confs), np.max(local_confs)]
                # local_confs = (local_confs - local_stats[0]) / (
                #     local_stats[1] - local_stats[0]
                # )
                local_stats = [np.mean(local_confs), np.std(local_confs)]
                local_confs = (local_confs - local_stats[0]) / local_stats[1]

                global_confs = np.array(global_confs)
                # global_stats = [np.min(global_confs), np.max(global_confs)]
                # global_confs = (global_confs - global_stats[0]) / (
                #     global_stats[1] - global_stats[0]
                # )
                global_stats = [np.mean(global_confs), np.std(global_confs)]
                global_confs = (global_confs - global_stats[0]) / global_stats[1]

                scores = local_confs + global_confs
                stats.append((local_stats, global_stats))
            elif self.local_conf:
                scores = np.array(local_confs)
            else:
                scores = np.array(global_confs)

            _, aurc, _ = RC_curve(df_val_residuals, scores)
            aurcs.append(aurc * 1000)

        return k_values, aurcs, stats

    def get_nn_dists(self, test_embs, k):
        """
        Returns the nearest neighbor distances and nearest neighbor training labels

        Parameters
        ----------
        test_embs : numpy.ndarray
            Test embeddings
        k : int
            Number of nearest neighbors

        Returns
        -------
        distances : dict
            Dictionary containing nearest neighbor distances for each class
        indexes : dict
            Dictionary containing indexes of nearest neighbors
        """

        distances = {}
        indexes = {}
        for label in self.classes:
            df_label = self.df_train.loc[self.df_train["label"] == label]
            og_idx = df_label.index.values
            nbrs = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric=self.local_distance_metric
            ).fit(self.local_train_embs[og_idx])
            dist, new_idx = nbrs.kneighbors(test_embs)
            idx = og_idx[new_idx.flatten()]
            idx = idx.reshape(new_idx.shape)
            distances[label] = dist
            indexes[label] = idx
        return distances, indexes

    def compute_conf(self, df_test):
        """
        Computes local and/or global confidence scores

        Parameters
        ----------
        df_test : pandas.DataFrame
            DataFrame containing test data with following columns: 'model_pred', 'embedding'

        Returns
        -------
        local_confs : numpy.ndarray
            Array containing local confidence scores
        global_confs : numpy.ndarray
            Array containing global confidence scores
        """

        test_embs = np.vstack(df_test["embedding"].values)
        test_preds = df_test["model_pred"].values.astype(int)

        if self.local_conf:
            if self.reduce_local_dim:
                local_test_embs = self.pca.transform(test_embs)
                distances, indexes = self.get_nn_dists(local_test_embs, self.k)
            else:
                distances, indexes = self.get_nn_dists(test_embs, self.k)

        if self.global_conf:
            mahal_scores = self.global_predict(test_embs)

        local_confs = []
        global_confs = []
        for i, pred in enumerate(test_preds):
            if self.local_conf:
                pred_dist = np.mean(distances[pred][i])
                other_dists = []
                for c in np.delete(self.classes, pred):
                    other_dists.append(distances[c][i])
                other_dist = np.mean(np.sort(np.concatenate(other_dists))[: self.k])
                local_confs.append(other_dist / pred_dist)
            if self.global_conf:
                m_pred = mahal_scores[i].flatten()[pred]
                m_other = np.delete(mahal_scores[i].flatten(), pred).min()
                global_confs.append(m_other / m_pred)
                # global_confs.append(m_other - m_pred)
                # global_confs.append(-1 * m_pred)
                # global_confs.append(1/m_pred)

        if self.local_conf and self.global_conf:
            local_confs = np.array(local_confs)
            # local_confs = (local_confs - self.local_stats[0]) / (
            #     self.local_stats[1] - self.local_stats[0]
            # )
            local_confs = (local_confs - self.local_stats[0]) / self.local_stats[1]
            global_confs = np.array(global_confs)
            # global_confs = (global_confs - self.global_stats[0]) / (
            #     self.global_stats[1] - self.global_stats[0]
            # )
            global_confs = (global_confs - self.global_stats[0]) / self.global_stats[1]
            return local_confs, global_confs
        elif self.local_conf:
            return np.array(local_confs)
        else:
            return np.array(global_confs)

    def global_predict(self, test_embs):
        if self.reduce_global_dim:
            test_embs = self.pca.transform(
                test_embs,
            )
        if self.global_norm:
            test_embs = normalize(test_embs, norm="l2", axis=1)

        preds = []
        start_idx = 0
        end_idx = self.global_batch_size
        for start_idx in tqdm(range(0, test_embs.shape[0], self.global_batch_size)):
            end_idx = min(test_embs.shape[0], start_idx + self.global_batch_size)
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
        if not self.rmd:
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
