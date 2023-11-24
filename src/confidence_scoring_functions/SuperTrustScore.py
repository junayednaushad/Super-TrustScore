import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.getcwd()))
from utils import RC_curve
from confidence_scoring_functions.Mahalanobis import Mahalanobis

plt.style.use("seaborn-v0_8")


def get_sts_scores(
    df_train,
    df_val,
    df_test,
    reduce_dim,
    n_components,
    norm,
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
        reduce_dim=reduce_dim,
        n_components=n_components,
        norm=norm,
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
        reduce_dim=True,
        n_components=0.9,
        norm=True,
        filter_training=True,
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
        reduce_dim : bool
            Determines if embedding dimensions should be reduced
        n_components: int or float
            If n_components > 1 then it is the number of features (with highest variability) to retain.
            If 0 < n_components < 1 then the number of features required to retain that percentage of explained
            variability will be kept
        norm : bool
            Determines if normalized (unit) embeddings will be used
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
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        self.norm = norm
        self.filter = filter_training
        self.local_conf = local_conf
        self.global_conf = global_conf

        self.train_embs = np.vstack(self.df_train["embedding"].values)
        if self.reduce_dim:
            # print('Reducing embedding dimensions')
            # print('Original embedding size:', self.train_embs[0].shape[0])
            self.pca = PCA(n_components=self.n_components, random_state=0)
            self.pca.fit(self.train_embs)
            self.train_embs = self.pca.transform(self.train_embs)
            # print('Reduced embedding size:', self.train_embs[0].shape[0])

        if self.norm:
            self.train_embs = normalize(self.train_embs, norm="l2", axis=1)

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

        if self.global_conf:
            self.mahalanobis = Mahalanobis(norm, reduce_dim, n_components)
            self.mahalanobis.fit(self.df_train)

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
        k_values, aurcs = self.search_k_values(df_val, min_k, max_k, k_step, N_samples)
        plt.plot(k_values, aurcs, label="AURC")
        plt.xlabel("k")
        plt.title(title)
        plt.legend()
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
        val_preds = df_val["model_pred"].values
        df_val_residuals = (df_val["label"] != df_val["model_pred"]).values.astype(int)
        distances, indexes = self.get_nn_dists(df_val, max_k)

        print("Searching k values")
        k_values = np.arange(min_k, max_k + 1, k_step)
        aurcs = []
        stats = []
        if self.global_conf:
            mahal_scores = self.mahalanobis.predict(df_val)
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
                    # global_confs.append(-1*m_pred)
                    # global_confs.append(1/m_pred)

            if self.local_conf and self.global_conf:
                local_confs = np.array(local_confs)
                local_stats = [np.mean(local_confs), np.std(local_confs)]
                local_confs = (local_confs - local_stats[0]) / local_stats[1]

                global_confs = np.array(global_confs)
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

    def get_nn_dists(self, df_test, k):
        """
        Returns the nearest neighbor distances and nearest neighbor training labels

        Parameters
        ----------
        df_test : pandas.DataFrame
            DataFrame containing test data with following columns: 'embedding'
        k : int
            Number of nearest neighbors

        Returns
        -------
        distances : dict
            Dictionary containing nearest neighbor distances for each class
        indexes : dict
            Dictionary containing indexes of nearest neighbors
        """
        test_embs = np.vstack(df_test["embedding"].values)
        if self.reduce_dim:
            test_embs = self.pca.transform(test_embs)
        if self.norm:
            test_embs = normalize(test_embs, norm="l2", axis=1)

        distances = {}
        indexes = {}
        for label in self.classes:
            df_label = self.df_train.loc[self.df_train["label"] == label]
            og_idx = df_label.index.values
            tree_label = KDTree(self.train_embs[og_idx], leaf_size=100, metric="l2")
            dist, new_idx = tree_label.query(test_embs, k=k)
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
        distances, indexes = self.get_nn_dists(df_test, self.k)
        test_preds = df_test["model_pred"].values
        if self.global_conf:
            mahal_scores = self.mahalanobis.predict(df_test)

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
                # global_confs.append(-1*m_pred)
                # global_confs.append(1/m_pred)

        if self.local_conf and self.global_conf:
            local_confs = np.array(local_confs)
            local_confs = (local_confs - self.local_stats[0]) / self.local_stats[1]
            global_confs = np.array(global_confs)
            global_confs = (global_confs - self.global_stats[0]) / self.global_stats[1]
            return local_confs, global_confs
        elif self.local_conf:
            return np.array(local_confs)
        else:
            return np.array(global_confs)
