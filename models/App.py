from base.base_model import BaseModel
from scipy.io import loadmat
from itertools import combinations_with_replacement
import faiss
import os
import pickle
import numpy as np
from utils_regression import RegressionMatrix


class App(BaseModel):
    def __init__(self, model_path):
        super().__init__("A++", model_path)
        self.model_data_folder = f"models/{self.name}"

    def download_model(self):
        # Download from GitHub
        URL = "https://github.com/EthanLinYitun/A_Plus_Plus_Spectral_Reconstruction/blob/main/train/trained_models/model_a_plus_plus_retrain.pkl"
        pass

    def get_model(self):
        self.knn_model, self.RegMat_a_plus_plus = self.load_a_plus_plus_model()

        # No need to load the model
        pass

    def unload_model(self):
        # Unload the model from memory
        pass

    def predict(self, data):
        return self.recover_a_plus_plus(data, self.knn_model, self.RegMat_a_plus_plus)

    def load_pr_rels(self):
        path_regmat_pr_rels = f"{self.model_data_folder}/model_pr_rels.pkl"
        with open(path_regmat_pr_rels, "rb") as handle:
            return pickle.load(handle)

    def load_a_plus_plus_model(self):

        # load K-SVD dictionary and normalize
        spec_rec_anchors_norm = normc(
            loadmat(f"{self.model_data_folder}/dictionary_a_plus_plus.mat")["anchors"].T
        )

        # setup knn
        knn_model = FaissKNeighbors(k=1)
        knn_model.fit(
            spec_rec_anchors_norm, np.arange(0, spec_rec_anchors_norm.shape[0], 1)
        )

        # load trained A++ local regression maps
        if os.path.exists(f"{self.model_data_folder}/model_a_plus_plus_retrain.pkl"):
            print(f"Loading {self.model_data_folder}/model_a_plus_plus_retrain.pkl")
        else:
            print(
                f"File {self.model_data_folder}/model_a_plus_plus_retrain.pkl does not exist"
            )
        path_regmat_a_plus_plus = (
            f"{self.model_data_folder}/model_a_plus_plus_retrain.pkl"
        )
        with open(path_regmat_a_plus_plus, "rb") as handle:
            RegMat_a_plus_plus = pickle.load(handle)

        return knn_model, RegMat_a_plus_plus

    def recover_a_plus_plus(self, rgb, knn_model, RegMat_a_plus_plus):
        # transform RGB to primary estimate and normalize
        pr_rels_model = self.load_pr_rels()

        primary_spec_rec = recover_pr_rels(rgb, pr_rels_model)
        dim, height, width = primary_spec_rec.shape
        primary_spec_rec = primary_spec_rec.reshape(31, -1).T
        rgb = rgb.reshape(3, -1).T

        # normalize the primary spectral estimates
        primary_spec_rec_norm = normc(primary_spec_rec)

        # kNN with adjustable batch size
        nearest_cluster_center = knn_model.predict(primary_spec_rec_norm)
        active_cluster_centers = np.unique(nearest_cluster_center).astype(int)

        # apply local linear maps
        recovery = np.zeros(primary_spec_rec_norm.shape)
        for i in active_cluster_centers:
            is_nearest = nearest_cluster_center == i
            rgb_nearest = rgb[is_nearest, :]

            recovery_part = rgb_nearest @ RegMat_a_plus_plus[i].get_matrix()
            recovery[is_nearest, :] = recovery_part
        return recovery.T.reshape(height, width, 31)


def get_polynomial_terms(num_of_var, highest_order, root):
    if highest_order == 1:
        all_set = np.eye(num_of_var)
        # final_set = [(1,0,0),(0,1,0),(0,0,1)]
        final_set = [tuple(all_set[i, :]) for i in range(num_of_var)]

        return final_set

    final_set = set()  # save the set of polynomial terms
    index_of_variables = [i for i in range(num_of_var)]

    for order in range(
        1, highest_order + 1
    ):  # consider all higher order terms from order 1, excluding the constant term

        # Each list member: one composition of the term of the assigned order, in terms of variable indices
        curr_polynomial_terms = list(
            combinations_with_replacement(index_of_variables, order)
        )

        for t in range(len(curr_polynomial_terms)):
            curr_term = curr_polynomial_terms[t]
            mapped_term = np.zeros(num_of_var)  # save the index value of each variables

            for var in curr_term:
                if root:
                    mapped_term[var] += 1.0 / order
                else:
                    mapped_term[var] += 1.0

            final_set.add(tuple(mapped_term))

    return list(sorted(final_set))


def rgb2poly(rgb_data, poly_order, root):
    dim_data, dim_variables = rgb_data.shape
    poly_term = get_polynomial_terms(dim_variables, poly_order, root)
    dim_poly = len(poly_term)

    out_mat = np.empty((dim_data, dim_poly))

    for term in range(dim_poly):
        new_col = np.ones((dim_data))  # DIM_DATA,
        for var in range(dim_variables):
            variable_vector = rgb_data[:, var]  # DIM_DATA,
            variable_index_value = poly_term[term][var]
            new_col = new_col * (variable_vector**variable_index_value)

        out_mat[:, term] = new_col

    return out_mat


def recover_pr_rels(rgb, RegMat_pr_rels):
    # get the polynomial expansion of RGB
    height, width, dim = rgb.shape
    rgb = rgb.reshape(3, -1).T
    poly_rgb = rgb2poly(rgb, 6, root=False)

    # apply regression matrix to polynomial RGB expansions to recover spectra
    recovery = poly_rgb @ RegMat_pr_rels.get_matrix()

    return recovery.T.reshape(31, height, width)


class FaissKNeighbors:
    """
    "Make kNN 300 times faster than Scikit-learn’s in 20 lines! -
    Using Facebook faiss library for REALLY fast kNN" - by Jakub Adamczyk

    url: https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
    """

    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


def normc(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)
