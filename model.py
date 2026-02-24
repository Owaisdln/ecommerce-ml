import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import pickle

class SVDModel:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.reconstructed_matrix = None

    def train(self, df):
        # 80/20 split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Create user-item matrix from train only
        self.user_item_matrix = train_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating'
        ).fillna(0)

        self.user_ids = self.user_item_matrix.index
        self.item_ids = self.user_item_matrix.columns

        # Fit SVD
        reduced_matrix = self.svd.fit_transform(self.user_item_matrix)

        # Reconstruct matrix
        self.reconstructed_matrix = np.dot(reduced_matrix, self.svd.components_)

        return train_df, test_df

    def predict(self, user_id, product_id):
        if user_id in self.user_ids and product_id in self.item_ids:
            user_idx = self.user_ids.get_loc(user_id)
            item_idx = self.item_ids.get_loc(product_id)
            return self.reconstructed_matrix[user_idx, item_idx]
        return 0

    def save_model(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path="model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)