"""Collaborative filtering movie recommendation model.

This module implements:
1. User-item rating matrix
2. KNN-based collaborative filtering
3. TruncatedSVD matrix factorization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """Collaborative filtering recommender using ratings behavior."""

    def __init__(
        self,
        n_neighbors: int = 20,
        n_components: int = 50,
        random_state: int = 42,
    ) -> None:
        """Initialize the collaborative recommender.

        Args:
            n_neighbors: Number of similar users for KNN.
            n_components: Number of latent factors for TruncatedSVD.
            random_state: Random seed for reproducibility.
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.random_state = random_state

        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.user_item_matrix: csr_matrix | None = None
        self.knn_model: NearestNeighbors | None = None
        self.svd_model: TruncatedSVD | None = None
        self.reconstructed_matrix: np.ndarray | None = None

        self.user_ids: np.ndarray | None = None
        self.movie_ids: np.ndarray | None = None
        self.user_to_index: Dict[int, int] = {}
        self.movie_to_index: Dict[int, int] = {}
        self.index_to_movie: Dict[int, int] = {}
        self.movie_metadata: Dict[int, Dict[str, object]] = {}
        self.movie_mean_ratings: Dict[int, float] = {}
        self.global_mean_rating: float = 3.0

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> "CollaborativeRecommender":
        """Train collaborative filtering models.

        Args:
            movies: Cleaned movies dataframe.
            ratings: Cleaned ratings dataframe.

        Returns:
            The fitted model instance.
        """
        logger.info("Training collaborative recommender")
        self.movies = movies.reset_index(drop=True).copy()
        self.ratings = ratings.copy()

        self.ratings = self.ratings[
            self.ratings["movieId"].isin(self.movies["movieId"])
        ].reset_index(drop=True)

        self.user_ids = np.sort(self.ratings["userId"].unique())
        self.movie_ids = np.sort(self.movies["movieId"].unique())

        self.user_to_index = {
            int(user_id): index for index, user_id in enumerate(self.user_ids)
        }
        self.movie_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
        }
        self.index_to_movie = {index: movie_id for movie_id, index in self.movie_to_index.items()}

        self.movie_metadata = {
            int(row.movieId): {"title": row.title, "genres": row.genres}
            for row in self.movies.itertuples(index=False)
        }

        self.global_mean_rating = float(self.ratings["rating"].mean())
        self.movie_mean_ratings = (
            self.ratings.groupby("movieId")["rating"].mean().astype(float).to_dict()
        )

        rows = self.ratings["userId"].map(self.user_to_index).to_numpy()
        cols = self.ratings["movieId"].map(self.movie_to_index).to_numpy()
        values = self.ratings["rating"].astype(float).to_numpy()

        self.user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        self._fit_knn()
        self._fit_svd()

        logger.info(
            "Collaborative model trained with %d users and %d movies",
            len(self.user_ids),
            len(self.movie_ids),
        )
        return self

    def _fit_knn(self) -> None:
        """Fit the user-user KNN model."""
        assert self.user_item_matrix is not None
        n_users = self.user_item_matrix.shape[0]
        n_neighbors = min(self.n_neighbors, n_users)

        self.knn_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=n_neighbors,
        )
        self.knn_model.fit(self.user_item_matrix)
        logger.info("KNN collaborative model fitted")

    def _fit_svd(self) -> None:
        """Fit the TruncatedSVD matrix factorization model."""
        assert self.user_item_matrix is not None
        n_users, n_movies = self.user_item_matrix.shape
        max_components = max(1, min(n_users, n_movies) - 1)
        n_components = min(self.n_components, max_components)

        self.svd_model = TruncatedSVD(
            n_components=n_components,
            random_state=self.random_state,
        )
        user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        item_factors = self.svd_model.components_
        self.reconstructed_matrix = np.dot(user_factors, item_factors)
        logger.info("TruncatedSVD collaborative model fitted")

    def _check_is_fitted(self) -> None:
        """Raise an error if the model is not trained."""
        if (
            self.movies is None
            or self.ratings is None
            or self.user_item_matrix is None
            or self.knn_model is None
            or self.reconstructed_matrix is None
        ):
            raise RuntimeError("Collaborative model is not fitted. Run fit() first.")

    def get_seen_movie_ids(self, user_id: int) -> Set[int]:
        """Return movies already rated by the given user."""
        self._check_is_fitted()
        assert self.ratings is not None

        if int(user_id) not in self.user_to_index:
            return set()

        seen = self.ratings[self.ratings["userId"] == int(user_id)]["movieId"].unique()
        return {int(movie_id) for movie_id in seen}

    def _format_recommendations(
        self, movie_indices: np.ndarray, scores: np.ndarray, top_n: int
    ) -> List[Dict[str, object]]:
        """Convert movie indexes and scores into API-friendly dictionaries."""
        recommendations: List[Dict[str, object]] = []

        for movie_index in movie_indices:
            movie_id = int(self.index_to_movie[int(movie_index)])
            metadata = self.movie_metadata.get(movie_id, {})
            recommendations.append(
                {
                    "movieId": movie_id,
                    "title": metadata.get("title", "Unknown Title"),
                    "genres": metadata.get("genres", ""),
                    "score": round(float(scores[movie_index]), 4),
                }
            )
            if len(recommendations) >= top_n:
                break

        return recommendations

    def _popular_recommendations(
        self, top_n: int = 10, exclude_movie_ids: Set[int] | None = None
    ) -> List[Dict[str, object]]:
        """Recommend popular movies as a fallback for unknown users."""
        self._check_is_fitted()
        assert self.ratings is not None

        exclude_movie_ids = exclude_movie_ids or set()
        stats = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        min_votes = max(5, int(stats["count"].quantile(0.60)))
        global_mean = float(self.ratings["rating"].mean())

        # Weighted rating avoids recommending a movie with one 5-star rating only.
        stats["weighted_score"] = (
            (stats["count"] / (stats["count"] + min_votes)) * stats["mean"]
            + (min_votes / (stats["count"] + min_votes)) * global_mean
        )
        stats = stats.drop(index=list(exclude_movie_ids), errors="ignore")
        stats = stats.sort_values("weighted_score", ascending=False).head(top_n)

        recommendations: List[Dict[str, object]] = []
        for movie_id, row in stats.iterrows():
            metadata = self.movie_metadata.get(int(movie_id), {})
            recommendations.append(
                {
                    "movieId": int(movie_id),
                    "title": metadata.get("title", "Unknown Title"),
                    "genres": metadata.get("genres", ""),
                    "score": round(float(row["weighted_score"]), 4),
                }
            )
        return recommendations

    def recommend_for_user(
        self, user_id: int, top_n: int = 10, method: str = "svd"
    ) -> List[Dict[str, object]]:
        """Recommend movies for a user.

        Args:
            user_id: MovieLens user ID.
            top_n: Number of recommendations.
            method: "svd" for matrix factorization or "knn" for user-user KNN.

        Returns:
            List of recommended movies.
        """
        self._check_is_fitted()
        method = method.lower().strip()

        if method == "knn":
            return self._recommend_knn(user_id=user_id, top_n=top_n)
        return self._recommend_svd(user_id=user_id, top_n=top_n)

    def _recommend_svd(self, user_id: int, top_n: int = 10) -> List[Dict[str, object]]:
        """Recommend movies using TruncatedSVD predictions."""
        self._check_is_fitted()
        assert self.reconstructed_matrix is not None

        user_id = int(user_id)
        if user_id not in self.user_to_index:
            logger.warning("Unknown userId %s. Returning popular recommendations.", user_id)
            return self._popular_recommendations(top_n=top_n)

        user_index = self.user_to_index[user_id]
        scores = self.reconstructed_matrix[user_index].copy()

        # Do not recommend movies the user has already rated.
        seen_movie_ids = self.get_seen_movie_ids(user_id)
        for movie_id in seen_movie_ids:
            movie_index = self.movie_to_index.get(movie_id)
            if movie_index is not None:
                scores[movie_index] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]
        return self._format_recommendations(top_indices, scores, top_n)

    def _recommend_knn(self, user_id: int, top_n: int = 10) -> List[Dict[str, object]]:
        """Recommend movies using ratings from similar users."""
        self._check_is_fitted()
        assert self.user_item_matrix is not None
        assert self.knn_model is not None

        user_id = int(user_id)
        if user_id not in self.user_to_index:
            logger.warning("Unknown userId %s. Returning popular recommendations.", user_id)
            return self._popular_recommendations(top_n=top_n)

        user_index = self.user_to_index[user_id]
        user_vector = self.user_item_matrix[user_index]
        n_neighbors = min(self.n_neighbors, self.user_item_matrix.shape[0])
        distances, neighbor_indices = self.knn_model.kneighbors(
            user_vector, n_neighbors=n_neighbors
        )

        distances = distances.flatten()
        neighbor_indices = neighbor_indices.flatten()
        similarities = 1 - distances

        # Remove the target user from neighbors if present.
        mask = neighbor_indices != user_index
        neighbor_indices = neighbor_indices[mask]
        similarities = similarities[mask]

        if len(neighbor_indices) == 0 or np.sum(similarities) == 0:
            return self._recommend_svd(user_id=user_id, top_n=top_n)

        neighbor_ratings = self.user_item_matrix[neighbor_indices].toarray()
        rated_mask = (neighbor_ratings > 0).astype(float)

        weighted_sum = np.dot(similarities, neighbor_ratings)
        similarity_sum = np.dot(similarities, rated_mask)
        scores = np.divide(
            weighted_sum,
            similarity_sum,
            out=np.zeros_like(weighted_sum, dtype=float),
            where=similarity_sum != 0,
        )

        # Fill unrated/zero KNN scores with SVD scores for a better fallback.
        assert self.reconstructed_matrix is not None
        svd_scores = self.reconstructed_matrix[user_index]
        scores = np.where(scores > 0, scores, svd_scores)

        seen_movie_ids = self.get_seen_movie_ids(user_id)
        for movie_id in seen_movie_ids:
            movie_index = self.movie_to_index.get(movie_id)
            if movie_index is not None:
                scores[movie_index] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]
        return self._format_recommendations(top_indices, scores, top_n)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a rating for a user/movie pair.

        Unknown users or movies fall back to movie mean or global mean.
        """
        self._check_is_fitted()
        assert self.reconstructed_matrix is not None

        user_id = int(user_id)
        movie_id = int(movie_id)

        if user_id in self.user_to_index and movie_id in self.movie_to_index:
            user_index = self.user_to_index[user_id]
            movie_index = self.movie_to_index[movie_id]
            prediction = float(self.reconstructed_matrix[user_index, movie_index])
        else:
            prediction = float(
                self.movie_mean_ratings.get(movie_id, self.global_mean_rating)
            )

        # MovieLens ratings are between 0.5 and 5.0.
        return float(np.clip(prediction, 0.5, 5.0))

    def get_collaborative_scores(self, user_id: int) -> Dict[int, float]:
        """Return normalized collaborative scores for all movies.

        Scores are normalized to the range [0, 1], which makes them easy to
        combine with cosine similarity in the hybrid model.
        """
        self._check_is_fitted()
        assert self.reconstructed_matrix is not None
        assert self.movie_ids is not None

        user_id = int(user_id)
        if user_id in self.user_to_index:
            user_index = self.user_to_index[user_id]
            scores = self.reconstructed_matrix[user_index].astype(float).copy()
        else:
            # Unknown user fallback: use movie average ratings.
            scores = np.array(
                [self.movie_mean_ratings.get(int(movie_id), self.global_mean_rating)
                 for movie_id in self.movie_ids],
                dtype=float,
            )

        min_score = float(np.nanmin(scores))
        max_score = float(np.nanmax(scores))
        if max_score - min_score == 0:
            normalized = np.zeros_like(scores, dtype=float)
        else:
            normalized = (scores - min_score) / (max_score - min_score)

        return {
            int(movie_id): float(score)
            for movie_id, score in zip(self.movie_ids, normalized)
        }

    def save(self, model_path: str | Path) -> None:
        """Save the trained collaborative model."""
        self._check_is_fitted()
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)
        logger.info("Saved collaborative model to %s", model_path)

    @staticmethod
    def load(model_path: str | Path) -> "CollaborativeRecommender":
        """Load a trained collaborative model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Collaborative model not found: {model_path}")
        return joblib.load(model_path)
