"""Content-based movie recommendation model.

The model recommends movies with similar genres using TF-IDF and cosine similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based recommender using movie genres as text features."""

    def __init__(self, max_features: int = 5000) -> None:
        """Initialize the recommender.

        Args:
            max_features: Maximum number of TF-IDF features.
        """
        self.max_features = max_features
        self.movies: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.title_to_index: Dict[str, int] = {}
        self.title_no_year_to_index: Dict[str, int] = {}

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """Train the content-based model.

        Args:
            movies: Cleaned movies dataframe.

        Returns:
            The fitted model instance.
        """
        required_columns = {"movieId", "title", "genres"}
        missing = required_columns.difference(movies.columns)
        if missing:
            raise ValueError(f"Movies dataframe missing columns: {sorted(missing)}")

        logger.info("Training content-based recommender")
        self.movies = movies.reset_index(drop=True).copy()

        if "genres_clean" not in self.movies.columns:
            self.movies["genres_clean"] = (
                self.movies["genres"]
                .fillna("")
                .str.replace("|", " ", regex=False)
                .str.lower()
            )

        if "title_no_year" not in self.movies.columns:
            self.movies["title_no_year"] = self.movies["title"].str.replace(
                r"\s*\(\d{4}\)\s*$", "", regex=True
            )

        feature_text = self.movies["genres_clean"].fillna("")

        # TF-IDF turns genre text such as "adventure animation children" into vectors.
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(feature_text)

        self.title_to_index = {
            str(title).lower(): index for index, title in enumerate(self.movies["title"])
        }
        self.title_no_year_to_index = {
            str(title).lower(): index
            for index, title in enumerate(self.movies["title_no_year"])
        }

        logger.info("Content model trained on %d movies", len(self.movies))
        return self

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been trained."""
        if self.movies is None or self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted. Run fit() first.")

    def find_movie_index(self, movie_title: str) -> int:
        """Find the best matching movie index from a title.

        The method supports both full titles like "Toy Story (1995)" and short
        titles like "Toy Story".
        """
        self._check_is_fitted()
        assert self.movies is not None

        query = str(movie_title).lower().strip()
        if query in self.title_to_index:
            return self.title_to_index[query]
        if query in self.title_no_year_to_index:
            return self.title_no_year_to_index[query]

        # Beginner-friendly fallback: partial matching.
        matches = self.movies[
            self.movies["title"].str.lower().str.contains(query, regex=False)
            | self.movies["title_no_year"].str.lower().str.contains(query, regex=False)
        ]
        if not matches.empty:
            return int(matches.index[0])

        raise ValueError(f"Movie title not found: {movie_title}")

    def recommend(self, movie_title: str, top_n: int = 10) -> List[Dict[str, object]]:
        """Recommend movies similar to a selected movie.

        Args:
            movie_title: Movie title selected by the user.
            top_n: Number of recommendations to return.

        Returns:
            List of recommendation dictionaries.
        """
        self._check_is_fitted()
        assert self.movies is not None
        assert self.tfidf_matrix is not None

        movie_index = self.find_movie_index(movie_title)
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[movie_index], self.tfidf_matrix
        ).flatten()

        # Sort movies by similarity, highest first.
        similar_indices = np.argsort(similarity_scores)[::-1]

        recommendations: List[Dict[str, object]] = []
        seen_movie_ids = set()

        for index in similar_indices:
            if index == movie_index:
                continue

            row = self.movies.iloc[index]
            movie_id = int(row["movieId"])
            if movie_id in seen_movie_ids:
                continue

            recommendations.append(
                {
                    "movieId": movie_id,
                    "title": row["title"],
                    "genres": row["genres"],
                    "score": round(float(similarity_scores[index]), 4),
                }
            )
            seen_movie_ids.add(movie_id)

            if len(recommendations) >= top_n:
                break

        return recommendations

    def get_similarity_scores(self, movie_title: str) -> Dict[int, float]:
        """Return content similarity scores for all movies.

        Args:
            movie_title: Movie used as the content seed.

        Returns:
            Mapping from movieId to cosine similarity score.
        """
        self._check_is_fitted()
        assert self.movies is not None
        assert self.tfidf_matrix is not None

        movie_index = self.find_movie_index(movie_title)
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[movie_index], self.tfidf_matrix
        ).flatten()

        return {
            int(movie_id): float(score)
            for movie_id, score in zip(self.movies["movieId"], similarity_scores)
        }

    def save(self, model_path: str | Path) -> None:
        """Save the trained model to disk."""
        self._check_is_fitted()
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)
        logger.info("Saved content model to %s", model_path)

    @staticmethod
    def load(model_path: str | Path) -> "ContentBasedRecommender":
        """Load a trained content-based model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Content model not found: {model_path}")
        return joblib.load(model_path)
