"""Hybrid movie recommendation model.

The hybrid model combines content-based scores and collaborative scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import joblib

from src.collaborative_model import CollaborativeRecommender
from src.content_model import ContentBasedRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Hybrid recommender using weighted content and collaborative scores."""

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collaborative_model: CollaborativeRecommender,
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5,
    ) -> None:
        """Initialize the hybrid recommender.

        Args:
            content_model: Trained content-based recommender.
            collaborative_model: Trained collaborative recommender.
            content_weight: Weight for content-based similarity.
            collaborative_weight: Weight for collaborative score.
        """
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

    def recommend(
        self,
        user_id: int,
        movie_title: str,
        top_n: int = 10,
        content_weight: float | None = None,
        collaborative_weight: float | None = None,
    ) -> List[Dict[str, object]]:
        """Generate hybrid recommendations.

        Final score formula:
        final_score = content_score * content_weight + collaborative_score * collaborative_weight
        """
        content_weight = self.content_weight if content_weight is None else content_weight
        collaborative_weight = (
            self.collaborative_weight
            if collaborative_weight is None
            else collaborative_weight
        )

        total_weight = content_weight + collaborative_weight
        if total_weight <= 0:
            raise ValueError("At least one hybrid weight must be greater than 0.")

        # Normalize weights so users can pass values like 70 and 30 too.
        content_weight = content_weight / total_weight
        collaborative_weight = collaborative_weight / total_weight

        content_scores = self.content_model.get_similarity_scores(movie_title)
        collaborative_scores = self.collaborative_model.get_collaborative_scores(user_id)

        selected_movie_index = self.content_model.find_movie_index(movie_title)
        assert self.content_model.movies is not None
        selected_movie_id = int(
            self.content_model.movies.iloc[selected_movie_index]["movieId"]
        )

        seen_movie_ids = self.collaborative_model.get_seen_movie_ids(user_id)
        all_movie_ids = set(content_scores).intersection(collaborative_scores)

        recommendations: List[Dict[str, object]] = []
        for movie_id in all_movie_ids:
            if movie_id == selected_movie_id or movie_id in seen_movie_ids:
                continue

            content_score = float(content_scores.get(movie_id, 0.0))
            collaborative_score = float(collaborative_scores.get(movie_id, 0.0))
            final_score = (
                content_score * content_weight
                + collaborative_score * collaborative_weight
            )

            metadata = self.collaborative_model.movie_metadata.get(movie_id, {})
            recommendations.append(
                {
                    "movieId": int(movie_id),
                    "title": metadata.get("title", "Unknown Title"),
                    "genres": metadata.get("genres", ""),
                    "content_score": round(content_score, 4),
                    "collaborative_score": round(collaborative_score, 4),
                    "final_score": round(final_score, 4),
                }
            )

        recommendations = sorted(
            recommendations,
            key=lambda item: item["final_score"],
            reverse=True,
        )
        return recommendations[:top_n]

    def save(self, model_path: str | Path) -> None:
        """Save the hybrid model."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)
        logger.info("Saved hybrid model to %s", model_path)

    @staticmethod
    def load(model_path: str | Path) -> "HybridRecommender":
        """Load a trained hybrid model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Hybrid model not found: {model_path}")
        return joblib.load(model_path)
