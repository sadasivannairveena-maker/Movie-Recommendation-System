"""Evaluation metrics for the recommendation system."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.collaborative_model import CollaborativeRecommender

logger = logging.getLogger(__name__)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_rating_predictions(
    model: CollaborativeRecommender,
    test_ratings: pd.DataFrame,
) -> Dict[str, float]:
    """Evaluate predicted ratings using RMSE and MAE."""
    y_true = test_ratings["rating"].astype(float).to_numpy()
    y_pred = np.array(
        [
            model.predict_rating(row.userId, row.movieId)
            for row in test_ratings.itertuples(index=False)
        ]
    )

    return {
        "rmse": round(calculate_rmse(y_true, y_pred), 4),
        "mae": round(calculate_mae(y_true, y_pred), 4),
    }


def precision_recall_at_k(
    model: CollaborativeRecommender,
    test_ratings: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> Dict[str, float]:
    """Calculate Precision@K and Recall@K.

    A movie is considered relevant if the user gave it a rating greater than
    or equal to relevance_threshold in the test set.
    """
    precisions = []
    recalls = []

    for user_id, user_test_data in test_ratings.groupby("userId"):
        relevant_movies = set(
            user_test_data[user_test_data["rating"] >= relevance_threshold]["movieId"]
        )
        if not relevant_movies:
            continue

        recommendations = model.recommend_for_user(int(user_id), top_n=k, method="svd")
        recommended_movies = {item["movieId"] for item in recommendations}

        hits = len(recommended_movies.intersection(relevant_movies))
        precisions.append(hits / k)
        recalls.append(hits / len(relevant_movies))

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0

    return {
        f"precision_at_{k}": round(precision, 4),
        f"recall_at_{k}": round(recall, 4),
    }


def train_test_split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test data."""
    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )
    return train_ratings.reset_index(drop=True), test_ratings.reset_index(drop=True)


def evaluate_collaborative_model(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    k: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """Train a temporary model and evaluate it on a test split.

    This function is used in train_model.py so the final project includes a
    complete evaluation pipeline.
    """
    logger.info("Starting collaborative model evaluation")
    train_ratings, test_ratings = train_test_split_ratings(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )

    eval_model = CollaborativeRecommender(random_state=random_state)
    eval_model.fit(movies, train_ratings)

    rating_metrics = evaluate_rating_predictions(eval_model, test_ratings)
    ranking_metrics = precision_recall_at_k(eval_model, test_ratings, k=k)

    metrics = {**rating_metrics, **ranking_metrics}
    logger.info("Evaluation metrics: %s", metrics)
    return metrics
