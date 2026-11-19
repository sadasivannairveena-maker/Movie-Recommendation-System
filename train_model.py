"""Train and save all movie recommendation models.

Run this file from the project root:
python train_model.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.collaborative_model import CollaborativeRecommender
from src.content_model import ContentBasedRecommender
from src.data_processing import preprocess_data, setup_logging
from src.evaluation import evaluate_collaborative_model
from src.hybrid_model import HybridRecommender

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

CONTENT_MODEL_PATH = MODELS_DIR / "content_model.pkl"
COLLABORATIVE_MODEL_PATH = MODELS_DIR / "collaborative_model.pkl"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.pkl"
EVALUATION_PATH = MODELS_DIR / "evaluation_results.json"
EDA_PATH = MODELS_DIR / "eda_summary.json"

logger = logging.getLogger(__name__)


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main() -> None:
    """Run the full machine learning training pipeline."""
    setup_logging()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Movie Recommendation System training pipeline")

    # 1. Data Collection + Data Cleaning + Feature Engineering
    movies, ratings, eda_summary = preprocess_data(DATA_DIR)
    save_json(eda_summary, EDA_PATH)
    logger.info("EDA summary saved to %s", EDA_PATH)

    # 2. Train content-based filtering model
    content_model = ContentBasedRecommender()
    content_model.fit(movies)
    content_model.save(CONTENT_MODEL_PATH)

    # 3. Train collaborative filtering model
    collaborative_model = CollaborativeRecommender(
        n_neighbors=20,
        n_components=50,
        random_state=42,
    )
    collaborative_model.fit(movies, ratings)
    collaborative_model.save(COLLABORATIVE_MODEL_PATH)

    # 4. Train/save hybrid model
    hybrid_model = HybridRecommender(
        content_model=content_model,
        collaborative_model=collaborative_model,
        content_weight=0.5,
        collaborative_weight=0.5,
    )
    hybrid_model.save(HYBRID_MODEL_PATH)

    # 5. Model Evaluation
    try:
        metrics = evaluate_collaborative_model(movies, ratings, k=10)
        save_json(metrics, EVALUATION_PATH)
        logger.info("Evaluation metrics saved to %s", EVALUATION_PATH)
    except Exception as exc:  # Keep training successful even if evaluation fails.
        logger.exception("Evaluation failed: %s", exc)

    # 6. Recommendation Generation Example
    logger.info("Training completed successfully!")
    logger.info("Saved models in %s", MODELS_DIR)

    if "Toy Story (1995)" in movies["title"].values:
        example_recs = content_model.recommend("Toy Story", top_n=5)
        logger.info("Example recommendations for Toy Story: %s", example_recs)


if __name__ == "__main__":
    main()
