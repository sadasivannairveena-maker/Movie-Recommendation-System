"""FastAPI backend for the Movie Recommendation System.

Run with:
uvicorn api.app:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.content_model import ContentBasedRecommender
from src.collaborative_model import CollaborativeRecommender
from src.database import add_rating, init_database
from src.hybrid_model import HybridRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

CONTENT_MODEL_PATH = MODELS_DIR / "content_model.pkl"
COLLABORATIVE_MODEL_PATH = MODELS_DIR / "collaborative_model.pkl"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.pkl"

app = FastAPI(
    title="Movie Recommendation System API",
    description="Content-based, collaborative, and hybrid movie recommendations.",
    version="1.0.0",
)

content_model: Optional[ContentBasedRecommender] = None
collaborative_model: Optional[CollaborativeRecommender] = None
hybrid_model: Optional[HybridRecommender] = None


class RatingRequest(BaseModel):
    """Request body for submitting a user rating."""

    user_id: int = Field(..., gt=0, description="User ID")
    movie_id: int = Field(..., gt=0, description="Movie ID")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating from 0.5 to 5.0")


@app.on_event("startup")
def startup_event() -> None:
    """Initialize database and load trained models."""
    init_database()
    load_models()


def load_models() -> None:
    """Load all recommendation models from disk."""
    global content_model, collaborative_model, hybrid_model

    try:
        content_model = ContentBasedRecommender.load(CONTENT_MODEL_PATH)
        collaborative_model = CollaborativeRecommender.load(COLLABORATIVE_MODEL_PATH)

        if HYBRID_MODEL_PATH.exists():
            hybrid_model = HybridRecommender.load(HYBRID_MODEL_PATH)
        else:
            hybrid_model = HybridRecommender(content_model, collaborative_model)

        logger.info("All models loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("Models are not trained yet: %s", exc)
        content_model = None
        collaborative_model = None
        hybrid_model = None


def ensure_models_loaded() -> None:
    """Raise an API error if models are not available."""
    if content_model is None or collaborative_model is None or hybrid_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are not trained yet. Run 'python train_model.py' "
                "from the project root first."
            ),
        )


@app.get("/")
def root() -> dict:
    """Return a welcome message."""
    return {
        "message": "Welcome to the Movie Recommendation System API",
        "docs": "Open /docs for interactive API documentation",
    }


@app.get("/movies")
def get_movies(
    limit: int = Query(100, ge=1, le=10000),
    search: Optional[str] = Query(None, description="Optional title search text"),
) -> dict:
    """Return available movies from the trained content model."""
    ensure_models_loaded()
    assert content_model is not None
    assert content_model.movies is not None

    movies = content_model.movies[["movieId", "title", "genres"]].copy()

    if search:
        search_lower = search.lower().strip()
        movies = movies[movies["title"].str.lower().str.contains(search_lower, regex=False)]

    movies = movies.head(limit)
    return {
        "count": int(len(movies)),
        "movies": movies.to_dict(orient="records"),
    }


@app.get("/recommend/movie/{movie_title:path}")
def recommend_by_movie(
    movie_title: str,
    top_n: int = Query(10, ge=1, le=50),
) -> dict:
    """Return content-based recommendations for a movie title."""
    ensure_models_loaded()
    assert content_model is not None

    try:
        recommendations = content_model.recommend(movie_title, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "movie": movie_title,
        "recommendations": [item["title"] for item in recommendations],
        "items": recommendations,
    }


@app.get("/recommend/user/{user_id}")
def recommend_by_user(
    user_id: int,
    top_n: int = Query(10, ge=1, le=50),
    method: str = Query("svd", pattern="^(svd|knn)$"),
) -> dict:
    """Return collaborative filtering recommendations for a user."""
    ensure_models_loaded()
    assert collaborative_model is not None

    recommendations = collaborative_model.recommend_for_user(
        user_id=user_id,
        top_n=top_n,
        method=method,
    )

    return {
        "user_id": user_id,
        "method": method,
        "recommendations": [item["title"] for item in recommendations],
        "items": recommendations,
    }


@app.get("/recommend/hybrid/{user_id}/{movie_title:path}")
def recommend_hybrid(
    user_id: int,
    movie_title: str,
    top_n: int = Query(10, ge=1, le=50),
    content_weight: float = Query(0.5, ge=0.0, le=1.0),
    collaborative_weight: float = Query(0.5, ge=0.0, le=1.0),
) -> dict:
    """Return hybrid recommendations for a user and movie title."""
    ensure_models_loaded()
    assert hybrid_model is not None

    try:
        recommendations = hybrid_model.recommend(
            user_id=user_id,
            movie_title=movie_title,
            top_n=top_n,
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "user_id": user_id,
        "movie": movie_title,
        "content_weight": content_weight,
        "collaborative_weight": collaborative_weight,
        "recommendations": [item["title"] for item in recommendations],
        "items": recommendations,
    }


@app.post("/rating")
def submit_rating(request: RatingRequest) -> dict:
    """Store a user rating in SQLite."""
    try:
        row_id = add_rating(
            user_id=request.user_id,
            movie_id=request.movie_id,
            rating=request.rating,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Rating stored successfully",
        "rating_id": row_id,
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "rating": request.rating,
    }
