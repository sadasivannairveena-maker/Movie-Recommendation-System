"""Data loading, cleaning, preprocessing, and EDA utilities.

This module works with the MovieLens small dataset files:
- data/movies.csv
- data/ratings.csv
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_MOVIES_COLUMNS = {"movieId", "title", "genres"}
REQUIRED_RATINGS_COLUMNS = {"userId", "movieId", "rating", "timestamp"}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple logging format for the whole project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_columns(df: pd.DataFrame, required_columns: set[str], file_name: str) -> None:
    """Validate that a dataframe contains all required columns.

    Args:
        df: Input dataframe.
        required_columns: Columns that must exist.
        file_name: Friendly file name used in the error message.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {sorted(missing)}"
        )


def load_data(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens movies and ratings CSV files.

    Args:
        data_dir: Folder containing movies.csv and ratings.csv.

    Returns:
        Tuple of movies dataframe and ratings dataframe.

    Raises:
        FileNotFoundError: If either CSV file is not found.
        ValueError: If required columns are missing.
    """
    data_path = Path(data_dir)
    movies_path = data_path / "movies.csv"
    ratings_path = data_path / "ratings.csv"

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Could not find {movies_path}. Download MovieLens small and place "
            "movies.csv inside the data/ folder."
        )

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Could not find {ratings_path}. Download MovieLens small and place "
            "ratings.csv inside the data/ folder."
        )

    logger.info("Loading movies from %s", movies_path)
    movies = pd.read_csv(movies_path)

    logger.info("Loading ratings from %s", ratings_path)
    ratings = pd.read_csv(ratings_path)

    validate_columns(movies, REQUIRED_MOVIES_COLUMNS, "movies.csv")
    validate_columns(ratings, REQUIRED_RATINGS_COLUMNS, "ratings.csv")

    return movies, ratings


def clean_title(title: object) -> str:
    """Clean a movie title by removing extra spaces."""
    if pd.isna(title):
        return "Unknown Title"
    title = str(title).strip()
    return re.sub(r"\s+", " ", title)


def extract_year(title: str) -> float:
    """Extract release year from a title like 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    if match:
        return float(match.group(1))
    return np.nan


def remove_year_from_title(title: str) -> str:
    """Remove the ending year from a MovieLens title."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean the movies dataframe.

    Cleaning steps:
    - remove missing movie IDs/titles
    - remove duplicates
    - normalize titles
    - create year, title_clean, title_no_year, genres_clean, genres_list
    """
    movies = movies.copy()

    logger.info("Cleaning movies dataframe")
    movies = movies.dropna(subset=["movieId", "title"])
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)

    movies["title"] = movies["title"].apply(clean_title)
    movies["title_no_year"] = movies["title"].apply(remove_year_from_title)
    movies["title_clean"] = movies["title_no_year"].str.lower().str.strip()
    movies["year"] = movies["title"].apply(extract_year)

    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["genres"] = movies["genres"].replace("", "(no genres listed)")

    movies["genres_list"] = movies["genres"].apply(
        lambda value: []
        if value == "(no genres listed)"
        else [genre.strip() for genre in str(value).split("|")]
    )

    movies["genres_clean"] = (
        movies["genres"]
        .replace("(no genres listed)", "")
        .str.replace("|", " ", regex=False)
        .str.lower()
        .str.strip()
    )

    movies = movies.drop_duplicates(subset=["movieId"], keep="first")
    movies = movies.drop_duplicates(subset=["title"], keep="first")
    movies = movies.sort_values("movieId").reset_index(drop=True)

    logger.info("Movies after cleaning: %d rows", len(movies))
    return movies


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Clean the ratings dataframe.

    Cleaning steps:
    - remove missing values
    - convert columns to numeric types
    - keep valid rating values from 0.5 to 5.0
    - remove duplicate user/movie ratings, keeping the latest timestamp
    - add human-readable rating_date
    """
    ratings = ratings.copy()

    logger.info("Cleaning ratings dataframe")
    for column in ["userId", "movieId", "rating", "timestamp"]:
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["timestamp"] = ratings["timestamp"].astype(int)

    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]
    ratings = ratings.sort_values("timestamp")
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")
    ratings["rating_date"] = pd.to_datetime(
        ratings["timestamp"], unit="s", errors="coerce"
    )

    ratings = ratings.reset_index(drop=True)
    logger.info("Ratings after cleaning: %d rows", len(ratings))
    return ratings


def merge_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Merge cleaned movies and ratings data on movieId."""
    merged = ratings.merge(movies, on="movieId", how="inner")
    logger.info("Merged dataframe contains %d rows", len(merged))
    return merged


def create_eda_summary(movies: pd.DataFrame, ratings: pd.DataFrame) -> Dict[str, object]:
    """Create a small EDA summary dictionary.

    The summary is useful for logging and README screenshots.
    """
    n_users = int(ratings["userId"].nunique())
    n_movies = int(movies["movieId"].nunique())
    n_ratings = int(len(ratings))
    possible_ratings = n_users * n_movies
    sparsity = 1 - (n_ratings / possible_ratings) if possible_ratings else 0

    genre_counts = (
        movies.explode("genres_list")
        .query("genres_list != ''")
        ["genres_list"]
        .value_counts()
        .head(10)
        .to_dict()
    )

    rating_distribution = (
        ratings["rating"].value_counts().sort_index().astype(int).to_dict()
    )

    summary = {
        "number_of_movies": n_movies,
        "number_of_users": n_users,
        "number_of_ratings": n_ratings,
        "average_rating": round(float(ratings["rating"].mean()), 3),
        "rating_sparsity": round(float(sparsity), 4),
        "top_10_genres": genre_counts,
        "rating_distribution": rating_distribution,
    }

    return summary


def preprocess_data(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Run the full data preprocessing pipeline.

    Args:
        data_dir: Folder containing MovieLens CSV files.

    Returns:
        Cleaned movies, cleaned ratings, and an EDA summary.
    """
    movies, ratings = load_data(data_dir)
    movies = clean_movies(movies)
    ratings = clean_ratings(ratings)

    # Keep only ratings for movies that exist in movies.csv.
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])].reset_index(drop=True)

    eda_summary = create_eda_summary(movies, ratings)
    logger.info("EDA summary: %s", eda_summary)

    return movies, ratings, eda_summary
