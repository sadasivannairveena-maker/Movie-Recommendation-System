"""SQLite database utilities for storing user ratings."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "user_ratings.db"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create a SQLite database connection."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_database(db_path: str | Path | None = None) -> None:
    """Initialize the ratings table if it does not exist."""
    with get_connection(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ratings_user_id
            ON ratings(user_id)
            """
        )
        connection.commit()
    logger.info("SQLite database initialized")


def add_rating(
    user_id: int,
    movie_id: int,
    rating: float,
    db_path: str | Path | None = None,
) -> int:
    """Insert a user rating into SQLite.

    Args:
        user_id: User ID.
        movie_id: Movie ID.
        rating: Rating from 0.5 to 5.0.
        db_path: Optional custom database path.

    Returns:
        The inserted row ID.
    """
    if rating < 0.5 or rating > 5.0:
        raise ValueError("Rating must be between 0.5 and 5.0")

    init_database(db_path)
    with get_connection(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO ratings (user_id, movie_id, rating)
            VALUES (?, ?, ?)
            """,
            (int(user_id), int(movie_id), float(rating)),
        )
        connection.commit()
        row_id = int(cursor.lastrowid)

    logger.info(
        "Stored rating: user_id=%s, movie_id=%s, rating=%s",
        user_id,
        movie_id,
        rating,
    )
    return row_id


def get_user_ratings(
    user_id: int,
    db_path: str | Path | None = None,
) -> List[dict]:
    """Return ratings submitted by one user."""
    init_database(db_path)
    with get_connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, user_id, movie_id, rating, created_at
            FROM ratings
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (int(user_id),),
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_all_ratings(db_path: str | Path | None = None) -> List[dict]:
    """Return all ratings stored in SQLite."""
    init_database(db_path)
    with get_connection(db_path) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, user_id, movie_id, rating, created_at
            FROM ratings
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]
