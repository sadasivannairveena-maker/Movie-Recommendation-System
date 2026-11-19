"""Streamlit frontend for the Movie Recommendation System.

Run with:
streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.getenv("MOVIE_API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data(ttl=300)
def fetch_movies() -> List[Dict[str, Any]]:
    """Fetch movie list from the FastAPI backend."""
    response = requests.get(f"{API_URL}/movies?limit=10000", timeout=10)
    response.raise_for_status()
    return response.json().get("movies", [])


def safe_get(endpoint: str) -> Dict[str, Any]:
    """Call a GET endpoint and return JSON data."""
    response = requests.get(f"{API_URL}{endpoint}", timeout=30)
    response.raise_for_status()
    return response.json()


def display_recommendations(items: List[Dict[str, Any]]) -> None:
    """Display recommendation results in Streamlit."""
    if not items:
        st.warning("No recommendations found.")
        return

    for rank, item in enumerate(items, start=1):
        title = item.get("title", "Unknown Title")
        genres = item.get("genres", "")
        score = item.get("final_score", item.get("score", "N/A"))
        st.write(f"**{rank}. {title}**")
        st.caption(f"Genres: {genres} | Score: {score}")


st.title("🎬 Movie Recommendation System")
st.write(
    "Get movie recommendations using content-based filtering, collaborative "
    "filtering, and a hybrid recommendation system."
)

st.sidebar.header("Backend Status")
st.sidebar.write(f"API URL: `{API_URL}`")
st.sidebar.info("Start the API first: `uvicorn api.app:app --reload`")

try:
    movies = fetch_movies()
except requests.exceptions.RequestException as exc:
    st.error(
        "Could not connect to the FastAPI backend. Make sure it is running with: "
        "uvicorn api.app:app --reload"
    )
    st.exception(exc)
    st.stop()

movie_titles = [movie["title"] for movie in movies]
movie_id_by_title = {movie["title"]: movie["movieId"] for movie in movies}

st.success(f"Loaded {len(movie_titles)} movies from the API.")

content_tab, collaborative_tab, hybrid_tab, rating_tab = st.tabs(
    [
        "Content-Based",
        "Collaborative",
        "Hybrid",
        "Submit Rating",
    ]
)

with content_tab:
    st.subheader("Content-Based Recommendations")
    st.write("Choose a movie and get similar movies based on genres.")

    selected_movie = st.selectbox(
        "Select a movie",
        movie_titles,
        key="content_movie",
    )
    top_n_content = st.slider("Number of recommendations", 1, 20, 10, key="top_n_content")

    if st.button("Generate Content-Based Recommendations"):
        try:
            encoded_title = quote(selected_movie, safe="")
            data = safe_get(f"/recommend/movie/{encoded_title}?top_n={top_n_content}")
            display_recommendations(data.get("items", []))
        except requests.exceptions.RequestException as exc:
            st.error("Failed to generate content-based recommendations.")
            st.exception(exc)

with collaborative_tab:
    st.subheader("Collaborative Filtering Recommendations")
    st.write("Enter a MovieLens user ID and get recommendations based on rating behavior.")

    user_id = st.number_input("User ID", min_value=1, value=1, step=1, key="collab_user")
    method = st.radio("Collaborative method", ["svd", "knn"], horizontal=True)
    top_n_collab = st.slider("Number of recommendations", 1, 20, 10, key="top_n_collab")

    if st.button("Generate Collaborative Recommendations"):
        try:
            data = safe_get(
                f"/recommend/user/{int(user_id)}?top_n={top_n_collab}&method={method}"
            )
            display_recommendations(data.get("items", []))
        except requests.exceptions.RequestException as exc:
            st.error("Failed to generate collaborative recommendations.")
            st.exception(exc)

with hybrid_tab:
    st.subheader("Hybrid Recommendations")
    st.write("Combine movie similarity and user rating behavior.")

    hybrid_user_id = st.number_input("User ID", min_value=1, value=1, step=1, key="hybrid_user")
    hybrid_movie = st.selectbox("Select a movie", movie_titles, key="hybrid_movie")

    content_weight = st.slider("Content weight", 0.0, 1.0, 0.5, 0.1)
    collaborative_weight = st.slider("Collaborative weight", 0.0, 1.0, 0.5, 0.1)
    top_n_hybrid = st.slider("Number of recommendations", 1, 20, 10, key="top_n_hybrid")

    if st.button("Generate Hybrid Recommendations"):
        try:
            encoded_title = quote(hybrid_movie, safe="")
            endpoint = (
                f"/recommend/hybrid/{int(hybrid_user_id)}/{encoded_title}"
                f"?top_n={top_n_hybrid}"
                f"&content_weight={content_weight}"
                f"&collaborative_weight={collaborative_weight}"
            )
            data = safe_get(endpoint)
            display_recommendations(data.get("items", []))
        except requests.exceptions.RequestException as exc:
            st.error("Failed to generate hybrid recommendations.")
            st.exception(exc)

with rating_tab:
    st.subheader("Submit a Rating")
    st.write("Store a user rating in the SQLite database.")

    rating_user_id = st.number_input("Rating User ID", min_value=1, value=1, step=1)
    rating_movie = st.selectbox("Movie to rate", movie_titles, key="rating_movie")
    rating_value = st.slider("Rating", 0.5, 5.0, 4.0, 0.5)

    if st.button("Submit Rating"):
        payload = {
            "user_id": int(rating_user_id),
            "movie_id": int(movie_id_by_title[rating_movie]),
            "rating": float(rating_value),
        }
        try:
            response = requests.post(f"{API_URL}/rating", json=payload, timeout=10)
            response.raise_for_status()
            st.success("Rating submitted successfully!")
            st.json(response.json())
        except requests.exceptions.RequestException as exc:
            st.error("Failed to submit rating.")
            st.exception(exc)
