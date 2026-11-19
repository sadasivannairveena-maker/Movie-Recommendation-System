# Movie-Recommendation-System

A beginner-friendly but production-style movie recommendation system built with Python.  
The project supports three recommendation approaches:

1. Content-based filtering
2. Collaborative filtering
3. Hybrid recommendation system

It includes data preprocessing, model training, evaluation, a FastAPI backend, a Streamlit web interface, and a SQLite database for storing user ratings.

---

## Features

- Load and clean the MovieLens small dataset
- Remove duplicates and handle missing values
- Basic exploratory data analysis summary
- Genre preprocessing for content-based filtering
- User-item rating matrix for collaborative filtering
- Content-based recommendations using TF-IDF and cosine similarity
- Collaborative recommendations using:
  - KNN user-user similarity
  - TruncatedSVD matrix factorization
- Hybrid recommendations using weighted scoring
- Model saving with Joblib
- REST API using FastAPI
- Streamlit frontend UI
- SQLite database for submitted user ratings
- Evaluation using RMSE, MAE, Precision@K, and Recall@K

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- SciPy
- FastAPI
- Uvicorn
- Streamlit
- SQLite
- Joblib
- Requests

---

## Dataset Information

This project uses the **MovieLens Latest Small** dataset.

Required files:

- `movies.csv`
- `ratings.csv`

Required columns:

`movies.csv`

```text
movieId,title,genres
```

`ratings.csv`

```text
userId,movieId,rating,timestamp
```

MovieLens small contains about 100,000 ratings and around 9,000 movies, which makes it easy to run locally on a normal laptop.

---

## Folder Structure

```text
Movie-Recommendation-System/
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── models/
│   ├── content_model.pkl
│   ├── collaborative_model.pkl
│   └── hybrid_model.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── content_model.py
│   ├── collaborative_model.py
│   ├── hybrid_model.py
│   ├── evaluation.py
│   └── database.py
│
├── api/
│   └── app.py
│
├── app/
│   └── streamlit_app.py
│
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

Note: The `.pkl` model files are generated after running `python train_model.py`. They are ignored by Git because model files can be large.

---

## Setup Instructions

### 1. Clone or create the project folder

```bash
git clone https://github.com/your-username/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

If you are creating the project manually, create the folder structure shown above and copy the code files into the correct locations.

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Download the MovieLens Dataset

1. Go to the official GroupLens MovieLens latest dataset page:

```text
https://grouplens.org/datasets/movielens/latest/
```

2. Download:

```text
ml-latest-small.zip
```

3. Extract the ZIP file.

4. Copy these two files into the `data/` folder:

```text
movies.csv
ratings.csv
```

Your folder should look like this:

```text
data/
├── movies.csv
└── ratings.csv
```

---

## How to Train the Models

Run this command from the project root:

```bash
python train_model.py
```

This will:

1. Load the dataset
2. Clean movies and ratings
3. Create an EDA summary
4. Train the content-based model
5. Train the collaborative filtering model
6. Train the hybrid model
7. Evaluate the collaborative model
8. Save models into the `models/` folder

Generated files:

```text
models/content_model.pkl
models/collaborative_model.pkl
models/hybrid_model.pkl
models/evaluation_results.json
models/eda_summary.json
```

---

## How to Run the FastAPI Backend

Run:

```bash
uvicorn api.app:app --reload
```

Open the API docs in your browser:

```text
http://127.0.0.1:8000/docs
```

---

## How to Run the Streamlit App

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app will open in your browser.

Important: Keep the FastAPI backend running while using Streamlit.

---

## API Endpoints

### 1. Welcome Endpoint

```http
GET /
```

Returns a welcome message.

---

### 2. Get Movies

```http
GET /movies
```

Optional query parameters:

```text
limit=100
search=toy
```

Example:

```http
GET /movies?limit=20&search=toy
```

---

### 3. Content-Based Recommendation

```http
GET /recommend/movie/{movie_title}
```

Example:

```http
GET /recommend/movie/Toy%20Story
```

Example response:

```json
{
    "movie": "Toy Story",
    "recommendations": [
        "Antz (1998)",
        "Toy Story 2 (1999)",
        "Adventures of Rocky and Bullwinkle, The (2000)"
    ]
}
```

---

### 4. Collaborative Filtering Recommendation

```http
GET /recommend/user/{user_id}
```

Example:

```http
GET /recommend/user/1
```

Optional parameters:

```text
top_n=10
method=svd
method=knn
```

Example:

```http
GET /recommend/user/1?top_n=10&method=svd
```

---

### 5. Hybrid Recommendation

```http
GET /recommend/hybrid/{user_id}/{movie_title}
```

Example:

```http
GET /recommend/hybrid/1/Toy%20Story
```

Optional parameters:

```text
content_weight=0.5
collaborative_weight=0.5
top_n=10
```

---

### 6. Submit Rating

```http
POST /rating
```

Request body:

```json
{
    "user_id": 1,
    "movie_id": 10,
    "rating": 4.5
}
```

Ratings are stored in:

```text
data/user_ratings.db
```

---

## Explanation of Recommendation Algorithms

### 1. Content-Based Filtering

Content-based filtering recommends movies similar to a movie selected by the user.

In this project:

1. Movie genres are cleaned.
2. Genres are converted into text features.
3. TF-IDF Vectorizer converts genre text into numerical vectors.
4. Cosine similarity compares movies.
5. The system returns movies with the highest similarity scores.

Example:

If the user selects `Toy Story`, the model looks for movies with similar genres such as animation, adventure, children, comedy, and fantasy.

---

### 2. Collaborative Filtering

Collaborative filtering recommends movies based on user rating patterns.

This project uses two approaches:

#### KNN Collaborative Filtering

1. A user-item rating matrix is created.
2. Similar users are found using cosine similarity.
3. Movies liked by similar users are recommended.

#### TruncatedSVD Matrix Factorization

1. The user-item matrix is decomposed into latent factors.
2. The model predicts ratings for movies the user has not watched.
3. Movies with the highest predicted ratings are recommended.

---

### 3. Hybrid Recommendation System

The hybrid model combines content-based and collaborative filtering.

Formula:

```text
final_score = (content_score * content_weight) + (collaborative_score * collaborative_weight)
```

Default weights:

```text
content_weight = 0.5
collaborative_weight = 0.5
```

This gives better recommendations because it considers both:

- similarity between movies
- user rating behavior

---

## Evaluation

The project evaluates the collaborative filtering model using:

### RMSE

Root Mean Squared Error measures how far predicted ratings are from actual ratings. Lower is better.

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual ratings. Lower is better.

### Precision@K

Precision@K measures how many of the top K recommended movies are actually relevant to the user.

### Recall@K

Recall@K measures how many relevant movies were successfully recommended in the top K results.

After training, evaluation results are saved to:

```text
models/evaluation_results.json
```

Example:

```json
{
    "rmse": 1.02,
    "mae": 0.78,
    "precision_at_10": 0.04,
    "recall_at_10": 0.01
}
```

Your exact results may be different because of the train/test split and local environment.

---

## Screenshots

Add your screenshots here after running the project.

### Streamlit Home Page

```text
screenshots/streamlit_home.png
```

### Content-Based Recommendations

```text
screenshots/content_recommendations.png
```

### API Documentation

```text
screenshots/api_docs.png
```

---

## Future Improvements

- Add movie posters using TMDB API
- Add login and user accounts
- Retrain models automatically after new ratings
- Add item-based collaborative filtering
- Add Surprise SVD
- Add Docker support
- Add unit tests with pytest
- Deploy API on Render or Railway
- Deploy Streamlit frontend on Streamlit Community Cloud
- Use PostgreSQL instead of SQLite for production deployment

---

## Final Commands

```bash
pip install -r requirements.txt
python train_model.py
uvicorn api.app:app --reload
streamlit run app/streamlit_app.py
```

---

## License

This project is for educational purposes. Check the MovieLens dataset license before using it commercially.
