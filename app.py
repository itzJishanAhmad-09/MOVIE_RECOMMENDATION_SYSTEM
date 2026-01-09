import streamlit as st
import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")


@st.cache_data
def prepare_model():
    if os.path.exists("movies.pkl") and os.path.exists("similarity.pkl"):
        with open("movies.pkl", "rb") as f:
            movies = pickle.load(f)
        with open("similarity.pkl", "rb") as f:
            similarity = pickle.load(f)
        return movies, similarity

    movies = pd.read_csv("tmdb_5000_movies.csv.zip", compression="zip")
    credits = pd.read_csv("tmdb_5000_credits.csv.zip", compression="zip")

    movies = movies.merge(credits, on="title")
    movies = movies[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]
    movies.dropna(inplace=True)

    def convert(text):
        return [i["name"] for i in ast.literal_eval(text)]

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)

    def convert_cast(text):
        return [i["name"] for i in ast.literal_eval(text)[:3]]

    movies["cast"] = movies["cast"].apply(convert_cast)

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i["job"] == "Director":
                return [i["name"]]
        return []

    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    with open("movies.pkl", "wb") as f:
        pickle.dump(movies, f)
    with open("similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    return movies, similarity


movies, similarity = prepare_model()


st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)


def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]
    movie_indices = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]
    return [movies.iloc[i[0]].title for i in movie_indices]


if st.button("Recommend"):
    for movie in recommend(selected_movie):
        st.write("👉", movie)
