import streamlit as st
import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")

# ---------------- BUILD / LOAD MODEL ----------------
@st.cache_data
def prepare_model():
    if os.path.exists("movies.pkl") and os.path.exists("similarity.pkl"):
        movies = pickle.load(open("movies.pkl", "rb"))
        similarity = pickle.load(open("similarity.pkl", "rb"))
        return movies, similarity

    # LOAD LOCAL FILES (NO DOWNLOAD)
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    def convert_cast(text):
        return [i['name'] for i in ast.literal_eval(text)[:3]]

    movies['cast'] = movies['cast'].apply(convert_cast)

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = (
        movies['overview']
        + movies['genres']
        + movies['keywords']
        + movies['cast']
        + movies['crew']
    )

    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    pickle.dump(movies, open("movies.pkl", "wb"))
    pickle.dump(similarity, open("similarity.pkl", "wb"))

    return movies, similarity


movies, similarity = prepare_model()

# ---------------- STREAMLIT UI ----------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]

if st.button("Recommend"):
    for movie in recommend(selected_movie):
        st.write("👉", movie)
