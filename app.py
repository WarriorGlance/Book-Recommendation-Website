import streamlit as st
import pandas as pd
from pathlib import Path
from book_recommender import ContentBasedRecommender, prepare_books

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_books(data_path, modified_time):
    books = pd.read_csv(data_path)
    books = prepare_books(books)
    books["Image-URL-M"] = books["Image-URL-M"].fillna("")
    books["Average_Rating"] = pd.to_numeric(
        books["Average_Rating"], errors="coerce"
    )
    books["Rating_Count"] = pd.to_numeric(
        books["Rating_Count"], errors="coerce"
    ).fillna(0)
    return books


data_path = Path("enriched_books.csv")
if not data_path.exists():
    data_path = Path("cleaned_books.csv")
books = load_books(str(data_path), data_path.stat().st_mtime)


@st.cache_resource
def create_recommender(data_path, modified_time):
    return ContentBasedRecommender(load_books(data_path, modified_time))


recommender = create_recommender(str(data_path), data_path.stat().st_mtime)

# ---------------- HEADER ----------------

st.title("📚 Book Recommendation System")

st.write(
    "Discover books using genre and description metadata when available, "
    "along with catalog information."
)

st.divider()

# ---------------- BOOK SELECTION ----------------

book_titles = sorted(books["Book-Title"].unique())

selected_book = st.selectbox(
    "Choose a book you like",
    book_titles
)

number_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# ---------------- DISPLAY SELECTED BOOK ----------------

if st.button("Find Recommended Books", type="primary"):

    st.subheader("Recommendations for You")

    recommendations = recommender.recommend(
        selected_book,
        number_of_recommendations
    )

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        columns = st.columns(3)

        for i, (_, book) in enumerate(
            recommendations.iterrows()
        ):
            with columns[i % 3]:

                with st.container(border=True):

                    image_url = book["Image-URL-M"]

                    if image_url.startswith(("http://", "https://")):
                        st.image(image_url, width=140)

                    st.markdown(f"**{book['Book-Title']}**")

                    st.caption(
                        f"By {book['Book-Author']}"
                    )

                    if pd.notna(book["Average_Rating"]):
                        st.write(
                            f"⭐ Rating: {book['Average_Rating']}"
                        )

                    st.caption(
                        f"Content similarity: "
                        f"{book['Similarity']:.2f}"
                    )

                    if book["Genres"]:
                        st.caption(f"Genre: {book['Genres']}")

                    st.caption(
                        f"Publisher: {book['Publisher'] or 'Unknown'}"
                    )

st.divider()

st.caption(
    "Similarity uses available catalog fields, descriptions, genres, and "
    "writing samples. A description or genre is not a substitute for an "
    "actual writing sample."
)
