import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURE_WEIGHTS = {
    "Catalog_Text": 0.15,
    "Description": 0.45,
    "Genres": 0.25,
    "Writing_Sample": 0.15,
}


def _text_column(books: pd.DataFrame, column: str) -> pd.Series:
    if column in books:
        return books[column].fillna("").astype(str).str.strip()
    return pd.Series("", index=books.index, dtype="string")


def prepare_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.dropna(subset=["Book-Title", "Book-Author"]).copy()
    for column in ("Publisher", "Description", "Genres", "Writing_Sample"):
        if column not in books:
            books[column] = ""
        books[column] = books[column].fillna("").astype(str)

    books["Normalised_Title"] = (
        books["Book-Title"]
        .astype(str)
        .str.casefold()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return books.drop_duplicates(
        subset=["Normalised_Title"]
    ).reset_index(drop=True)


class ContentBasedRecommender:
    def __init__(self, books: pd.DataFrame):
        self.books = prepare_books(books)
        self.matrices = {}
        self.available = {}

        self.books["Catalog_Text"] = (
            self.books["Book-Title"].astype(str)
            + " "
            + self.books["Book-Author"].astype(str)
            + " "
            + self.books["Publisher"]
        )

        for column in FEATURE_WEIGHTS:
            texts = _text_column(self.books, column)
            self.available[column] = texts.ne("").to_numpy()
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                dtype=np.float32,
            )
            analyzer = vectorizer.build_analyzer()
            has_tokens = self.available[column].any() and any(
                analyzer(text) for text in texts[self.available[column]]
            )
            if has_tokens:
                self.matrices[column] = vectorizer.fit_transform(texts)
            else:
                self.available[column] = np.zeros(len(self.books), dtype=bool)
                self.matrices[column] = csr_matrix(
                    (len(self.books), 0), dtype=np.float32
                )

        self.title_lookup = {
            title.casefold(): index
            for index, title in enumerate(self.books["Book-Title"])
        }

    def recommend(self, book_title: str, number: int = 5) -> pd.DataFrame:
        book_index = self.title_lookup.get(book_title.casefold())
        if book_index is None or number < 1:
            return pd.DataFrame()

        numerator = np.zeros(len(self.books), dtype=np.float32)
        denominator = np.zeros(len(self.books), dtype=np.float32)

        for column, weight in FEATURE_WEIGHTS.items():
            if not self.available[column][book_index]:
                continue

            similarities = (
                self.matrices[column].getrow(book_index)
                @ self.matrices[column].T
            ).toarray().ravel()
            candidates_with_feature = self.available[column]
            numerator[candidates_with_feature] += (
                similarities[candidates_with_feature] * weight
            )
            denominator[candidates_with_feature] += weight

        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        different_titles = (
            self.books["Normalised_Title"].to_numpy()
            != self.books.iloc[book_index]["Normalised_Title"]
        )
        candidate_indices = np.flatnonzero(different_titles & (denominator > 0))
        if candidate_indices.size == 0:
            return pd.DataFrame()

        top_count = min(number, candidate_indices.size)
        top_candidates = candidate_indices[
            np.argpartition(scores[candidate_indices], -top_count)[-top_count:]
        ]
        ranked_indices = top_candidates[
            np.argsort(scores[top_candidates])[::-1]
        ]

        recommendations = self.books.iloc[ranked_indices].copy()
        recommendations["Similarity"] = scores[ranked_indices]
        return recommendations
