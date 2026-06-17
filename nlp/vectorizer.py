import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

_vectorizer = None


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
    return _vectorizer


def fit_transform(docs):
    vec = get_vectorizer()
    matrix = vec.fit_transform(docs)
    return matrix


def transform(query):
    vec = get_vectorizer()
    if isinstance(query, str):
        return vec.transform([query])
    return vec.transform(query)


def save(path=VECTORIZER_PATH):
    with open(path, "wb") as f:
        pickle.dump(get_vectorizer(), f)


def load(path=VECTORIZER_PATH):
    global _vectorizer
    with open(path, "rb") as f:
        _vectorizer = pickle.load(f)
