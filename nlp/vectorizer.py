import os
import pickle
import threading
from sklearn.feature_extraction.text import TfidfVectorizer

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(_MODULE_DIR, "..", "models", "tfidf_vectorizer.pkl")

_lock = threading.RLock()
_vectorizer = None


def get_vectorizer():
    global _vectorizer
    with _lock:
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
    with _lock:
        with open(path, "wb") as f:
            pickle.dump(get_vectorizer(), f)


def load(path=VECTORIZER_PATH):
    global _vectorizer
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    with _lock:
        _vectorizer = loaded