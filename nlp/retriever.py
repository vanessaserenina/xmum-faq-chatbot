import os
import pickle
import threading
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_VECTORS_PATH = os.path.join(_MODULE_DIR, "..", "models", "faq_vectors.pkl")
FAQ_METADATA_PATH = os.path.join(_MODULE_DIR, "..", "models", "faq_metadata.pkl")

_lock = threading.Lock()
_faq_vectors = None
_faq_metadata = None


def load(vectors_path=FAQ_VECTORS_PATH, metadata_path=FAQ_METADATA_PATH):
    global _faq_vectors, _faq_metadata
    with open(vectors_path, "rb") as f:
        loaded_vectors = pickle.load(f)
    with open(metadata_path, "rb") as f:
        loaded_metadata = pickle.load(f)
    with _lock:
        _faq_vectors = loaded_vectors
        _faq_metadata = loaded_metadata


def save(vectors, metadata, vectors_path=FAQ_VECTORS_PATH, metadata_path=FAQ_METADATA_PATH):
    with open(vectors_path, "wb") as f:
        pickle.dump(vectors, f)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def retrieve(query_vector, predicted_intent=None):
    with _lock:
        faq_vectors = _faq_vectors
        faq_metadata = _faq_metadata

    if predicted_intent == "out_of_scope":
        return None, 0.0, None, None, None

    if predicted_intent is not None:
        indices = [
            i for i, m in enumerate(faq_metadata)
            if m["intent"] == predicted_intent
        ]
    else:
        indices = list(range(len(faq_metadata)))

    if not indices:
        return None, 0.0, None, None, None

    filtered_vectors = faq_vectors[indices]
    scores = cosine_similarity(query_vector, filtered_vectors)[0]
    best_local = int(np.argmax(scores))
    best_global = indices[best_local]
    best_score = float(scores[best_local])

    matched = faq_metadata[best_global]
    return (
        matched["answer"],
        best_score,
        matched["question"],
        matched["id"],
        matched["intent"]
    )