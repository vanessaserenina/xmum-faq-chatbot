import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FAQ_VECTORS_PATH = "models/faq_vectors.pkl"
FAQ_METADATA_PATH = "models/faq_metadata.pkl"

_faq_vectors = None
_faq_metadata = None


def load(vectors_path=FAQ_VECTORS_PATH, metadata_path=FAQ_METADATA_PATH):
    global _faq_vectors, _faq_metadata
    with open(vectors_path, "rb") as f:
        _faq_vectors = pickle.load(f)
    with open(metadata_path, "rb") as f:
        _faq_metadata = pickle.load(f)


def save(vectors, metadata, vectors_path=FAQ_VECTORS_PATH, metadata_path=FAQ_METADATA_PATH):
    with open(vectors_path, "wb") as f:
        pickle.dump(vectors, f)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def retrieve(query_vector, predicted_intent=None):
    if predicted_intent is not None:
        indices = [
            i for i, m in enumerate(_faq_metadata)
            if m["intent"] == predicted_intent
        ]
    else:
        indices = list(range(len(_faq_metadata)))

    if not indices:
        return None, 0.0, None, None, None

    filtered_vectors = _faq_vectors[indices]

    scores = cosine_similarity(query_vector, filtered_vectors)[0]
    best_local = int(np.argmax(scores))
    best_global = indices[best_local]
    best_score = float(scores[best_local])

    matched = _faq_metadata[best_global]
    return (
        matched["answer"],
        best_score,
        matched["question"],
        matched["id"],
        matched["intent"]
    )
