import os
import pickle
import threading
from sklearn.svm import SVC

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(_MODULE_DIR, "..", "models", "svm_classifier.pkl")

_lock = threading.RLock()
_model = None


def get_model():
    global _model
    with _lock:
        if _model is None:
            _model = SVC(kernel="linear", C=1.0, probability=True)
        return _model


def train(X, y):
    model = get_model()
    model.fit(X, y)
    return model


def predict(vector):
    model = get_model()
    intent = model.predict(vector)[0]
    probas = model.predict_proba(vector)[0]
    confidence = float(max(probas))
    return intent, confidence


def save(path=CLASSIFIER_PATH):
    with _lock:
        with open(path, "wb") as f:
            pickle.dump(get_model(), f)


def load(path=CLASSIFIER_PATH):
    global _model
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    with _lock:
        _model = loaded