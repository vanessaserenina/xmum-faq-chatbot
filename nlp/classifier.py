import pickle
from sklearn.svm import SVC

CLASSIFIER_PATH = "models/svm_classifier.pkl"

_model = None


def get_model():
    global _model
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
    classes = model.classes_
    confidence = float(max(probas))
    return intent, confidence


def save(path=CLASSIFIER_PATH):
    with open(path, "wb") as f:
        pickle.dump(get_model(), f)


def load(path=CLASSIFIER_PATH):
    global _model
    with open(path, "rb") as f:
        _model = pickle.load(f)
