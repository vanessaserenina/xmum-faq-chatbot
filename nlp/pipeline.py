import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from nlp import preprocessor, vectorizer, classifier, retriever

CONFIDENCE_THRESHOLD = 0.3

FALLBACK_ANSWER = (
    "Sorry, I could not find a relevant answer to your question. "
    "Please contact the XMUM Admissions Office at admissions@xmu.edu.my "
    "for further information."
)


def load_pipeline():
    vectorizer.load(os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.pkl"))
    classifier.load(os.path.join(ROOT_DIR, "models", "svm_classifier.pkl"))
    retriever.load(
        os.path.join(ROOT_DIR, "models", "faq_vectors.pkl"),
        os.path.join(ROOT_DIR, "models", "faq_metadata.pkl")
    )


def get_response(user_query):
    cleaned = preprocessor.preprocess(user_query)

    if not cleaned:
        return {
            "answer": FALLBACK_ANSWER,
            "intent": "out_of_scope",
            "confidence": 0.0,
            "matched_question": None,
            "id": None
        }

    query_vector = vectorizer.transform(cleaned)

    predicted_intent, svm_confidence = classifier.predict(query_vector)

    answer, similarity_score, matched_question, matched_id, _ = retriever.retrieve(
        query_vector, predicted_intent
    )

    if similarity_score >= CONFIDENCE_THRESHOLD and predicted_intent != "out_of_scope":
        return {
            "answer": answer,
            "intent": predicted_intent,
            "confidence": round(similarity_score, 4),
            "matched_question": matched_question,
            "id": matched_id
        }

    return {
        "answer": FALLBACK_ANSWER,
        "intent": "out_of_scope",
        "confidence": round(similarity_score, 4),
        "matched_question": None,
        "id": None
    }
