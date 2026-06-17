import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import preprocessor, vectorizer, classifier, retriever

CONFIDENCE_THRESHOLD = 0.3

FALLBACK_ANSWER = (
    "Sorry, I could not find a relevant answer to your question. "
    "Please contact the XMUM Admissions Office at admissions@xmu.edu.my "
    "for further assistance."
)


def load_pipeline():
    vectorizer.load()
    classifier.load()
    retriever.load()


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

    answer, similarity_score, matched_question, matched_id, matched_intent = retriever.retrieve(
        query_vector, predicted_intent
    )

    if similarity_score < CONFIDENCE_THRESHOLD or predicted_intent == "out_of_scope":
        # Global fallback retrieval
        gb_answer, gb_score, gb_question, gb_id, gb_intent = retriever.retrieve(query_vector, None)
        if gb_score >= CONFIDENCE_THRESHOLD and gb_intent != "out_of_scope":
            answer = gb_answer
            similarity_score = gb_score
            matched_question = gb_question
            matched_id = gb_id
            matched_intent = gb_intent

    if similarity_score >= CONFIDENCE_THRESHOLD and matched_intent != "out_of_scope":
        return {
            "answer": answer,
            "intent": matched_intent,
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
