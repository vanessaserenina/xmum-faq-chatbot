import os
import sys
import json
import csv
import numpy as np
import pickle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from nlp import preprocessor, vectorizer, classifier, retriever

CORPUS_PATH = os.path.join(ROOT_DIR, "data", "uni_faq_corpus.json")
FLAT_CSV_PATH = os.path.join(ROOT_DIR, "data", "corpus_flat.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def load_and_flatten(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for block in data["corpus"]:
        intent = block["intent"]
        for qa in block["qa_pairs"]:
            for question in qa["questions"]:
                rows.append({
                    "id": qa["id"],
                    "intent": intent,
                    "question": question,
                    "answer": qa["answer"]
                })
    return rows


def export_flat_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "intent", "question", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading corpus...", flush=True)
    rows = load_and_flatten(CORPUS_PATH)
    print(f"Loaded {len(rows)} question rows across {len(set(r['intent'] for r in rows))} intents.", flush=True)

    export_flat_csv(rows, FLAT_CSV_PATH)
    print(f"Flat CSV exported to {FLAT_CSV_PATH}", flush=True)

    print("Preprocessing questions...", flush=True)
    preprocessed_questions = [preprocessor.preprocess(r["question"]) for r in rows]
    intents = [r["intent"] for r in rows]

    print("Entering fit_transform", flush=True)
    X = vectorizer.fit_transform(preprocessed_questions)
    print("fit_transform finished", flush=True)

    print("About to save vectorizer", flush=True)
    vectorizer.save(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("Vectorizer saved", flush=True)

    print("Training SVM classifier...", flush=True)
    classifier.train(X, intents)
    classifier.save(os.path.join(MODELS_DIR, "svm_classifier.pkl"))
    print("SVM classifier saved.", flush=True)

    print("Saving FAQ vectors and metadata...", flush=True)
    faq_metadata = [
        {
            "id": r["id"],
            "intent": r["intent"],
            "question": r["question"],
            "answer": r["answer"]
        }
        for r in rows
    ]
    retriever.save(
        X,
        faq_metadata,
        os.path.join(MODELS_DIR, "faq_vectors.pkl"),
        os.path.join(MODELS_DIR, "faq_metadata.pkl")
    )
    print("FAQ vectors and metadata saved.", flush=True)

    intent_counts = {}
    for intent in intents:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print("\nTraining complete.", flush=True)
    print(f"Total: {len(rows)} questions across {len(intent_counts)} intents", flush=True)
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}", flush=True)


if __name__ == "__main__":
    main()