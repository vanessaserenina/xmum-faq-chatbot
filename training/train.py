import os
import sys
import json
import csv
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import preprocessor, vectorizer, classifier, retriever

CORPUS_PATH = "data/uni_faq_corpus.json"
FLAT_CSV_PATH = "data/corpus_flat.csv"
MODELS_DIR = "models"


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

    print("Loading corpus...")
    rows = load_and_flatten(CORPUS_PATH)
    print(f"Loaded {len(rows)} question rows across {len(set(r['intent'] for r in rows))} intents.")

    export_flat_csv(rows, FLAT_CSV_PATH)
    print(f"Flat CSV exported to {FLAT_CSV_PATH}")

    print("Preprocessing questions...")
    preprocessed_questions = [preprocessor.preprocess(r["question"]) for r in rows]
    intents = [r["intent"] for r in rows]

    print("Fitting TF-IDF and transforming corpus...")
    X = vectorizer.fit_transform(preprocessed_questions)
    vectorizer.save()
    print(f"TF-IDF vectorizer saved. Matrix shape: {X.shape}")

    print("Training SVM classifier...")
    classifier.train(X, intents)
    classifier.save()
    print("SVM classifier saved.")

    print("Saving FAQ vectors and metadata...")
    faq_metadata = [
        {
            "id": r["id"],
            "intent": r["intent"],
            "question": r["question"],
            "answer": r["answer"]
        }
        for r in rows
    ]
    retriever.save(X, faq_metadata)
    print("FAQ vectors and metadata saved.")

    intent_counts = {}
    for intent in intents:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print("\nTraining complete.")
    print(f"Total: {len(rows)} questions across {len(intent_counts)} intents")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()
