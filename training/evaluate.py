import os
import sys
import json
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import preprocessor, vectorizer, classifier, retriever

DATA_PATH = "data/corpus_flat.csv"
EVAL_DIR = "results"
os.makedirs(EVAL_DIR, exist_ok=True)

def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = []
        intents = []
        for row in reader:
            questions.append(row["question"])
            intents.append(row["intent"])
    return questions, intents

def main():
    print("Loading preprocessed data...")
    questions, intents = load_data()

    print("Preprocessing and vectorizing...")
    vectorizer.load()
    X = vectorizer.transform([preprocessor.preprocess(q) for q in questions])

    print("Loading trained SVM classifier...")
    classifier.load()

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, intents, test_size=0.2, random_state=42, stratify=intents
    )

    print("Evaluating on test set...")
    y_pred = [classifier.predict(vec)[0] for vec in X_test]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    with open(os.path.join(EVAL_DIR, "evaluation_results.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")

    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(intents)))
    with open(os.path.join(EVAL_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f)

    print("Evaluation results saved to results/ directory.")

if __name__ == "__main__":
    main()