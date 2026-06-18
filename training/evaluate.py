import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from nlp import preprocessor, vectorizer as vec_module

CORPUS_PATH = "data/uni_faq_corpus.json"
REPORTS_DIR = "training"


def load_and_flatten(path):
    with open(path, "r", encoding="utf-8") as f:
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


def main():
    print("Loading and flattening corpus...")
    rows = load_and_flatten(CORPUS_PATH)
    questions = [r["question"] for r in rows]
    intents = [r["intent"] for r in rows]

    print("Preprocessing...")
    preprocessed = [preprocessor.preprocess(q) for q in questions]

    print("Fitting TF-IDF for evaluation (fresh fit, not the saved model)...")
    X = vec_module.fit_transform(preprocessed)
    y = intents

    model = SVC(kernel="linear", C=1.0, probability=True)

    print("\nRunning 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]
        m = SVC(kernel="linear", C=1.0, probability=True)
        m.fit(X_train, y_train)
        acc = m.score(X_test, y_test)
        fold_accuracies.append(acc)
        print(f"  Fold {fold}: {acc:.4f}")

    print(f"\nMean CV Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Dev:          {np.std(fold_accuracies):.4f}")

    print("\nGenerating classification report (cross-val predictions)...")
    y_pred = cross_val_predict(model, X, y, cv=skf)
    print(classification_report(y, y_pred, zero_division=0))

    print("Saving confusion matrix...")
    labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues"
    )
    plt.title("SVM Intent Classifier - Confusion Matrix (5-Fold CV)")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    print("\nTesting confidence threshold values...")
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    import pickle

    model.fit(X, y)

    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx_list, test_idx_list = [], []
    for tr, te in skf2.split(X, y):
        train_idx_list.append(tr)
        test_idx_list.append(te)

    tr = train_idx_list[0]
    te = test_idx_list[0]
    X_train_t = X[tr]
    y_train_t = [y[i] for i in tr]
    X_test_t = X[te]
    y_test_t = [y[i] for i in te]

    thresh_model = SVC(kernel="linear", C=1.0, probability=True)
    thresh_model.fit(X_train_t, y_train_t)

    predicted_intents = thresh_model.predict(X_test_t)

    scores = []
    for i, q_vec in enumerate(X_test_t):
        pred_intent = predicted_intents[i]
        mask = [j for j, label in enumerate(y_train_t) if label == pred_intent]
        if not mask:
            scores.append(0.0)
            continue
        filtered = X_train_t[mask]
        sims = cos_sim(q_vec, filtered)[0]
        scores.append(float(np.max(sims)))

    print(f"\n{'Threshold':<12} {'Fallback %':<14} {'Answered %'}")
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        fallback = sum(1 for s in scores if s < threshold)
        pct_fallback = fallback / len(scores) * 100
        pct_answered = 100 - pct_fallback
        print(f"{threshold:<12} {pct_fallback:<14.1f} {pct_answered:.1f}")

if __name__ == "__main__":
    main()
