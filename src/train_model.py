"""
train_model.py

Trains a Random Forest classifier to flag network connections as
"normal" or "attack" based on connection-level features (NSL-KDD-style
schema). Saves the trained pipeline (preprocessing + model) and a
confusion-matrix plot to disk.

Usage:
    python src/train_model.py
"""

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/network_traffic.csv"
MODEL_PATH = "models/nids_model.joblib"

CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "num_failed_logins", "land", "wrong_fragment", "urgent",
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="attack")
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (attack class): {f1:.4f}")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=["normal", "attack"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "attack"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Network Intrusion Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to results/confusion_matrix.png")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    with open("results/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 (attack class): {f1:.4f}\n\n{report}\n")


if __name__ == "__main__":
    main()
