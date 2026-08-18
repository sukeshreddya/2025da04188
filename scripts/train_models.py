"""Train required classifiers on preprocessed credit card dataset.

Saves trained models to `model/` and metrics to `results/model_metrics.csv`.

Usage:
  python scripts/train_models.py --input data/eda_creditcard/preprocessed.csv
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
import joblib


def train_and_evaluate(input_csv: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "model")
    os.makedirs(models_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    # Assume target column is 'Class' or last column
    target = "Class" if "Class" in df.columns else df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state),
    }

    records = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = np.nan

        rec = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": auc,
            "mcc": matthews_corrcoef(y_test, y_pred),
        }
        records.append(rec)

        # save model
        joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))

        # save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(results_dir, f"cm_{name}.csv")
        pd.DataFrame(cm).to_csv(cm_path, index=False)

    results_df = pd.DataFrame(records).set_index("model")
    results_df.to_csv(os.path.join(results_dir, "model_metrics.csv"))
    print("Training complete. Metrics saved to:", os.path.join(results_dir, "model_metrics.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eda_creditcard/preprocessed.csv", help="Preprocessed CSV input")
    parser.add_argument("--output_dir", default=".", help="Output base dir to store model/ and results/")
    args = parser.parse_args()
    train_and_evaluate(args.input, args.output_dir)


if __name__ == "__main__":
    main()
