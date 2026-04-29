"""Train a Titanic classifier and save the trained model.

Supports two model types via command-line flag:
    python -m src.train --model random_forest
    python -m src.train --model logistic
"""

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import (
    get_train_val_split,
    load_train,
    split_features_and_targets,
)
from src.pipeline import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def train_and_save(model_type="random_forest"):
    """
    Train a classifier and save the trained model to weights/.

    Parameters
    ----------
    model_type : str
        Either "random_forest" or "logistic".
    """
    # Load and split
    train = load_train()
    X, y = split_features_and_targets(train)
    X_train, X_val, y_train, y_val = get_train_val_split(X, y)

    # Build and fit pipeline
    print(f"Training model: {model_type}")
    model = build_pipeline(X_train, model_type=model_type)
    model.fit(X_train, y_train)

    # Predictions for evaluation
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    print("\n📊 Model Evaluation Metrics")
    print("-" * 40)
    print(f"Accuracy  : {accuracy_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_val, y_proba):.4f}")
    print(f"Precision : {precision_score(y_val, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_val, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_val, y_pred):.4f}")

    print("\n🧱 Confusion Matrix")
    print(confusion_matrix(y_val, y_pred))

    print("\n📋 Classification Report")
    print(classification_report(y_val, y_pred))

    # Save the trained model — filename includes model type
    WEIGHTS_DIR.mkdir(exist_ok=True)
    model_path = WEIGHTS_DIR / f"model_{model_type}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Titanic classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic"],
        help="Which classifier to train (default: random_forest)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(model_type=args.model)