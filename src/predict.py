"""Load a trained Titanic model and generate a Kaggle submission file."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data import load_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def generate_submission(model_type="random_forest"):
    """Load a trained model and create a Kaggle submission CSV."""
    model_path = WEIGHTS_DIR / f"model_{model_type}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. "
            f"Run `python -m src.train --model {model_type}` first."
        )

    print(f"Loading model: {model_type}")
    model = joblib.load(model_path)
    test = load_test()

    test_ids = test["PassengerId"]
    X_test = test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": predictions,
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
    print(f"Predictions: {predictions.sum()} survived / {len(predictions)} total")

    return submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train Titanic classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic", "xgboost"],   # ← add xgboost
        help="Which classifier to train (default: random_forest)",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    generate_submission(model_type=args.model)