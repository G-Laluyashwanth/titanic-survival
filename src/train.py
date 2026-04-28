"""Train the titanic classification pipeline and save the trained model."""

import warnings
warnings.filterwarnings("ignore")


from pathlib import Path

import joblib
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from src.data import(
    get_train_val_split,
    load_train,
    split_features_and_targets
)

from src.pipeline import build_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_PATH = WEIGHTS_DIR / "model.joblib"


def train_and_save():

    train = load_train()
    X, y = split_features_and_targets(train)
    X_train, X_val, y_train, y_val = get_train_val_split(X, y)

    clf = build_pipeline(X_train)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cr = classification_report(y_val, y_pred)

    print("\n📊 Model Evaluation Metrics")
    print("-" * 35)

    print(f"Accuracy      : {accuracy:.4f}")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    print("\n🧱 Confusion Matrix")
    print(cm)

    print("\n📄 Classification Report")
    print(cr)


    WEIGHTS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    return clf



# if __name__ == "__main__":
#     train_and_save(tune=True)